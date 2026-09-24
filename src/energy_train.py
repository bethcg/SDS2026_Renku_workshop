"""Train an hourly energy-demand forecasting model (GEO4CIVHIC demo sites).

Uses the mounted Zenodo dataset doi:10.5281/zenodo.10568762:
an 8760-row annual load profile with columns [Hour, Load_kW].

The model predicts the load for a given hour from calendar features
(hour of day, day of week, month, weekend) plus lagged load values
(previous hour, same hour yesterday, same hour last week). The last part of
the year is held out as a chronological test set, so there is no leakage from
the future into training.

Run it as a Renku session command or a one-off job:

    python src/energy_train.py
    python src/energy_train.py --model gbr --test-days 30 --output-dir /home/renku/work/outputs/energy
    OUTPUT_DIR=/home/renku/work/outputs/energy python src/energy_train.py

Outputs:
  If an output directory is given (--output-dir, or the OUTPUT_DIR /
  ENERGY_OUTPUT_DIR environment variable), these files are written there:
    energy_model.joblib   fitted model + feature list, ready for inference
    metrics.json          MAE / RMSE / MAPE / R2 for the model and a naive baseline
    test_predictions.csv  timestamp, actual and predicted load on the test period
  If no output directory is given, nothing is written to disk: the metrics and
  the test predictions are printed to stdout (the job logs) instead, and the
  model itself is not persisted.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Same default location as energy_app.py (Renku data connector mount point).
DEFAULT_DATA_PATH = (
    "/home/renku/work/energy-demand-of-geo4civhic-de-doi-10.5281-zenodo.10568762/"
    "Energy demand_GEO4CIVHIC demo sites.xlsx"
)
LAGS = {"lag_1h": 1, "lag_24h": 24, "lag_168h": 168}
FEATURES = ["hour", "dayofweek", "month", "is_weekend", *LAGS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-path", default=os.environ.get("ENERGY_DATA_PATH", DEFAULT_DATA_PATH))
    p.add_argument("--output-dir",
                   default=os.environ.get("OUTPUT_DIR") or os.environ.get("ENERGY_OUTPUT_DIR"),
                   help="where to save model/metrics/predictions; if omitted, results go to the logs only")
    p.add_argument("--model", choices=["rf", "gbr"], default="rf",
                   help="rf = RandomForest, gbr = GradientBoosting")
    p.add_argument("--test-days", type=int, default=28,
                   help="number of final days held out for testing")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_data(path: str) -> pd.DataFrame:
    """Load the load profile exactly as energy_app.py does."""
    if not os.path.exists(path):
        sys.exit(f"Data not found at: {path}\n"
                 "Check the Renku data connector mount or pass --data-path.")
    df = pd.read_excel(path)
    df = df.iloc[:, :2]
    df.columns = ["Hour", "Load_kW"]
    df["Load_kW"] = pd.to_numeric(df["Load_kW"], errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["Hour"] - 1, unit="h", origin="2026-01-01")
    return df.dropna(subset=["Load_kW"]).sort_values("Timestamp").reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = out["Timestamp"].dt
    out["hour"] = ts.hour
    out["dayofweek"] = ts.dayofweek
    out["month"] = ts.month
    out["is_weekend"] = (ts.dayofweek >= 5).astype(int)
    for name, lag in LAGS.items():
        out[name] = out["Load_kW"].shift(lag)
    return out.dropna(subset=list(LAGS)).reset_index(drop=True)


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    nonzero = y_true != 0
    return {
        "mae_kW": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse_kW": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "mape_pct": round(float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        print("No output directory set (--output-dir / OUTPUT_DIR): results will be logged only.")

    print(f"Loading data from {args.data_path}")
    data = build_features(load_data(args.data_path))

    split = data["Timestamp"].max() - pd.Timedelta(days=args.test_days)
    train, test = data[data["Timestamp"] <= split], data[data["Timestamp"] > split]
    if train.empty or test.empty:
        sys.exit("Train/test split is empty - reduce --test-days.")
    print(f"Train: {len(train)} rows (until {split:%Y-%m-%d %H:%M}), test: {len(test)} rows")

    if args.model == "rf":
        model = RandomForestRegressor(n_estimators=args.n_estimators, min_samples_leaf=2,
                                      n_jobs=-1, random_state=args.seed)
    else:
        model = GradientBoostingRegressor(n_estimators=args.n_estimators, learning_rate=0.05,
                                          max_depth=4, random_state=args.seed)

    t0 = time.time()
    model.fit(train[FEATURES], train["Load_kW"])
    train_seconds = round(time.time() - t0, 2)

    pred = model.predict(test[FEATURES])
    results = {
        "model": args.model,
        "params": model.get_params(),
        "data_path": args.data_path,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_start": str(test["Timestamp"].min()),
        "train_seconds": train_seconds,
        "model_metrics": metrics(test["Load_kW"], pred),
        # Naive seasonal baseline: "same hour last week"
        "baseline_lag_168h_metrics": metrics(test["Load_kW"], test["lag_168h"]),
    }
    if hasattr(model, "feature_importances_"):
        results["feature_importance"] = {
            f: round(float(v), 4) for f, v in sorted(
                zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
        }

    predictions = pd.DataFrame({"Timestamp": test["Timestamp"], "actual_kW": test["Load_kW"],
                                "predicted_kW": pred.round(3)})

    print(f"Model:    {results['model_metrics']}")
    print(f"Baseline: {results['baseline_lag_168h_metrics']}")

    if out_dir:
        joblib.dump({"model": model, "features": FEATURES, "lags": LAGS}, out_dir / "energy_model.joblib")
        (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
        predictions.to_csv(out_dir / "test_predictions.csv", index=False)
        print(f"Saved model, metrics and predictions to {out_dir}")
    else:
        print("\n===== metrics.json =====")
        print(json.dumps(results, indent=2, default=str))
        print("\n===== test_predictions.csv =====")
        print(predictions.to_csv(index=False), end="")
        print("\n===== model =====")
        print(model)
        print("Model not saved to disk (set --output-dir or OUTPUT_DIR to persist it).")


if __name__ == "__main__":
    main()
