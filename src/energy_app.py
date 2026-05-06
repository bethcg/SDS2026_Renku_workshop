import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Energy Demand Dashboard", layout="wide")

st.title("⚡ GEO4CIVHIC demo sites: Energy demand")
st.markdown("This dashboard visualizes annual load profiles from the Renku-connected dataset stemming from Zenodo.")


# 1. Path Management
# Update your data path accourding to the mount point in your project
# https://zenodo.org/records/10568762
DATA_PATH = '/home/renku/work/energy-demand-of-geo4civhic-de-doi-10.5281-zenodo.10568762/Energy demand_GEO4CIVHIC demo sites.xlsx'

@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        return None
    
    # Load Excel: Skip rows if there's a header/metadata (adjust as needed)
    df = pd.read_excel(path)
    
    # Rename columns for easier access if they match your description
    df.columns = ['Hour', 'Load_kW']
    
    # Convert 'Hour' to a pseudo-datetime for better plotting
    # Assuming the 8760 rows represent a non-leap year starting Jan 1st
    df['Timestamp'] = pd.to_datetime(df['Hour'] - 1, unit='h', origin='2026-01-01')
    return df

# 2. Data Loading Logic
df = load_data(DATA_PATH)

if df is not None:
    # 3. Sidebar Filters
    st.sidebar.header("Filter options")
    
    # Allow users to view a specific month or the full year
    month = st.sidebar.select_slider(
        "Select month", 
        options=df['Timestamp'].dt.month_name().unique()
    )
    
    filtered_df = df[df['Timestamp'].dt.month_name() == month]

    # 4. Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Peak load", f"{df['Load_kW'].max()} kW")
    col2.metric("Avg load", f"{round(df['Load_kW'].mean(), 2)} kW")
    col3.metric("Total consumption", f"{int(df['Load_kW'].sum())} kWh")

    # 5. Visualization
    fig = px.line(
        filtered_df, 
        x='Timestamp', 
        y='Load_kW', 
        title=f"Demand profile for {month}",
        labels={'Load_kW': 'Load [kW]', 'Timestamp': 'Time of day'},
        template="plotly_white"
    )
    
    fig.update_traces(line_color='#2ca02c')
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error(f"**Data not found!** Expected file at: `{DATA_PATH}`")
    st.info("Check your Renku data connector or ensure the file is named correctly.")
