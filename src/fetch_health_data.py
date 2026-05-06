from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

# INSTRUCTIONS: Ensure that the route directs to the data connector with read & write access
# Create a new directory if it doesn't exist
os.makedirs('data/health', exist_ok=True)

# Fetch dataset
print("Fetching heart disease dataset from UCI...")
heart_disease = fetch_ucirepo(id=45) 

# Combine features and targets into one dataframe
df = heart_disease.data.original

# Save to CSV for the R thread
df.to_csv('data/health/heart_disease.csv', index=False)
print("Data saved to data/health/heart_disease.csv")
