import os
import pandas as pd

file_name = "ed_patient_data.csv"

print("Current working directory:")
print(os.getcwd())

if os.path.exists(file_name):
    df = pd.read_csv(file_name)

    print("\nDataset loaded successfully.")
    print("Shape:", df.shape)
    print("\nAvailable columns:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head())
else:
    print("\nDataset not found.")
    print("Please obtain the approved dataset")
    print("or select a suitable public dataset.")
