
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "data" / "nirf_cleaned.csv"

df = pd.read_csv(CSV_PATH)

print("\n========== DATASET INFORMATION ==========")

print("\nShape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 10 records:")
print(df.head(10).to_string(index=False))

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

if "Year" in df.columns:
    print("\nAvailable years:")
    print(sorted(df["Year"].dropna().unique()))

if "Category" in df.columns:
    print("\nAvailable categories:")
    print(sorted(df["Category"].dropna().unique()))

print("\n=========================================")
