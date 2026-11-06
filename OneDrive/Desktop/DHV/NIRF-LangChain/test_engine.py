
import pandas as pd

from pathlib import Path
from analytics.engine import (
    prepare_data,
    top_institutions,
    score_distribution,
    state_top100,
    parameter_correlation
)

# Load the actual dataset
BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "data" / "nirf_cleaned.csv"

df = pd.read_csv(CSV_PATH)

# Prepare dataset
df = prepare_data(df)

latest_year = int(df["year"].max())

category = df["category"].dropna().mode()[0]

print("\nLatest year:", latest_year)
print("Most common category:", category)

# TEST 1: Top institutions
print("\n========== TOP INSTITUTIONS ==========")

result = top_institutions(
    df,
    year=latest_year,
    category=category,
    limit=10
)

print(result["answer"])
print("Chart type:", result["chart"]["type"])

# TEST 2: Score distribution
print("\n========== SCORE DISTRIBUTION ==========")

result = score_distribution(
    df,
    year=latest_year,
    category=category
)

print(result["answer"])
print(result["statistics"])

# TEST 3: State-wise top 100
print("\n========== STATE ANALYSIS ==========")

result = state_top100(
    df,
    year=latest_year,
    category=category
)

print(result["answer"])
print("Chart type:", result["chart"]["type"])

# TEST 4: Parameter correlation
print("\n========== PARAMETER CORRELATION ==========")

result = parameter_correlation(
    df,
    year=latest_year,
    category=category,
    x_parameter="tlr",
    y_parameter="rpc"
)

print(result["answer"])
print("Correlation:", result.get("correlation"))
print("Chart type:", result["chart"]["type"])

print("\nALL TESTS COMPLETED!")
