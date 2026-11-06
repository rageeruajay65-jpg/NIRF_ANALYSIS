
import pandas as pd

from analytics.engine import prepare_data, run_analysis

# Load dataset
df = pd.read_csv("data/nirf_cleaned.csv")
df = prepare_data(df)

print("Dataset loaded:", df.shape)


# TEST 1: Compare Tamil Nadu and Kerala rankings
print("\nTEST 1: STATE COMPARISON")

plan1 = {
    "analysis_type": "comparison",
    "metric": "rank",
    "group_by": ["state"],
    "aggregation": "mean",
    "states": ["Tamil Nadu", "Kerala"],
    "year": 2025,
    "limit": 10
}

result1 = run_analysis(plan1, df)

print(result1["answer"])
print(result1["table"])


# TEST 2: Top 10 Engineering institutions
print("\nTEST 2: TOP INSTITUTIONS")

plan2 = {
    "analysis_type": "ranking",
    "metric": "score",
    "group_by": ["name"],
    "aggregation": "mean",
    "category": "engineering",
    "year": 2025,
    "sort_order": "descending",
    "limit": 10
}

result2 = run_analysis(plan2, df)

print(result2["answer"])
print(result2["table"])


# TEST 3: Average RPC by state
print("\nTEST 3: STATE-WISE RPC")

plan3 = {
    "analysis_type": "aggregation",
    "metric": "rpc",
    "group_by": ["state"],
    "aggregation": "mean",
    "year": 2025
}

result3 = run_analysis(plan3, df)

print(result3["answer"])
print(result3["table"][:5])


# TEST 4: IIT Madras ranking trend
print("\nTEST 4: RANKING TREND")

plan4 = {
    "analysis_type": "trend",
    "metric": "rank",
    "group_by": ["year"],
    "aggregation": "mean",
    "institution": "IIT Madras",
    "start_year": 2017,
    "end_year": 2025
}

result4 = run_analysis(plan4, df)

print(result4["answer"])
print(result4["table"])


# TEST 5: Score distribution
print("\nTEST 5: SCORE DISTRIBUTION")

plan5 = {
    "analysis_type": "distribution",
    "metric": "score",
    "year": 2025,
    "category": "engineering"
}

result5 = run_analysis(plan5, df)

print(result5["answer"])
print(result5["statistics"])


print("\nALL UNIVERSAL ENGINE TESTS COMPLETED!")
