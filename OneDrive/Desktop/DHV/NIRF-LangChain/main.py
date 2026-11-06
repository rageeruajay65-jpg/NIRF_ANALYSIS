
import pandas as pd
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


# STEP 1: Load API key
load_dotenv()


# STEP 2: Load NIRF dataset
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "nirf_sample.csv"

df = pd.read_csv(CSV_PATH)

# Clean column values
df["Category"] = df["Category"].astype(str).str.strip()
df["State"] = df["State"].astype(str).str.strip()
df["Institution"] = df["Institution"].astype(str).str.strip()
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")

print("NIRF dataset loaded successfully!")


# STEP 3: Create ranking comparison tool
@tool
def compare_rankings(
    category: str,
    start_year: int,
    end_year: int,
    states: str = "all states"
) -> str:
    """
    Compare NIRF institutions between two years.

    If no state is specified, compare all states in India.
    Multiple states can be provided, separated by commas or 'and'.

    Find the institution with the largest absolute rank difference.
    Positive improvement means the institution moved up.
    """

    # Filter category
    filtered = df[
        df["Category"].str.lower() == category.strip().lower()
    ].copy()

    # Filter states only when specified
    states_input = states.strip().lower()

    if states_input not in ["all", "all states", "india", "overall"]:

        state_names = re.split(
            r",|\band\b|\bvs\b|\bversus\b",
            states_input
        )

        state_names = [s.strip() for s in state_names if s.strip()]

        filtered = filtered[
            filtered["State"].str.lower().isin(state_names)
        ]

    # Filter selected years
    filtered = filtered[
        filtered["Year"].isin([start_year, end_year])
    ]

    if filtered.empty:
        return "No matching NIRF records found."

    # Compare institutions
    pivot = filtered.pivot_table(
        index=["Institution", "State"],
        columns="Year",
        values="Rank",
        aggfunc="first"
    )

    if start_year not in pivot.columns or end_year not in pivot.columns:
        return "Data for one or both selected years is unavailable."

    pivot = pivot.dropna(subset=[start_year, end_year])

    if pivot.empty:
        return "No institutions have records in both selected years."

    # Calculate rank difference
    pivot["Rank Improvement"] = pivot[start_year] - pivot[end_year]

    # Absolute difference considers both improvement and decline
    pivot["Absolute Difference"] = pivot["Rank Improvement"].abs()

    pivot["Status"] = pivot["Rank Improvement"].apply(
        lambda x: "Improved" if x > 0
        else "Declined" if x < 0
        else "No Change"
    )

    # Find the largest absolute difference
    maximum_difference = pivot["Absolute Difference"].max()

    top_institutions = pivot[
        pivot["Absolute Difference"] == maximum_difference
    ]

    result = top_institutions.reset_index()

    return (
        f"Category: {category}\n"
        f"Years: {start_year} to {end_year}\n"
        f"States: {states_input}\n\n"
        f"Largest absolute rank difference: {maximum_difference}\n\n"
        f"{result.to_string(index=False)}"
    )


def compare_state_improvements(start_year=2020, end_year=2025,
                               category="Engineering"):

    filtered = df[
        (df["Category"].str.lower() == category.lower()) &
        (df["Year"].isin([start_year, end_year]))
    ].copy()

    if filtered.empty:
        return "No matching records found."

    pivot = filtered.pivot_table(
        index=["Institution", "State"],
        columns="Year",
        values="Rank",
        aggfunc="first"
    )

    if start_year not in pivot.columns or end_year not in pivot.columns:
        return "Data for one or both years is unavailable."

    pivot = pivot.dropna(subset=[start_year, end_year])

    if pivot.empty:
        return "No institutions found in both years."

    pivot["Rank Improvement"] = pivot[start_year] - pivot[end_year]

    state_results = pivot.groupby("State").agg(
        Average_Improvement=("Rank Improvement", "mean"),
        Improved_Institutions=("Rank Improvement", lambda x: (x > 0).sum()),
        Declined_Institutions=("Rank Improvement", lambda x: (x < 0).sum()),
        Total_Institutions=("Rank Improvement", "count")
    ).reset_index()

    state_results = state_results.sort_values(
        by="Average_Improvement",
        ascending=False
    )

    print("\nState-wise ranking improvement:")
    print(state_results.to_string(index=False))

    winner = state_results.iloc[0]

    return (
        f"\nState with highest average rank improvement: {winner['State']}\n"
        f"Average improvement: {winner['Average_Improvement']:.2f}"
    )



# STEP 4: Connect to Google Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    max_retries=3
)


# STEP 5: Create system prompt
system_prompt = """
You are an intelligent NIRF Ranking Analysis Assistant.

Understand natural-language questions about NIRF rankings.

Instructions:

1. Identify the NIRF category.
2. Identify the start and end years.
3. If the user does not mention a state, use all states in India.
4. If multiple states are mentioned, compare all those states.
5. Never ask for a state when the user wants an overall comparison.
6. For questions about the largest rank difference, call compare_rankings.
7. Explain the institution with the largest rank difference.
8. Clearly explain whether it improved or declined.
9. Never invent ranking values.
10. Use only the results returned by Python.



For questions such as:
"Which institute has the highest rank difference in engineering
category from 2020 to 2025?"

Use:
category = Engineering
start_year = 2020
end_year = 2025
states = all states
"""


# STEP 6: Create LangChain agent
agent = create_agent(
    model=llm,
    tools=[compare_rankings],
    system_prompt=system_prompt
)


# STEP 7: Ask the user
# STEP 6: Ask the user a question
print("\n===================================")
print("       NIRF RANKING ANALYSIS")
print("===================================")

question = input("\nEnter your NIRF question: ")

# STEP 7: Process the question
try:
    print("\n========== ANALYSIS RESULT ==========\n")

    question_lower = question.lower()

    # State-wise improvement question
    if (
        "state" in question_lower
        and (
            "improvement" in question_lower
            or "improved" in question_lower
        )
    ):

        result = compare_state_improvements(
            start_year=2020,
            end_year=2025,
            category="Engineering"
        )

        print(result)

    # Other questions go to Gemini + LangChain
    else:

        response = agent.invoke({
            "messages": [
                {"role": "user", "content": question}
            ]
        })

        content = response["messages"][-1].content

        if isinstance(content, list):
            answer = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
            )
        else:
            answer = str(content)

        print(answer)

except Exception as error:
    print("\nSomething went wrong:")
    print(error)


