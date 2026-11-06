
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv

from analytics.engine import prepare_data
from ai.planner import plan_question


# Load API key from .env
load_dotenv()

# Load dataset
BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "data" / "nirf_cleaned.csv"

df = pd.read_csv(CSV_PATH)

df = prepare_data(df)


# Questions to test
questions = [
    "Show the rank trend of IIT Madras from 2017 to 2025",

    "Show the top 10 Engineering institutions in 2025",

    "Which states have the most institutions in the top 100 in 2025?",

    "Show the score distribution of Engineering institutions in 2025",

    "Show the correlation between TLR and RPC in Engineering in 2025",

    "Compare the average RPC score across states in 2025",

    "Which institutions improved the most between 2020 and 2025?"
]


for question in questions:

    print("\n========================================")
    print("QUESTION:")
    print(question)

    try:
        plan = plan_question(question, df)

        print("\nGEMINI ANALYSIS PLAN:")
        print(plan)

    except Exception as error:
        print("\nERROR:")
        print(error)
