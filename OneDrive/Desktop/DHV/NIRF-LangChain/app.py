from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pathlib import Path


import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from analytics.engine import prepare_data, run_analysis
from ai.planner import plan_question


# --------------------------------------------------
# 1. FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="NIRF Ranking Analysis API",
    description="AI-powered universal NIRF analytics chatbot",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# FRONTEND CONFIGURATION

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(directory="templates")


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "nirf_cleaned.csv"

df = pd.read_csv(CSV_PATH)
df = prepare_data(df)

print("NIRF dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Available years:", sorted(df["year"].dropna().unique()))


# --------------------------------------------------
# 3. REQUEST MODEL
# --------------------------------------------------

class QuestionRequest(BaseModel):
    question: str


# --------------------------------------------------
# 4. HOME ROUTE
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "NIRF Universal Analytics API is running!",
        "status": "success"
    }


# --------------------------------------------------
# 5. UNIVERSAL QUESTION ENDPOINT
# --------------------------------------------------

@app.post("/api/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    try:

        # STEP 1: Ask Gemini to understand the question
        plan = plan_question(question, df)

        print("\nUSER QUESTION:", question)
        print("GEMINI ANALYSIS PLAN:", plan)

        # STEP 2: Handle unsupported questions
        if plan.get("analysis_type") == "unsupported":

            return {
                "question": question,
                "plan": plan,
                "answer": (
                    "Sorry, this question cannot be answered "
                    "using the available NIRF dataset."
                ),
                "chart": None,
                "table": [],
                "statistics": None
            }

        # STEP 3: Validate requested years
        available_years = set(
            df["year"].dropna().astype(int).unique()
        )

        for field in ["year", "start_year", "end_year"]:

            value = plan.get(field)

            if value is not None and int(value) not in available_years:

                raise HTTPException(
                    status_code=400,
                    detail=f"Year {value} is not available in the dataset."
                )

        start_year = plan.get("start_year")
        end_year = plan.get("end_year")

        if (
            start_year is not None
            and end_year is not None
            and start_year > end_year
        ):

            raise HTTPException(
                status_code=400,
                detail="Start year cannot be greater than end year."
            )

        # STEP 4: Validate category
        category = plan.get("category")

        available_categories = set(
            df["category"].dropna().astype(str).str.lower().unique()
        )

        if category and category.lower() != "all":

            if category.lower() not in available_categories:

                raise HTTPException(
                    status_code=400,
                    detail=f"Category '{category}' is not available."
                )

        # STEP 5: Execute universal Pandas analytics
        result = run_analysis(plan, df)

        # STEP 6: Return results to frontend
        return {
            "question": question,
            "plan": plan,
            "answer": result.get("answer"),
            "chart": result.get("chart"),
            "table": result.get("table", []),
            "statistics": result.get("statistics")
        }

    except HTTPException:
        raise

    except ValueError as error:

        print("VALIDATION ERROR:", str(error))

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        print("API ERROR:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="Unable to process your question. Check the server terminal."
        )


# --------------------------------------------------
# 6. DATASET STATUS
# --------------------------------------------------

@app.get("/api/status")
def status():

    return {
        "status": "online",
        "dataset_rows": len(df),
        "available_years": sorted(
            df["year"].dropna().astype(int).unique().tolist()
        ),
        "available_categories": sorted(
            df["category"].dropna().astype(str).unique().tolist()
        ),
        "available_states": sorted(
            df["state"].dropna().astype(str).unique().tolist()
        ),
        "available_columns": df.columns.tolist()
    }

# DASHBOARD ROUTE

@app.get("/dashboard")
def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

