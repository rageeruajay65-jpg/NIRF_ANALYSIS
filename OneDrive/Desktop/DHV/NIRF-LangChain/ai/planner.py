import os
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


# Universal analysis plan
class AnalysisPlan(BaseModel):

    analysis_type: Literal[
        "ranking",
        "comparison",
        "aggregation",
        "trend",
        "distribution",
        "correlation",
        "statistics",
        "unsupported"
    ] = Field(description="The type of analysis requested")

    metric: str | None = Field(
        default="score",
        description="Numeric dataset column to analyze, such as score, rank, tlr, rpc, go, oi, perception, qlr, qnr or sfc"
    )

    group_by: list[str] = Field(
        default_factory=list,
        description="Columns used to group or compare results, such as state, city, category, name or year"
    )

    aggregation: Literal[
        "mean", "median", "sum", "min", "max", "count", "std"
    ] = Field(
        default="mean",
        description="Aggregation method"
    )

    year: int | None = Field(default=None)
    start_year: int | None = Field(default=None)
    end_year: int | None = Field(default=None)

    category: str | None = Field(default=None)
    states: list[str] = Field(default_factory=list)
    cities: list[str] = Field(default_factory=list)
    institution: str | None = Field(default=None)

    x_metric: str | None = Field(default=None)
    y_metric: str | None = Field(default=None)

    sort_order: Literal["ascending", "descending"] = "descending"
    limit: int = Field(default=10, ge=1, le=100)

    explanation: str = Field(
        default="",
        description="Brief explanation of the intended analysis"
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
     google_api_key=os.getenv("GOOGLE_API_KEY"),
    timeout=60,
    max_retries=2
)

planner_model = llm.with_structured_output(AnalysisPlan)


def get_dataset_context(df):

    return {
        "years": sorted(df["year"].dropna().astype(int).unique().tolist()),
        "categories": sorted(df["category"].dropna().astype(str).unique().tolist()),
        "columns": df.columns.tolist(),
        "numeric_columns": df.select_dtypes(include="number").columns.tolist()
    }


def plan_question(question, df):

    context = get_dataset_context(df)

    system_prompt = f"""
You are an intelligent analytics planner for an Indian NIRF ranking dataset.

Your task is to convert ANY natural-language question into a structured
analysis plan.

AVAILABLE DATASET INFORMATION:
Years: {context['years']}
Categories: {context['categories']}
Columns: {context['columns']}
Numeric columns: {context['numeric_columns']}

SUPPORTED ANALYSIS TYPES:

1. ranking:
   Find top or bottom institutions, states, cities or categories.

2. comparison:
   Compare two or more states, cities, institutions, categories or years.

3. aggregation:
   Calculate average, median, sum, count, minimum or maximum grouped by
   a requested dimension.

4. trend:
   Analyze changes over multiple years.
   Use group_by=["year"].
   Extract start_year and end_year when supplied.

5. distribution:
   Analyze the distribution of a numeric parameter.

6. correlation:
   Analyze the relationship between two numeric parameters.
   Fill x_metric and y_metric.

7. statistics:
   Calculate descriptive statistics for a numeric parameter.

8. unsupported:
   Use ONLY when the question genuinely cannot be answered using this dataset.

IMPORTANT RULES:

- Do not use unsupported just because the question is phrased differently.
- Understand synonyms and natural-language variations.
- "Compare Tamil Nadu and Kerala" means comparison grouped by state.
- Extract both states into the states field.
- Recognize Tamilnadu as Tamil Nadu.
- Recognize average, mean, typical as mean unless another aggregation is specified.
- Recognize highest as descending and lowest as ascending.
- For ranking by NIRF score, use metric="score".
- For ranking by NIRF rank, use metric="rank".
- For institution rankings, use group_by=["name"].
- For state comparisons, use group_by=["state"].
- For city comparisons, use group_by=["city"].
- For category comparisons, use group_by=["category"].
- For year trends, use group_by=["year"].
- If no year is specified, use the latest available year.
- If a year range is specified, extract start_year and end_year.
- If the user asks to compare rankings or rank positions,
  use metric="rank".
- If the user asks to compare scores, use metric="score".
- If the user asks to compare two or more states,
  use group_by=["state"].
- If the user asks to compare two or more cities,
  use group_by=["city"].
- If the user asks to compare two or more institutions,
  use group_by=["name"].
- For comparisons of rankings, use aggregation="mean"
  unless another aggregation is explicitly requested.
- Do not invent dataset columns.
- Do not invent data or calculations.
- Use only the available dataset columns.
- For correlation, select the two requested numeric parameters.
- Do not confuse rank and score: lower rank is better, higher score is better.
- If the user asks for all institutions, do not arbitrarily restrict the analysis to 10.
- Keep the explanation short.

PARAMETER ABBREVIATIONS:
TLR = tlr
RPC = rpc
GO = go
OI = oi
Perception = perception
QLR = qlr
QNR = qnr
SFC = sfc

Return only the structured analysis plan.
"""

    response = planner_model.invoke([
        ("system", system_prompt),
        ("human", question)
    ])

    plan = response.model_dump()

    if not plan.get("limit") or plan["limit"] < 1:
        plan["limit"] = 10

    if not plan.get("metric"):
        plan["metric"] = "score"

    return plan
