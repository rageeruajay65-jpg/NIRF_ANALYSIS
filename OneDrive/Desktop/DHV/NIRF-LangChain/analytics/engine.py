import pandas as pd
import numpy as np


# --------------------------------------------------
# 1. DATA PREPARATION
# --------------------------------------------------

NUMERIC_COLUMNS = [
    "score", "rank", "tlr", "rpc", "go",
    "oi", "perception", "qlr", "qnr", "sfc"
]

GROUP_COLUMNS = [
    "name", "state", "city", "category", "year"
]


def prepare_data(df):

    df = df.copy()

    # Convert numeric columns safely
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Clean text columns
    for column in ["name", "state", "city", "category"]:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    df["category"] = df["category"].str.lower()

    return df


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def normalize_text(value):
    """Normalize text for matching user-provided filters."""

    value = str(value).strip().lower()

    # Handle common state-name variations
    replacements = {
        "tamilnadu": "tamil nadu",
        "jammu & kashmir": "jammu and kashmir"
    }

    return replacements.get(value, value)


def filter_data(
    df,
    year=None,
    start_year=None,
    end_year=None,
    category=None,
    states=None,
    cities=None,
    institution=None
):

    result = df.copy()

    if year is not None:
        result = result[result["year"] == int(year)]

    if start_year is not None:
        result = result[result["year"] >= int(start_year)]

    if end_year is not None:
        result = result[result["year"] <= int(end_year)]

    if category and str(category).lower() != "all":
        result = result[
            result["category"].str.lower() == str(category).lower()
        ]

    if states:
        requested_states = {
            normalize_text(state) for state in states
        }

        result = result[
            result["state"].apply(
                lambda x: normalize_text(x) in requested_states
                if pd.notna(x) else False
            )
        ]

    if cities:
        requested_cities = {
            normalize_text(city) for city in cities
        }

        result = result[
            result["city"].apply(
                lambda x: normalize_text(x) in requested_cities
                if pd.notna(x) else False
            )
        ]

    if institution:
        result = result[
            result["name"].str.contains(
                str(institution),
                case=False,
                na=False,
                regex=False
            )
        ]

    return result


def clean_records(dataframe):
    """Convert DataFrame records into JSON-safe Python values."""

    records = dataframe.to_dict(orient="records")
    cleaned = []

    for record in records:
        clean_record = {}

        for key, value in record.items():

            if pd.isna(value):
                clean_record[key] = None

            elif isinstance(value, (np.integer,)):
                clean_record[key] = int(value)

            elif isinstance(value, (np.floating,)):
                clean_record[key] = round(float(value), 3)

            else:
                clean_record[key] = value

        cleaned.append(clean_record)

    return cleaned


def make_chart(chart_type, title, x, y, x_label, y_label):

    return {
        "type": chart_type,
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "x": list(x),
        "y": list(y)
    }


def get_latest_year(df):
    years = df["year"].dropna()

    if years.empty:
        raise ValueError("No valid years are available.")

    return int(years.max())


# --------------------------------------------------
# 3. UNIVERSAL ANALYTICS ENGINE
# --------------------------------------------------

def run_analysis(plan, df):

    analysis_type = plan.get("analysis_type", "unsupported")

    metric = plan.get("metric") or "score"

    aggregation = plan.get("aggregation", "mean")

    group_by = plan.get("group_by") or []

    limit = plan.get("limit", 10)

    sort_order = plan.get("sort_order", "descending")

    # Validate metric
    if metric not in NUMERIC_COLUMNS:
        raise ValueError(f"Unsupported metric: {metric}")

    # Validate aggregation
    allowed_aggregations = [
        "mean", "median", "sum", "min", "max", "count", "std"
    ]

    if aggregation not in allowed_aggregations:
        raise ValueError("Unsupported aggregation method.")

    # Validate grouping columns
    for column in group_by:
        if column not in GROUP_COLUMNS:
            raise ValueError(f"Unsupported grouping column: {column}")

    # Apply filters
    data = filter_data(
        df=df,
        year=plan.get("year"),
        start_year=plan.get("start_year"),
        end_year=plan.get("end_year"),
        category=plan.get("category"),
        states=plan.get("states"),
        cities=plan.get("cities"),
        institution=plan.get("institution")
    )

    # Default to latest year for analyses without a year range
    if (
        plan.get("year") is None
        and plan.get("start_year") is None
        and plan.get("end_year") is None
        and analysis_type not in ["trend"]
    ):
        latest_year = get_latest_year(df)
        data = data[data["year"] == latest_year]

    if data.empty:
        return {
            "answer": "No matching records were found for your question.",
            "chart": None,
            "table": [],
            "statistics": None
        }

    # --------------------------------------------------
    # A. CORRELATION
    # --------------------------------------------------

    if analysis_type == "correlation":

        x_metric = plan.get("x_metric")
        y_metric = plan.get("y_metric")

        if x_metric not in NUMERIC_COLUMNS:
            raise ValueError("Invalid X-axis parameter.")

        if y_metric not in NUMERIC_COLUMNS:
            raise ValueError("Invalid Y-axis parameter.")

        correlation_data = data[[x_metric, y_metric]].dropna()

        if len(correlation_data) < 2:
            raise ValueError("Not enough valid records to calculate correlation.")

        correlation = correlation_data[x_metric].corr(
            correlation_data[y_metric]
        )

        chart = {
            "type": "scatter",
            "title": f"{x_metric.upper()} vs {y_metric.upper()}",
            "x_label": x_metric.upper(),
            "y_label": y_metric.upper(),
            "x": correlation_data[x_metric].tolist(),
            "y": correlation_data[y_metric].tolist()
        }

        return {
            "answer": (
                f"Correlation between {x_metric.upper()} and "
                f"{y_metric.upper()} is {correlation:.3f}."
            ),
            "chart": chart,
            "table": clean_records(correlation_data),
            "statistics": {"correlation": round(float(correlation), 3)}
        }

    # --------------------------------------------------
    # B. DISTRIBUTION
    # --------------------------------------------------

    if analysis_type == "distribution":

        values = data[metric].dropna()

        if values.empty:
            raise ValueError(f"No valid values available for {metric}.")

        chart = {
            "type": "histogram",
            "title": f"Distribution of {metric.upper()}",
            "x_label": metric.upper(),
            "y_label": "Frequency",
            "x": values.tolist()
        }

        statistics = {
            "count": int(values.count()),
            "mean": round(float(values.mean()), 3),
            "median": round(float(values.median()), 3),
            "minimum": round(float(values.min()), 3),
            "maximum": round(float(values.max()), 3),
            "standard_deviation": round(float(values.std()), 3)
        }

        return {
            "answer": f"Generated the distribution of {metric.upper()} using {len(values)} valid records.",
            "chart": chart,
            "table": [],
            "statistics": statistics
        }

    # --------------------------------------------------
    # C. DESCRIPTIVE STATISTICS
    # --------------------------------------------------

    if analysis_type == "statistics":

        values = data[metric].dropna()

        if values.empty:
            raise ValueError(f"No valid values available for {metric}.")

        statistics = {
            "count": int(values.count()),
            "mean": round(float(values.mean()), 3),
            "median": round(float(values.median()), 3),
            "minimum": round(float(values.min()), 3),
            "maximum": round(float(values.max()), 3),
            "standard_deviation": round(float(values.std()), 3)
        }

        return {
            "answer": f"Calculated descriptive statistics for {metric.upper()}.",
            "chart": None,
            "table": [statistics],
            "statistics": statistics
        }

    # --------------------------------------------------
    # D. GROUPING AND AGGREGATION
    # --------------------------------------------------

    if analysis_type in [
        "ranking", "comparison", "aggregation", "trend"
    ]:

        # Automatically choose a grouping column when needed
        if not group_by:

            if plan.get("states"):
                group_by = ["state"]

            elif plan.get("cities"):
                group_by = ["city"]

            elif analysis_type == "trend":
                group_by = ["year"]

            else:
                group_by = ["name"]

        # Ranking institutions requires institution names
        if analysis_type == "ranking" and not group_by:
            group_by = ["name"]

        if aggregation == "count":
            result = (
                data.groupby(group_by, dropna=False)
                .size()
                .reset_index(name="count")
            )

            value_column = "count"

        else:
            result = (
                data.groupby(group_by, dropna=False)[metric]
                .agg(aggregation)
                .reset_index()
            )

            value_column = metric

        result = result.dropna(subset=[value_column])

        if result.empty:
            raise ValueError("No valid results could be calculated.")

        # Lower NIRF rank is better
        ascending = (
            True if metric == "rank"
            else sort_order == "ascending"
        )

        result = result.sort_values(
            by=value_column,
            ascending=ascending
        )

        # Limit results
        if analysis_type in ["ranking", "comparison"]:
            result = result.head(int(limit))

        # Keep all groups for a trend
        if analysis_type == "trend":
            result = result.sort_values(by=group_by[0])

        # Prepare chart
        x_column = group_by[0]

        if analysis_type == "trend":
            chart_type = "line"
        else:
            chart_type = "bar"

        chart = make_chart(
            chart_type=chart_type,
            title=f"{aggregation.title()} {metric.upper()} by {x_column.title()}",
            x=result[x_column].astype(str).tolist(),
            y=result[value_column].tolist(),
            x_label=x_column.title(),
            y_label=value_column.title()
        )

        records = clean_records(result)

        answer = (
            f"Successfully analyzed {len(records)} groups "
            f"using {aggregation} of {metric.upper()}."
        )

        return {
            "answer": answer,
            "chart": chart,
            "table": records,
            "statistics": {
                "groups_analyzed": len(records),
                "records_analyzed": len(data)
            }
        }

    # --------------------------------------------------
    # E. UNSUPPORTED QUESTIONS
    # --------------------------------------------------

    return {
        "answer": (
            "This question cannot currently be analyzed "
            "using the available dataset."
        ),
        "chart": None,
        "table": [],
        "statistics": None
    }