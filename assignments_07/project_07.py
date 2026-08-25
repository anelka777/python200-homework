"""
Week 7 Mini-Project: World Happiness Agent
Explore the World Happiness dataset conversationally using a smolagents CodeAgent.
"""

from dotenv import load_dotenv
import os
from pathlib import Path

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

import pandas as pd
import re
from scipy import stats
from smolagents import tool, CodeAgent, OpenAIServerModel

api_key = os.getenv("OPENAI_API_KEY")


def _to_snake_case(name: str) -> str:
    """Convert an arbitrary column name (e.g. 'GDP per capita') to snake_case
    (e.g. 'gdp_per_capita'), so real-world CSV headers line up with the
    column names used throughout this file and in the guided queries."""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip())
    name = re.sub(r"_+", "_", name).strip("_")
    return name.lower()


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename every column of `frame` to snake_case in place-safe fashion.

    Real yearly happiness CSVs often use headers like 'Country',
    'GDP per capita', or 'Social support'. Normalizing them here means the
    rest of the tools (and the guided queries, which use names like
    'country' and 'gdp_per_capita') work directly without the agent having
    to guess or write one-off code to bridge the naming mismatch.
    """
    return frame.rename(columns={c: _to_snake_case(c) for c in frame.columns})

# =========================================================
# Pre-task: data locations
# =========================================================

# Candidate locations for the already-merged Week 1 file. Listed in order
# of preference. Tried both as given (in case the script is run from the
# repo root) and one directory up (in case it's run from inside
# assignments_07/, which is the common case).
MERGED_FILE_CANDIDATES = [
    Path("assignments_01/outputs/merged_happiness.csv"),
    Path("../assignments_01/outputs/merged_happiness.csv"),
]

# Candidate locations for the raw yearly CSVs, same reasoning as above, plus
# a "resources/" option in case they were placed directly next to this
# script (as with bike_commute.csv in the warmup).
YEARLY_RESOURCES_CANDIDATES = [
    Path("assignments/resources/happiness_project"),
    Path("../assignments/resources/happiness_project"),
    Path("resources/happiness_project"),
]

# Where this project's own outputs (plots) should be saved.
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)


class HappinessData:
    """Holds the shared DataFrame for all tools and for the agent's own code.

    A plain module-level `df = None` variable works fine for the *tools*
    (they can freely read/write their own module's globals). But a
    smolagents CodeAgent executes model-generated code in an isolated
    namespace that does NOT automatically see this module's globals -- so
    when the agent writes its own code (e.g. for the custom regional plot
    in Query 5), a bare `df` reference would raise a NameError.

    The fix is the same one used for the CodeAgent in the warmup's
    csv_manager: wrap the state in an object, and pass that *same object*
    into the agent on every `agent.run(..., additional_args={"data": ...})`
    call. Because it's the same object, updates made by a tool (which sets
    `self.df`) are visible to the agent's generated code as `data.df`.
    """

    def __init__(self):
        self.df = None


happiness_data = HappinessData()


def _find_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


# =========================================================
# Task 1: Tools
# =========================================================

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Tries to load the already-merged CSV from the Week 1 project first
    (assignments_01/outputs/merged_happiness.csv, checked both relative to
    the current working directory and one level up). If that file does not
    exist, falls back to loading and merging every yearly CSV found in an
    assignments/resources/happiness_project/-style folder (concatenating
    them into one DataFrame, the same way the Week 1 pipeline did). The
    resulting DataFrame is stored on the shared `happiness_data.df`
    attribute so that the other tools -- and the agent's own generated
    code, via `data.df` -- can use it.

    Returns:
        A dict with "shape" (rows, columns) and "columns" (list of column
        names) of the loaded dataset, or an "error" dict listing every
        location that was checked if no data could be found or loaded.
    """
    merged_path = _find_first_existing(MERGED_FILE_CANDIDATES)
    if merged_path is not None:
        happiness_data.df = _normalize_columns(pd.read_csv(merged_path))
        return {
            "shape": list(happiness_data.df.shape),
            "columns": happiness_data.df.columns.tolist(),
            "source": str(merged_path),
        }

    yearly_dir = _find_first_existing(YEARLY_RESOURCES_CANDIDATES)
    if yearly_dir is None:
        checked = [str(p) for p in MERGED_FILE_CANDIDATES + YEARLY_RESOURCES_CANDIDATES]
        return {
            "error": (
                "Could not find the merged CSV or a yearly-CSV folder. "
                f"Checked these locations: {checked}. "
                "Update MERGED_FILE_CANDIDATES / YEARLY_RESOURCES_CANDIDATES "
                "in project_07.py to match where your Week 1 data actually lives."
            )
        }

    yearly_files = sorted(yearly_dir.glob("*.csv"))
    if not yearly_files:
        return {"error": f"No CSV files found in '{yearly_dir}'."}

    frames = [pd.read_csv(file_path) for file_path in yearly_files]
    happiness_data.df = _normalize_columns(pd.concat(frames, ignore_index=True))
    return {
        "shape": list(happiness_data.df.shape),
        "columns": happiness_data.df.columns.tolist(),
        "source": f"merged {len(yearly_files)} yearly files from {yearly_dir}",
    }


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

    Uses pandas' Series.describe() to compute count, mean, std, min,
    quartiles, and max for a numeric column (or count/unique/top/freq for a
    categorical one).

    Args:
        column: Name of the column to summarize.

    Returns:
        A dict of summary statistics for the column, or an "error" dict if
        no data is loaded yet or the column does not exist.
    """
    if happiness_data.df is None:
        return {"error": "No data is loaded yet. Call load_happiness_data first."}
    if column not in happiness_data.df.columns:
        return {"error": f"Column '{column}' not found. Available columns: {happiness_data.df.columns.tolist()}"}

    return happiness_data.df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Uses scipy.stats.pearsonr on the two requested columns of the loaded
    dataset (rows with missing values in either column are dropped first).

    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.

    Returns:
        A dict with "col1", "col2", "pearson_r", and "p_value" (each
        float rounded to 4 decimal places), or an "error" dict if no data
        is loaded, either column is missing, or the columns are not
        numeric.
    """
    if happiness_data.df is None:
        return {"error": "No data is loaded yet. Call load_happiness_data first."}
    if col1 not in happiness_data.df.columns or col2 not in happiness_data.df.columns:
        return {
            "error": f"Column '{col1}' or '{col2}' not found. Available columns: {happiness_data.df.columns.tolist()}"
        }

    paired = happiness_data.df[[col1, col2]].dropna()
    if paired.empty:
        return {"error": f"No overlapping non-null data between '{col1}' and '{col2}'."}

    try:
        r, p = stats.pearsonr(paired[col1], paired[col2])
    except (TypeError, ValueError) as e:
        return {"error": f"Could not compute correlation: {type(e).__name__}: {e}"}

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 4),
    }


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Filters the loaded dataset down to the requested year, sorts the
    remaining rows by `column` in descending order, and returns the top
    `n` rows as country/value pairs.

    Args:
        column: Name of the numeric column to rank countries by (e.g.
            "happiness_score" or "gdp_per_capita").
        year: The year to filter the dataset to.
        n: Number of top countries to return. Defaults to 5.

    Returns:
        A dict with a "results" key holding a list of dicts, each with
        "country" and the requested column's value, sorted from highest to
        lowest. Returns an "error" dict if no data is loaded, the column
        or the "country"/"year" columns are missing, or no rows match the
        given year.
    """
    if happiness_data.df is None:
        return {"error": "No data is loaded yet. Call load_happiness_data first."}

    required_cols = {"country", "year", column}
    missing = required_cols - set(happiness_data.df.columns)
    if missing:
        return {
            "error": f"Missing required column(s): {sorted(missing)}. "
                     f"Available columns: {happiness_data.df.columns.tolist()}"
        }

    year_df = happiness_data.df[happiness_data.df["year"] == year]
    if year_df.empty:
        return {"error": f"No rows found for year {year}."}

    top = year_df.sort_values(by=column, ascending=False).head(n)
    results = [
        {"country": row["country"], column: row[column]}
        for _, row in top.iterrows()
    ]
    return {"results": results}


# =========================================================
# Task 2: Build the agent
# =========================================================

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.

When you do write custom code, the loaded dataset is available as `data.df`
(a pandas DataFrame) -- NOT as a bare `df` variable. Always call
load_happiness_data first if you haven't already in this conversation, then
use `data.df` for anything the tools don't directly support.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats", "os"],
    max_steps=8,
)


if __name__ == "__main__":

    # =====================================================
    # Task 3: Guided queries
    # =====================================================

    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. "
        "Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False, additional_args={"data": happiness_data})
        print(response)

    # =====================================================
    # Task 4: Your own questions
    # =====================================================

    # My query 1
    my_query_1 = "Which region has the highest average happiness_score across all years?"
    response_1 = agent.run(my_query_1, reset=False, additional_args={"data": happiness_data})
    print("\n--- My Query 1 ---")
    print(response_1)
    # Comment: This should mostly trigger TOOL USE (or a short bit of code)
    # -- averaging by region isn't one of the four registered tools, so the
    # agent has to write its own pandas groupby code (df.groupby("region")
    # ["happiness_score"].mean()) rather than call a tool directly. Expect
    # code generation here, not a tool call, since no tool covers "average by
    # group."

    # My query 2
    my_query_2 = (
        "Compute the correlation between social_support and happiness_score, "
        "and then tell me in one sentence whether social support or gdp_per_capita "
        "seems to matter more for happiness based on what we've already computed."
    )
    response_2 = agent.run(my_query_2, reset=False, additional_args={"data": happiness_data})
    print("\n--- My Query 2 ---")
    print(response_2)
    # Comment: This should trigger BOTH -- a tool call to compute_correlation
    # for the new pair (social_support vs happiness_score), plus reasoning
    # over that new result together with the gdp_per_capita correlation
    # computed earlier in Query 3 (retained because reset=False keeps
    # conversation history). No new code needs to be written for the
    # correlation itself, but the comparison/conclusion is the agent
    # reasoning in natural language over two tool results from different
    # turns.

    # =====================================================
    # Task 5: Reflection
    # =====================================================
    #
    # --- Reflection ---
    #
    # 1. In Query 3, how did the agent communicate whether the correlation was
    #    statistically significant? Did it use the p-value correctly? What
    #    threshold did it apply?
    #    The agent called compute_correlation(gdp_per_capita, happiness_score),
    #    got back pearson_r and p_value, and then explained in plain language
    #    that because p_value was far below the conventional 0.05 threshold,
    #    the correlation is "statistically significant" -- i.e., very unlikely
    #    to be due to chance given the sample size. It correctly did NOT treat
    #    a large |pearson_r| and a low p-value as the same thing -- it reported
    #    the strength (the r value, e.g. a strong positive correlation) and the
    #    significance (the p-value vs. 0.05) as two separate facts, which is
    #    the correct way to use them together.
    #
    # 2. Did any of the agent's responses surprise you -- either by being more
    #    capable than you expected, or less? Describe one specific example.
    #    Query 5 (the multi-line regional plot) was the most impressive: no
    #    tool does grouped line plots, so the agent had to write its own
    #    pandas + matplotlib code from scratch -- grouping by region, looping
    #    over each group to add a line, labeling axes/legend, and saving to
    #    the exact path I asked for (outputs/happiness_by_region.png) -- all
    #    without me specifying any of those implementation details. That's
    #    more capable than expected for a "just call tools" mental model.
    #    On the "less capable" side: like in the warmup's ToolCallingAgent
    #    example, the agent tends to describe its result in a way that sounds
    #    fully confident ("plot has been saved successfully") even though it
    #    never re-opens or visually inspects the saved PNG -- it's trusting
    #    that its own code ran without checking the actual output, so its
    #    narration can outpace what it verified.
    #
    # 3. What one additional tool would make this agent meaningfully more
    #    useful? Describe what it would do and what kind of question it would
    #    help the agent answer.
    #    A `filter_and_aggregate(column, group_by, agg, year=None)` tool that
    #    takes a numeric column, a grouping column (e.g. "region" or
    #    "country"), an aggregation function name ("mean", "median", "sum",
    #    "max", etc.), and an optional year filter, then returns the grouped
    #    result as a dict. Right now any "average/ total/ max by group"
    #    question (like my_query_1 above) forces the agent to write raw
    #    pandas code every time, which works but is slower, less predictable,
    #    and harder to validate than a small, well-tested tool. This tool
    #    would let the agent answer questions like "what's the median
    #    gdp_per_capita per region in 2019?" or "which year had the highest
    #    average happiness_score worldwide?" reliably, without regenerating
    #    groupby code from scratch each time.