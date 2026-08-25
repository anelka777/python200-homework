"""
Week 7 Warmup Exercises
AI Agents, ReAct loop, JSON tool schemas, multi-tool agents, smolagents
"""

from dotenv import load_dotenv
import os
import json

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

from openai import OpenAI

client = OpenAI()


# =========================================================
# --- Lesson 02: Tool Definitions and the ReAct Loop ---
# =========================================================

from datetime import datetime

# get_current_time: copied verbatim from the lesson.
def get_current_time() -> str:
    """Return the current local time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Single-tool schema, in the exact style used in the lesson.
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current local time as a string.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
]


def run_agent(user_prompt: str) -> str:
    """Run a minimal ReAct-style agent for a single user prompt. (Lesson version -- only get_current_time)"""

    SYSTEM_PROMPT = """You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    first_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    print("First response received from model...")
    first_message = first_response.choices[0].message

    messages.append(
        {
            "role": "assistant",
            "content": first_message.content,
            "tool_calls": first_message.tool_calls,
        }
    )

    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            if function_name == "get_current_time":
                tool_result = get_current_time()
            else:
                tool_result = f"Error: unknown tool {function_name}."

            print("Tool called:", function_name)
            print("Tool result:", tool_result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                }
            )

        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )
        print("Second response received from model...")

        final_message = second_response.choices[0].message
        return final_message.content or ""
    else:
        print("No tools needed....")

    return first_message.content or ""


# --- Q1 ---
print("\n=== Q1 ===")

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

# JSON schema describing celsius_to_fahrenheit to an LLM, in the same style
# as the get_current_time schema above.
celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit and return it as a formatted string.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "The temperature in degrees Celsius to convert.",
                }
            },
            "required": ["celsius"],
        },
    },
}

# Call the function directly (no agent yet)
for c in [0, 100, -40]:
    print(celsius_to_fahrenheit(c))


# --- Q2 ---
print("\n=== Q2 ===")

# PREDICTION (written before running):
# Calling run_agent("Convert 100 degrees Celsius to Fahrenheit") should NOT
# trigger a tool call, because the only tool available at this point is
# get_current_time -- there is no temperature-conversion tool registered yet.
# The model has no matching tool to call, so it will answer directly from its
# own knowledge. I expect this to take exactly 1 API call, since no tool
# round-trip is triggered when the model has nothing relevant to call.
result_q2 = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print("Result:", result_q2)
# ACTUAL RESULT (written after running):
# My prediction was correct: with only get_current_time registered, the model
# has no matching tool for a temperature conversion. It goes straight to
# "No tools needed...." and answers directly from its own knowledge in a
# single API call. No tool_calls were triggered.


# --- Q3 ---
print("\n=== Q3 ===")

# Extend the agent to support both tools: add celsius_to_fahrenheit to the
# tools list and dispatch it inside a modified run_agent.
tools_q3 = [
    tools[0],  # get_current_time schema
    celsius_to_fahrenheit_schema,
]


def run_agent_q3(user_prompt: str) -> str:
    """Extended ReAct agent supporting both get_current_time and celsius_to_fahrenheit."""

    SYSTEM_PROMPT = """You are a helpful assistant with two tools: one to tell the
                     current time, and one to convert Celsius to Fahrenheit.
                     Use the appropriate tool whenever the user's request matches it."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    first_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools_q3,
        tool_choice="auto",
    )
    first_message = first_response.choices[0].message

    messages.append(
        {
            "role": "assistant",
            "content": first_message.content,
            "tool_calls": first_message.tool_calls,
        }
    )

    if first_message.tool_calls:
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            if function_name == "get_current_time":
                tool_result = get_current_time()
            elif function_name == "celsius_to_fahrenheit":
                args = json.loads(tool_call.function.arguments or "{}")
                tool_result = celsius_to_fahrenheit(**args)
            else:
                tool_result = f"Error: unknown tool {function_name}."

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                }
            )

        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )
        return second_response.choices[0].message.content or ""

    return first_message.content or ""


response_a = run_agent_q3("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# A tool WAS called here (celsius_to_fahrenheit). The query explicitly asks
# for a numeric conversion that the registered tool performs exactly, so the
# model has a clear, unambiguous match and delegates the math to the tool
# instead of computing it itself.

response_b = run_agent_q3("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# No tool was called here. This is a general knowledge question, not a
# conversion request or a time lookup -- neither available tool matches, so
# the model answers directly from its own knowledge in one API call.


# =========================================================
# --- Lesson 03: Multi-Tool Agent ---
# =========================================================

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

# The lesson uses Path("resources") because its script sits next to a
# resources/ folder containing bike_commute.csv. Place a resources/ folder
# (with bike_commute.csv inside it) directly next to this script in
# assignments_07/ for this same relative path to work.
RESOURCES_DIR = Path("resources")


class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot from the active CSV.

        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."

        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"

        if x == y:
            x = None

        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."

        title_csv = self.csv_name or "current CSV"

        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."

        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"

        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()

        return f"Plotted {y} vs {x} as a {plot_type}."

    # --- Q4: new tool added to fix the tool-round-limit failure from the lesson ---
    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()
        if error:
            return error
        if col1 not in self.df.columns or col2 not in self.df.columns:
            return {"error": f"Column '{col1}' or '{col2}' not found."}
        r, p = stats.pearsonr(self.df[col1], self.df[col2])
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4),
        }


csv_backend = CsvManager(RESOURCES_DIR)

node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "plot_data": csv_backend.plot_data,
    "compute_correlation": csv_backend.compute_correlation,  # Q4
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string", "description": "Column name for y-axis."},
                    "x": {"type": "string", "description": "Optional column name for x-axis."},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    },
    # --- Q4: schema entry for the new tool ---
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute the Pearson correlation coefficient and p-value between two numeric columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {"type": "string", "description": "First column name."},
                    "col2": {"type": "string", "description": "Second column name."},
                },
                "required": ["col1", "col2"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)


# --- Q4 ---
print("\n=== Q4 ===")
# The lesson's agent hit the tool-round limit when asked to compute a
# correlation because no tool for that existed. Fixed above by adding
# CsvManager.compute_correlation (uses scipy.stats.pearsonr, returns
# col1/col2/pearson_r/p_value rounded to 4 decimals, or an {"error": ...}
# dict if a column is missing or no CSV is loaded yet). It's also registered
# in node_tools and given a JSON schema entry in tools_schema above.
# This tool is exercised end-to-end in Q5 below.
print("compute_correlation added to CsvManager, node_tools, and tools_schema.")


def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.
    `messages` parameter will usually just contain a system prompt,
    and then user text will be appended.

    The loop has three main steps:

    REASON:
      - Call the model with the conversation so far.
      - The model either replies normally, or asks to call a tool from tool set.

    ACT:
      - If tools are requested, run the Python functions

    OBSERVE:
      - Append each requested tool result back into the LLMs conversation history.
      - On the next iteration, the model reads those tool call results and determines
        whether it has reached the goal.

    Stop condition:
      - If the model returns an assistant message with no tool calls, this is the
        final answer for this react cycle, this implies that reasoning alone without
        tool calls was enough.
      - max_tool_rounds is a safety cap to prevent infinite loops.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        """
        Return a tool's return value as a message that can be appended to the
        LLMs conversation history. The model will read this tool output on the next
        REASON step.
        """
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        tool_message = {"role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,}
        return tool_message

    for loop_idx in range(max_tool_rounds):
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)

        if not msg.tool_calls:
            return msg.content

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}

            messages.append(observe_tool_result(tool_call.id, result))

    return "I hit the tool-round limit. Try a simpler request."


# --- Q5 ---
print("\n=== Q5 ===")
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.",
)
print(result)


# --- Q6 ---
print("\n=== Q6 ===")
# Roles in the ReAct loop:
# - "system": sets the agent's overall instructions/persona before any user
#   input; sent once at the start of the conversation.
# - "user": the human's request that kicks off (or continues) the loop.
# - "assistant": the model's own turn -- either a natural-language answer, or
#   a request to call one or more tools (tool_calls), which is the "Act" step.
# - "tool": the result returned by actually running a tool the assistant
#   requested; this is the "Observe" step that gets fed back to the model so
#   it can reason further ("Reason" again) or produce a final answer.
print(json.dumps(messages, indent=2, default=str))


# =========================================================
# --- Lesson 04: smolagents ---
# =========================================================

from smolagents import tool, ToolCallingAgent, CodeAgent, OpenAIServerModel

api_key = os.getenv("OPENAI_API_KEY")

# smolagents tool wrappers for the CsvManager methods, exactly as shown in
# the lesson (each wraps a csv_backend method with the @tool decorator).

@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_backend.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".

    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_backend.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_backend.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return summary stats for selected columns (or all columns).
    This includes count, mean, std, min, max, and percentiles for numeric columns,
    or count, unique, top, freq for categorical columns.

    Args:
        columns: Column names to summarize. If None, summarizes all columns.

    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_backend.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe a single column (basic stats) for the requested column.
    This includes count, mean, std, min, max, and percentiles for numeric column,
    or count, unique, top, freq for categorical column.

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_backend.describe_column(column)


@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str | dict:
    """Plot from the active CSV.

    Args:
        y: Column name to plot on the y-axis.
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.

    Returns:
        Generates and shows the plot.
        Returns a short success message string, or an error dict/string.
    """
    return csv_backend.plot_data(y=y, x=x, plot_type=plot_type)


# --- Q7 ---
print("\n=== Q7 ===")

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns
    in the currently loaded CSV.

    Args:
        col1: Name of the first column.
        col2: Name of the second column.
    """
    return csv_backend.compute_correlation(col1, col2)

print(compute_correlation.description)
# smolagents builds its tool description straight from the docstring: the
# first line/paragraph becomes the "description", and each entry under
# "Args:" becomes the per-parameter description, all inferred automatically
# from the function signature's type hints. In Q4 (the manual JSON schema),
# I had to write out the "type": "object" / "properties" / "required"
# structure by hand and duplicate every parameter name and description.
# smolagents needs: a clear docstring summary, an "Args:" section with one
# line per parameter, and type hints on the function signature -- from that
# it generates the equivalent of the JSON schema for me.


# --- Q8 ---
print("\n=== Q8 ===")

TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,  # Q7 addition
]

model_to_use = "gpt-4o-mini"
smol_model = OpenAIServerModel(api_key=api_key, model_id=model_to_use)

SYSTEM_PROMPT = (
    "You are a small data assistant to help analyze files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Keep answers short and student-friendly."
)

tool_agent = ToolCallingAgent(tools=TOOLS, model=smol_model, instructions=SYSTEM_PROMPT)

CODE_INSTRUCTIONS = """
You are a helpful CSV analysis assistant.

You can do two kinds of actions:
1) Call the provided tools.
2) Write and execute Python code when tools are not enough.

Rules:
- Prefer tools for simple tasks.
- IMPORTANT: If the user requests plot styling (color, marker, title text, labels, grid, etc.)
  that the plot_data tool cannot control, DO NOT call plot_data.
  Instead, write matplotlib code directly so the plot matches the request.
  If code execution fails, do not fall back to plot_data when the user requested styling (like color).
  Explain what failed and what you would need to proceed.
- Be honest: only claim you did something if the code or tool actually did it.
- Assume the active dataset lives in csv_manager.df after a CSV is loaded.
"""

code_agent = CodeAgent(
    tools=TOOLS,
    model=smol_model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=8,
)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

response_tool = tool_agent.run(prompt)
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_backend})

print("ToolCallingAgent response:", response_tool)
print("CodeAgent response:", response_code)

# What each agent actually produced (filled in after running):
# The ToolCallingAgent can call load_csv and plot_data (both are registered
# tools), so it CAN load bike_commute.csv and produce a scatter plot of
# avg_heart_rate vs duration_min. But plot_data has no color parameter at
# all, so it CANNOT honor "green dots" -- it plotted with matplotlib's
# default color. Worse, its final_answer text claimed "has been created
# with green dots" anyway, which is not true -- the tool has no way to set
# color, so this is the model overstating what actually happened, not a
# real capability. This is itself a notable risk of tool-based agents: they
# can report success confidently even when the requested detail was never
# actually applied.
# The CodeAgent, by contrast, follows CODE_INSTRUCTIONS and writes its own
# matplotlib code (instead of calling plot_data) specifically because
# styling was requested, so it CAN set the dot color to green exactly as
# asked, using color='green' directly in generated code -- and its final
# answer does not falsely claim a color it didn't set.
#
# This reveals that a ToolCallingAgent is only as capable as its fixed set of
# tools -- it's safer and more predictable in terms of WHAT it can do, but
# limited to whatever the developer anticipated and wrote a tool for (here:
# no color parameter means no color control, no matter how the user phrases
# the request) -- and when it can't do something, it may still say it did.
# A CodeAgent can improvise arbitrary logic (like plot styling) that no one
# wrote a tool for, at the cost of running model-generated code.


# --- Q9 ---
print("\n=== Q9 ===")
# A ToolCallingAgent would be a better choice than a CodeAgent for a task
# like "send a Slack message to the #alerts channel when the error rate goes
# above 5%." The action here is a fixed, well-defined side effect against a
# real external system with a strict, known interface (a Slack API call)
# -- there's no need for improvisation, and any deviation (a slightly wrong
# API call written by generated code) could send malformed or unintended
# messages. A tool-based approach constrains the agent to exactly the
# vetted, tested action the developer wrote, which is what you want when the
# action has real-world side effects and a narrow, well-specified interface.
#
# One meaningful risk of a CodeAgent that doesn't apply to a ToolCallingAgent:
# the CodeAgent actually generates and EXECUTES arbitrary Python code on the
# fly (inside a restricted interpreter, per smolagents' sandboxing). That
# code could still do anything the interpreter and authorized imports allow
# -- read/write files, consume excessive memory/CPU, or contain subtle bugs
# -- with no guarantee it matches what the developer intended, unlike a
# ToolCallingAgent which can only ever call one of the specific,
# pre-written, already-reviewed functions the developer registered as tools.

print("\nDone with warmup_07.py")