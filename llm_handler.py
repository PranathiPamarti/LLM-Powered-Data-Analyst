# llm_handler.py
import os
import httpx
import json
import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"

def generate_code_with_llm(
    questions_text: str,
    db_path: str,
    table_schemas: Dict[str, List[str]],
    sample_rows: Dict[str, list] = None,
    timeout: float = 60.0,
) -> str:
    """
    Request the LLM to generate Python code that queries the SQLite database at db_path
    and answers the given questions in order.
    The generated code must produce a variable `result` that is a Python list of answers.
    Plots must be returned as base64-encoded PNG strings within `result`.
    The code should gracefully handle small datasets, missing data, and avoid errors with `.str` and correlation.
    """
    token = os.getenv("AIPIPE_TOKEN")
    if not token:
        raise RuntimeError("AIPIPE_TOKEN not set in environment")

    # Construct detailed and explicit prompt for robust, safe code generation
    context = (
        "You are a skilled data analyst and Python developer.\n"
        "You will be given the path to a SQLite database, a set of natural language questions, "
        "and metadata about tables and columns in the database.\n\n"

        "Your task is to write complete, executable Python code (without additional explanations) that please im begging you do a good job:\n"
        "- Uses only these imports: pandas, numpy, sqlite3, matplotlib.pyplot as plt, io, base64.\n"
        "- Connects to the SQLite database at the given path using sqlite3.\n"
        "- Loads data from tables using pandas.read_sql_query.\n"
        "- Performs all calculations in pandas (do NOT use unsupported SQLite functions like MEDIAN).\n"
        "- Handles missing or non-numeric data gracefully, dropping or filling as needed.\n"
        "- Uses `.str` methods ONLY on columns confirmed to be string dtype to avoid errors.\n"
        "- Computes correlations only when columns exist and contain sufficient numeric data; return None otherwise.\n"
        "- Calculates medians via pandas.\n"
        "- For requested plots, generates matplotlib figures with proper axis labels and titles, saves them to bytes buffers as PNGs, encodes to base64 strings, and includes them in the `result` list.\n"
        "- Closes figures after saving to avoid memory leaks or empty plots.\n"
        "- Returns the final answers as a Python list named `result`, where each element corresponds to each question in order.\n"
        "- Uses informative variable names and comments sparingly to keep code clear.\n\n"

        f"Database path: {db_path}\n"
        f"Questions:\n{questions_text.strip()}\n\n"
        "Available tables and their columns:\n"
    )

    for table_name, columns in table_schemas.items():
        context += f"- {table_name}: {', '.join(columns)}\n"
        if sample_rows and table_name in sample_rows:
            try:
                sample_json = json.dumps(sample_rows[table_name], ensure_ascii=False, indent=2)
                context += f"  Sample rows:\n{sample_json}\n"
            except Exception:
                pass

    system_msg = {
        "role": "system",
        "content": "You are a helpful data analyst. Respond with ONLY the Python code. Enclose it within triple backticks or provide raw code."
    }
    user_msg = {"role": "user", "content": context}

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [system_msg, user_msg],
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(AIPIPE_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            code = data["choices"][0]["message"]["content"]

            # Strip code fences if present
            code = re.sub(r"```(?:python)?\n(.*?)```", r"\1", code, flags=re.DOTALL).strip()
            return code

    except Exception as e:
        logger.exception("Error calling LLM: %s", e)
        raise
