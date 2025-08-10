# safe_exec.py
import io
import base64
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sqlite3
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def safe_execute_code_with_db(code: str, db_path: str, questions_text: str) -> Dict[str, Any]:
    """
    Execute generated code in a controlled namespace.
    Returns dict: {"answer": <converted>, "plot_data_uri": <str or None>, "error": <str or None>}
    """
    local_vars = {}
    plot_data_uri = None
    error = None
    answer = None

    # prepare globals
    exec_globals = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "sqlite3": sqlite3,
        "conn": None,
        "db_path": db_path,
        "io": io,
        "base64": base64,
    }

    try:
        # open a sqlite connection and put into namespace
        conn = sqlite3.connect(db_path)
        exec_globals["conn"] = conn

        # override input to avoid LLM generated blocking input()
        exec_globals["input"] = lambda *a, **k: ""

        # execute
        exec(code, exec_globals, local_vars)

        # attempt to fetch result-like variables
        for name in ("result", "results", "output", "answer", "answers"):
            if name in local_vars:
                result_var = local_vars[name]
                break
        else:
            result_var = None

        def convert_item(item):
            # pandas DataFrame
            if isinstance(item, pd.DataFrame):
                if item.shape == (1,1):
                    val = item.iloc[0,0]
                    try:
                        return val.item()
                    except:
                        return val
                return item.to_dict(orient="records")
            if isinstance(item, pd.Series):
                return item.tolist()
            if isinstance(item, np.generic):
                return item.item()
            try:
                import json
                json.dumps(item)
                return item
            except:
                return str(item)

        if isinstance(result_var, (list, tuple)):
            answer = [convert_item(it) for it in result_var]
        elif result_var is not None:
            answer = convert_item(result_var)
        else:
            answer = None

        # capture matplotlib figure if any
        figs = [plt.figure(n) for n in plt.get_fignums()]
        if figs:
            buf = io.BytesIO()
            figs[0].savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_bytes = buf.read()
            plot_data_uri = "data:image/png;base64," + base64.b64encode(img_bytes).decode()
            plt.close('all')

        # also check local variables for base64 strings
        if not plot_data_uri:
            for k,v in local_vars.items():
                if isinstance(v, str) and v.startswith("data:image"):
                    plot_data_uri = v
                    break

        conn.close()
    except Exception as e:
        error = str(e) + "\n" + traceback.format_exc()
        logger.exception("Error executing generated code: %s", e)
    return {"answer": answer, "plot_data_uri": plot_data_uri, "error": error}
