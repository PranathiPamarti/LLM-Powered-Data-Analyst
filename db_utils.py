# db_utils.py
import sqlite3
import tempfile
import os
import re
import pandas as pd
from typing import Dict

def store_dataframes_in_db(dataframes: Dict[str, pd.DataFrame]) -> str:
    """
    Store all dataframes in a temporary SQLite database and return the db path.
    Names come from keys of dataframes (safe table names will be created).
    """
    db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_path = db_file.name
    db_file.close()

    conn = sqlite3.connect(db_path)
    for name, df in dataframes.items():
        # sanitize table name
        base = os.path.splitext(os.path.basename(name))[0]
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', base)
        if table_name and table_name[0].isdigit():
            table_name = "t_" + table_name
        if not table_name:
            table_name = "table_1"
        try:
            df.to_sql(table_name, conn, index=False, if_exists="replace")
        except Exception as e:
            # fall back: store as CSV in a single-table called 'raw_table'
            print(f"[db_utils] Failed to store {name} as table {table_name}: {e}")
            try:
                df2 = pd.DataFrame({"error": [str(e)]})
                df2.to_sql(f"{table_name}_error", conn, index=False, if_exists="replace")
            except:
                pass
    conn.close()
    return db_path

def create_temp_sqlite_db_from_single_df(df, table_name="table1") -> str:
    tmp = tempfile
