import pandas as pd
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)

def to_snake_case(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'[^\w\s]', '', s)  # remove punctuation except underscore
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'__+', '_', s)
    return s

def clean_numeric_series(series: pd.Series) -> pd.Series:
    """Clean a series potentially containing numeric values with noise."""
    # Remove bracketed references [#1], commas, currency symbols, and whitespace
    cleaned = series.astype(str).str.replace(r'\[.*?\]', '', regex=True)
    cleaned = cleaned.str.replace(r'[^\d\.\-\,]', '', regex=True)  # keep digits, dot, minus, comma
    cleaned = cleaned.str.replace(',', '')  # remove thousands separators

    # Convert to numeric, coercing errors to NaN
    numeric = pd.to_numeric(cleaned, errors='coerce')

    # Try to cast to nullable Int64 if all values are integers
    if numeric.notna().all() and (numeric.dropna() % 1 == 0).all():
        numeric = numeric.astype("Int64")
    return numeric

def apply_basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Normalize column names
    df.columns = [to_snake_case(str(c)) for c in df.columns]

    # 2. Remove empty columns and rows (all null or whitespace)
    df = df.dropna(axis=1, how='all')
    df = df.loc[~df.apply(lambda row: row.astype(str).str.strip().eq('').all(), axis=1)]

    # 3. Clean string/object columns: remove references, extra whitespace, weird unicode
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r'\[.*?\]', '', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
            .replace({'nan': None, 'none': None, '': None, 'unknown': None}, regex=False)
        )

    # 4. Detect and convert numeric-like columns robustly
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(50).astype(str)
            if sample.str.contains(r'[\d]').mean() > 0.5:
                try:
                    df[col] = clean_numeric_series(df[col])
                except Exception as e:
                    logger.debug(f"Failed numeric cleaning for column '{col}': {e}")

    # 5. Detect and convert date/time columns by name hints and content
    date_cols = [c for c in df.columns if any(k in c for k in ['date', 'time', 'day', 'year'])]
    for col in date_cols:
        try:
            if 'year' in col:
                # Parse year as integer, not datetime
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            else:
                parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=True, infer_datetime_format=True)
                if parsed.notna().sum() > len(parsed) * 0.5:
                    df[col] = parsed
        except Exception as e:
            logger.debug(f"Failed date parsing for column '{col}': {e}")

    # 6. Drop exact duplicates (rows)
    df = df.drop_duplicates()

    # 7. Reset index after all drops
    df = df.reset_index(drop=True)

    return df
