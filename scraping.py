import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
import logging
from typing import Dict
from data_cleaning import apply_basic_cleaning
import httpx

logger = logging.getLogger(__name__)

AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"

def call_llm_infer_data_type(questions: str, timeout: float = 30.0) -> str:
    token = os.getenv("AIPIPE_TOKEN")
    if not token:
        raise RuntimeError("AIPIPE_TOKEN not set in environment")

    system_msg = {
        "role": "system",
        "content": (
            "You are an expert data analyst. "
            "Given a set of project questions, "
            "answer only with one of: 'table', 'text', or 'both' "
            "depending on what kind of data is needed to answer."
        )
    }
    user_msg = {
        "role": "user",
        "content": f"Here are the questions:\n{questions}\n\nAnswer with only one word: table, text, or both."
    }

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [system_msg, user_msg],
        "temperature": 0.0,
        "max_tokens": 5
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(AIPIPE_URL, json=payload, headers=headers)
            r.raise_for_status()
            resp = r.json()
            content = resp["choices"][0]["message"]["content"].strip().lower()
            if content in ("table", "text", "both"):
                return content
            else:
                logger.warning(f"Unexpected LLM data type response: {content}, defaulting to 'both'")
                return "both"
    except Exception as e:
        logger.error(f"LLM call failed for data type inference: {e}")
        return "both"

def score_text(text: str, keywords: set) -> int:
    return sum(1 for k in keywords if k in text.lower())

def scrape_url_smart(url: str, questions: str, top_n_tables=1, top_n_texts=3) -> Dict[str, Dict]:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "lxml")

    data_type = call_llm_infer_data_type(questions)
    logger.info(f"Inferred data type needed by LLM: {data_type}")

    keywords = set(re.findall(r'\w+', questions.lower()))

    result = {}

    if data_type in ("table", "both"):
        tables = soup.find_all("table")
        scored_tables = []
        for i, table in enumerate(tables):
            headers_list = [th.get_text(strip=True).lower() for th in table.find_all("th")]
            header_text = " ".join(headers_list)
            score = score_text(header_text, keywords)
            scored_tables.append((score, i, table))

        scored_tables.sort(key=lambda x: x[0], reverse=True)

        selected_tables = [t[2] for t in scored_tables if t[0] > 0][:top_n_tables]
        if not selected_tables and tables:
            selected_tables = [tables[0]]

        cleaned_tables = {}
        for idx, table in enumerate(selected_tables):
            try:
                raw_df = pd.read_html(str(table), flavor="lxml")[0]

                raw_csv_path = os.path.join(os.getcwd(), f"scraped_raw_table_{idx+1}.csv")
                raw_df.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")
                logger.info(f"Saved raw scraped table to {raw_csv_path}")

                cleaned_df = apply_basic_cleaning(raw_df)

                cleaned_csv_path = os.path.join(os.getcwd(), f"scraped_cleaned_table_{idx+1}.csv")
                cleaned_df.to_csv(cleaned_csv_path, index=False, encoding="utf-8-sig")
                logger.info(f"Saved cleaned scraped table to {cleaned_csv_path}")

                cleaned_tables[f"table_{idx+1}.csv"] = cleaned_df

            except Exception as e:
                logger.warning(f"Failed to parse or clean table {idx}: {e}")

        if cleaned_tables:
            result["tables"] = cleaned_tables

    if data_type in ("text", "both"):
        candidates = soup.find_all(["p", "li"])
        scored_texts = []
        for tag in candidates:
            txt = tag.get_text(strip=True)
            if len(txt) < 20:
                continue
            score = score_text(txt, keywords)
            if score > 0:
                scored_texts.append((score, txt))
        scored_texts.sort(key=lambda x: x[0], reverse=True)
        result["text"] = [t[1] for t in scored_texts[:top_n_texts]]

    return result