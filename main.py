import os
import io
import re
import json
import logging
import base64
from typing import Tuple, Any, Dict
import pandas as pd
import sqlite3
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from PIL import Image  # Pillow for image processing
import httpx  # needed for async LLM calls

load_dotenv()

from db_utils import store_dataframes_in_db
from scraping import scrape_url_smart  # Your new smart scraper function
from llm_handler import generate_code_with_llm
from safe_exec import safe_execute_code_with_db
from data_cleaning import apply_basic_cleaning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Refactored Data Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def is_df_empty(df: pd.DataFrame) -> bool:
    if df.empty or df.shape[1] == 0 or df.dropna(how="all").empty:
        return True
    return False

def is_base64_image_string(s: str) -> bool:
    # Rough heuristic: very long string with base64 chars and starts with typical image prefix
    if not isinstance(s, str):
        return False
    if len(s) < 100:
        return False
    # Common base64 prefixes for png/jpg/gif (without data URI prefix)
    prefixes = ['iVBOR', '/9j/', 'R0lGOD']  # PNG, JPG, GIF signatures base64 start
    return any(s.startswith(p) for p in prefixes)

def compress_base64_image(b64_str: str, max_width: int = 800, quality: int = 70) -> str:
    try:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_bytes))

        # Resize preserving aspect ratio if too wide
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.LANCZOS)

        buffer = io.BytesIO()
        # Save as JPEG or PNG depending on original format
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(buffer, format="JPEG", quality=quality)
        compressed_bytes = buffer.getvalue()
        compressed_b64 = base64.b64encode(compressed_bytes).decode('utf-8')
        return compressed_b64
    except Exception as e:
        logger.warning(f"Failed to compress image: {e}")
        # Return original if compression fails
        return b64_str

def separate_and_compress_images(data: Any) -> Tuple[Any, Dict[str, str]]:
    """
    Recursively walk through `data` (dict or list),
    extract and compress base64 image strings,
    return tuple (data_without_images, dict_of_images).
    """
    images = {}

    def recurse(obj, path=""):
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                new_path = f"{path}.{k}" if path else k
                if isinstance(v, str) and is_base64_image_string(v):
                    # Compress and save in images dict, replace with placeholder
                    compressed = compress_base64_image(v)
                    images[new_path] = compressed
                    new_obj[k] = f"<<base64_image_removed:{new_path}>>"
                else:
                    new_obj[k] = recurse(v, new_path)
            return new_obj
        elif isinstance(obj, list):
            new_list = []
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                if isinstance(item, str) and is_base64_image_string(item):
                    compressed = compress_base64_image(item)
                    images[new_path] = compressed
                    new_list.append(f"<<base64_image_removed:{new_path}>>")
                else:
                    new_list.append(recurse(item, new_path))
            return new_list
        else:
            return obj

    cleaned_data = recurse(data)
    return cleaned_data, images

def restore_images(data: Any, images: Dict[str, str]) -> Any:
    """
    Recursively replace placeholders <<base64_image_removed:path>> with actual compressed base64 string.
    """
    def recurse(obj):
        if isinstance(obj, dict):
            return {k: recurse(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recurse(i) for i in obj]
        elif isinstance(obj, str):
            m = re.match(r"<<base64_image_removed:(.+)>>", obj)
            if m:
                path = m.group(1)
                return images.get(path, obj)
            else:
                return obj
        else:
            return obj

    return recurse(data)

async def call_llm_with_prompt(prompt: str, timeout: float = 60.0) -> str:
    token = os.getenv("AIPIPE_TOKEN")
    if not token:
        raise RuntimeError("AIPIPE_TOKEN not set in environment")

    url = "https://aipipe.org/openrouter/v1/chat/completions"
    system_msg = {
        "role": "system",
        "content": "You are a helpful assistant. Respond ONLY with the requested JSON."
    }
    user_msg = {"role": "user", "content": prompt}

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [system_msg, user_msg],
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

async def structure_answer_with_llm(questions_text: str, raw_answer) -> dict:
    prompt = f"""
You are a helpful assistant.

Given these questions:

{questions_text}

And these raw answers (as JSON):

{json.dumps(raw_answer, indent=2)}

Please respond with a JSON object that maps each question exactly to its corresponding answer ONLY IF such a schema is mentioned explicitly.

Respond ONLY with the JSON object.
"""
    try:
        structured_response_text = await call_llm_with_prompt(prompt)
        structured = json.loads(structured_response_text)
        return structured
    except Exception as e:
        logger.error(f"Failed to parse structured JSON from LLM: {e}")
        # fallback: return raw answer as is
        return raw_answer

@app.post("/api/")
async def analyze_data(request: Request):
    try:
        form = await request.form()

        # 1. Get questions text
        questions_text = ""
        if "questions.txt" in form:
            f = form["questions.txt"]
            content = await f.read()
            questions_text = content.decode("utf-8")
        elif "questions" in form:
            questions_text = str(form["questions"])
        else:
            return JSONResponse({"error": "questions.txt file or 'questions' field required"}, status_code=400)

        # 2. Collect uploaded dataframes
        dataframes = {}
        for key, item in form.items():
            if key in ("questions.txt", "questions", "url"):
                continue
            try:
                if hasattr(item, "filename") and item.filename:
                    filename = item.filename
                    content = await item.read()
                    if filename.lower().endswith(".csv"):
                        df = pd.read_csv(io.BytesIO(content))
                    elif filename.lower().endswith(".json"):
                        df = pd.read_json(io.BytesIO(content))
                    elif filename.lower().endswith(".parquet"):
                        df = pd.read_parquet(io.BytesIO(content))
                    else:
                        continue
                    if not is_df_empty(df):
                        dataframes[filename] = df
                    else:
                        logger.info(f"Uploaded file {filename} resulted in empty dataframe, skipping.")
            except Exception as e:
                logger.debug(f"Skipping non-file or unreadable form item {key}: {e}")
                continue

        # 3. Scrape if no data uploaded
        scraped_text_snippets = []
        if not dataframes:
            urls_to_scrape = set()

            if "url" in form and str(form["url"]).strip():
                urls_to_scrape.add(str(form["url"]).strip())

            url_pattern = re.compile(r'https?://[^\s)]+')
            urls_found = url_pattern.findall(questions_text)
            for u in urls_found:
                urls_to_scrape.add(u.strip())

            if urls_to_scrape:
                logger.info(f"Detected URLs for scraping: {urls_to_scrape}")
                for u in urls_to_scrape:
                    scraped = scrape_url_smart(u, questions_text)

                    if "tables" in scraped:
                        for k, df in scraped["tables"].items():
                            if not is_df_empty(df):
                                dataframes[k] = df

                    if "text" in scraped:
                        scraped_text_snippets.extend(scraped["text"])

                    if not scraped.get("tables") and not scraped.get("text"):
                        logger.warning(f"No relevant data found for URL: {u}")

        if not dataframes:
            return JSONResponse({"error": "No dataframes provided via file upload or URL scraping"}, status_code=400)

        # 5. Clean dataframes
        cleaned = {}
        for name, df in dataframes.items():
            logger.info(f"Cleaning dataframe '{name}' shape before cleaning: {df.shape}")
            try:
                cleaned_df = apply_basic_cleaning(df)
                if is_df_empty(cleaned_df):
                    logger.warning(f"Dataframe '{name}' is empty after cleaning, skipping.")
                    continue
                cleaned[name] = cleaned_df
                logger.info(f"Dataframe '{name}' shape after cleaning: {cleaned_df.shape}")
            except Exception as e:
                logger.error(f"Error cleaning dataframe '{name}': {e}")

        if not cleaned:
            return JSONResponse({"error": "All dataframes were empty or invalid after cleaning"}, status_code=400)

        # 6. Store cleaned dataframes in SQLite
        db_path = store_dataframes_in_db(cleaned)

        # 7. Get SQLite table schemas
        conn_tables = {}
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [r[0] for r in cur.fetchall()]
            for t in tables:
                cur.execute(f"PRAGMA table_info({t});")
                cols = [c[1] for c in cur.fetchall()]
                conn_tables[t] = cols
            conn.close()
        except Exception as e:
            logger.debug(f"Could not fetch schemas: {e}")

        # 8. Prepare sample rows
        sample_rows = {}
        for k, df in cleaned.items():
            base = re.sub(r'[^a-zA-Z0-9_]', '_', os.path.splitext(os.path.basename(k))[0])
            if base and base[0].isdigit():
                base = "t_" + base
            if not base:
                base = "table_1"
            try:
                sample_rows[base] = df.head(5).to_dict(orient="records")
            except Exception:
                sample_rows[base] = []

        # 9. Generate code with LLM
        code = generate_code_with_llm(
            questions_text,
            db_path=db_path,
            table_schemas=conn_tables,
            sample_rows=sample_rows,
            timeout=120.0
        )

        logger.info(f"Generated code:\n{code}")

                # 10. Execute code safely
        exec_result = safe_execute_code_with_db(code, db_path, questions_text)
        ans = exec_result.get("answer")

        # 11. Separate and compress any base64 images inside answer
        ans_without_images, compressed_images = separate_and_compress_images(ans)

        # 12. Send only answer part to LLM for mapping
        mapped = await structure_answer_with_llm(questions_text, ans_without_images)

        # 13. Restore compressed images separately at top-level keys
        # Remove any existing image placeholders from mapped to avoid duplicates
        # Then add compressed images back as top-level keys
        # 13. Restore compressed images inside the structured mapped answer recursively
        mapped = restore_images(mapped, compressed_images)


        return JSONResponse(content=mapped)

    except Exception as e:
        logger.exception("Failed to analyze data")
        return JSONResponse(content={"error": str(e)}, status_code=500)
