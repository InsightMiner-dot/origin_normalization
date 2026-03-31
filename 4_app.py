import io
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= CONFIG =================
DEFAULT_ADDRESS_COLUMN = "Address"
DEFAULT_OUTPUT_FILE = "output_with_locations.xlsx"
DEFAULT_MASTER_DATABASE_FILE = "origin_normalization_master_database.csv"

load_dotenv(override=True)

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# ================= CLIENT =================
@st.cache_resource
def get_openai_client():
    return AzureOpenAI(
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
    )

# ================= NORMALIZATION =================
def normalize_country(country):
    if not country:
        return None
    key = country.strip().lower()
    return {
        "us": "United States",
        "usa": "United States",
        "canada": "Canada",
    }.get(key, country)


def normalize_city(city):
    if not city:
        return None
    return city.strip()


def expand_state_or_province(sp, country):
    return sp


def infer_country_from_state(sp):
    return None


def preprocess_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ================= MASTER DB =================
def empty_master_database():
    return pd.DataFrame(columns=["lookup_key", "city", "state_or_province", "country"])


def load_master_database(path):
    if not os.path.exists(path):
        return empty_master_database()
    try:
        return pd.read_csv(path)
    except:
        return empty_master_database()


def save_master_database(df, path):
    df.to_csv(path, index=False)


def build_lookup_key(text):
    return preprocess_text(text).lower()


def get_master_match(master_df, text):
    key = build_lookup_key(text)
    if master_df.empty:
        return None
    match = master_df[master_df["lookup_key"] == key]
    if match.empty:
        return None
    row = match.iloc[-1]
    return {
        "city": row["city"],
        "state_or_province": row["state_or_province"],
        "country": row["country"],
        "extraction_method": "master_database",
    }


def build_master_record(text, res):
    return {
        "lookup_key": build_lookup_key(text),
        "city": res.get("city"),
        "state_or_province": res.get("state_or_province"),
        "country": res.get("country"),
    }


# ================= LLM =================
SYSTEM_PROMPT = "Extract CITY, STATE/PROVINCE, COUNTRY. Return JSON."

def call_llm_batch(texts):
    client = get_openai_client()

    combined = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])

    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Return JSON:\n{combined}"},
        ],
    )

    try:
        return json.loads(response.choices[0].message.content)["results"]
    except:
        return [{"city": None, "state_or_province": None, "country": None}] * len(texts)


def call_llm_batch_with_retry(texts, retries=3):
    for i in range(retries):
        try:
            return call_llm_batch(texts)
        except:
            time.sleep(2 ** i)
    return [{"city": None, "state_or_province": None, "country": None}] * len(texts)


def process_batch(batch):
    outputs = call_llm_batch_with_retry(batch)
    results = []

    for out in outputs:
        country = normalize_country(out.get("country"))
        results.append({
            "city": normalize_city(out.get("city")),
            "state_or_province": out.get("state_or_province"),
            "country": country,
            "extraction_method": "batch_llm",
            "error": None,
        })

    return results


# ================= CORE =================
def process_dataframe(df, address_column, master_path):
    master_df = load_master_database(master_path)

    total = len(df)
    progress = st.progress(0)
    status = st.empty()

    start_time = time.time()

    # Deduplicate
    unique_vals = df[address_column].fillna("").astype(str).unique()

    cache = {}
    to_process = []

    for val in unique_vals:
        match = get_master_match(master_df, val)
        if match:
            cache[val] = match
        else:
            to_process.append(val)

    BATCH_SIZE = 8
    MAX_WORKERS = 4

    batches = [to_process[i:i+BATCH_SIZE] for i in range(0, len(to_process), BATCH_SIZE)]

    results_map = {}
    records = []
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_batch, b): b for b in batches}

        for future in as_completed(futures):
            batch = futures[future]
            res = future.result()

            for t, r in zip(batch, res):
                results_map[t] = r
                records.append(build_master_record(t, r))

            completed += len(batch)

            # ⏱ TIME
            elapsed = time.time() - start_time
            speed = completed / elapsed if elapsed else 0
            remaining = len(to_process) - completed
            eta = remaining / speed if speed else 0

            progress.progress(min(completed / max(len(to_process), 1), 1))

            status.text(
                f"Processed {completed}/{len(to_process)} | "
                f"Speed {speed:.2f}/s | Elapsed {elapsed:.1f}s | ETA {eta:.1f}s"
            )

    # Map back
    city, state, country = [], [], []

    for val in df[address_column].fillna("").astype(str):
        res = cache.get(val) or results_map.get(val, {})
        city.append(res.get("city"))
        state.append(res.get("state_or_province"))
        country.append(res.get("country"))

    df["City"] = city
    df["State"] = state
    df["Country"] = country

    # Save DB
    master_df = pd.concat([master_df, pd.DataFrame(records)]).drop_duplicates("lookup_key", keep="last")
    save_master_database(master_df, master_path)

    st.success(f"Completed in {time.time() - start_time:.2f}s 🚀")

    return df


# ================= UI =================
def main():
    st.title("🚀 Location Extraction (Batch + Parallel + Timer)")

    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if not file:
        return

    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

    st.dataframe(df.head())

    if st.button("Run Extraction"):
        result = process_dataframe(df, DEFAULT_ADDRESS_COLUMN, DEFAULT_MASTER_DATABASE_FILE)

        st.dataframe(result.head())

        st.download_button(
            "Download CSV",
            data=result.to_csv(index=False),
            file_name="output.csv",
        )


if __name__ == "__main__":
    main()
