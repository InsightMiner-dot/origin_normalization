import os
import json
import time
import tempfile
from typing import List, Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

# =========================================================
# 🔧 CONFIG
# =========================================================
class Config:
    DEFAULT_SHEET = "Sheet1"
    DEFAULT_ADDRESS_COL = "Address"
    BATCH_SIZE = 20  # 🔥 KEY PERFORMANCE PARAM
    SLEEP_TIME = 0.0
    USER_OVERRIDE = True
    HISTORY_FILE = "history_output.xlsx"

# =========================================================
# 🔐 ENV + CLIENT
# =========================================================
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# =========================================================
# 🧠 PROMPT
# =========================================================
SYSTEM_PROMPT = """
You are a location extraction engine.

Extract CITY, STATE/PROVINCE, COUNTRY.

Return STRICT JSON ARRAY:
[
 {"id": 0, "city": "...", "state_or_province": "...", "country": "..."}
]

RULES:
- Keep same id
- Do NOT mix rows
- If unknown → null
- No explanation
"""

# =========================================================
# 🚀 BATCH LLM FUNCTION (SAFE)
# =========================================================
def batch_extract_locations(texts: List[str]) -> List[Dict]:

    results = []

    for i in range(0, len(texts), Config.BATCH_SIZE):
        batch = texts[i:i + Config.BATCH_SIZE]

        payload = [{"id": idx, "text": txt} for idx, txt in enumerate(batch)]

        user_prompt = f"""
INPUT:
{json.dumps(payload)}
"""

        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
            )

            raw = response.choices[0].message.content
            parsed = json.loads(raw)

            mapped = {r["id"]: r for r in parsed}

            for idx in range(len(batch)):
                results.append(mapped.get(idx, {
                    "city": None,
                    "state_or_province": None,
                    "country": None
                }))

        except Exception:
            # 🔁 FALLBACK (row-by-row safe)
            for txt in batch:
                results.append({
                    "city": None,
                    "state_or_province": None,
                    "country": None
                })

        if Config.SLEEP_TIME > 0:
            time.sleep(Config.SLEEP_TIME)

    return results


# =========================================================
# 🔄 PROCESS LOGIC
# =========================================================
def process_dataframe(df: pd.DataFrame, address_col: str):

    texts = df[address_col].fillna("").astype(str).tolist()

    progress = st.progress(0)
    status = st.empty()

    batch_results = []
    total = len(texts)

    for i in range(0, total, Config.BATCH_SIZE):
        chunk = texts[i:i + Config.BATCH_SIZE]

        chunk_results = batch_extract_locations(chunk)
        batch_results.extend(chunk_results)

        progress.progress(min((i + Config.BATCH_SIZE) / total, 1.0))
        status.text(f"Processed {min(i + Config.BATCH_SIZE, total)}/{total}")

    df["City"] = [r.get("city") for r in batch_results]
    df["State/Province"] = [r.get("state_or_province") for r in batch_results]
    df["Country"] = [r.get("country") for r in batch_results]

    df["City & State"] = [
        f"{c}, {s}" if c and s else None
        for c, s in zip(df["City"], df["State/Province"])
    ]

    return df


# =========================================================
# 💾 STORAGE
# =========================================================
def append_history(df: pd.DataFrame):
    if os.path.exists(Config.HISTORY_FILE):
        old = pd.read_excel(Config.HISTORY_FILE)
        df = pd.concat([old, df], ignore_index=True)

    df.to_excel(Config.HISTORY_FILE, index=False)


# =========================================================
# ✏️ USER OVERRIDE
# =========================================================
def apply_user_override(df):
    st.subheader("✏️ Edit / Override Results")
    return st.data_editor(df, use_container_width=True)


# =========================================================
# 📥 DOWNLOAD
# =========================================================
def download_section(df):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        df.to_excel(tmp.name, index=False)

    with open(tmp.name, "rb") as f:
        st.download_button("📥 Download Output", f, "output.xlsx")


# =========================================================
# 🎨 UI
# =========================================================
def main():

    st.set_page_config(page_title="Location Extractor", layout="wide")
    st.title("🌍 AI Location Extraction (Batch Optimized)")

    # Sidebar
    st.sidebar.header("⚙️ Settings")

    sheet = st.sidebar.text_input("Sheet Name", Config.DEFAULT_SHEET)
    address_col = st.sidebar.text_input("Address Column", Config.DEFAULT_ADDRESS_COL)

    Config.BATCH_SIZE = st.sidebar.slider("Batch Size", 5, 50, 20)

    uploaded_file = st.file_uploader("📂 Upload Excel / CSV", type=["xlsx", "csv"])

    if uploaded_file:

        try:
            # Load
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, sheet_name=sheet)

            st.success("✅ File Loaded")

            with st.expander("🔍 Preview"):
                st.dataframe(df.head(), use_container_width=True)

            if address_col not in df.columns:
                st.error(f"❌ Column '{address_col}' not found")
                return

            if st.button("🚀 Run Extraction"):

                with st.spinner("Processing with batch LLM..."):
                    result_df = process_dataframe(df, address_col)

                st.success("✅ Extraction Completed")

                # ✏️ Override
                if Config.USER_OVERRIDE:
                    result_df = apply_user_override(result_df)

                st.subheader("📊 Final Output")
                st.dataframe(result_df, use_container_width=True)

                # Save
                append_history(result_df)
                st.info("📁 Saved to history")

                # Download
                download_section(result_df)

        except Exception as e:
            st.error(str(e))


# =========================================================
# ▶️ ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()
