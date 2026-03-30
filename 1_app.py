import os
import json
import re
import time
import tempfile
from typing import Optional, Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

# =========================================================
# 🔧 CONFIGURATION
# =========================================================
class Config:
    DEFAULT_SHEET = "Sheet1"
    DEFAULT_ADDRESS_COL = "Address"
    SLEEP_TIME = 0.0
    USER_OVERRIDE = True   # 🔥 IMPORTANT FLAG
    HISTORY_FILE = "history_output.xlsx"

# =========================================================
# 🔐 LOAD ENV
# =========================================================
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# =========================================================
# 🧠 LLM LAYER
# =========================================================
SYSTEM_PROMPT = """
Extract city, state_or_province, country.

Return JSON only:
{
 "city": "...",
 "state_or_province": "...",
 "country": "..."
}
"""

def call_llm(text: str) -> Dict:
    try:
        res = client.chat.completions.create(
            model=DEPLOYMENT,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ]
        )
        return json.loads(res.choices[0].message.content)
    except:
        return {"city": None, "state_or_province": None, "country": None}


# =========================================================
# 🔄 BUSINESS LOGIC
# =========================================================
def extract_location(text: str) -> Dict:
    if not text:
        return {"city": None, "state_or_province": None, "country": None}

    data = call_llm(text)

    return {
        "city": data.get("city"),
        "state_or_province": data.get("state_or_province"),
        "country": data.get("country"),
    }


def process_dataframe(df: pd.DataFrame, address_col: str):
    city_list, state_list, city_state_list = [], [], []

    progress = st.progress(0)
    status = st.empty()

    total = len(df)

    for i, val in enumerate(df[address_col].fillna(""), start=1):

        result = extract_location(str(val))

        city = result["city"]
        state = result["state_or_province"]

        city_state = f"{city}, {state}" if city and state else None

        city_list.append(city)
        state_list.append(state)
        city_state_list.append(city_state)

        progress.progress(i / total)
        status.text(f"Processing {i}/{total}")

        if Config.SLEEP_TIME > 0:
            time.sleep(Config.SLEEP_TIME)

    df["City"] = city_list
    df["State/Province"] = state_list
    df["City & State"] = city_state_list

    return df


# =========================================================
# 💾 STORAGE LAYER
# =========================================================
def append_history(df: pd.DataFrame):
    if os.path.exists(Config.HISTORY_FILE):
        old = pd.read_excel(Config.HISTORY_FILE)
        df = pd.concat([old, df], ignore_index=True)

    df.to_excel(Config.HISTORY_FILE, index=False)


# =========================================================
# 🎨 UI LAYER
# =========================================================
def render_sidebar():
    st.sidebar.header("⚙️ Configuration")

    sheet = st.sidebar.text_input("Sheet Name", Config.DEFAULT_SHEET)
    col = st.sidebar.text_input("Address Column", Config.DEFAULT_ADDRESS_COL)
    sleep = st.sidebar.number_input("Delay (sec)", 0.0)

    return sheet, col, sleep


def render_upload():
    return st.file_uploader("📂 Upload Excel / CSV", type=["xlsx", "csv"])


def render_preview(df):
    with st.expander("🔍 Data Preview"):
        st.dataframe(df.head(), use_container_width=True)


# =========================================================
# ✏️ USER OVERRIDE FEATURE
# =========================================================
def apply_user_override(df):
    st.subheader("✏️ User Override (Editable Table)")

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True
    )

    return edited_df


# =========================================================
# 📥 DOWNLOAD
# =========================================================
def download_section(df):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        df.to_excel(tmp.name, index=False)

    with open(tmp.name, "rb") as f:
        st.download_button(
            "📥 Download Output",
            f,
            file_name="location_output.xlsx"
        )


# =========================================================
# 🚀 MAIN APP
# =========================================================
def main():
    st.set_page_config(page_title="Location Extractor", layout="wide")
    st.title("🌍 Location Extraction (Structured App)")

    sheet, address_col, sleep = render_sidebar()
    Config.SLEEP_TIME = sleep

    uploaded_file = render_upload()

    if uploaded_file:

        try:
            # Load file
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, sheet_name=sheet)

            st.success("✅ File Loaded")

            render_preview(df)

            if address_col not in df.columns:
                st.error(f"❌ Column '{address_col}' not found")
                return

            if st.button("🚀 Run Extraction"):

                with st.spinner("Processing..."):
                    result_df = process_dataframe(df, address_col)

                st.success("✅ Extraction Complete")

                # ---------------------------------------
                # USER OVERRIDE (KEY FEATURE)
                # ---------------------------------------
                if Config.USER_OVERRIDE:
                    result_df = apply_user_override(result_df)

                st.subheader("📊 Final Output")
                st.dataframe(result_df, use_container_width=True)

                # Save history
                append_history(result_df)
                st.info("📁 Saved to history file")

                # Download
                download_section(result_df)

        except Exception as e:
            st.error(str(e))


# =========================================================
# ▶️ ENTRY POINT (MAIN STYLE)
# =========================================================
if __name__ == "__main__":
    main()