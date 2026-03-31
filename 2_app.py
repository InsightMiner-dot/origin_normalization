import io
import json
import os
import re
import time
from typing import Dict, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

DEFAULT_SHEET_NAME = "Sheet1"
DEFAULT_ADDRESS_COLUMN = "Address"
DEFAULT_OUTPUT_FILE = "output_with_locations.xlsx"
DEFAULT_SLEEP_BETWEEN_ROWS_SEC = 0.0

load_dotenv(override=True)

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

US_STATES = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia",
}
US_STATE_NAMES = {value.lower(): value for value in US_STATES.values()}

CA_PROVINCES = {
    "AB": "Alberta",
    "BC": "British Columbia",
    "MB": "Manitoba",
    "NB": "New Brunswick",
    "NL": "Newfoundland and Labrador",
    "NS": "Nova Scotia",
    "NT": "Northwest Territories",
    "NU": "Nunavut",
    "ON": "Ontario",
    "PE": "Prince Edward Island",
    "QC": "Quebec",
    "SK": "Saskatchewan",
    "YT": "Yukon",
}
CA_PROVINCE_NAMES = {value.lower(): value for value in CA_PROVINCES.values()}

COUNTRY_NORMALIZE = {
    "us": "United States",
    "usa": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "united states": "United States",
    "canada": "Canada",
    "ca": "Canada",
}

CITY_NORMALIZE = {
    "nyc": "New York City",
}

ORG_SUFFIXES = (
    "inc",
    "inc.",
    "llc",
    "l.l.c.",
    "ltd",
    "ltd.",
    "plc",
    "corp",
    "corp.",
    "co",
    "co.",
    "company",
    "gmbh",
    "s.a.",
    "s.p.a.",
    "pte",
    "bv",
    "sarl",
    "ag",
    "oy",
    "ab",
    "sa",
    "sas",
    "sl",
    "oyj",
)
INDUSTRY_ONLY_WORDS = {
    "industry",
    "industries",
    "manufacturing",
    "logistics",
    "pharma",
    "pharmaceuticals",
    "steel",
    "paper",
    "plant",
    "mill",
    "factory",
}

SYSTEM_PROMPT = """
You are a location extraction engine.

TASK:
Extract CITY, STATE/PROVINCE, and COUNTRY from the given text. Most addresses are from the USA and CANADA.

STRICT RULES:
1. Extract only if location words appear in the text.
2. Do NOT guess based on company name, EXCEPT when the entire text is only a company/brand name and you know its official headquarters with high confidence.
3. You may use widely-known city/state/province-to-country knowledge, mainly for USA or Canada.
4. If a field is missing and cannot be inferred with high confidence, return null for that field.
5. Accept informal hints (e.g., "near airport VA", "AUGUSTA GA 30906", "Toronto ON", "NYC, NY", "Vancouver, BC Canada").
6. Normalize:
   - Expand obvious city short forms (e.g., "NYC" -> "New York City") when unambiguous.
   - "2500 W.S.R. 60-Bartow, FL 33830" -> city "Bartow", state "Florida".
   - "PEACE RIVER (BARTOW, FL) (02/26/2025)" -> city "Bartow", state "Florida".
   - When company names, plant names, suite numbers, dates, invoice text, or other noise appears alongside an address, ignore the noise and extract the embedded location if present.
   - If a full address fragment clearly contains city/state/postal information, extract from that fragment even if other unrelated tokens appear before or after it.
   - If U.S. city and state/province is present but country isn't, set country to "United States".
   - If Canadian city and province is present but country isn't, set country to "Canada".
   - Prefer the most specific/complete location when multiple appear in context; otherwise null.

EXAMPLES:
Input: "EQ DETROIT INC 1923 FREDERICK ST, DETROIT, MI 48211"
Output: {"city": "Detroit", "state_or_province": "Michigan", "country": "United States"}

Input: "PEACE RIVER (BARTOW, FL) (02/26/2025)"
Output: {"city": "Bartow", "state_or_province": "Florida", "country": "United States"}

Input: "Toronto ON M5V 2T6"
Output: {"city": "Toronto", "state_or_province": "Ontario", "country": "Canada"}

Input: "Cycle chem, Inc"
Output: {"city": null, "state_or_province": null, "country": null}

OUTPUT FORMAT (JSON ONLY):
{
  "city": string or null,
  "state_or_province": string or null,
  "country": string or null
}

No explanation. No extra words.
"""

COMPANY_HQ_PROMPT = """
You are a company headquarters resolver.

TASK:
Given a single company or brand name, return the city, state/province, and country of its primary corporate headquarters.

RULES:
- If you are unsure, or if the name looks generic or is an industry, check for the HQ address and extract City & State.
- Prefer U.S./Canada formatting for state/province abbreviations or full names if applicable.

OUTPUT FORMAT (JSON ONLY):
{
  "city": string or null,
  "state_or_province": string or null,
  "country": string or null
}

No explanation. No extra words.
"""


@st.cache_resource
def get_openai_client() -> AzureOpenAI:
    if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT):
        raise RuntimeError(
            "Missing required Azure OpenAI environment variables: "
            "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT."
        )

    return AzureOpenAI(
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
    )


def normalize_country(country: Optional[str]) -> Optional[str]:
    if not country:
        return None
    key = country.strip().lower()
    return COUNTRY_NORMALIZE.get(key, country.strip())


def expand_state_or_province(sp: Optional[str], country: Optional[str]) -> Optional[str]:
    if not sp:
        return None

    raw = sp.strip()
    lower = raw.lower()

    if lower in US_STATE_NAMES:
        return US_STATE_NAMES[lower]
    if lower in CA_PROVINCE_NAMES:
        return CA_PROVINCE_NAMES[lower]

    if 1 < len(raw) <= 3:
        up = raw.replace(".", "").upper()
        if not country:
            if up in US_STATES:
                return US_STATES[up]
            if up in CA_PROVINCES:
                return CA_PROVINCES[up]
        else:
            normalized_country = normalize_country(country)
            if normalized_country == "United States" and up in US_STATES:
                return US_STATES[up]
            if normalized_country == "Canada" and up in CA_PROVINCES:
                return CA_PROVINCES[up]

    return raw


def infer_country_from_state(sp: Optional[str]) -> Optional[str]:
    if not sp:
        return None

    token = sp.strip().replace(".", "")
    upper_token = token.upper()
    lower_token = token.lower()

    if upper_token in US_STATES or lower_token in US_STATE_NAMES:
        return "United States"
    if upper_token in CA_PROVINCES or lower_token in CA_PROVINCE_NAMES:
        return "Canada"
    return None


def normalize_city(city: Optional[str]) -> Optional[str]:
    if not city:
        return None
    cleaned_city = city.strip()
    return CITY_NORMALIZE.get(cleaned_city.lower(), cleaned_city)


def preprocess_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"[|;/]+", ", ", cleaned)
    cleaned = re.sub(r"\((\d{1,2}/\d{1,2}/\d{2,4})\)", " ", cleaned)
    cleaned = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", " ", cleaned)
    cleaned = re.sub(r"\b(invoice|inv|po|purchase order|ref|reference)\b[:#\- ]*\w*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" ,")


def has_location_signal(text: str) -> bool:
    if not text:
        return False
    upper_text = text.upper()
    has_us_state = any(f", {abbr}" in upper_text or f" {abbr} " in upper_text for abbr in US_STATES)
    has_ca_province = any(f", {abbr}" in upper_text or f" {abbr} " in upper_text for abbr in CA_PROVINCES)
    has_zip = re.search(r"\b\d{5}(?:-\d{4})?\b", text) is not None
    has_ca_postal = re.search(r"\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b", upper_text) is not None
    return has_us_state or has_ca_province or has_zip or has_ca_postal


def looks_like_company_name(text: str) -> bool:
    if not text:
        return False

    stripped_text = text.strip()
    if len(stripped_text) > 80:
        return False

    tokens = re.findall(r"[A-Za-z]+\.?", stripped_text)
    lower_tokens = [token.lower() for token in tokens]

    if any(token in INDUSTRY_ONLY_WORDS for token in lower_tokens) and len(tokens) <= 3:
        return False

    has_org_suffix = any(token in ORG_SUFFIXES for token in lower_tokens)
    few_words_title_case = (1 <= len(tokens) <= 5) and (
        sum(token[0].isupper() for token in tokens if token) >= max(1, len(tokens) - 1)
    )
    single_token_brand = len(tokens) == 1 and tokens[0][0].isupper()

    return has_org_suffix or few_words_title_case or single_token_brand


def call_llm(system_prompt: str, user_text: str) -> Dict[str, Optional[str]]:
    client = get_openai_client()
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except Exception:
        return {"city": None, "state_or_province": None, "country": None}


def extract_location(
    text: str,
    allow_company_hq_fallback: bool,
) -> Dict[str, Optional[str]]:
    if not text or not isinstance(text, str):
        return {
            "city": None,
            "state_or_province": None,
            "country": None,
            "extraction_method": "empty",
            "error": None,
        }

    cleaned_text = preprocess_text(text)
    data = call_llm(SYSTEM_PROMPT, cleaned_text)
    city = data.get("city")
    state_or_province = data.get("state_or_province")
    country = data.get("country")
    extraction_method = "llm"

    if not any([city, state_or_province, country]) and cleaned_text != text:
        retry_data = call_llm(
            SYSTEM_PROMPT,
            f"Focus on the most likely address fragment and ignore company/noise text.\nInput: {text}",
        )
        city = retry_data.get("city")
        state_or_province = retry_data.get("state_or_province")
        country = retry_data.get("country")
        extraction_method = "llm_retry"

    if (
        allow_company_hq_fallback
        and not any([city, state_or_province, country])
        and looks_like_company_name(text)
    ):
        fallback_data = call_llm(COMPANY_HQ_PROMPT, text)
        city = fallback_data.get("city")
        state_or_province = fallback_data.get("state_or_province")
        country = fallback_data.get("country")
        extraction_method = "hq_fallback"

    country = normalize_country(country)
    if not country:
        inferred_country = infer_country_from_state(state_or_province)
        if inferred_country:
            country = inferred_country

    normalized_state = expand_state_or_province(state_or_province, country)
    normalized_city = normalize_city(city)

    if not any([normalized_city, normalized_state, country]):
        extraction_method = "unresolved_with_signal" if has_location_signal(text) else "unresolved"

    return {
        "city": normalized_city,
        "state_or_province": normalized_state,
        "country": country,
        "extraction_method": extraction_method,
        "error": None,
    }


def process_dataframe(
    df: pd.DataFrame,
    address_column: str,
    sleep_between_rows_sec: float,
    allow_company_hq_fallback: bool,
) -> pd.DataFrame:
    if address_column not in df.columns:
        raise ValueError(
            f"Column '{address_column}' not found. Available columns: {list(df.columns)}"
        )

    result_df = df.copy()
    city_out = []
    state_out = []
    country_out = []
    city_state_out = []
    method_out = []
    error_out = []

    total = len(result_df)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, value in enumerate(result_df[address_column].tolist(), start=1):
        try:
            result = extract_location(
                str(value) if pd.notna(value) else "",
                allow_company_hq_fallback=allow_company_hq_fallback,
            )
            city = result.get("city")
            state_or_province = result.get("state_or_province")
            country = result.get("country")
            extraction_method = result.get("extraction_method")
            city_state = f"{city}, {state_or_province}" if city and state_or_province else None

            city_out.append(city)
            state_out.append(state_or_province)
            country_out.append(country)
            city_state_out.append(city_state)
            method_out.append(extraction_method)
            error_out.append(result.get("error"))
        except Exception as exc:
            city_out.append(None)
            state_out.append(None)
            country_out.append(None)
            city_state_out.append(None)
            method_out.append("error")
            error_out.append(str(exc))

        if sleep_between_rows_sec > 0:
            time.sleep(sleep_between_rows_sec)

        progress_bar.progress(idx / total if total else 1.0)
        if idx % 25 == 0 or idx == total:
            status_text.text(f"Processed {idx}/{total} rows")

    result_df["City"] = city_out
    result_df["State/Province"] = state_out
    result_df["Country"] = country_out
    result_df["City & State"] = city_state_out
    result_df["Extraction Method"] = method_out
    result_df["Extraction Error"] = error_out
    return result_df


def load_input_file(uploaded_file, sheet_name: str) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="openpyxl")


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output.getvalue()


def main() -> None:
    st.set_page_config(page_title="Location Extraction Tool", layout="wide")
    st.title("Location Extraction Tool")
    st.caption("Upload an Excel or CSV file, extract location fields from the address text, and download the enriched output.")

    with st.sidebar:
        st.header("Settings")
        sheet_name = st.text_input("Sheet name", value=DEFAULT_SHEET_NAME)
        address_column = st.text_input("Address column", value=DEFAULT_ADDRESS_COLUMN)
        sleep_between_rows_sec = st.number_input(
            "Sleep between rows (seconds)",
            min_value=0.0,
            value=DEFAULT_SLEEP_BETWEEN_ROWS_SEC,
            step=0.1,
        )
        allow_company_hq_fallback = st.checkbox(
            "Allow company HQ fallback",
            value=True,
            help="If no location is found and the text looks like only a company name, try resolving the company's headquarters.",
        )
        output_filename = st.text_input("Output file name", value=DEFAULT_OUTPUT_FILE)

    uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])

    if not uploaded_file:
        st.info("Upload a file to begin.")
        return

    try:
        source_df = load_input_file(uploaded_file, sheet_name)
    except Exception as exc:
        st.error(f"Unable to read the uploaded file: {exc}")
        return

    st.success("File loaded successfully.")
    st.write(f"Rows: {len(source_df)}")
    st.write(f"Columns: {', '.join(source_df.columns.astype(str))}")

    with st.expander("Preview input data", expanded=True):
        st.dataframe(source_df.head(20), use_container_width=True)

    if address_column not in source_df.columns:
        st.error(f"Column '{address_column}' not found in the uploaded data.")
        return

    if st.button("Run extraction", type="primary"):
        try:
            with st.spinner("Extracting locations..."):
                result_df = process_dataframe(
                    source_df,
                    address_column=address_column,
                    sleep_between_rows_sec=sleep_between_rows_sec,
                    allow_company_hq_fallback=allow_company_hq_fallback,
                )
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
            return

        st.success("Extraction completed.")

        with st.expander("Preview output data", expanded=True):
            st.dataframe(result_df.head(50), use_container_width=True)

        st.download_button(
            label="Download output Excel",
            data=dataframe_to_excel_bytes(result_df),
            file_name=output_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
