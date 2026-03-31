import io
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI, BadRequestError

DEFAULT_ADDRESS_COLUMN = "Address"
DEFAULT_SLEEP_BETWEEN_ROWS_SEC = 0.0
DEFAULT_MASTER_DATABASE_FILE = "origin_normalization_master_database.csv"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIT_DIR = os.path.join(APP_DIR, "audit")
os.makedirs(AUDIT_DIR, exist_ok=True)
DEFAULT_LOG_FILE = os.path.join(AUDIT_DIR, "origin_normalization_app.log")

load_dotenv(override=True)

logging.basicConfig(
    filename=DEFAULT_LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

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
2. Do NOT guess based on company name, EXCEPT when the entire text is only a company/brand name and you know its official headquarters with moderate confidence.
3. You may use widely-known city/state/province-to-country knowledge, mainly for USA or Canada.
4. If a field is missing and cannot be inferred with moderate confidence, return null for that field.
5. Accept informal hints and noisy prefixes/suffixes (e.g., "near airport VA", "AUGUSTA GA 30906", "Toronto ON", "NYC, NY", "Vancouver, BC Canada", "Attn:", "Ship To:", "Job Site:", "Job site:", "Customer:", "Plant:", "Site:").
6. Normalize:
   - Expand obvious city short forms (e.g., "NYC" -> "New York City") when unambiguous.
   - "2500 W.S.R. 60-Bartow, FL 33830" -> city "Bartow", state "Florida".
   - "PEACE RIVER (BARTOW, FL) (02/26/2025)" -> city "Bartow", state "Florida".
   - When company names, plant names, suite numbers, dates, invoice text, or other noise appears alongside an address, ignore the noise and extract the embedded location if present.
   - If the text contains a company name followed by a street address and then a city/state/postal code, treat it as an address record, not as a company-only row.
   - In strings like "COMPANY NAME 1923 FREDERICK ST, DETROIT, MI 48211", ignore the company name and street if needed, and extract city "Detroit", state "Michigan", country "United States".
   - When labels like "Attn:", "Ship To:", "Job Site:", "Job site:", or similar routing text appear, ignore those labels and extract the actual address location.
   - When digits and company names are concatenated with the address text, separate the noise mentally and extract the most likely city/state/province from the address fragment.
   - If a full address fragment clearly contains city/state/postal information, extract from that fragment even if other unrelated tokens appear before or after it.
   - If U.S. city and state/province is present but country isn't, set country to "United States".
   - If Canadian city and province is present but country isn't, set country to "Canada".
   - If the text contains only an organization, company, or industry name and no address is present, use the primary corporate headquarters with moderate confidence instead of leaving fields null.
   - Prefer the most specific/complete location when multiple appear in context; otherwise null.

EXAMPLES:
Input: "EQ DETROIT INC 1923 FREDERICK ST, DETROIT, MI 48211"
Output: {"city": "Detroit", "state_or_province": "Michigan", "country": "United States"}

Input: "ABC INDUSTRIES 500 MAIN ST, CLEVELAND, OH 44114"
Output: {"city": "Cleveland", "state_or_province": "Ohio", "country": "United States"}

Input: "PEACE RIVER (BARTOW, FL) (02/26/2025)"
Output: {"city": "Bartow", "state_or_province": "Florida", "country": "United States"}

Input: "Toronto ON M5V 2T6"
Output: {"city": "Toronto", "state_or_province": "Ontario", "country": "Canada"}

Input: "Job site:Reworld9400 STRANG RDLA PORTE TX 77571United state"
Output: {"city": "La Porte", "state_or_province": "Texas", "country": "United States"}

Input: "107197100 Vexor Technology Inc 955 West Smith Road Medina, OH, 44256"
Output: {"city": "Medina", "state_or_province": "Ohio", "country": "United States"}

Input: "Cycle Chem, Inc"
Output: {"city": "Elizabeth", "state_or_province": "New Jersey", "country": "United States"}

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
- Answer when you are moderately or highly confident.
- If you are unsure, or if the name looks generic or is an industry (e.g., "paper mill", "pharma"), return nulls.
- Prefer U.S./Canada formatting for state/province abbreviations or full names if applicable.
- If the text contains only an organization, company, or industry name, prefer returning the primary corporate headquarters instead of null when you have moderate confidence.

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
    cleaned = "".join(ch if ch.isprintable() or ch in "\n\t " else " " for ch in cleaned)
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = cleaned.replace("\u200b", " ")
    cleaned = cleaned.replace("\ufeff", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"[|;/]+", ", ", cleaned)
    cleaned = re.sub(r"\((\d{1,2}/\d{1,2}/\d{2,4})\)", " ", cleaned)
    cleaned = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", " ", cleaned)
    cleaned = re.sub(r"\b(invoice|inv|po|purchase order|ref|reference)\b[:#\- ]*\w*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(attn|ship to|job site|job site address|jobsite|site|plant|customer)\s*:", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"([A-Za-z])(\d)", r"\1 \2", cleaned)
    cleaned = re.sub(r"(\d)([A-Za-z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"\b(usa|us|united state)\b", "United States", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" ,")


def build_lookup_key(text: str) -> str:
    return preprocess_text(text).lower()


def has_location_signal(text: str) -> bool:
    if not text:
        return False
    upper_text = text.upper()
    has_us_state = any(f", {abbr}" in upper_text or f" {abbr} " in upper_text for abbr in US_STATES)
    has_ca_province = any(f", {abbr}" in upper_text or f" {abbr} " in upper_text for abbr in CA_PROVINCES)
    has_zip = re.search(r"\b\d{5}(?:-\d{4})?\b", text) is not None
    has_ca_postal = re.search(r"\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b", upper_text) is not None
    return has_us_state or has_ca_province or has_zip or has_ca_postal


def aggressive_sanitize_text(text: str) -> str:
    cleaned = preprocess_text(text)
    cleaned = re.sub(r"[^A-Za-z0-9,\-#&()/. ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" ,")


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


def empty_usage() -> Dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def merge_usage(*usage_items: Dict[str, int]) -> Dict[str, int]:
    merged = empty_usage()
    for usage in usage_items:
        for key in merged:
            merged[key] += int(usage.get(key, 0))
    return merged


def call_llm(system_prompt: str, user_text: str) -> Tuple[Dict[str, Optional[str]], Dict[str, int]]:
    client = get_openai_client()
    logger.info("Calling Azure OpenAI for location extraction.")
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    )
    usage = {
        "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
    }
    raw = response.choices[0].message.content
    try:
        return json.loads(raw), usage
    except Exception:
        return {"city": None, "state_or_province": None, "country": None}, usage


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
            "usage": empty_usage(),
        }

    cleaned_text = preprocess_text(text)
    try:
        data, usage = call_llm(SYSTEM_PROMPT, cleaned_text)
    except BadRequestError:
        logger.warning("BadRequestError on primary extraction. Retrying with sanitized text.")
        cleaned_text = aggressive_sanitize_text(text)
        data, usage = call_llm(SYSTEM_PROMPT, cleaned_text)

    city = data.get("city")
    state_or_province = data.get("state_or_province")
    country = data.get("country")
    extraction_method = "llm"

    if not any([city, state_or_province, country]) and cleaned_text != text:
        try:
            retry_data, retry_usage = call_llm(
                SYSTEM_PROMPT,
                f"Focus on the most likely address fragment and ignore company/noise text.\nInput: {text}",
            )
        except BadRequestError:
            logger.warning("BadRequestError on retry extraction. Retrying with aggressively sanitized text.")
            retry_data, retry_usage = call_llm(
                SYSTEM_PROMPT,
                f"Focus on the most likely address fragment and ignore company/noise text.\nInput: {aggressive_sanitize_text(text)}",
            )
        usage = merge_usage(usage, retry_usage)
        city = retry_data.get("city")
        state_or_province = retry_data.get("state_or_province")
        country = retry_data.get("country")
        extraction_method = "llm_retry"

    if (
        allow_company_hq_fallback
        and not any([city, state_or_province, country])
        and looks_like_company_name(text)
    ):
        try:
            logger.info("Trying HQ fallback for organization-like text.")
            fallback_data, fallback_usage = call_llm(COMPANY_HQ_PROMPT, text)
        except BadRequestError:
            logger.warning("BadRequestError on HQ fallback. Retrying with sanitized text.")
            fallback_data, fallback_usage = call_llm(COMPANY_HQ_PROMPT, aggressive_sanitize_text(text))
        usage = merge_usage(usage, fallback_usage)
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
        "usage": usage,
    }


def empty_master_database() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "lookup_key",
            "source_text",
            "city",
            "state_or_province",
            "country",
            "city_state",
            "extraction_method",
            "extraction_error",
            "created_at",
            "updated_at",
        ]
    )


def load_master_database(master_database_path: str) -> pd.DataFrame:
    if not os.path.exists(master_database_path):
        logger.info("Master database file not found. Starting with empty database.")
        return empty_master_database()

    try:
        master_df = pd.read_csv(master_database_path)
    except Exception:
        logger.exception("Failed to read master database. Starting with empty database.")
        return empty_master_database()

    expected_columns = empty_master_database().columns.tolist()
    for column in expected_columns:
        if column not in master_df.columns:
            master_df[column] = None

    return master_df[expected_columns].copy()


def save_master_database(master_df: pd.DataFrame, master_database_path: str) -> None:
    master_df.to_csv(master_database_path, index=False)
    logger.info("Master database saved to %s with %s records.", master_database_path, len(master_df))


def build_master_record(text: str, result: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    city = result.get("city")
    state_or_province = result.get("state_or_province")
    city_state = f"{city}, {state_or_province}" if city and state_or_province else None

    return {
        "lookup_key": build_lookup_key(text),
        "source_text": text,
        "city": city,
        "state_or_province": state_or_province,
        "country": result.get("country"),
        "city_state": city_state,
        "extraction_method": result.get("extraction_method"),
        "extraction_error": result.get("error"),
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def should_store_in_master_database(result: Dict[str, Optional[str]]) -> bool:
    city = result.get("city")
    state_or_province = result.get("state_or_province")
    country = result.get("country")
    extraction_method = result.get("extraction_method")

    if extraction_method in {"error", "empty", "unresolved", "unresolved_with_signal"}:
        return False

    return any([city, state_or_province, country])


def upsert_master_database(
    master_df: pd.DataFrame,
    records: list[Dict[str, Optional[str]]],
) -> pd.DataFrame:
    if not records:
        return master_df

    existing_by_key = {
        str(row["lookup_key"]): idx
        for idx, row in master_df.iterrows()
        if pd.notna(row["lookup_key"])
    }

    for record in records:
        key = str(record["lookup_key"])
        if key in existing_by_key:
            idx = existing_by_key[key]
            record["created_at"] = master_df.at[idx, "created_at"]
            record["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for column, value in record.items():
                master_df.at[idx, column] = value
        else:
            master_df.loc[len(master_df)] = record
            existing_by_key[key] = len(master_df) - 1

    return master_df


def get_master_match(master_df: pd.DataFrame, text: str) -> Optional[Dict[str, Optional[str]]]:
    lookup_key = build_lookup_key(text)
    if not lookup_key or master_df.empty:
        return None

    matches = master_df[master_df["lookup_key"] == lookup_key]
    if matches.empty:
        return None

    match = matches.iloc[-1]
    return {
        "city": match.get("city"),
        "state_or_province": match.get("state_or_province"),
        "country": match.get("country"),
        "extraction_method": "master_database",
        "error": match.get("extraction_error"),
    }


def filter_master_database(master_df: pd.DataFrame, search_text: str) -> pd.DataFrame:
    if master_df.empty or not search_text.strip():
        return master_df

    search_value = search_text.strip().lower()
    mask = pd.Series(False, index=master_df.index)

    searchable_columns = [
        "source_text",
        "city",
        "state_or_province",
        "country",
        "city_state",
        "extraction_method",
    ]

    for column in searchable_columns:
        series = master_df[column].fillna("").astype(str).str.lower()
        mask = mask | series.str.contains(search_value, regex=False)

    return master_df[mask].copy()


def render_master_database_section(master_database_path: str) -> None:
    master_df = load_master_database(master_database_path)

    with st.expander("Master Database Preview", expanded=False):
        st.write(f"CSV path: `{master_database_path}`")
        st.write(f"Cached unique records: {len(master_df)}")

        if master_df.empty:
            st.info("The master database is currently empty.")
            return

        search_text = st.text_input(
            "Search master database",
            value="",
            placeholder="Search by source text, city, state, country, or method",
            key="master_db_search",
        )
        preview_limit = st.number_input(
            "Rows to preview",
            min_value=5,
            max_value=200,
            value=25,
            step=5,
            key="master_db_preview_limit",
        )

        filtered_df = filter_master_database(master_df, search_text)
        st.write(f"Matching records: {len(filtered_df)}")
        st.dataframe(filtered_df.head(preview_limit), use_container_width=True)

        st.download_button(
            label="Download master database CSV",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name=os.path.basename(master_database_path),
            mime="text/csv",
        )


def process_dataframe(
    df: pd.DataFrame,
    address_column: str,
    sleep_between_rows_sec: float,
    allow_company_hq_fallback: bool,
    master_database_path: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if address_column not in df.columns:
        raise ValueError(
            f"Column '{address_column}' not found. Available columns: {list(df.columns)}"
        )

    logger.info(
        "Processing started. Rows=%s, address_column=%s, master_database=%s",
        len(df),
        address_column,
        master_database_path,
    )

    result_df = df.copy()
    city_out = []
    state_out = []
    country_out = []
    city_state_out = []
    method_out = []
    error_out = []
    master_records_to_upsert = []
    master_hits = 0
    llm_runs = 0
    token_usage = empty_usage()

    master_df = load_master_database(master_database_path)

    total = len(result_df)
    start_time = time.time()
    progress_bar = st.progress(0, text="Preparing file processing...")
    status_text = st.empty()

    for idx, value in enumerate(result_df[address_column].tolist(), start=1):
        raw_text = str(value) if pd.notna(value) else ""
        try:
            result = get_master_match(master_df, raw_text)
            if result:
                master_hits += 1
                logger.info("Master database hit for row %s.", idx)
            else:
                result = extract_location(
                    raw_text,
                    allow_company_hq_fallback=allow_company_hq_fallback,
                )
                token_usage = merge_usage(token_usage, result.get("usage", empty_usage()))
                if should_store_in_master_database(result):
                    master_records_to_upsert.append(build_master_record(raw_text, result))
                    logger.info(
                        "Stored new extraction candidate from row %s with method %s.",
                        idx,
                        result.get("extraction_method"),
                    )
                else:
                    logger.info(
                        "Skipped master database storage for row %s with method %s.",
                        idx,
                        result.get("extraction_method"),
                    )
                llm_runs += 1

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
            logger.exception("Row %s failed during processing.", idx)

        if sleep_between_rows_sec > 0:
            time.sleep(sleep_between_rows_sec)

        progress_value = idx / total if total else 1.0
        elapsed_time = time.time() - start_time
        speed = (idx / elapsed_time) if elapsed_time > 0 else 0.0
        remaining_time = ((total - idx) / speed) if speed > 0 else 0.0

        progress_bar.progress(progress_value, text="Processing file...")
        status_text.text(
            f"Processed {idx}/{total} | Speed: {speed:.1f}/s | ETA: {remaining_time:.1f}s"
        )

    result_df["City"] = city_out
    result_df["State/Province"] = state_out
    result_df["Country"] = country_out
    result_df["City & State"] = city_state_out
    result_df["Extraction Method"] = method_out
    result_df["Extraction Error"] = error_out

    updated_master_df = upsert_master_database(master_df, master_records_to_upsert)
    save_master_database(updated_master_df, master_database_path)
    logger.info(
        "Processing completed. Rows=%s, master_hits=%s, llm_runs=%s, stored_records=%s, prompt_tokens=%s, completion_tokens=%s, total_tokens=%s",
        total,
        master_hits,
        llm_runs,
        len(master_records_to_upsert),
        token_usage["prompt_tokens"],
        token_usage["completion_tokens"],
        token_usage["total_tokens"],
    )
    return result_df, token_usage


def get_excel_sheet_names(uploaded_file) -> list[str]:
    uploaded_file.seek(0)
    excel_file = pd.ExcelFile(uploaded_file, engine="openpyxl")
    sheet_names = excel_file.sheet_names
    uploaded_file.seek(0)
    return sheet_names


def load_input_file(uploaded_file, sheet_name: Optional[str]) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="openpyxl")


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output.getvalue()


def build_output_filename(uploaded_filename: str) -> str:
    base_name, _ = os.path.splitext(os.path.basename(uploaded_filename))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.xlsx"


def main() -> None:
    st.set_page_config(page_title="Location Extraction Tool V2", layout="wide")
    st.title("Location Extraction Tool V2")
    st.caption("Upload an Excel or CSV file, reuse cached master-database results, run extraction for missing rows, and download the enriched output.")

    with st.sidebar:
        st.header("Settings")
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
        master_database_path = st.text_input(
            "Master database CSV",
            value=DEFAULT_MASTER_DATABASE_FILE,
        )
        st.caption(f"Audit log folder: `{AUDIT_DIR}`")

    uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])

    if not uploaded_file:
        st.info("Upload a file to begin.")
        return

    logger.info("User uploaded file: %s", uploaded_file.name)

    selected_sheet_name = None
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        try:
            sheet_names = get_excel_sheet_names(uploaded_file)
        except Exception as exc:
            st.error(f"Unable to inspect Excel sheets: {exc}")
            return

        if len(sheet_names) == 1:
            selected_sheet_name = sheet_names[0]
            st.info(f"Detected sheet: {selected_sheet_name}")
        else:
            selected_sheet_name = st.selectbox("Select sheet", options=sheet_names)

    try:
        source_df = load_input_file(uploaded_file, selected_sheet_name)
    except Exception as exc:
        logger.exception("Failed to load uploaded file: %s", uploaded_file.name)
        st.error(f"Unable to read the uploaded file: {exc}")
        return

    output_filename = build_output_filename(uploaded_file.name)

    st.success("File loaded successfully.")
    st.write(f"Rows: {len(source_df)}")
    st.write(f"Columns: {', '.join(source_df.columns.astype(str))}")
    render_master_database_section(master_database_path)

    with st.expander("Preview input data", expanded=True):
        st.dataframe(source_df.head(20), use_container_width=True)

    if address_column not in source_df.columns:
        st.error(f"Column '{address_column}' not found in the uploaded data.")
        return

    if st.button("Run extraction", type="primary"):
        try:
            logger.info("Run extraction button clicked for file: %s", uploaded_file.name)
            with st.spinner("Extracting locations..."):
                result_df, token_usage = process_dataframe(
                    source_df,
                    address_column=address_column,
                    sleep_between_rows_sec=sleep_between_rows_sec,
                    allow_company_hq_fallback=allow_company_hq_fallback,
                    master_database_path=master_database_path,
                )
        except Exception as exc:
            logger.exception("Processing failed for file: %s", uploaded_file.name)
            st.error(f"Processing failed: {exc}")
            return

        st.success("Extraction completed.")
        st.caption(f"Master database updated at: {master_database_path}")
        st.caption(
            f"LLM tokens used | Input: {token_usage['prompt_tokens']} | Output: {token_usage['completion_tokens']} | Total: {token_usage['total_tokens']}"
        )
        st.caption(f"Audit log file: {DEFAULT_LOG_FILE}")
        st.caption(f"Download file name: {output_filename}")

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
