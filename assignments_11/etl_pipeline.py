# Video: https://youtu.be/OLjeHfvFXWg

"""
Full Extract -> Load (raw) -> Transform -> Load (enriched) pipeline for the
"good day for running" weather project, orchestrated with Prefect.
  2. load_raw       -> upsert raw records into weather_raw (Supabase)

Pipeline steps:
  1. extract       -> pull 2023 daily weather from the Open-Meteo archive API
  3. transform      -> incremental ML classification + LLM one-sentence summary
  4. load_enriched  -> upsert enrichment records into weather_enriched (Supabase)
"""

import os
import json
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI
from prefect import flow, task
from prefect.cache_policies import NO_CACHE

load_dotenv()

# --- Config ---

MODEL_PATH = os.path.join("models", "weather_classifier.pkl")
METADATA_PATH = os.path.join("models", "weather_classifier_metadata.json")

LLM_MODEL = "gpt-4o-mini"

# Same city/variables as Week 4 so the model's learned thresholds still apply.
CITY_NAME = "Sacramento"
LATITUDE = 38.5816
LONGITUDE = -121.4944
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

SYSTEM_PROMPT = """
You are a running coach assistant. Given a day's weather data and a
machine learning model's prediction of whether it is a good day for
running, write exactly one sentence recommending whether or not to run
that day, referencing the weather conditions that support the
recommendation. Do not include anything besides that one sentence.
"""


# --- Clients ---

def get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError(
            "Missing SUPABASE_URL or SUPABASE_KEY environment variable. "
            "Check your .env file."
        )
    return create_client(url, key)


def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing OPENAI_API_KEY environment variable. Check your .env file."
        )
    return OpenAI(api_key=api_key)


def call_with_retry(client, messages, max_retries=3):
    """Call the chat completions endpoint, retrying on any exception."""
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
            )
        except Exception as e:
            print(f"  LLM call failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2)
    return None


def build_user_message(raw_record, enrichment_record):
    return (
        f"Date: {raw_record['date']}\n"
        f"Max temperature: {raw_record['temperature_2m_max']} C\n"
        f"Min temperature: {raw_record['temperature_2m_min']} C\n"
        f"Precipitation: {raw_record['precipitation_sum']} mm\n"
        f"Max wind speed: {raw_record['wind_speed_10m_max']} km/h\n"
        f"Model prediction: "
        f"{'good day for running' if enrichment_record['good_for_running'] else 'not a good day for running'} "
        f"(confidence: {enrichment_record['confidence']:.2f})"
    )


# --- Task 1: Extract ---

@task(retries=2, retry_delay_seconds=10)
def extract():
    """
    Fetch 2023 daily weather data for CITY_NAME from the Open-Meteo
    historical archive API and return it as a list of row dicts.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    daily = payload["daily"]
    dates = daily["time"]

    records = []
    for i, date in enumerate(dates):
        record = {"date": date}
        for var in DAILY_VARIABLES:
            record[var] = daily[var][i]
        records.append(record)

    print(
        f"=== Task: Extract ===\n"
        f"Fetched {len(records)} daily records for {CITY_NAME} "
        f"({START_DATE} to {END_DATE}).\n"
    )
    return records


# --- Task 2: Load raw ---

@task(retries=2, retry_delay_seconds=5, cache_policy=NO_CACHE)
def load_raw(supabase, raw_records):
    """Upsert raw weather records into weather_raw."""
    response = (
        supabase.table("weather_raw")
        .upsert(raw_records, on_conflict="date")
        .execute()
    )
    print(f"=== Task: Load Raw ===\nUpserted {len(response.data)} rows into weather_raw.\n")


# --- Task 3: Transform ---

@task(cache_policy=NO_CACHE)
def transform(supabase, openai_client, raw_records):
    """
    Incrementally classify + summarize days that aren't already in
    weather_enriched. Returns the list of new enrichment records
    (each with date, good_for_running, confidence, llm_summary).
    """
    # Incremental check
    enriched_response = supabase.table("weather_enriched").select("date").execute()
    enriched_dates = {row["date"] for row in enriched_response.data}
    unprocessed = [r for r in raw_records if r["date"] not in enriched_dates]

    print(
        f"=== Task: Transform ===\n"
        f"Raw records fetched:      {len(raw_records)}\n"
        f"Already enriched:         {len(enriched_dates)}\n"
        f"To be processed this run: {len(unprocessed)}"
    )

    if not unprocessed:
        print("Nothing to transform.\n")
        return []

    # --- ML classification ---
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    model = joblib.load(MODEL_PATH)
    feature_names = metadata["feature_names"]

    df = pd.DataFrame(unprocessed)
    X = df[feature_names]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    enrichment_records = []
    for i, record in enumerate(unprocessed):
        good_for_running = bool(predictions[i])
        confidence = float(max(probabilities[i]))
        enrichment_records.append({
            "date": record["date"],
            "good_for_running": good_for_running,
            "confidence": confidence,
        })

    good_count = sum(1 for r in enrichment_records if r["good_for_running"])
    confidences = [r["confidence"] for r in enrichment_records]
    print(
        f"Days classified good_for_running: {good_count} / {len(enrichment_records)}\n"
        f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}"
    )

    # --- LLM summaries ---
    raw_by_date = {r["date"]: r for r in unprocessed}

    for i, enrichment_record in enumerate(enrichment_records, start=1):
        raw_record = raw_by_date[enrichment_record["date"]]
        user_message = build_user_message(raw_record, enrichment_record)

        response = call_with_retry(
            openai_client,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        if response is not None:
            enrichment_record["llm_summary"] = response.choices[0].message.content.strip()
        else:
            enrichment_record["llm_summary"] = (
                "Recommendation unavailable due to an API error."
            )

        if i % 50 == 0:
            print(f"  Processed {i}/{len(enrichment_records)} records...")

    print(f"Finished transform for {len(enrichment_records)} records.\n")
    return enrichment_records


# --- Task 4: Load enriched ---

@task(retries=2, retry_delay_seconds=5, cache_policy=NO_CACHE)
def load_enriched(supabase, enrichment_records):
    """Upsert enrichment records into weather_enriched."""
    if not enrichment_records:
        print("=== Task: Load Enriched ===\nNothing to upsert.\n")
        return

    now = datetime.now(timezone.utc).isoformat()
    for record in enrichment_records:
        record["enriched_at"] = now

    response = (
        supabase.table("weather_enriched")
        .upsert(enrichment_records, on_conflict="date")
        .execute()
    )
    print(f"=== Task: Load Enriched ===\nUpserted {len(response.data)} rows into weather_enriched.\n")


# --- Flow ---

@flow(log_prints=True)
def weather_etl_pipeline():
    supabase = get_supabase_client()
    openai_client = get_openai_client()

    raw_records = extract()
    load_raw(supabase, raw_records)
    enrichment_records = transform(supabase, openai_client, raw_records)
    load_enriched(supabase, enrichment_records)

    print("Pipeline run complete: extract -> load_raw -> transform -> load_enriched.")


if __name__ == "__main__":
    weather_etl_pipeline()