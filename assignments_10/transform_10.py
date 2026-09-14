# Video: https://youtu.be/wJWllhBTaJM

import os
import json
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

load_dotenv()

MODEL_PATH = os.path.join("models", "weather_classifier.pkl")
METADATA_PATH = os.path.join("models", "weather_classifier_metadata.json")

LLM_MODEL = "gpt-4o-mini"

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


# --- Step 1: Incremental Read ---

def incremental_read(supabase, metadata):
    """
    Fetch weather_raw and weather_enriched, and return only the raw
    records whose date isn't already present in weather_enriched.
    """
    raw_response = supabase.table("weather_raw").select("*").execute()
    raw_records = raw_response.data

    enriched_response = supabase.table("weather_enriched").select("date").execute()
    enriched_dates = {row["date"] for row in enriched_response.data}

    unprocessed = [r for r in raw_records if r["date"] not in enriched_dates]

    print("=== Step 1: Incremental Read ===")
    print(f"Raw records in weather_raw:      {len(raw_records)}")
    print(f"Already enriched:                {len(enriched_dates)}")
    print(f"To be processed this run:        {len(unprocessed)}")
    print()

    return unprocessed


# --- Step 2: ML Transform ---

def ml_transform(model, metadata, unprocessed_records):
    """
    Run the trained classifier on each unprocessed record and build a
    list of enrichment dicts with date, good_for_running, confidence.
    """
    if not unprocessed_records:
        print("=== Step 2: ML Transform ===")
        print("Nothing to process.\n")
        return []

    feature_names = metadata["feature_names"]

    df = pd.DataFrame(unprocessed_records)
    X = df[feature_names]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    enrichment_records = []
    for i, record in enumerate(unprocessed_records):
        good_for_running = bool(predictions[i])
        # Confidence = the probability the model assigned to whichever
        # class it actually predicted (works whether it predicted
        # good or not-good).
        confidence = float(max(probabilities[i]))
        enrichment_records.append({
            "date": record["date"],
            "good_for_running": good_for_running,
            "confidence": confidence,
        })

    good_count = sum(1 for r in enrichment_records if r["good_for_running"])
    confidences = [r["confidence"] for r in enrichment_records]

    print("=== Step 2: ML Transform ===")
    print(f"Days classified good_for_running: {good_count} / {len(enrichment_records)}")
    print(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
    print()

    return enrichment_records


# --- Step 3: LLM Transform ---

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


def llm_transform(openai_client, unprocessed_records, enrichment_records):
    """
    For each enrichment record, ask the LLM for a one-sentence
    recommendation and attach it as llm_summary. Falls back to a
    placeholder string if the call fails after retries.
    """
    if not enrichment_records:
        print("=== Step 3: LLM Transform ===")
        print("Nothing to process.\n")
        return enrichment_records

    raw_by_date = {r["date"]: r for r in unprocessed_records}

    print("=== Step 3: LLM Transform ===")
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

    print(f"Finished LLM transform for {len(enrichment_records)} records.\n")
    return enrichment_records


# --- Step 4: Load ---

def load_enriched(supabase, enrichment_records):
    if not enrichment_records:
        print("=== Step 4: Load ===")
        print("Nothing to upsert.\n")
        return

    # Stamp enriched_at explicitly rather than relying on a column
    # default: a DEFAULT now() only fires on INSERT, so if upsert() ever
    # updates an existing row instead of inserting a new one, the
    # timestamp would silently stay stale unless we set it ourselves.
    now = datetime.now(timezone.utc).isoformat()
    for record in enrichment_records:
        record["enriched_at"] = now

    response = (
        supabase.table("weather_enriched")
        .upsert(enrichment_records, on_conflict="date")
        .execute()
    )
    print("=== Step 4: Load ===")
    print(f"Upserted {len(response.data)} rows into weather_enriched.\n")


# --- Step 5: Verify ---

def verify(supabase):
    all_rows = supabase.table("weather_enriched").select("*").execute().data

    print("=== Step 5: Verify ===")
    print(f"Total rows in weather_enriched: {len(all_rows)}")

    sample = sorted(all_rows, key=lambda r: r["date"])[:5]
    print("Sample rows:")
    for row in sample:
        print(
            f"  {row['date']} | good_for_running={row['good_for_running']} "
            f"| confidence={row['confidence']:.2f} | {row['llm_summary']}"
        )

    good_count = sum(1 for r in all_rows if r["good_for_running"])
    print(f"Days classified good for running: {good_count} / {len(all_rows)}")
    print()

    # Looking at the sample rows printed above:
    #   - 2023-01-05 is a particularly good summary: the model confidently
    #     predicted good_for_running=False, and the LLM's sentence directly
    #     ties that back to the actual numbers behind the call — "high
    #     precipitation of 34.8 mm and strong wind speed of 27.8 km/h" —
    #     instead of a vague statement. That specificity makes the
    #     recommendation easy to trust and verify against the raw data.
    #   - 2023-01-01 feels slightly off: the model predicted
    #     good_for_running=True with confidence=1.00, but the LLM's summary
    #     still adds "just be cautious of the high wind speed" — which reads
    #     as mildly contradictory next to a confident "great day for
    #     running." This is likely because the LLM reacts to the raw
    #     wind_speed_10m_max value itself, even when that value is still
    #     comfortably under the model's threshold (< 30 km/h), rather than
    #     treating "under threshold" as fully safe the way the classifier
    #     effectively does.


# --- Step 6: Reflect ---
#
# 1. The model was trained on weather data from Sacramento, CA (that's
#    the city listed in the metadata), and the data I loaded in Week 9
#    is also Sacramento, so for this specific project I'd expect the
#    predictions to be fairly accurate — training and inference data
#    match. But if I fed it weather data from a completely different
#    city, I wouldn't trust the predictions nearly as much. The
#    thresholds the model learned (like the 7-26 C range for max
#    temperature) came from patterns specific to Sacramento's climate,
#    and those same numeric cutoffs don't necessarily mean the same
#    thing somewhere with a very different climate — a humid or
#    high-altitude city, for example, could have a completely different
#    relationship between "good running weather" and these four raw
#    numbers, even though the columns look identical.
#
# 2. The LLM can't override anything the classifier decided — it's
#    strictly additive. It only ever receives the model's prediction as
#    part of its input and is asked to justify/explain it in a sentence,
#    never to make its own independent call on whether the day is good
#    for running. So the good_for_running and confidence values that
#    actually get written to the database come entirely from the ML
#    model; the LLM just adds commentary on top. The downside of this
#    is that if the classifier gets something wrong, the LLM will still
#    write a smooth, confident-sounding sentence defending that wrong
#    answer — it has no way to flag "actually, this prediction looks
#    off." That's worth being careful about, because a fluent sentence
#    can come across as more trustworthy than a raw probability number,
#    even when it's just dressing up a mistake.
#
# 3. Scaling this up to 50,000 records, cost is what would worry me
#    most, with latency right behind it. Step 2 (the ML transform) would
#    barely notice the difference — predict/predict_proba runs on the
#    whole batch at once and is fast regardless of size. Step 3 is the
#    bottleneck: it calls the OpenAI API once per row, so 50,000 records
#    means 50,000 separate calls. Even at roughly a second each, run
#    one after another that's many hours, plus real API cost that adds
#    up fast at that volume. To fix this I'd look at running the LLM
#    calls concurrently (threads or async, staying under the API's rate
#    limits) instead of one at a time, and lean hard on the incremental
#    read from Step 1 so a re-run never re-processes rows that already
#    have a valid llm_summary — that's really what keeps the ongoing
#    cost manageable as the dataset grows.


def main():
    supabase = get_supabase_client()
    openai_client = get_openai_client()

    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    model = joblib.load(MODEL_PATH)

    unprocessed = incremental_read(supabase, metadata)
    enrichment_records = ml_transform(model, metadata, unprocessed)
    enrichment_records = llm_transform(openai_client, unprocessed, enrichment_records)
    load_enriched(supabase, enrichment_records)
    verify(supabase)


if __name__ == "__main__":
    main()