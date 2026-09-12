# Video: https://youtu.be/dvLV5Qhui2I

import os
import requests
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

CITY = "Sacramento"
LATITUDE = 38.58
LONGITUDE = -121.49

START_DATE = "2023-01-01"
END_DATE = "2023-12-31"


def get_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError(
            "Missing SUPABASE_URL or SUPABASE_KEY environment variable. "
            "Check your .env file."
        )
    return create_client(url, key)


def extract_weather_data():
    """Step 1: Extract daily weather data for 2023 from Open-Meteo."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
        ],
        "timezone": "auto",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    print(f"Fetched weather data for {CITY} ({START_DATE} to {END_DATE})")
    print(f"Days returned: {len(data['daily']['time'])}")
    return data


def transform_weather_data(raw_data):
    """Step 2: Convert the columnar API response into row dictionaries."""
    daily = raw_data["daily"]
    dates = daily["time"]
    records = []
    for i, day in enumerate(dates):
        records.append({
            "date": day,
            "temperature_2m_max": daily["temperature_2m_max"][i],
            "temperature_2m_min": daily["temperature_2m_min"][i],
            "precipitation_sum": daily["precipitation_sum"][i],
            "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
        })

    print("First record:", records[0])
    print("Last record:", records[-1])
    # A full non-leap year (2023) should have 365 records. If the count
    # returned differs from 365, it's most likely due to missing
    # measurements for a handful of days at the source weather station
    # (Open-Meteo interpolates some gaps but not all), rather than the
    # API cutting off part of the requested range.
    return records


def load_weather_data(supabase, records):
    """Step 3: Upsert records into weather_raw."""
    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )
    print(f"Upserted {len(response.data)} rows into weather_raw")
    # Running this script a second time upserts the same 365 rows again.
    # Because `date` is the conflict key, upsert() updates the existing
    # rows in place instead of inserting duplicates, so the total row
    # count in weather_raw stays exactly the same. That's the practical
    # meaning of idempotency: re-running the pipeline is safe and has
    # no side effect beyond re-confirming the data is correct.
    return response


def verify_load(supabase):
    """Step 4: Verify the data landed correctly."""
    all_rows = supabase.table("weather_raw").select("date").execute().data
    print(f"Total rows in weather_raw: {len(all_rows)}")

    earliest = (
        supabase.table("weather_raw").select("*").order("date").limit(1).execute().data
    )
    latest = (
        supabase.table("weather_raw")
        .select("*")
        .order("date", desc=True)
        .limit(1)
        .execute()
        .data
    )
    print("Earliest date:", earliest[0]["date"] if earliest else None)
    print("Latest date:", latest[0]["date"] if latest else None)

    target_date = "2023-07-04"
    exact_match = (
        supabase.table("weather_raw")
        .select("*")
        .eq("date", target_date)
        .execute()
        .data
    )
    if exact_match:
        print(f"Row for {target_date}:", exact_match[0])
    else:
        # Fall back to the nearest available date on or after the target
        nearest = (
            supabase.table("weather_raw")
            .select("*")
            .gte("date", target_date)
            .order("date")
            .limit(1)
            .execute()
            .data
        )
        print(f"No row for {target_date}. Nearest date:", nearest[0] if nearest else None)


if __name__ == "__main__":
    supabase = get_client()
    raw = extract_weather_data()
    records = transform_weather_data(raw)
    load_weather_data(supabase, records)
    verify_load(supabase)