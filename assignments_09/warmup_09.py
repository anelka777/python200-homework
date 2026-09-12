import os
from datetime import date
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()


# --- Supabase Connection ---

# Q1
# supabase-py needs two pieces of information to connect:
#   1. SUPABASE_URL — the project's API URL (Project Settings > API > Project URL)
#   2. SUPABASE_KEY — the anon/service API key (Project Settings > API > Project API keys)
# These should never be hardcoded because:
#   - committing them to git exposes them to anyone with repo access (public repos = anyone)
#   - keys can be rotated/revoked without touching code if kept in env vars
#   - different environments (dev/staging/prod) need different credentials without code changes

# Q2
def get_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError(
            "Missing SUPABASE_URL or SUPABASE_KEY environment variable. "
            "Check your .env file."
        )
    return create_client(url, key)

# Q3
# Row Level Security (RLS) is a Postgres feature that restricts which rows
# a query can see/modify based on rules tied to the requesting user/role.
# It's disabled for this course because we're connecting with a single
# trusted key from a local script (no end users, no auth layer) — RLS
# would just add friction while we're learning the CRUD basics.
# In a real-world app (e.g., a multi-tenant SaaS where each user should
# only see their own data), you'd want RLS enabled so the database itself
# enforces isolation, even if application code has a bug.


# --- supabase-py CRUD ---

# CRUD Q1
def insert_test_record(supabase):
    row = {
        "date": date.today().isoformat(),
        "temperature_2m_max": 30.5,
        "temperature_2m_min": 18.2,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 12.3,
    }
    response = supabase.table("weather_raw").insert(row).execute()
    return response

# If run twice, insert() would try to add a second row with the same date,
# creating a duplicate (unless there's a unique constraint on `date`, in
# which case it would raise an error instead). To make it safe to run
# multiple times, use upsert() instead of insert(), with `date` as the
# conflict key.

# CRUD Q2
def get_records_by_date_range(supabase, start, end):
    response = (
        supabase.table("weather_raw")
        .select("*")
        .gte("date", start)
        .lte("date", end)
        .execute()
    )
    return response.data

# CRUD Q3
# insert() always creates a new row and fails (or duplicates) if a row
# with the same unique key already exists. Use it when you know the
# record is brand new (e.g., logging a one-time event).
# upsert() inserts if the row doesn't exist, or updates it if it does,
# based on a conflict key. Use it when re-running a script should be
# safe — like reloading the same date range of weather data.
def safe_upsert(supabase, records):
    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )
    print(f"Upserted {len(response.data)} rows")
    return response


# --- Idempotency ---

# Idempotency Q1
# Idempotency matters because pipelines fail and get re-run — a network
# blip, a crashed process, a retry from a scheduler. If an operation
# isn't idempotent, re-running it changes the outcome instead of just
# re-confirming it.
# Example: a pipeline inserts 365 rows of weather data, but crashes
# after row 200 due to a network error. On restart, if the script uses
# insert() from the beginning again, rows 1-200 get duplicated while
# 201-365 get added fresh — the table now has extra, wrong data instead
# of just the correct 365 rows.


if __name__ == "__main__":
    supabase = get_client()

    # Q1 — insert a test record with today's date, then confirm it's there
    insert_test_record(supabase)

    # Q2 — read it back with a date range query that includes today
    today_str = date.today().isoformat()
    results = get_records_by_date_range(supabase, today_str, today_str)
    print(f"Found {len(results)} record(s) in range:")
    print(results)

    # Q3 — safe upsert of a small batch (re-runnable without duplicates)
    sample_records = [
        {
            "date": today_str,
            "temperature_2m_max": 31.0,
            "temperature_2m_min": 19.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 14.1,
        },
    ]
    safe_upsert(supabase, sample_records)
