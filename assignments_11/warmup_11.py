"""
Week 11 Warmup — Prefect Orchestration & Production Patterns
"""

# ============================================================
# Prefect Question 1
# ============================================================
# @task vs @flow:
#   - @flow is the top-level entry point of a Prefect pipeline. It's the
#     orchestrator: it defines the run, shows up as a single "flow run" in
#     the UI, and is responsible for calling tasks in the right order and
#     handling the overall run state (Completed/Failed).
#   - @task is a single unit of work *inside* a flow. Each task gets its
#     own state, its own retries, its own logs, and its own box in the
#     Prefect UI graph, so you can see exactly which step succeeded or
#     failed independently of the others.
#
#   Would I decorate a pure Celsius->Fahrenheit converter with @task?
#   No. It's a pure, in-memory, deterministic calculation with no I/O,
#   no external dependency that can fail, and nothing worth retrying or
#   observing separately in the UI. Wrapping it in @task would only add
#   overhead (a task run entry, scheduling overhead) without any benefit.
#   Plain Python helper functions should stay plain Python helper
#   functions; @task is reserved for units of work with I/O, side
#   effects, or failure modes worth tracking (API calls, DB writes,
#   model inference on external data, etc.).

# ============================================================
# Prefect Question 2
# ============================================================
# @task(retries=3, retry_delay_seconds=30)
# def call_api():
#     ...

# ============================================================
# Prefect Question 3
# ============================================================
# Where to look: open the flow run in the Prefect UI (localhost:4200),
# click into the "transform" task run that shows Failed, and open its
# "Logs" tab. That's where the actual Python traceback for the failure
# is printed (exception type, message, and stack trace pointing at the
# line that raised).
#
# What I'd expect to find there:
#   - The exception class and message (e.g. FileNotFoundError for a
#     missing models/weather_classifier.pkl, or an OpenAI API error)
#   - A full traceback showing which line inside the transform task
#     raised it
#   - Any print()/logger output the task produced before it crashed
#     (useful to see how far it got — e.g. "processed 120/200 records")
#   - The task's final state details (Failed) and how many retries, if
#     any, were attempted and also failed
# The reason load_enriched never ran is that it depends on transform's
# output; since transform never returned, Prefect never scheduled the
# downstream task.

# ============================================================
# Production Question 1
# ============================================================
# response.raise_for_status() checks the HTTP status code and raises an
# HTTPError immediately if it's a 4xx/5xx response. Writing
# "if response.status_code != 200: print('error')" only *logs* the
# problem — the code keeps executing past that line with whatever
# response.json() returns (which may be an error payload, or nothing
# useful), silently corrupting downstream steps.
#
# What happens on a 500 in each case:
#   - With raise_for_status(): the task raises an exception right away.
#     Because it's decorated with @task(retries=...), Prefect catches
#     that exception, retries the task after the configured delay, and
#     if all retries are exhausted, marks the task (and the flow) as
#     Failed. Downstream tasks are never scheduled, so bad data never
#     propagates.
#   - With just a print statement: the task returns "successfully" with
#     garbage/partial data. Prefect sees it as Completed, so downstream
#     tasks (load_raw, transform, load_enriched) run anyway on bad
#     input, and the failure only surfaces much later as confusing data
#     problems rather than a clear pipeline failure.

# ============================================================
# Production Question 2
# ============================================================
# upsert with on_conflict="date" makes load_raw idempotent: if a row
# for a given date already exists, it's overwritten in place instead of
# inserted a second time. So if the pipeline crashes halfway through
# transform and I fix the bug and re-run everything from the start,
# extract/load_raw will re-fetch and re-upsert the same 2023 dates —
# and because it's an upsert, the table ends up with exactly one row per
# date, no duplicates, no matter how many times I re-run it.
#
# If I had used plain insert instead: re-running from the beginning
# would try to insert rows for dates that are already in weather_raw.
# Depending on the table's constraints this either throws a duplicate
# key / unique constraint error (and the whole re-run fails again) or,
# if there's no unique constraint on date, it silently creates duplicate
# rows for the same date — corrupting any later aggregation or join.

# ============================================================
# Production Question 3
# ============================================================
from prefect import task, get_run_logger


@task
def log_enrichment_upsert(enrichment_records: list):
    logger = get_run_logger()
    logger.info(f"Upserted {len(enrichment_records)} enrichment records")

# ============================================================
# Production Question 4
# ============================================================
# The incremental check in transform (fetching dates already present in
# weather_enriched and skipping them) is what makes the transform step
# idempotent: re-running the flow never re-does work that was already
# done successfully. It only processes the delta — records in
# weather_raw that don't have a matching row in weather_enriched yet.
#
# If I removed that check and ran ML + LLM on all 365 records every
# single run:
#   - Cost: every re-run would burn 365 extra OpenAI API calls even for
#     records that already have a perfectly good LLM summary — multiply
#     that by however many times you re-run during development/debugging
#     or on a daily schedule, and the cost grows unnecessarily and
#     linearly with re-runs instead of staying flat.
#   - Time: the flow would take roughly the same, full runtime on every
#     re-run (ML predict + 365 sequential/batched LLM calls) instead of
#     finishing almost instantly once most dates are already enriched.
#   - Data correctness: with on_conflict="date" upserts this wouldn't
#     create duplicate rows, but it could still overwrite existing
#     enriched rows with a *new*, possibly different LLM summary each
#     time (since LLM output isn't perfectly deterministic), making the
#     enrichment non-reproducible and harder to audit/trust over time.