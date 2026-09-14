"""
Week 10 Warmup — ML vs. LLM in Pipelines, Prompt Design
"""

import time


# ============================================================
# ML vs. LLM in Pipelines
# ============================================================

# --- ML/LLM Question 1 ---
#
# The ML classifier (weather_classifier.pkl) produces a deterministic,
# structured output: a binary label (good_for_running: True/False) plus
# a calibrated probability (confidence), derived purely from the four
# numeric weather features it was trained on. It does this because it
# was fit on hundreds of labeled examples where the relationship between
# temperature/precipitation/wind and the "good day" label is statistical
# and numeric — exactly the kind of pattern a trained classifier is good
# at generalizing.
#
# The LLM produces unstructured, generated text: a one-sentence,
# human-readable recommendation that explains *why* the day is good or
# bad in natural language, using the ML prediction and the raw features
# as context. It does this because language generation — picking words
# that read naturally and reference the right features — is not a task
# a tabular classifier can do; the LLM was trained on huge amounts of
# text and is good at producing fluent, contextual sentences instead of
# a single fixed-format number.
#
# If you swapped them:
#   - Using the LLM for the binary good/skip prediction would make the
#     pipeline non-deterministic and unreliable: the same weather could
#     get different labels on different calls, there's no calibrated
#     probability/confidence score, and you'd lose the accuracy the
#     classifier gets from being trained specifically on labeled
#     historical data for this exact task.
#   - Using the ML model to "write" the recommendation doesn't even make
#     sense — a logistic regression pipeline has no language generation
#     capability. It could at best output another number, not a
#     sentence. You'd have to hand-craft a template, which loses the
#     natural, varied phrasing an LLM provides.


# --- ML/LLM Question 2 ---
#
# 1. Converting a date string like "2023-07-04" to day-of-week:
#    Deterministic code — this is a pure, well-defined transformation
#    (datetime.strptime + .strftime("%A")) with no ambiguity, so writing
#    a rule is faster, free, and always 100% correct. No model needed.
#
# 2. Classifying a job posting as "entry-level"/"mid-level"/"senior"
#    from freeform text:
#    LLM — the input is unstructured natural language with huge
#    variation in phrasing, and there's usually no large labeled
#    training set on hand. An LLM can reason over the free text
#    (years of experience mentioned, seniority-signaling words, etc.)
#    without needing a custom-trained classifier.
#
# 3. Predicting customer churn given 15 numeric features and a labeled
#    training dataset:
#    Trained ML model — structured numeric features plus an existing
#    labeled dataset is exactly the classic supervised-learning setup.
#    A model like logistic regression / gradient boosting will be more
#    accurate, cheaper to run at scale, and faster than calling an LLM
#    for every row.
#
# 4. Normalizing inconsistent city names ("NYC", "New York City",
#    "New York, NY") to a canonical form:
#    Deterministic code — a lookup table / mapping dictionary (or a
#    fuzzy-matching library) handles this reliably and is free to run.
#    An LLM could technically do it, but it's overkill, slower, and can
#    hallucinate on edge cases where a fixed mapping would just work.
#
# 5. Summing a column of revenue figures:
#    Deterministic code — this is pure arithmetic (pandas .sum()).
#    Never use ML or an LLM for something that has one exact, provable
#    correct answer.


# --- ML/LLM Question 3 ---
#
# Incremental processing means only processing the records that haven't
# been processed yet — in this pipeline, comparing the dates already
# present in weather_enriched against all the dates in weather_raw, and
# running the ML + LLM transform only on the difference (the dates
# missing from weather_enriched).
#
# It matters for this pipeline for two reasons: cost and correctness.
# The LLM call in Step 3 costs money and takes time per row. If the
# script reprocessed all 365 records on every run, every re-run would
# re-pay for 365 LLM calls even though 364 of them (or all of them,
# once the table is fully enriched) already have a correct, unchanged
# result sitting in weather_enriched — that's wasted cost and wasted
# time for no benefit. It doesn't affect *correctness* directly, since
# upsert() with the date as the conflict key would just overwrite the
# same rows with (probably) the same values — but at scale, reprocessing
# everything every run turns a cheap few-second job into an expensive,
# slow one that gets worse the more historical data accumulates.


# ============================================================
# Prompt Design
# ============================================================

# --- Prompt Question 1 ---
#
# Alternative system prompt for a two-sentence recommendation:
#
# TWO_SENTENCE_SYSTEM_PROMPT = """
# You are a running coach assistant. Given a day's weather data and a
# machine learning model's prediction of whether it is a good day for
# running, write exactly two sentences:
# 1. The first sentence states the prediction plainly (good day to run
#    or not a good day to run).
# 2. The second sentence explains the reasoning, referencing the
#    specific weather features (temperature, precipitation, wind) that
#    led to that prediction.
# Do not include any text besides these two sentences.
# """
#
# Validation logic changes needed:
# The current validation (checking the LLM response is a single
# sentence — e.g. splitting on "." and expecting exactly one non-empty
# segment, or checking there's exactly one sentence-ending punctuation
# mark) would need to instead check for exactly two sentences. That
# means splitting the response into sentences (e.g. on ". ", or with a
# simple regex on sentence-ending punctuation) and validating the count
# is 2, not 1. It would also be worth validating that the first sentence
# contains a clear yes/no signal (e.g. checking for words like "good"/
# "not" ) so a malformed two-sentence response that omits the actual
# prediction still gets caught, rather than just counting sentences.


# --- Prompt Question 2 ---

def call_with_retry(client, messages, max_retries=3):
    """
    Call the chat completions endpoint, retrying on any exception up to
    max_retries times, waiting 2 seconds between attempts. Returns the
    response object on success, or None if every attempt fails.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
        except Exception as e:
            print(f"Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(2)
    return None

# When to use this in production:
# Any time you're calling an external API inside a loop over many
# records (exactly like the LLM transform step in transform_10.py).
# Network blips, rate limits, and transient 5xx errors from the
# provider are common and usually resolve themselves on a retry a
# couple seconds later. Without retry logic, a single flaky call in
# the middle of processing 365+ records would either crash the whole
# pipeline or silently skip a record, when a short wait-and-retry
# would have succeeded. Returning None on final failure (rather than
# raising) also lets the caller fall back gracefully — e.g. using a
# placeholder llm_summary instead of losing the whole run.