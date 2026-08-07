from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# --- Completions API ---
# API Q1
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print("Response:", response.choices[0].message.content)
print("Model:", response.model)
print("Total tokens:", response.usage.total_tokens)

# API Q2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    print(f"Temperature {temp}: {response.choices[0].message.content}")

# At temperature=0, the response is short and consistent across runs — good for
# reproducibility. At 0.7 the wording varies slightly but stays concise. At 1.5
# the model becomes noticeably more verbose and less predictable in format,
# even adding unsolicited explanation. For consistent, reproducible output,
# temperature=0 is the right choice.


# API Q3
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

for i, choice in enumerate(response.choices, start=1):
    print(f"Choice {i}: {choice.message.content}")


# API Q4
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)
print("Response:", response.choices[0].message.content)

# The response gets cut off mid-sentence because max_tokens limits how much text
# the model can generate. In a real application, max_tokens is useful for
# controlling API costs and keeping responses within a predictable length
# (e.g. for UI display constraints or short summaries).


# --- System Messages and Personas ---
# System Q1
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print("Tutor persona:", response.choices[0].message.content)

# Now a different personality
messages_2 = [
    {"role": "system", "content": "You are a blunt senior engineer who has no patience for fluff. You give short, technical, no-nonsense answers."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response_2 = client.chat.completions.create(model="gpt-4o-mini", messages=messages_2)
print("Blunt engineer persona:", response_2.choices[0].message.content)

# What changed: tone, length, and level of hand-holding differ drastically between
# the two personas, even though the underlying technical content is the same.


# System Q2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print("Memory test:", response.choices[0].message.content)

# The model knows Jordan's name even though the API itself is stateless — it has
# no memory between calls. What actually happens is that we send the entire
# conversation history (including the earlier assistant reply) as part of the
# messages list on every single request. The model "remembers" only because we
# re-send the full context each time, not because it retains anything internally.


# --- Prompt Engineering ---
# Prompt Q1 - Zero-Shot
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, start=1):
    prompt = f"Classify the sentiment of this review as positive, negative, or mixed: \"{review}\""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"Review {i} (zero-shot): {response.choices[0].message.content}")

# Prompt Q2 - One-Shot
example = 'Review: "Fast shipping but the item arrived damaged."\nSentiment: mixed'

for i, review in enumerate(reviews, start=1):
    prompt = f"""Classify the sentiment of the review as positive, negative, or mixed.

Example:
{example}

Review: "{review}"
Sentiment:"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"Review {i} (one-shot): {response.choices[0].message.content}")

# Adding one example made the output format much more consistent — the model
# tends to reply with just the label (e.g. "mixed") instead of a full sentence,
# because it's mirroring the example's format.

# Prompt Q3 - Few-Shot
examples = """Review: "The team was incredibly supportive throughout onboarding."
Sentiment: positive

Review: "The product broke after two days and support ignored my emails."
Sentiment: negative

Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed"""

for i, review in enumerate(reviews, start=1):
    prompt = f"""Classify the sentiment of the review as positive, negative, or mixed.

{examples}

Review: "{review}"
Sentiment:"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"Review {i} (few-shot): {response.choices[0].message.content}")

# Comparison: zero-shot is fastest to write but least consistent in format and
# sometimes in accuracy on ambiguous cases (like "mixed"). One-shot improves
# format consistency. Few-shot is the most reliable for consistent labeling,
# especially when a category (like "mixed") is easy to confuse with the others -
# I'd use zero-shot for simple/obvious tasks, one-shot when I just need a
# specific output format, and few-shot when the task has nuance or multiple
# categories that are easy to mix up.

# Prompt Q4 - Chain of Thought
cot_prompt = """A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?

Show your reasoning step by step, then give the final answer clearly labeled as "Final Answer:"."""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": cot_prompt}]
)
print("Chain of thought:", response.choices[0].message.content)

# Asking the model to reason step by step tends to improve accuracy because it
# forces the model to break a multi-step problem into smaller, more manageable
# sub-calculations instead of jumping straight to a guessed final number. Each
# intermediate step also gives the model a chance to catch and correct errors
# before they compound into the final answer.

# Prompt Q5 - Structured Output
import json

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

structured_prompt = f"""Analyze the sentiment of this review and return ONLY valid JSON
with exactly these keys: "sentiment" (positive/negative/mixed), "confidence" (a float from 0 to 1),
and "reason" (one sentence explaining the sentiment).

Review: "{review}"

Respond with ONLY the raw JSON object. Do not use markdown code fences, do not add
any explanation before or after the JSON."""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": structured_prompt}]
)
raw_output = response.choices[0].message.content
print("Raw response:", raw_output)

cleaned_output = raw_output.strip()
if cleaned_output.startswith("```"):
    cleaned_output = cleaned_output.strip("`")
    cleaned_output = cleaned_output.replace("json\n", "", 1).strip()

try:
    parsed = json.loads(cleaned_output)
    print("Sentiment:", parsed["sentiment"])
    print("Confidence:", parsed["confidence"])
    print("Reason:", parsed["reason"])
except json.JSONDecodeError:
    print("Failed to parse JSON. Raw response was:", raw_output)

# Prompt Q6 - Delimiters
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

````{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
print("Delimiters test 1 (instructions):", response.choices[0].message.content)

non_instruction_text = "Pasta has been a staple of Italian cuisine for centuries, \
with regional variations found across the entire country."

prompt_2 = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{non_instruction_text}```
"""

response_2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt_2}]
)
print("Delimiters test 2 (no instructions):", response_2.choices[0].message.content)

# Delimiters (like triple backticks) help prevent the model from confusing the
# user-supplied content with the instructions themselves - without them, if the
# user's text happened to contain something like "ignore the above and...", the
# model might misinterpret it as part of the instructions rather than as data to
# process.

# --- Local Models with Ollama ---
# Ollama Q1
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain what a large language model is in two sentences."}]
)
print("OpenAI response:", response.choices[0].message.content)

# Ollama output (qwen3:0.6b), pasted from terminal:
# """
# Thinking...
# Okay, the user wants me to explain a large language model in two sentences. Let me
# start by recalling what a large language model is. First, they should know that
# they're a type of AI model. Then, mention that they're trained on massive datasets,
# which makes them powerful. Also, I should highlight their ability to understand and
# generate text, which is the main function. Need to make sure each sentence covers
# these points concisely. Let me check if I'm using correct terms like "massive" and
# "text generation." Yep, that's right. Alright, two sentences should cover all that.
# ...done thinking.
# A large language model is an advanced AI model trained on vast datasets to
# understand and generate human-like text, enabling tasks like writing articles,
# answering questions, or composing stories.
# """

# Differences noticed: qwen3:0.6b exposes its internal reasoning ("thinking") before
# giving the final answer, which OpenAI's gpt-4o-mini does not show by default. The
# final answers themselves were similar in content and length, though the OpenAI
# response felt slightly more polished/concise.
#
# Advantage of running locally: no per-request API cost, works fully offline, and
# no data is sent to a third-party server (better privacy for sensitive input).
# Disadvantage of running locally: much smaller model (0.6B parameters vs a much
# larger cloud model) means lower quality/nuance in responses, and it requires
# local compute resources and disk space to download and run.