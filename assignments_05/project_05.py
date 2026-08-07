from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

SYSTEM_PROMPT = """You are a supportive job application coach helping career changers and
job seekers improve their resumes, cover letters, and overall application materials.

Your job is to help with:
- Rewriting resume bullet points to be more specific and results-oriented
- Drafting cover letter openings
- Answering general questions about job applications (tone, structure, common mistakes)

Rules you must follow:
- Stay focused on job application materials. If asked about something unrelated
  (e.g. general life advice, coding help, unrelated topics), politely redirect the
  conversation back to job applications.
- Always remind the user to review and personally edit anything you generate before
  submitting it to a real employer - your output is a draft, not a final product.
- You do not know the user's specific industry norms, company culture, or the exact
  expectations of the role they're applying to. Be clear that the user should use
  their own judgment and, where possible, verify norms specific to their field.
- Be encouraging and constructive, especially with career changers who may be unsure
  how their prior experience translates to a new field.
"""

# Deliberate choice: I explicitly instructed the model to redirect off-topic
# questions back to job applications rather than refusing outright. This keeps
# the tool focused on its purpose without being unhelpfully rigid if the user
# drifts slightly off-topic mid-conversation.


# --- Task 2: Bullet Point Rewriter ---
def rewrite_bullets(bullets: list[str]) -> list[dict]:
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).
    Do not include markdown code fences or any text outside the JSON list.

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    raw_output = get_completion(messages)

    # Clean up in case the model wraps the JSON in code fences anyway
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        print("Failed to parse JSON. Raw response was:", raw_output)
        return []

    for item in result:
        print(f"Original: {item['original']}")
        print(f"Improved: {item['improved']}\n")

    return result


# What makes these bullets weak: they use vague verbs ("helped", "made", "worked"),
# have no measurable outcomes (no numbers, no scale, no impact), and don't specify
# what tools, skills, or context were involved. The model's rewrites typically add
# stronger action verbs (e.g. "resolved", "delivered", "collaborated"), imply scale
# or impact where reasonable, and make the responsibility more concrete.


# --- Task 3: Cover Letter Generator ---
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    result = get_completion(messages)
    print(result)
    return result

# I chose examples with a clear "before/after" narrative structure - both connect
# a very different prior career (nursing, banking) to the new technical role through
# a specific, concrete detail rather than a generic claim like "I'm a hard worker."
# The few-shot pattern helps control tone (confident, not apologetic about the career
# change), length (3-5 sentences), and structure (a personal hook -> concrete skill ->
# tie-in to the target role) - without examples, the model tends to default to more
# generic, cliché-heavy openings.


# --- Task 4: Moderation Check ---
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged

    if flagged:
        print("Job Application Helper: I'm not able to help with that message. "
              "Could you rephrase it?")
        return False

    return True


# --- Manual tests from Tasks 2-4 (kept for reference, no longer run automatically) ---
# bullets = [
#     "Helped customers with their problems",
#     "Made reports for the management team",
#     "Worked with a team to finish the project on time"
# ]
# rewrite_bullets(bullets)
#
# job_title = "Junior Data Engineer"
# background = "Five years of experience as a middle school math teacher; recently completed \
# a Python course and built data pipelines using Prefect and Pandas."
# generate_cover_letter(job_title, background)
#
# safe_test = is_safe("Can you help me rewrite my resume bullet points?")
# print("Safe test result:", safe_test)
#
# flagged_test = is_safe("I want to hurt my manager for rejecting my application.")
# print("Flagged test result:", flagged_test)


# --- Task 5: Chatbot Loop ---
def run_chatbot():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        if not user_input:
            continue

        if not is_safe(user_input):
            continue

        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            rewrite_bullets(raw_bullets)

        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            generate_cover_letter(job_title, background)

        else:
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            print(f"\nJob Application Helper: {reply}\n")
            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_chatbot()


# --- Task 6: Ethics Reflection ---
# Format chosen: Option A - comment block

# 1. Bias: This bot was trained on text written mostly by people who already work in
# white-collar, English-speaking professional environments, so its sense of a "strong"
# resume bullet or cover letter reflects a fairly narrow set of conventions - confident
# first-person framing, quantified achievements, and a direct, individualistic tone.
# That style doesn't map equally well onto every culture or industry: some cultures
# value modesty over self-promotion, some fields (trades, caregiving, nonprofit work)
# don't naturally produce "metrics" to quantify, and non-native English speakers may
# get pushed toward phrasing that sounds fluent but erases their own voice. The bot
# could end up favoring people who already write and self-market the way tech/corporate
# hiring expects, while making career changers from other backgrounds sound like
# everyone else instead of highlighting what's actually distinctive about them.

# 2. Risk of unreviewed output: While testing the bullet rewriter, the model rewrote
# "Built dashboards for the team" into a version that added "improving team productivity
# by 30%" - a specific number I never provided. If a job-seeker submitted that directly
# without reviewing it, they'd be putting a fabricated statistic on their resume, which
# could be flagged as dishonest in an interview when asked to explain it, or simply be
# untrue and damage their credibility if checked. This is exactly why the system prompt
# requires reminding users to review and edit everything before submitting it anywhere.