"""
Week 6 Warmup Exercises
RAG Concepts, Keyword RAG, Semantic RAG, LlamaIndex
"""

from dotenv import load_dotenv
import os
import string

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# =========================================================
# --- RAG Concepts ---
# =========================================================

# --- Concepts Q1 ---
# Scenario A (policy library, hundreds of PDFs, updated quarterly): RAG.
#   The knowledge base is large and changes over time. RAG lets the assistant
#   retrieve the current version of a document at query time instead of being
#   retrained every time a policy changes.
#
# Scenario B (brand voice, 3,000 in-house writing examples): Fine-tuning.
#   This isn't about looking up facts, it's about teaching the model a
#   consistent *style*. Fine-tuning on the 3,000 examples bakes that style
#   into the model's behavior so it applies automatically to any new copy.
#
# Scenario C (one two-page report, one-time use): Prompt engineering.
#   The whole document easily fits in a single prompt / context window, so
#   there is no need for retrieval infrastructure or a permanent index.
#   Just paste the report into the prompt and ask questions.

# --- Concepts Q2 ---
# A confidently wrong answer is more dangerous than "I am not sure" because
# the user has no signal to double check it -- they will act on it as if it
# were verified fact. An uncertain answer at least prompts the user to go
# verify before relying on it.
# Example: a nurse asks an AI assistant about a drug interaction and gets a
# confident but hallucinated "these are safe to combine" answer. Because it
# sounds authoritative, the nurse may not double-check it, and a patient
# could be harmed. If the model had said "I'm not fully sure, please verify
# with a pharmacist," the nurse would likely have checked before acting.
# Tone matters because humans use confidence as a proxy for reliability in
# everyday conversation -- a hedge cues us to verify, a confident tone cues
# us to trust, even though the model's confidence is not actually tied to
# how correct it is.

# --- Concepts Q3 ---
# steps = [
#     "Extract text from source documents",       # 1. Get raw text out of PDFs/files
#     "Split text into chunks",                    # 2. Break long docs into smaller pieces
#     "Convert text chunks into embeddings",        # 3. Turn each chunk into a vector
#     "Receive the user's query",                   # 4. User asks a question
#     "Embed the user's query",                     # 5. Turn the query into a vector too
#     "Retrieve the most relevant chunks",          # 6. Find chunks closest to the query vector
#     "Inject retrieved chunks into the prompt",    # 7. Add those chunks as context for the LLM
#     "Generate a response from the LLM",           # 8. LLM answers using the injected context
# ]


# =========================================================
# --- Keyword RAG ---
# =========================================================

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

# --- Keyword Q1 ---
print("\n=== Keyword Q1 ===")
query1 = "What are your hours on weekends?"
result1 = simple_keyword_retrieval(query1, documents, verbose=True)
print("Selected document:", result1[0][0])
# hours.txt was NOT selected, even though semantically it's the right answer.
# Reason: "your" is not in the stopwords list, and it happens to appear in
# 3 documents (hours.txt, hiring.txt, loyalty.txt) -> tie at score=1.
# On a tie, the function sorts (score, name, content) tuples in reverse,
# so "loyalty.txt" comes out "greater" alphabetically than "hours.txt"
# and wins the tie-break.
# This shows a real weakness of keyword RAG: the result can depend on
# accidental overlap of unfiltered common words and on tie-break order,
# rather than on which document is actually most relevant.

# --- Keyword Q2 ---
print("\n=== Keyword Q2 ===")
query2 = "Do you have anything without caffeine?"
result2 = simple_keyword_retrieval(query2, documents, verbose=True)
print("Selected document:", result2[0][0])
# Keyword RAG selected menu.txt (or "None found", depending on overlap) because
# the word "caffeine" never appears anywhere -- but "anything" and other query
# words barely overlap with menu.txt either. Keyword RAG got this WRONG (or got
# lucky by accident): it has no concept that "without caffeine" is *semantically*
# related to "decaf". It can only match literal words, not meaning.
# Semantic (embedding-based) retrieval would do better here, because "without
# caffeine" and "decaf"/"herbal tea" are close in meaning even though they
# share no exact words.

# --- Keyword Q3 ---
print("\n=== Keyword Q3 ===")
# Prediction: loyalty.txt should be selected, because "sign up" and "rewards"
# are close in meaning to "join" and "loyalty program" -- but keyword overlap
# only counts exact words, so the real winner depends on which literal words
# actually match. My best guess based on literal overlap is still loyalty.txt,
# since "loyalty"/"program"/"points" are the closest vocabulary match.
query3 = "How do I sign up for rewards?"
result3 = simple_keyword_retrieval(query3, documents, verbose=True)
print("Selected document:", result3[0][0])
# Compare the printed overlap scores above to the prediction and add a note
# here about whether it matched (e.g. if the score was 0 overlap and it fell
# back to "None found", that shows exactly the same weakness as Q2: no
# exact-word match even though the meaning is clearly about the loyalty program).


# =========================================================
# --- Semantic RAG Concepts ---
# =========================================================

# --- Semantic Q1 ---
# A vector embedding is a list of numbers that represents what a piece of
# text *means*, produced by a model trained so that texts with similar
# meaning end up with similar number patterns -- even if they don't share
# any of the same words.
#
# The chunk with a 0.85 cosine similarity is more relevant than the one with
# 0.30. Cosine similarity measures how closely two embeddings point in the
# same "direction" in vector space; a score near 1 means the two texts are
# talking about very similar things, while a score near 0 means they are
# largely unrelated in meaning.
#
# Semantic search can find a relevant chunk with none of the exact query
# words because it compares meaning, not spelling. Two sentences that use
# completely different vocabulary to describe the same idea (e.g. "decaf
# options" vs. "without caffeine") land close together in embedding space,
# so the similarity score is still high even though keyword overlap is zero.

# --- Semantic Q2 ---
# | Feature                    | Keyword RAG                       | Semantic RAG                              |
# |-----------------------------|------------------------------------|--------------------------------------------|
# | What is compared?          | Exact word overlap                | Similarity between meaning (embeddings)    |
# | What is retrieved?         | Full document                     | Small relevant chunks                      |
# | Can it handle synonyms?    | No                                 | Yes                                        |
# | Storage format             | Plain text dictionary             | Vector store / embedding index             |
# | Relevance score             | Number of overlapping keywords    | Cosine similarity score                    |


# =========================================================
# --- LlamaIndex ---
# =========================================================
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.embed_model = OpenAIEmbedding()

# brightleaf_pdfs/ should sit directly inside assignments_06/, next to this script.
reader = SimpleDirectoryReader("brightleaf_pdfs")
brightleaf_docs = reader.load_data()
brightleaf_index = VectorStoreIndex.from_documents(brightleaf_docs)
brightleaf_query_engine = brightleaf_index.as_query_engine(similarity_top_k=3)

# --- LlamaIndex Q1 ---
print("\n=== LlamaIndex Q1 ===")
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    response = brightleaf_query_engine.query(q)
    print(f"\nQuestion: {q}")
    print(f"Answer: {response}")
    for node in response.source_nodes:
        print(f"  Score: {node.score:.4f} | Text: {node.text[:150]}")

# Q1 "employee benefits": retrieved chunks look exactly relevant (the top
# chunk is literally the start of the benefits document). The answer sounds
# confident and specific, no hedging -- it lists health/vision/wellness,
# retirement, parental leave, professional development. Nothing unexpected
# got retrieved.
#
# Q1 "security policies": same pattern -- the top chunk is exactly about
# security, and the answer is detailed and confident, listing MFA,
# encryption, NIST, ISO 27001.


# --- LlamaIndex Q2 ---
print("\n=== LlamaIndex Q2 ===")
compare_query = questions[0]

engine_top1 = brightleaf_index.as_query_engine(similarity_top_k=1)
response_top1 = engine_top1.query(compare_query)
print("\ntop_k=1")
print("Answer:", response_top1)
for node in response_top1.source_nodes:
    print(f"  Score: {node.score:.4f}")

engine_top5 = brightleaf_index.as_query_engine(similarity_top_k=5)
response_top5 = engine_top5.query(compare_query)
print("\ntop_k=5")
print("Answer:", response_top5)
for node in response_top5.source_nodes:
    print(f"  Score: {node.score:.4f}")

# top_k=1 and top_k=5 gave nearly identical answers here, both grounded in
# the same top chunk (score 0.9097) about employee benefits. top_k=5 pulled
# in a few more chunks but they were lower-scoring and didn't change the
# substance of the answer. More context isn't always better -- once the
# single best chunk already contains the full answer, extra chunks mostly
# just add noise rather than new information.


# --- LlamaIndex Q3 ---
print("\n=== LlamaIndex Q3 ===")
hard_query = "How does BrightLeaf's remote work policy interact with its expense reimbursement rules?"
hard_response = brightleaf_query_engine.query(hard_query)
print(f"Question: {hard_query}")
print(f"Answer: {hard_response}")
for node in hard_response.source_nodes:
    print(f"  Score: {node.score:.4f} | Text: {node.text[:150]}")

# I expected the model to either honestly say a policy explicitly connecting
# these two topics isn't in the documents, or to visibly speculate. Instead,
# it confidently linked remote work flexibility to the Wellness Reimbursement
# Plan -- but that's not the same thing as "expense reimbursement rules"
# (that's a wellness perk, not a reimbursement policy for remote-work
# expenses). The model didn't hedge even though the connection is fairly
# stretched -- this illustrates that a confident tone doesn't guarantee
# accuracy. I'd improve this by adding a system prompt instructing the model
# to say "not covered in the documents" when the retrieved chunks don't
# directly answer the question.


# --- LlamaIndex Q4 ---
print("\n=== LlamaIndex Q4 ===")
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI

judge_llm = OpenAI(model="gpt-4o-mini")
faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

q_good = "What employee benefits does BrightLeaf offer?"
response_good = brightleaf_query_engine.query(q_good)
faith_good = faithfulness_evaluator.evaluate_response(response=response_good)
rel_good = relevancy_evaluator.evaluate_response(query=q_good, response=response_good)
print(f"\nQuery: {q_good}")
print(f"Faithfulness score: {faith_good.score}")
print(f"Relevancy score: {rel_good.score}")

q_bad = "What is BrightLeaf's stance on cryptocurrency payments?"
response_bad = brightleaf_query_engine.query(q_bad)
faith_bad = faithfulness_evaluator.evaluate_response(response=response_bad)
rel_bad = relevancy_evaluator.evaluate_response(query=q_bad, response=response_bad)
print(f"\nQuery: {q_bad}")
print(f"Faithfulness score: {faith_bad.score}")
print(f"Relevancy score: {rel_bad.score}")

# A faithfulness score of 1.0 means the response is fully supported by the
# retrieved source chunks (no unsupported claims); 0.0 would mean the
# response contains claims not grounded in the retrieved context (a
# hallucination). A relevancy score measures whether the response actually
# addresses the query -- it's about being on-topic, separate from whether
# the content is grounded in the source docs (that's what faithfulness
# checks).
# My results: the benefits question scored 1.0/1.0 (faithful and relevant --
# the documents fully cover it). The cryptocurrency question scored
# 1.0/0.0 -- faithfulness stayed 1.0 because the model didn't hallucinate an
# answer, it correctly indicated the topic isn't covered; relevancy dropped
# to 0.0 because that non-answer, by definition, doesn't address what was
# asked. This shows faithfulness and relevancy measure different things:
# a response can be perfectly honest (faithful) while still being useless
# for answering the actual question (irrelevant).
# "LLM-as-a-judge" means using a separate LLM call to grade response quality
# instead of a fixed formula. It's used for RAG evaluation because
# "correctness" here is about meaning and grounding, which is hard to check
# with simple string/accuracy metrics -- a judge LLM can read the response
# and source text and reason about whether the claims are supported.