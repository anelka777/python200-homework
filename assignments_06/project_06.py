"""
Week 6 Mini-Project
Groundwork Coffee Co. RAG-powered Q&A assistant
"""

# =========================================================
# Step 1: Setup
# =========================================================
from dotenv import load_dotenv
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

Settings.embed_model = OpenAIEmbedding()

# groundwork_docs/ should sit directly inside assignments_06/, next to this script.
docs_dir = Path("groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"


# =========================================================
# Step 2: Load the Documents
# =========================================================
reader = SimpleDirectoryReader(str(docs_dir))
groundwork_docs = reader.load_data()

print(f"\nLoaded {len(groundwork_docs)} documents:")
for doc in groundwork_docs:
    print(f"  - {doc.metadata.get('file_name')}")


# =========================================================
# Step 3: Build the Index and Query Engine
# =========================================================
groundwork_index = VectorStoreIndex.from_documents(groundwork_docs)
groundwork_query_engine = groundwork_index.as_query_engine(similarity_top_k=3)
print("\nIndex built successfully. Ready to answer questions.")


# =========================================================
# Step 4: Query the Assistant
# =========================================================
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

print("\n=== Step 4: Querying the assistant ===")
for question in questions:
    response = groundwork_query_engine.query(question)
    top_node = response.source_nodes[0]

    print(f"\nQuestion: {question}")
    print(f"Answer: {response}")
    print(f"Top source: {top_node.metadata.get('file_name')}")
    print(f"  Score: {top_node.score:.4f}")
    print(f"  Text: {top_node.text[:200]}")

# The assistant sounded confident and accurate across all five questions,
# with high-scoring top chunks (0.76-0.90) directly matching the topic asked.
# Nothing felt thin or vague -- even the loyalty program and dairy-free
# questions, which weren't in the top-scoring document (faq.txt still won on
# the loyalty question despite the exact numbers coming from a different part
# of the same file), pulled clean, correct-sounding answers. Nothing
# surprised me; the retrieval consistently picked the right source document
# for each question.


# =========================================================
# Step 5: Find a Failure
# =========================================================
print("\n=== Step 5: Failure case ===")
# This question asks the assistant to combine information that likely lives
# in two different documents (loyalty program rules + catering/wholesale
# terms), which single top-k retrieval often struggles to stitch together.
failure_question = "If I order catering for a company event, do I still earn loyalty points on that purchase?"
failure_response = groundwork_query_engine.query(failure_question)

print(f"Question: {failure_question}")
print(f"Answer: {failure_response}")
for node in failure_response.source_nodes:
    print(f"\n  Document: {node.metadata.get('file_name')}")
    print(f"  Score: {node.score:.4f}")
    print(f"  Text: {node.text[:200]}")

# What I asked and why I expected it to be hard: the question combines a
# catering/wholesale detail with a loyalty-program detail -- two separate
# documents (wholesale_catering.txt and faq.txt) -- and the exact combination
# ("do catering purchases earn loyalty points?") is not written down anywhere
# in either document.
#
# What went wrong: this wasn't a retrieval failure -- the top chunk
# (wholesale_catering.txt, score 0.7671) is genuinely the most relevant
# document, and faq.txt (with the loyalty program details) was retrieved
# too. The real issue is a missing-information problem: the documents never
# state whether catering purchases earn loyalty points, so there's no
# grounded answer to give. Despite that, the model didn't say "this isn't
# specified in the documents" -- it flatly answered "You do not earn loyalty
# points on catering purchases," which is not something either source
# document actually states. The model appears to have guessed/inferred this
# from the fact that wholesale_catering.txt makes no mention of loyalty
# points, treating an absence of information as a definitive "no."
#
# Did the tone change? No -- the answer was just as short and confident as
# the well-grounded answers in Step 4, with zero hedging. This is exactly
# the core trust issue with RAG: a fluent, confident tone doesn't guarantee
# a grounded answer, and this response would look identical to a correct one
# without checking it against the source text.
#
# What I'd change: add an explicit instruction (via a custom prompt template)
# telling the model to say "not specified in the available documents" when
# the source chunks don't directly state an answer, rather than inferring
# from silence. I'd also add a line to the loyalty program document
# explicitly stating whether catering/wholesale purchases count toward
# points, so the cross-cutting policy is stated in one place instead of
# left implicit.


# =========================================================
# Step 6: Reflection
# =========================================================
# - The manual semantic RAG build in the lesson (chunking, embedding calls,
#   writing a cosine-similarity search function, storing/indexing vectors)
#   took a good number of lines just for the retrieval machinery. Here,
#   SimpleDirectoryReader + VectorStoreIndex + as_query_engine() replaced
#   all of that in roughly 5-6 lines. That's the value of a framework: the
#   chunking strategy, embedding calls, vector storage, and similarity
#   search are all handled internally, so you can focus on the application
#   logic (which documents, which questions, how to evaluate) instead of
#   re-implementing retrieval infrastructure every time.
# - Different use case: a hospital's internal onboarding assistant that
#   answers new-employee questions from HR policy documents, benefits
#   guides, and department handbooks -- the same "folder of documents ->
#   accurate Q&A without manually searching every PDF" value applies to any
#   organization with a large, evolving body of internal documentation
#   (legal firms, universities, government agencies, software companies with
#   large internal wikis, etc.).
# - One failure mode RAG cannot fully prevent even when retrieval works
#   correctly: the LLM can still misinterpret or slightly distort the
#   retrieved text when generating the final answer -- retrieval only
#   guarantees the *right chunks were found*, not that the generation step
#   summarizes them perfectly. Also, if the documents themselves are
#   outdated or wrong, RAG will confidently retrieve and repeat that
#   incorrect information -- retrieval doesn't fact-check the source data.