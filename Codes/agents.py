import os
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
#from openai import OpenAI

# So we can import from src/ later
sys.path.append("..")


from groq import Groq
#GROQ

GROQ_API_KEY = "****************"       # comment before committing
groq_client  = Groq(api_key=GROQ_API_KEY)
MODEL        = "llama-3.1-8b-instant"


#GEMINI 
import google.generativeai as genai

genai.configure(api_key="**************")

vision_model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash"
)

print("Groq   ready — classifier, responder, evaluator, guardrails")
print("Gemini ready — vision agent only")


embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="../vector_store")
collection = chroma_client.get_or_create_collection(
    name="support_kb",
    metadata={"hnsw:space": "cosine"}
)

print(f"Collection loaded with {collection.count()} docs")
print("All setup done!")



def classifier_agent(query: str) -> dict:
    prompt = f"""Classify this customer support query.

Query: "{query}"

Choose EXACTLY ONE category from this list:
- ACCOUNT (login, password, email, profile, account access)
- BILLING (charges, payments, invoices, subscriptions)
- ORDER (placing, cancelling, modifying orders)
- REFUND (refund requests, returns, refund status)
- SHIPPING (delivery, tracking, lost packages)
- TECHNICAL (app errors, crashes, bugs, error codes)
- GENERAL (anything else)

Rules:
- Reply ONLY in the exact format below
- Category must be one of the 7 words above exactly
- No extra words or explanation

CATEGORY: <one word from the list>
INTENT: <2-3 word description like reset_password or cancel_order>
CONFIDENCE: <HIGH or MEDIUM or LOW>"""

    response = groq_client.chat.completions.create(
        model       = MODEL,
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0.0,
        max_tokens  = 150
    )
    raw    = response.choices[0].message.content.strip()
    result = {"category": "GENERAL", "intent": "unknown", "confidence": "LOW"}

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("CATEGORY:"):
            result["category"]   = line.split(":", 1)[1].strip().upper()
        elif line.startswith("INTENT:"):
            result["intent"]     = line.split(":", 1)[1].strip().lower()
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line.split(":", 1)[1].strip().upper()

    # Force valid category — catch any model hallucinations
    valid = ["ACCOUNT","BILLING","ORDER","REFUND","SHIPPING","TECHNICAL","GENERAL"]
    if result["category"] not in valid:
        result["category"] = "GENERAL"

    return result

test = classifier_agent("I was charged twice for my order")
print(test)

# Run this diagnostic first
test_queries = [
    ("I can't reset my password",        "ACCOUNT"),
    ("I was charged twice",              "BILLING"),
    ("Where is my package",              "SHIPPING"),
    ("App keeps crashing",               "TECHNICAL"),
    ("I want a refund",                  "REFUND"),
    ("Cancel my order",                  "ORDER"),
]

print(f"{'Query':<35} {'Expected':<12} {'Got':<12} {'Match'}")
print("-" * 70)
for query, expected in test_queries:
    result    = classifier_agent(query)
    predicted = result["category"].upper().strip()
    match     = "Yes" if predicted == expected else "No"
    print(f"{query:<35} {expected:<12} {predicted:<12} {match}")



def retriever_agent(query: str, category: str = None, top_k: int = 5) -> dict:
    q_embedding = embedder.encode([query]).tolist()

    try:
        results     = collection.query(
            query_embeddings = q_embedding,
            n_results        = top_k
        )
        docs        = results["documents"][0]
        metadatas   = results["metadatas"][0]
        distances   = results["distances"][0]

        if not docs:
            raise ValueError("Empty")

        similarities = [round(1 - d, 3) for d in distances]

        # Rerank — boost docs that match the detected category
        if category:
            combined = list(zip(docs, metadatas, similarities))
            combined.sort(
                key=lambda x: (
                    1 if x[1].get("category","").upper() == category.upper() else 0,
                    x[2]
                ),
                reverse=True
            )
            docs, metadatas, similarities = zip(*combined)
            docs        = list(docs)
            metadatas   = list(metadatas)
            similarities = list(similarities)

        # Return top 3 after reranking
        return {
            "docs":         docs[:3],
            "metadatas":    metadatas[:3],
            "similarities": similarities[:3]
        }

    except Exception as e:
        print(f"[Retriever] {e}")
        return {
            "docs":         ["No relevant documents found."],
            "metadatas":    [{"category": "GENERAL", "intent": "unknown"}],
            "similarities": [0.0]
        }


# Test it

retrieved = retriever_agent("I was charged twice", category="BILLING")
for i, (doc, sim) in enumerate(zip(retrieved["docs"], retrieved["similarities"])):
    print(f"[{i+1}] similarity={sim}")
    print(f"     {doc[:120]}...")
    print()



def responder_agent(query: str, retrieved: dict, category: str, intent: str) -> str:
    """
    Generates a grounded answer using retrieved docs as context.
    """
    context = "\n\n".join([
        f"Doc {i+1} (similarity={retrieved['similarities'][i]}):\n{doc}"
        for i, doc in enumerate(retrieved["docs"])
    ])

    prompt = f"""You are a professional customer support agent.
Use the knowledge base excerpts below to answer the customer's question.
Be concise, empathetic, and helpful.
Do NOT make up information not present in the excerpts.
If the excerpts don't fully answer the question, say what you know and ask for more details.

Customer Category: {category}
Detected Intent: {intent}

Knowledge Base:
{context}

Customer Question: {query}

Your Answer:"""

    response = groq_client.chat.completions.create(
        model       = MODEL,
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0.1
    )

    return response.choices[0].message.content.strip()



# Test it
answer = responder_agent(
    query     = "I was charged twice for my order",
    retrieved = retrieved,
    category  = "BILLING",
    intent    = "double_charge"
)
print(answer)


def evaluator_agent(query: str, answer: str, similarities: list) -> dict:
    if not similarities:
        return {
            "avg_similarity": 0.0,
            "llm_score":      "LOW",
            "final_score":    "LOW",
            "reason":         "No docs retrieved",
            "relevant":       "no",
            "helpful":        "no",
            "professional":   "no"
        }

    similarities   = [max(0.0, min(1.0, s)) for s in similarities]
    avg_similarity = round(sum(similarities) / len(similarities), 3)

    # Fast path — very high similarity, skip LLM call
    if avg_similarity >= 0.70:
        return {
            "avg_similarity": avg_similarity,
            "llm_score":      "HIGH",
            "final_score":    "HIGH",
            "reason":         "High similarity fast path",
            "relevant":       "yes",
            "helpful":        "yes",
            "professional":   "yes"
        }

    # Fast path — very low similarity, escalate immediately
    if avg_similarity < 0.30:
        return {
            "avg_similarity": avg_similarity,
            "llm_score":      "LOW",
            "final_score":    "LOW",
            "reason":         "Low similarity fast path",
            "relevant":       "no",
            "helpful":        "no",
            "professional":   "yes"
        }

    # Middle range — ask LLM
    prompt = f"""Evaluate this customer support response quality.

Question: "{query}"
Answer: "{answer}"

RELEVANT: <yes or no>
HELPFUL: <yes or no>
PROFESSIONAL: <yes or no>
SCORE: <HIGH or MEDIUM or LOW>
REASON: <one sentence>"""

    response = groq_client.chat.completions.create(
        model       = MODEL,
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0.0
    )
    raw    = response.choices[0].message.content.strip()
    result = {
        "avg_similarity": avg_similarity,
        "llm_score":      "MEDIUM",
        "relevant":       "unknown",
        "helpful":        "unknown",
        "professional":   "unknown",
        "reason":         ""
    }

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("RELEVANT:"):
            result["relevant"]     = line.split(":",1)[1].strip()
        elif line.startswith("HELPFUL:"):
            result["helpful"]      = line.split(":",1)[1].strip()
        elif line.startswith("PROFESSIONAL:"):
            result["professional"] = line.split(":",1)[1].strip()
        elif line.startswith("SCORE:"):
            result["llm_score"]    = line.split(":",1)[1].strip()
        elif line.startswith("REASON:"):
            result["reason"]       = line.split(":",1)[1].strip()

    # Balanced scoring
    if avg_similarity >= 0.60 and result["llm_score"] == "HIGH":
        result["final_score"] = "HIGH"
    elif avg_similarity >= 0.40 and result["llm_score"] in ["HIGH","MEDIUM"]:
        result["final_score"] = "MEDIUM"
    else:
        result["final_score"] = "LOW"

    return result


# ── Testing ──────

eval_result = evaluator_agent(
    query        = "I was charged twice for my order",
    answer       = answer,
    similarities = retrieved["similarities"]   # comes from Step 3
)
print(eval_result)


# Check a sample from the collection
sample = collection.peek(5)
print("Sample documents:")
for i, doc in enumerate(sample["documents"]):
    print(f"\n[{i+1}] {doc[:100]}...")
    print(f"     metadata: {sample['metadatas'][i]}")





