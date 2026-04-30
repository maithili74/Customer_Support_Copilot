import google.generativeai as genai
from groq import Groq


import os
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
#from openai import OpenAI
from tqdm import tqdm

print("All imports successful!")


#GROQ

GROQ_API_KEY = "*********************"       # comment before committing
groq_client  = Groq(api_key=GROQ_API_KEY)
MODEL        = "llama-3.1-8b-instant"


#GEMINI 
import google.generativeai as genai

genai.configure(api_key="*******************")

vision_model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash"
)

print("Groq   ready — classifier, responder, evaluator, guardrails")
print("Gemini ready — vision agent only")



print("Loading dataset from hugging face transformers")

#ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
df = pd.read_csv("../data/raw/support_dataset.csv")
#df = ds["train"].to_pandas()

print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nCategories:\n{df['category'].value_counts()}")



df.head()


df['category'].value_counts()


df.isnull().sum() #no missing values


df[["instruction", "response", "category", "intent"]].head(5)



os.makedirs("../data/raw", exist_ok=True)
df.to_csv("../data/raw/support_dataset.csv", index=False)
print(f"Saved {len(df)} rows to data/raw/support_dataset.csv")



# Cleaning the dataset - since most of our dataset is already clean only few necessary changes

# Drop duplicates, strip whitespace, keep only what we need
df_clean = df[["instruction", "response", "category", "intent"]].copy()
df_clean = df_clean.drop_duplicates(subset=["response"])
df_clean = df_clean.dropna()
df_clean["instruction"] = df_clean["instruction"].str.strip()
df_clean["response"]    = df_clean["response"].str.strip()

os.makedirs("../data/processed", exist_ok=True)
df_clean.to_csv("../data/processed/support_clean.csv", index=False)

print(f"Cleaned shape: {df_clean.shape}")
df_clean.head(3)


# Building the knowledge base

# The knowledge base means the "responses" column, these are the good answers the system will retrieve from

os.makedirs("../data/knowledge_base", exist_ok=True)

kb_docs = []
for _, row in df_clean.iterrows():
    kb_docs.append({
        "text":     row["response"],
        "category": row["category"],
        "intent":   row["intent"]
    })

print(f"Knowledge base size: {len(kb_docs)} documents")
print(f"\nExample document:")
print(kb_docs[0])

#Creating embedding and storing in ChromaDB

print("Loading embedding model (downloads once ~80MB)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready!")

chroma_client = chromadb.PersistentClient(path="../vector_store")
collection = chroma_client.get_or_create_collection(
    name="support_kb",
    metadata={"hnsw:space": "cosine"}
)

print(f"Collection ready. Current doc count: {collection.count()}")


# Only embed if collection is empty (avoids re-embedding on restart)
if collection.count() == 0:
    print("Embedding and storing documents...")
    batch_size = 100

    for i in tqdm(range(0, len(kb_docs), batch_size)):
        batch     = kb_docs[i : i + batch_size]
        texts     = [doc["text"] for doc in batch]
        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            documents  = texts,
            embeddings = embeddings,
            metadatas  = [{"category": d["category"], "intent": d["intent"]} for d in batch],
            ids        = [f"doc_{i+j}" for j in range(len(batch))]
        )

    print(f"Done! Stored {collection.count()} documents.")
else:
    print(f"Collection already has {collection.count()} docs. Skipping embedding.")



def retrieve_docs(query: str, top_k: int = 3):
    q_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=top_k
    )
    return results["documents"][0], results["metadatas"][0]


def generate_answer(query: str, context_docs: list) -> str:
    context = "\n\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(context_docs)])

    prompt = f"""You are a helpful customer support agent.
Use the knowledge base excerpts below to answer the customer's question.
If the answer isn't in the excerpts, say so honestly — do not make things up.

Knowledge Base:
{context}

Customer Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model       = MODEL,
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0.3
    )
    return response.choices[0].message.content.strip()


def rag_pipeline(query: str):
    print(f"Query: {query}\n")
    docs, metas = retrieve_docs(query)
    answer = generate_answer(query, docs)

    print(f"Answer:\n{answer}")
    print(f"\nSources used:")
    for i, m in enumerate(metas):
        print(f"  [{i+1}] category={m['category']}  intent={m['intent']}")
    return answer



rag_pipeline("I can't reset my password")


rag_pipeline("I was charged twice for my order")


import os

vs_path = "../vector_store"
files = os.listdir(vs_path)
print(f"Vector store files: {files}")
print(f"Total KB docs in collection: {collection.count()}")



rag_pipeline("How do I cancel my subscription?")


# Data layer — you downloaded a real dataset of 26,000+ customer support conversations from HuggingFace, cleaned it (removed duplicates, nulls, whitespace), and saved both raw and processed versions to disk.
# 
# Knowledge base — you took the response column from the dataset and turned it into a searchable knowledge base. These are the "good answers" your system learns from.
# 
# Embeddings — you used all-MiniLM-L6-v2 (a free, local model) to convert every KB document into a vector — a list of numbers that captures the meaning of the text. Similar questions get similar vectors.
# 
# Vector store — you stored all those vectors in ChromaDB, persisted to disk. This means when someone asks a question, you can instantly find the most semantically similar KB documents without scanning all 26,000 rows one by one.
# 
# RAG pipeline — you connected everything into one function: question comes in → gets embedded → top 3 similar docs retrieved → docs + question sent to LLM → grounded answer comes out. The LLM never guesses — it answers based on real retrieved knowledge.




