## Multimodal Customer Support Copilot

A production-grade multi-agent RAG system for intelligent customer support. Handles text queries, error screenshots, and system logs with confidence-based routing, 3-layer guardrails, and human escalation.

## Architecture

```
Customer Input (text + optional image/logs)
            ↓
    ┌─────────────────┐
    │   Guardrails    │  ← PII masking, injection detection, toxicity filter
    └────────┬────────┘
             │ ALLOW / MASK / BLOCK
             ↓
    ┌─────────────────┐
    │   Classifier    │  ← Intent + category detection (Llama 3.1)
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │    Retriever    │  ← Semantic search over 26K KB docs (ChromaDB)
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │    Responder    │  ← Grounded answer generation (Llama 3.1)
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │    Evaluator    │  ← Confidence scoring (similarity + LLM check)
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │     Router      │  ← RESPOND or ESCALATE TO HUMAN
    └─────────────────┘
```

## Features

- **Multi-agent RAG pipeline** - Classifier → Retriever → Responder → Evaluator → Router
- **Multimodal inputs** - text queries, error screenshots (Gemini Vision), and system error logs
- **3-layer guardrails** - regex fast path, LLM-based injection + toxicity detection, PII masking
- **Confidence-based routing** - automatically escalates low-confidence queries to human agents
- **Conversation memory** - remembers last 3 exchanges for natural follow-up questions
- **Evaluator fast path** - skips LLM call for high/low similarity cases to reduce latency
- **Category-aware reranking** - boosts retrieved docs that match the classified category
- **Gradio chat UI** - clean interface with screenshot upload, log paste, and example queries

## Project Structure

```
customer-support-copilot/
│
├── notebooks/
│   ├── agents.py           ← classifier, retriever, responder, evaluator, router
│   ├── multimodal.py       ← vision agent, log analyzer, enrich query, pipeline
│   ├── guardrails.py       ← PII detector, injection detector, toxicity filter
│   ├── ui.py               ← Gradio chat interface
│   └── metrics.py          ← evaluation pipeline and benchmark
│
├── data/
│   ├── raw/                ← downloaded dataset + uploaded images
│   ├── processed/          ← cleaned CSV, eval results
│   └── knowledge_base/     ← FAQ text chunks
│
├── vector_store/           ← ChromaDB persisted embeddings (auto-created)
├── requirements.txt
└── README.md
```

##  Tech Stack

| Component | Technology |
|-----------|------------|
| LLM (text) | Llama 3.1 8B via Groq API |
| LLM (vision) | Gemini 2.5 Flash |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector DB | ChromaDB (local, persisted) |
| Dataset | Bitext Customer Support (26K conversations) |
| UI | Gradio 6.x |
| Guardrails | Regex + LLM classification |

## How RAG Works Here

1. **Knowledge base** - 26,000 real customer support responses embedded using `all-MiniLM-L6-v2` and stored in ChromaDB
2. **Query embedding** - customer question converted to the same vector space
3. **Similarity search** - ChromaDB finds top 5 most semantically similar KB documents
4. **Reranking** - documents matching the classified category are boosted, top 3 returned
5. **Grounded generation** - Llama 3.1 reads the 3 retrieved documents as context and generates an answer based on them

## Evaluation Results

| Metric | Score |
|--------|-------|
| Intent Classification Accuracy | 83.3% |
| Answer Quality Rate | 75.0% |
| Avg Retrieval Similarity | 0.563 |
| Guardrail Precision | 100% |
| Escalation Rate | 25.0% |
| Avg End-to-End Latency | ~2s |
