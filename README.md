## Multimodal Customer Support Copilot

A production-grade multi-agent RAG system for intelligent customer support. Handles text queries, error screenshots, and system logs with confidence-based routing, 3-layer guardrails, and human escalation.

## Evaluation Results

| Metric | Score |
|--------|-------|
| Intent Classification Accuracy | 83.3% |
| Answer Quality Rate | 75.0% |
| Avg Retrieval Similarity | 0.563 |
| Guardrail Precision | 100% |
| Escalation Rate | 25.0% |
| Avg End-to-End Latency | ~2s |

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
