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
