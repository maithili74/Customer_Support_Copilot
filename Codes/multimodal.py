

import os
import sys
import base64
import json
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
#from openai import OpenAI
from PIL import Image

sys.path.append("..")
print("Imports done!")


from groq import Groq
#GROQ

GROQ_API_KEY = "g***************"       # comment before committing
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





embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="../vector_store")
collection = chroma_client.get_or_create_collection(
    name="support_kb",
    metadata={"hnsw:space": "cosine"}
)

print(f"Collection loaded: {collection.count()} docs")



from agents import classifier_agent, retriever_agent, responder_agent, evaluator_agent



def encode_image(image_path: str) -> str:
    """Convert image to base64 string for the API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime_type(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif":  "image/gif",
        ".webp": "image/webp"
    }
    return mime_map.get(ext, "image/jpeg")


def validate_image(image_path: str) -> bool:
    """Check image exists and is under 5MB."""
    path = Path(image_path)
    if not path.exists():
        print(f"Image not found: {image_path}")
        return False
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > 5:
        print(f"Image too large: {size_mb:.1f}MB (max 5MB)")
        return False
    return True


# Test helpers
print("Image helpers ready!")



def vision_agent(query: str, image_path: str) -> dict:
    """
    Gemini handles the image.
    Auto-retries on rate limit, falls back gracefully if all retries fail.
    """
    if not validate_image(image_path):
        return {
            "success":         False,
            "summary":         "Image could not be processed.",
            "error_messages":  [],
            "ui_section":      "unknown",
            "problem_visible": "No image context available.",
            "relevant_text":   []
        }

    img = Image.open(image_path)
    print(f"[Vision/Gemini] Processing: {Path(image_path).name} | size={img.size}")

    prompt = f"""You are analyzing a customer support screenshot.
Customer issue: "{query}"

Respond ONLY in this JSON format:
{{
    "error_messages":  ["list any error messages visible"],
    "ui_section":      "which part of the app is shown",
    "problem_visible": "describe the problem in one sentence",
    "relevant_text":   ["any other relevant text on screen"],
    "summary":         "one paragraph summary for a support agent"
}}"""

    max_retries = 3
    wait_times  = [30, 60, 120]

    for attempt in range(max_retries):
        try:
            response = vision_model.generate_content([prompt, img])
            raw      = response.text.strip()

            try:
                clean  = re.sub(r"```json|```", "", raw).strip()
                parsed = json.loads(clean)
                parsed["success"] = True
                print(f"[Vision/Gemini] Analysis complete")
                return parsed
            except json.JSONDecodeError:
                return {
                    "success":         True,
                    "error_messages":  [],
                    "ui_section":      "unknown",
                    "problem_visible": raw,
                    "relevant_text":   [],
                    "summary":         raw
                }

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "ResourceExhausted" in error_str or "quota" in error_str.lower():
                retry_match = re.search(r"retry in (\d+)", error_str.lower())
                wait        = int(retry_match.group(1)) + 5 if retry_match else wait_times[attempt]
                print(f"[Vision/Gemini] Rate limited. Waiting {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            else:
                print(f"[Vision/Gemini] Error: {error_str[:100]}")
                break

    # Fallback — Groq describes the situation without the image
    print("[Vision/Gemini] All retries failed — using Groq fallback")
    fallback_prompt = f"""A customer submitted a support ticket with an image attachment 
that could not be analyzed automatically.
Their text query is: "{query}"
Describe what kind of issue they might be facing and what information 
a support agent should ask for."""

    fallback = groq_client.chat.completions.create(
        model       = MODEL,
        messages    = [{"role": "user", "content": fallback_prompt}],
        temperature = 0.3
    )
    fallback_text = fallback.choices[0].message.content.strip()

    return {
        "success":         False,
        "error_messages":  [],
        "ui_section":      "unknown",
        "problem_visible": "Image analysis unavailable",
        "relevant_text":   [],
        "summary":         fallback_text
    }




def log_analyzer_agent(log_text: str, query: str) -> dict:
    """
    Analyzes error logs or system output pasted by the customer.
    Extracts key errors and maps them to likely causes.
    """
    prompt = f"""You are a technical support specialist analyzing error logs.
The customer's issue is: "{query}"

Analyze these logs and extract key information:

LOGS:
{log_text}

Respond in this exact JSON format:
{{
    "error_codes":    ["list of error codes found, e.g. 500, 404, NullPointerException"],
    "error_messages": ["list of error messages found"],
    "likely_cause":   "most likely root cause in one sentence",
    "severity":       "LOW, MEDIUM, or HIGH",
    "suggested_fix":  "brief technical suggestion",
    "summary":        "one paragraph summary for a support agent"
}}

Return ONLY the JSON, no extra text."""

    response = groq_client.chat.completions.create(
        model       = MODEL,
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0.1
    )
    raw    = response.choices[0].message.content.strip()

    try:
        clean  = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(clean)
        parsed["success"] = True
        return parsed
    except json.JSONDecodeError:
        return {
            "success":        True,
            "error_codes":    [],
            "error_messages": [raw],
            "likely_cause":   "Could not parse logs automatically",
            "severity":       "MEDIUM",
            "suggested_fix":  "Manual review needed",
            "summary":        raw
        }



# Test with a synthetic log
test_log = """
[2024-01-15 10:23:41] ERROR 500: Payment service timeout
[2024-01-15 10:23:41] Failed to connect to payment-gateway.internal:8080
[2024-01-15 10:23:42] Retrying... attempt 1/3
[2024-01-15 10:23:45] Retrying... attempt 2/3
[2024-01-15 10:23:48] Retrying... attempt 3/3
[2024-01-15 10:23:48] CRITICAL: Transaction ID txn_8821 failed after 3 retries
[2024-01-15 10:23:48] Customer charge may be in inconsistent state
"""

log_result = log_analyzer_agent(test_log, "my payment failed")
print(json.dumps(log_result, indent=2))



def enrich_query(
    query:      str,
    image_path: str  = None,
    log_text:   str  = None
) -> dict:
    """
    Takes optional image and/or log and enriches the query with context.
    Returns enriched query string + all extracted context.
    """
    context_parts  = [f"Customer query: {query}"]
    image_context  = None
    log_context    = None

    if image_path:
        print("[Multimodal] Analyzing image...")
        image_context = vision_agent(query, image_path)
        if image_context.get("success"):
            context_parts.append(
                f"Visual context from screenshot:\n"
                f"- UI section: {image_context.get('ui_section', 'unknown')}\n"
                f"- Problem visible: {image_context.get('problem_visible', '')}\n"
                f"- Error messages seen: {', '.join(image_context.get('error_messages', []) or ['none'])}\n"
                f"- Summary: {image_context.get('summary', '')}"
            )

    if log_text:
        print("[Multimodal] Analyzing logs...")
        log_context = log_analyzer_agent(log_text, query)
        if log_context.get("success"):
            context_parts.append(
                f"Log analysis:\n"
                f"- Error codes: {', '.join(log_context.get('error_codes', []) or ['none'])}\n"
                f"- Likely cause: {log_context.get('likely_cause', '')}\n"
                f"- Severity: {log_context.get('severity', 'MEDIUM')}\n"
                f"- Suggested fix: {log_context.get('suggested_fix', '')}"
            )

    enriched_query = "\n\n".join(context_parts)

    return {
        "original_query": query,
        "enriched_query": enriched_query,
        "has_image":      image_path is not None,
        "has_logs":       log_text   is not None,
        "image_context":  image_context,
        "log_context":    log_context
    }


def router(final_score: str, query: str, answer: str) -> dict:
    if final_score == "HIGH":
        action  = "RESPOND"
        message = answer

    elif final_score == "MEDIUM":
        action  = "RESPOND"
        message = answer

    else:
        # Only LOW escalates
        action  = "ESCALATE"
        message = (
            "I'm sorry, I wasn't able to find a confident answer. "
            "I'm escalating this to a human agent who will get "
            "back to you shortly."
        )

    return {"action": action, "message": message}

print("Router defined!")

def multimodal_pipeline(
    query:      str,
    image_path: str  = None,
    log_text:   str  = None,
    verbose:    bool = True
) -> dict:

    if verbose:
        print(f"{'='*60}")
        print(f"Query: {query}")
        if image_path: print(f"Image: {image_path}")
        if log_text:   print(f"Logs:  yes ({len(log_text)} chars)")
        print(f"{'='*60}")

    # Step 1: Enrich query with multimodal context
    enriched = enrich_query(query, image_path, log_text)
    enriched_query = enriched["enriched_query"]

    if verbose and (image_path or log_text):
        print(f"\n[Enriched Query Preview]\n{enriched_query[:300]}...\n")

    # Step 2: Run through all 4 agents using enriched query
    classification = classifier_agent(enriched_query)
    if verbose:
        print(f"[Classifier] category={classification['category']}  intent={classification['intent']}")

    retrieved = retriever_agent(enriched_query, category=classification["category"])
    if verbose:
        print(f"[Retriever]  similarities={retrieved['similarities']}")

    answer = responder_agent(
        query     = enriched_query,
        retrieved = retrieved,
        category  = classification["category"],
        intent    = classification["intent"]
    )
    if verbose:
        print(f"\n[Responder]\n{answer}")

    evaluation = evaluator_agent(enriched_query, answer, retrieved["similarities"])
    if verbose:
        print(f"\n[Evaluator]  final_score={evaluation['final_score']}  reason={evaluation['reason']}")

    decision = router(evaluation["final_score"], enriched_query, answer)
    if verbose:
        print(f"\n[Router]     action={decision['action']}")
        print(f"\n{'='*60}")
        print(f"Final Response:\n{decision['message']}")
        print(f"{'='*60}")

    return {
        "query":          query,
        "enriched":       enriched,
        "classification": classification,
        "retrieved":      retrieved,
        "answer":         answer,
        "evaluation":     evaluation,
        "decision":       decision
    }


import time

def timed_pipeline(query):
    start  = time.time()
    result = multimodal_pipeline(query=query, verbose=False)
    end    = time.time()
    return result, round(end - start, 2)



result1 = multimodal_pipeline(
    query = "I can't reset my password"
)




# Save any screenshot to data/ folder and point to it here
# If you don't have one, skip this cell for now

result3 = multimodal_pipeline(
    query      = "the checkout page shows an error",
    image_path = "../data/raw/pay_error.png"   # swap with your path
)




payment_log = """
[2024-01-15 10:23:41] ERROR 500: Payment service timeout
[2024-01-15 10:23:41] Failed to connect to payment-gateway.internal:8080
[2024-01-15 10:23:48] CRITICAL: Transaction ID txn_8821 failed after 3 retries
[2024-01-15 10:23:48] Customer charge may be in inconsistent state
"""

result2 = multimodal_pipeline(
    query    = "my payment failed and I'm not sure if I was charged",
    log_text = payment_log
)






