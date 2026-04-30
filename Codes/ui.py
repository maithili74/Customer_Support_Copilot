
from pathlib import Path
import jinja2
print(f"jinja2: {jinja2.__version__}")
print("Gradio installed!")


import gradio as gr
import time
import json
import re
import os
import sys
from groq import Groq
import google.generativeai as genai
from PIL import Image
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
from datetime import datetime

GROQ_API_KEY   = "*******************"    # comment before committing
GEMINI_API_KEY = "*****************"    # comment before committing

groq_client  = Groq(api_key=GROQ_API_KEY)
MODEL        = "llama-3.1-8b-instant"

genai.configure(api_key=GEMINI_API_KEY)
vision_model = genai.GenerativeModel("models/gemini-2.0-flash-001")

embedder      = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="../vector_store")
collection    = chroma_client.get_or_create_collection(
    name     = "support_kb",
    metadata = {"hnsw:space": "cosine"}
)

GEMINI_AVAILABLE = True

print(f"Collection: {collection.count()} docs")
print("Setup done!")

from agents import classifier_agent, retriever_agent, responder_agent, evaluator_agent
from multimodal import router, multimodal_pipeline, validate_image, vision_agent, log_analyzer_agent, enrich_query, timed_pipeline
from guardrails import pii_detector,injection_detector,toxicity_filter,run_guardrails_fast, safe_pipeline


review_queue = []

def add_to_queue(query, answer, reason, evaluation):
    ticket = {
        "id":         len(review_queue) + 1,
        "time":       datetime.now().strftime("%H:%M:%S"),
        "query":      query[:60],
        "reason":     reason,
        "confidence": evaluation.get("final_score", "N/A"),
        "status":     "PENDING"
    }
    review_queue.append(ticket)
    return ticket

def get_queue_df():
    if not review_queue:
        return pd.DataFrame(
            columns=["ID","Time","Query","Reason","Confidence","Status"]
        )
    return pd.DataFrame([{
        "ID":         t["id"],
        "Time":       t["time"],
        "Query":      t["query"],
        "Reason":     t["reason"],
        "Confidence": t["confidence"],
        "Status":     t["status"]
    } for t in review_queue])

print("Queue ready!")


from pathlib import Path
import os

# Redefine validate_image with Path import built in
def validate_image(image_path: str) -> bool:
    from pathlib import Path
    path = Path(image_path)
    if not path.exists():
        print(f"[Vision] Image not found: {image_path}")
        return False
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > 5:
        print(f"[Vision] Image too large: {size_mb:.1f}MB")
        return False
    return True

# Redefine vision_agent with all imports built in
def vision_agent(query: str, image_path: str) -> dict:
    from pathlib import Path
    import re, json, time
    from PIL import Image

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

    max_retries = 2
    wait_times  = [30, 60]

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
            if "429" in error_str or "quota" in error_str.lower():
                wait = wait_times[attempt] if attempt < len(wait_times) else 60
                print(f"[Vision] Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            else:
                print(f"[Vision] Error: {error_str[:100]}")
                break

    # Groq fallback
    print("[Vision] Using Groq fallback...")
    try:
        fallback_prompt = (
            f"A customer submitted a support ticket with an image. "
            f"Their message: '{query}'. "
            f"Image filename: '{Path(image_path).name}'. "
            f"What issue might they be facing and what should support ask? "
            f"Reply in JSON: {{\"error_messages\":[], \"ui_section\":\"\", "
            f"\"problem_visible\":\"\", \"relevant_text\":[], \"summary\":\"\"}}"
        )
        resp = groq_client.chat.completions.create(
            model       = MODEL,
            messages    = [{"role": "user", "content": fallback_prompt}],
            temperature = 0.1,
            max_tokens  = 200
        )
        raw = resp.choices[0].message.content.strip()
        try:
            clean  = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(clean)
            parsed["success"]  = False
            parsed["fallback"] = True
            return parsed
        except:
            return {
                "success":         False,
                "fallback":        True,
                "error_messages":  [],
                "ui_section":      "unknown",
                "problem_visible": f"Customer reports: {query}",
                "relevant_text":   [],
                "summary":         raw
            }
    except Exception as e:
        return {
            "success":         False,
            "error_messages":  [],
            "ui_section":      "unknown",
            "problem_visible": "Image analysis unavailable",
            "relevant_text":   [],
            "summary":         f"Customer query: {query}. Manual review needed."
        }

print("validate_image ✓")
print("vision_agent   ✓")


def chat(message, history, log_input, image_input):
    """
    message     = current user message
    history     = list of [user, bot] pairs — Gradio handles this
    log_input   = optional error logs pasted by user
    image_input = optional screenshot uploaded by user
    """
    if not message or not message.strip():
        return "Please type a question."

    # ── Handle image ──────────────────────────────────────────
    # NEW — handles all Gradio image types correctly
    image_path = None
    if image_input is not None:
        if isinstance(image_input, str) and os.path.exists(image_input):
            image_path = image_input
            print(f"[UI] Image: {image_path}")
        else:
            print(f"[UI] Image invalid or not found: {image_input}")
            

    # ── Handle logs ───────────────────────────────────────────
    logs = log_input.strip() if log_input and log_input.strip() else None

    # ── Build conversation context from history ───────────────
    # Replace the old for user_msg, bot_msg in recent block with this:
    context = ""
    if history:
        recent  = history[-6:]
        context = "Previous conversation:\n"
        for msg in recent:
            if msg["role"] == "user":
                # content can be string or list (when image was attached)
                content = msg["content"]
                if isinstance(content, list):
                    # extract just the text parts, skip image objects
                    text_parts = [
                        part if isinstance(part, str)
                        else part.get("text", "") if isinstance(part, dict)
                        else ""
                        for part in content
                    ]
                    content = " ".join([p for p in text_parts if p]).strip()
                if content:
                    context += f"Customer: {content}\n"

            elif msg["role"] == "assistant":
                content = msg["content"]
                if isinstance(content, str):
                    clean_bot = content.split("\n\n---\n")[0]
                    context  += f"Agent: {clean_bot}\n"
        context += "\n"

    # ── Run guardrails on current message only ────────────────
    guard = run_guardrails_fast(message)

    if guard["overall_action"] == "BLOCK":
        reason = guard.get("block_reason", "Policy violation")
        add_to_queue(
            query      = message,
            answer     = "",
            reason     = reason,
            evaluation = {"final_score": "BLOCKED"}
        )
        return (
            f"I'm sorry, your message could not be processed.\n\n"
            f"**Reason:** {reason}\n\n"
            f"Please rephrase or contact support directly."
        )

    # ── Use masked text if PII found ──────────────────────────
    clean_message = (
        guard["processed_text"]
        if guard["overall_action"] == "MASK_AND_PROCEED"
        else message
    )

    # ── Add conversation context to query ─────────────────────
    full_query = (
        f"{context}Current question: {clean_message}"
        if context else clean_message
    )

    # ── Run pipeline ──────────────────────────────────────────
    start  = time.time()
    result = multimodal_pipeline(
        query      = full_query,
        image_path = image_path,
        log_text   = logs,
        verbose    = False
    )
    latency = round(time.time() - start, 2)

    # ── Extract results ───────────────────────────────────────
    decision   = result.get("decision",       {})
    evaluation = result.get("evaluation",     {})
    classify   = result.get("classification", {})
    retrieved  = result.get("retrieved",      {})

    answer     = decision.get("message", "I could not generate a response.")
    action     = decision.get("action",  "UNKNOWN")
    score      = evaluation.get("final_score",    "N/A")
    category   = classify.get("category",         "N/A")
    intent     = classify.get("intent",           "N/A")
    avg_sim    = evaluation.get("avg_similarity", "N/A")

    # ── Add to review queue if needed ─────────────────────────
    if action == "ESCALATE":
        add_to_queue(
            query      = message,
            answer     = answer,
            reason     = "Low confidence — needs human review",
            evaluation = evaluation
        )

    # ── Build response with metadata footer ───────────────────
    pii_note = (
        "\n\n⚠️ *Note: Personal information was detected "
        "and masked for your privacy.*"
        if guard["overall_action"] == "MASK_AND_PROCEED"
        else ""
    )

    footer = (
        f"\n\n---\n"
        f"*🏷️ {category} → {intent} | "
        f"📊 Confidence: {score} | "
        f"🔍 Similarity: {avg_sim} | "
        f"⚡ {latency}s*"
    )

    return answer + pii_note + footer


from pathlib import Path
import builtins

# Inject Path into builtins so ALL functions can access it
# regardless of where they were defined
builtins.Path = Path

# Also inject other commonly missing names
import re, json, time
from PIL import Image
builtins.re    = re
builtins.json  = json
builtins.time  = time
builtins.Image = Image

print("Patched builtins — Path, re, json, time, Image now globally available")

# Verify
test = Path(".")
print(f"Path works: {test.resolve()}")


def build_ui():
    # Remove theme from here
    with gr.Blocks(title="Customer Support Copilot") as demo:

        # ── Header ────────────────────────────────────────────
        gr.Markdown("# 🤖 Customer Support Copilot")
        gr.Markdown(
            "Ask your support question below. "
            "Follow-up questions are supported. "
            "You can also attach a screenshot or paste error logs."
        )

        # ── Main layout ───────────────────────────────────────
        with gr.Row():

            # Left — chat
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label      = "Conversation",
                    height     = 500,
                    show_label = False
                )
                msg_input = gr.Textbox(
                    label       = "",
                    placeholder = "Type your question here... (Enter to send)",
                    lines       = 2,
                    show_label  = False
                )
                with gr.Row():
                    submit_btn = gr.Button("Send",  variant="primary",   scale=3)
                    clear_btn  = gr.Button("Clear", variant="secondary", scale=1)

            # Right — optional inputs
            with gr.Column(scale=1):
                gr.Markdown("### Optional")
                image_input = gr.Image(
                    label   = "Attach Screenshot",
                    type    = "filepath",   # ← Gradio saves it and gives you the path directly
                    sources = ["upload"]
                )
                log_input = gr.Textbox(
                    label       = "Paste Error Logs",
                    placeholder = "ERROR 500: Payment timeout...",
                    lines       = 6
                )
                gr.Markdown(
                    "*Attach a screenshot or paste error logs "
                    "for more accurate support.*"
                )

        # ── Examples ──────────────────────────────────────────
        gr.Markdown("---")
        gr.Markdown("### 💬 Example Questions")
        gr.Examples(
            examples = [
                ["I can't reset my password",              "", None],
                ["I was charged twice for my order",       "", None],
                ["How do I cancel my subscription?",       "", None],
                ["Where is my package?",                   "", None],
                ["The app keeps crashing on my phone",     "", None],
                ["How do I get a refund?",                 "", None],
                ["my email is test@gmail.com need refund", "", None],
            ],
            inputs  = [msg_input, log_input, image_input],
            label   = "Click any example to try it"
        )

        gr.Markdown("---")
        gr.Markdown(
            "*Powered by Llama 3.1 + ChromaDB + Guardrails | "
            "Built as a Multimodal Customer Support Copilot*"
        )

        # ── Interactions ──────────────────────────────────────
        def respond(message, history, log_input, image_input):
            if not message or not message.strip():
                return history, ""

            bot_message = chat(
                message     = message,
                history     = history,
                log_input   = log_input,
                image_input = image_input
            )

            history = history + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": bot_message}
            ]
            return history, ""

        submit_btn.click(
            fn      = respond,
            inputs  = [msg_input, chatbot, log_input, image_input],
            outputs = [chatbot, msg_input]
        )

        msg_input.submit(
            fn      = respond,
            inputs  = [msg_input, chatbot, log_input, image_input],
            outputs = [chatbot, msg_input]
        )

        clear_btn.click(
            fn      = lambda: ([], ""),
            outputs = [chatbot, msg_input]
        )

    return demo

print("UI built!")


import inspect

# This shows you EXACTLY which file/cell each function came from
print("vision_agent source:")
print(inspect.getfile(vision_agent))
print()
print("validate_image source:")
print(inspect.getfile(validate_image))
print()
print("multimodal_pipeline source:")
print(inspect.getfile(multimodal_pipeline))


demo = build_ui()
demo.launch(
    share      = True,
    show_error = True
)


# Simulate a multi-turn conversation
test_history = []

turns = [
    "I can't reset my password",
    "I didn't receive the reset email",
    "how long does it usually take?",
    "what if I still don't get it after 10 minutes?"
]

for message in turns:
    print(f"\nUser: {message}")
    response = chat(
        message     = message,
        history     = test_history,
        log_input   = "",
        image_input = None
    )
    # Strip footer for cleaner display
    clean = response.split("\n\n---\n")[0]
    print(f"Bot:  {clean[:200]}")
    test_history.append([message, response])


###