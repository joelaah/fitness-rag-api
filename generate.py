"""
RAG Generation Module
=====================
Generates a final answer to the user's query using the google-genai SDK.
The model is configured with temperature=0.0 and a strict system instruction
that constrains it to answer *only* from the provided context.  If the answer
is not present in the context, the model must respond with "I do not know".
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GENERATION_MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = (
    "You are a strict question-answering assistant. "
    "You must answer the user's question using ONLY the provided context. "
    "Do not use any prior knowledge or information outside of the context. "
    "If the answer cannot be found in the provided context, you must respond "
    "with exactly: 'I do not know'. "
    "Do not speculate, infer, or hallucinate any information beyond what is "
    "explicitly stated in the context."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client helper
# ---------------------------------------------------------------------------


def init_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """Return an authenticated Google GenAI client."""
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")
    return genai.Client(api_key=key)


# ---------------------------------------------------------------------------
# Generation function
# ---------------------------------------------------------------------------


FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-flash-latest",
]

def generate_answer(
    query: str,
    context: str,
    gemini: Optional[genai.Client] = None,
    model: str = GENERATION_MODEL,
    temperature: float = 0.0,
    system_instruction: str = SYSTEM_INSTRUCTION,
    history: Optional[list] = None,
) -> str:
    """
    Generate a grounded answer to a user query using retrieved context.

    The model receives a strict system instruction that forbids it from using
    any knowledge outside the provided context.  If the context does not
    contain the answer, the model must reply with "I do not know".

    Args:
        query:              The user's original natural-language question.
        context:            Concatenated text retrieved from the vector store.
        gemini:             Authenticated GenAI client (lazily initialised if
                            omitted).
        model:              Gemini model to use for generation
                            (default: "gemini-2.5-flash").
        temperature:        Sampling temperature (default: 0.0 for
                            deterministic output).
        system_instruction: System prompt that constrains the model behaviour.
        history:            Optional list of prior chat turns (ChatMessage or dict).

    Returns:
        The model's generated answer as a plain-text string.
    """
    if gemini is None:
        gemini = init_gemini_client()

    # Build the user prompt with the context and query clearly separated
    user_prompt = (
        f"Context:\n"
        f"---\n"
        f"{context}\n"
        f"---\n\n"
        f"Question: {query}"
    )

    contents: list[types.Content] | str = user_prompt
    if history:
        raw_turns: list[tuple[str, str]] = []
        for msg in history:
            role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "user")
            model_role = "model" if role in ("assistant", "model") else "user"
            content_text = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else str(msg))
            if content_text and str(content_text).strip():
                raw_turns.append((model_role, str(content_text).strip()))

        # Google GenAI requires:
        # 1. The first content turn MUST have role 'user' (drop leading assistant/welcome messages).
        while raw_turns and raw_turns[0][0] == "model":
            raw_turns.pop(0)

        # 2. Alternating roles (merge consecutive turns of same role)
        merged_turns: list[tuple[str, str]] = []
        for r, t in raw_turns:
            if merged_turns and merged_turns[-1][0] == r:
                merged_turns[-1] = (r, merged_turns[-1][1] + "\n\n" + t)
            else:
                merged_turns.append((r, t))

        # 3. If prior history ended with 'user', remove it so user_prompt is the active turn
        if merged_turns and merged_turns[-1][0] == "user":
            merged_turns.pop()

        if merged_turns:
            msg_contents: list[types.Content] = []
            for r, t in merged_turns:
                msg_contents.append(
                    types.Content(
                        role=r,
                        parts=[types.Part.from_text(text=t)],
                    )
                )
            msg_contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_prompt)],
                )
            )
            contents = msg_contents

    models_to_try = [model]
    if model == GENERATION_MODEL:
        for fb in FALLBACK_MODELS:
            if fb not in models_to_try:
                models_to_try.append(fb)

    last_exc = None
    for target_model in models_to_try:
        try:
            log.info(
                "Generating answer with %s (temperature=%.1f)...",
                target_model,
                temperature,
            )
            response = gemini.models.generate_content(
                model=target_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                ),
            )
            answer = response.text or ""
            log.info("Generation complete (%d chars).", len(answer))
            return answer
        except Exception as exc:
            last_exc = exc
            log.warning("Generation attempt failed with %s: %s", target_model, exc)

    if last_exc:
        raise last_exc
    return ""


# ---------------------------------------------------------------------------
# Demo / CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    user_query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is Qdrant and what language is it written in?"
    )

    # Import the retrieval function to build a full RAG demo
    from retrieve import retrieve

    print(f"\nQuery: {user_query}\n")

    try:
        context = retrieve(user_query)
        print("--- Retrieved Context ---")
        print(context if context else "[No matching documents found]")
        print()

        answer = generate_answer(user_query, context)
        print("--- Generated Answer ---")
        print(answer)
    except Exception as exc:
        print(f"RAG demo ended with: {exc}")
