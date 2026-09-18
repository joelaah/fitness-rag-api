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
GENERATION_MODEL = "gemini-3.6-flash"

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


def generate_answer(
    query: str,
    context: str,
    gemini: Optional[genai.Client] = None,
    model: str = GENERATION_MODEL,
    temperature: float = 0.0,
    system_instruction: str = SYSTEM_INSTRUCTION,
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
                            (default: "gemini-3.6-flash").
        temperature:        Sampling temperature (default: 0.0 for
                            deterministic output).
        system_instruction: System prompt that constrains the model behaviour.

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

    log.info(
        "Generating answer with %s (temperature=%.1f)...",
        model,
        temperature,
    )

    response = gemini.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        ),
    )

    answer = response.text or ""
    log.info("Generation complete (%d chars).", len(answer))
    return answer


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
