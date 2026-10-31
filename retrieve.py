"""
RAG Retrieval Module (with Cohere Reranking)
=============================================
Provides a two-stage retrieval function for the RAG pipeline:

Stage 1 — Broad Recall:
    Embeds the user query with gemini-embedding-2 (task_type="RETRIEVAL_QUERY"),
    searches the Qdrant collection for the top 25 candidate chunks.

Stage 2 — Precision Reranking:
    Passes the 25 candidate chunks to the Cohere Rerank API, which uses a
    cross-encoder model to re-score each chunk against the query.
    Only the top 3 highest-scoring chunks are returned to the generator.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Any

import cohere
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Load environment variables
load_dotenv()

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")

COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMS = 768
RETRIEVAL_TOP_K = 25     # Broad recall: pull 25 candidates from Qdrant
RERANK_TOP_N = 3         # Precision: return only the 3 best after reranking
RERANK_MODEL = "rerank-v3.5"  # Cohere's latest rerank model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client Initialisation
# ---------------------------------------------------------------------------


def init_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """Return an authenticated Google GenAI client."""
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")
    return genai.Client(api_key=key)


def init_qdrant_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> QdrantClient:
    """Return an authenticated Qdrant client."""
    target_url = url or QDRANT_URL
    target_key = api_key or QDRANT_API_KEY
    return QdrantClient(
        url=target_url,
        api_key=target_key if target_key else None,
    )


def init_cohere_client(api_key: Optional[str] = None) -> cohere.ClientV2:
    """Return an authenticated Cohere V2 client."""
    key = api_key or COHERE_API_KEY
    if not key:
        raise ValueError(
            "COHERE_API_KEY is not set. Please add it to your .env file. "
            "Get a free key at https://dashboard.cohere.com/api-keys"
        )
    return cohere.ClientV2(api_key=key)


# ---------------------------------------------------------------------------
# Reranking Helper
# ---------------------------------------------------------------------------


def rerank_chunks(
    query: str,
    chunks: list[str],
    co: Optional[cohere.ClientV2] = None,
    model: str = RERANK_MODEL,
    top_n: int = RERANK_TOP_N,
) -> list[str]:
    """
    Re-score *chunks* against *query* using the Cohere Rerank API
    and return only the top_n highest-scoring chunks.

    The Cohere cross-encoder reads the full (query, chunk) pair together,
    producing a much more accurate relevance score than cosine similarity
    alone, which compares independent embeddings.

    Args:
        query: The user's natural language question.
        chunks: Candidate text chunks retrieved from Qdrant.
        co: Authenticated Cohere V2 client (lazily initialized if omitted).
        model: Cohere rerank model identifier (default: rerank-v3.5).
        top_n: Number of top results to return after reranking.

    Returns:
        A list of the top_n chunk strings, ordered by relevance score
        (highest first).
    """
    if not chunks:
        return []

    if co is None:
        co = init_cohere_client()

    log.info(
        "Reranking %d chunks with Cohere %s (top_n=%d)...",
        len(chunks), model, top_n,
    )

    response = co.rerank(
        model=model,
        query=query,
        documents=chunks,
        top_n=top_n,
    )

    # Each result has .index (into the original chunks list) and .relevance_score
    reranked: list[str] = []
    for result in response.results:
        original_text = chunks[result.index]
        reranked.append(original_text)
        log.info(
            "  Rerank #%d  score=%.4f  (original index %d, %d chars)",
            len(reranked), result.relevance_score, result.index, len(original_text),
        )

    return reranked


# ---------------------------------------------------------------------------
# Main Retrieval Function
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    gemini: Optional[genai.Client] = None,
    qdrant: Optional[QdrantClient] = None,
    co: Optional[cohere.ClientV2] = None,
    collection_name: str = COLLECTION_NAME,
    top_k: int = RETRIEVAL_TOP_K,
    rerank_top_n: int = RERANK_TOP_N,
    category: Optional[str] = None,
    separator: str = "\n\n",
    return_metadata: bool = False,
) -> Any:
    """
    Two-stage retrieval: broad vector recall → precision Cohere reranking.

    Stage 1 — Vector Search (Broad Recall):
        1. Embeds the user query with gemini-embedding-2
           (task_type="RETRIEVAL_QUERY", output_dimensionality=768).
        2. Optionally pre-filters the collection by category using Qdrant's
           Filter and FieldCondition.
        3. Retrieves candidate chunks and their payloads from Qdrant via cosine
           similarity.

    Stage 2 — Reranking (Precision):
        4. Passes candidate chunks to Cohere Rerank API (rerank-v3.5).
        5. Preserves payload metadata (document_name, category) associated with
           each chunk.
        6. Formats each chunk with its document citation header if available.

    Args:
        query: User's natural language question or search query.
        gemini: Authenticated GenAI client (lazily initialized if omitted).
        qdrant: Qdrant client instance (lazily initialized if omitted).
        co: Cohere V2 client instance (lazily initialized if omitted).
        collection_name: Target collection name (default: "documents").
        top_k: Number of nearest matches for Stage 1 recall (default: 25).
        rerank_top_n: Number of chunks to keep after reranking (default: 3).
        category: Optional category string to pre-filter results.
        separator: String delimiter used to join extracted payloads.
        return_metadata: If True, returns tuple (joined_text, list_of_payload_dicts).

    Returns:
        Concatenated reranked string, or (joined_text, payloads) if return_metadata=True.
    """
    if not query or not query.strip():
        return ("", []) if return_metadata else ""

    if gemini is None:
        gemini = init_gemini_client()
    if qdrant is None:
        qdrant = init_qdrant_client()

    log.info("Embedding query with %s (task_type=RETRIEVAL_QUERY)...", EMBEDDING_MODEL)

    # ── Stage 1: Vector search (broad recall) ──────────────────────────

    # 1. Embed query
    embed_response = gemini.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIMS,
        ),
    )
    query_vector = embed_response.embeddings[0].values

    # 2. Build optional category filter
    query_filter = None
    if category:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="category",
                    match=MatchValue(value=category),
                )
            ]
        )
        log.info(
            "Applying category filter: '%s' on collection '%s' (top_k=%d)...",
            category, collection_name, top_k,
        )
    else:
        log.info(
            "Searching Qdrant collection '%s' (top_k=%d, no filter)...",
            collection_name, top_k,
        )

    # 3. Query Qdrant for top_k (25) candidate vectors
    search_response = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    # Extract text and preserve payload metadata from results
    candidate_payloads: list[dict] = []
    candidate_chunks: list[str] = []
    for point in search_response.points:
        if point.payload and "text" in point.payload:
            candidate_payloads.append(dict(point.payload))
            candidate_chunks.append(str(point.payload["text"]))

    log.info(
        "Stage 1 complete: retrieved %d candidate chunks from Qdrant.",
        len(candidate_chunks),
    )

    if not candidate_chunks:
        return ("", []) if return_metadata else ""

    # ── Stage 2: Cohere reranking (precision) with payload preservation ─────────

    reranked_chunks: list[str] = []
    reranked_payloads: list[dict] = []

    try:
        if co is None:
            co = init_cohere_client()

        log.info(
            "Reranking %d chunks with Cohere %s (top_n=%d)...",
            len(candidate_chunks), RERANK_MODEL, rerank_top_n,
        )

        response = co.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=candidate_chunks,
            top_n=rerank_top_n,
        )

        for result in response.results:
            payload = candidate_payloads[result.index]
            reranked_payloads.append(payload)

            doc_name = payload.get("document_name") or payload.get("title") or ""
            text = str(payload.get("text", "")).strip()

            if doc_name:
                clean_title = doc_name.replace(".txt", "").replace("_", " ").title()
                chunk_str = f"## {clean_title}\n{text}" if not text.startswith("## ") else text
            else:
                chunk_str = text

            reranked_chunks.append(chunk_str)
            log.info(
                "  Rerank #%d  score=%.4f  (original index %d, %d chars)",
                len(reranked_chunks), result.relevance_score, result.index, len(chunk_str),
            )
    except Exception as exc:
        log.warning("Cohere rerank fallback to top vectors: %s", exc)
        reranked_chunks = []
        reranked_payloads = candidate_payloads[:rerank_top_n]
        for payload in reranked_payloads:
            doc_name = payload.get("document_name") or payload.get("title") or ""
            text = str(payload.get("text", "")).strip()
            if doc_name:
                clean_title = doc_name.replace(".txt", "").replace("_", " ").title()
                chunk_str = f"## {clean_title}\n{text}" if not text.startswith("## ") else text
            else:
                chunk_str = text
            reranked_chunks.append(chunk_str)

    log.info(
        "Stage 2 complete: %d chunks after reranking (from %d candidates).",
        len(reranked_chunks), len(candidate_chunks),
    )

    joined_text = separator.join(reranked_chunks)
    if return_metadata:
        return joined_text, reranked_payloads
    return joined_text


if __name__ == "__main__":
    import sys

    user_query = sys.argv[1] if len(sys.argv) > 1 else "What is Qdrant and what is it written in?"
    print(f"\nQuery: {user_query}\n")
    try:
        context = retrieve(user_query)
        print("--- Retrieved Context (Reranked Top 3) ---")
        print(context if context else "[No matching documents found]")
    except Exception as exc:
        print(f"Retrieval demo ended with: {exc}")
