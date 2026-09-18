"""
RAG Ingestion Pipeline (Async)
==============================
Connects to Qdrant Cloud, creates a vector collection configured for
Cosine similarity (768 dims), embeds text documents with the Gemini
gemini-embedding-2 model using fully asynchronous concurrency, and
batch-upserts the resulting vectors + metadata payloads into the collection.

Async architecture:
    1. Chunks are divided into embedding batches.
    2. Each batch is embedded concurrently via client.aio.models.embed_content.
    3. Resulting points are batch-upserted via AsyncQdrantClient.upsert
       to minimise network round-trips.

Required environment variables (see .env.example):
    GEMINI_API_KEY   – Google AI Studio API key
    QDRANT_URL       – Qdrant Cloud cluster URL  (e.g. https://xyz.cloud.qdrant.io:6333)
    QDRANT_API_KEY   – Qdrant Cloud API key
"""

from __future__ import annotations

import asyncio
import os
import uuid
import logging
from typing import Optional, Sequence

from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from retrieve import retrieve

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
QDRANT_URL: str = os.environ["QDRANT_URL"]
QDRANT_API_KEY: str = os.environ["QDRANT_API_KEY"]

COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMS = 768
EMBED_BATCH_SIZE = 20    # Number of texts per concurrent embedding request
UPSERT_BATCH_SIZE = 100  # Max points per Qdrant upsert call
MAX_CONCURRENCY = 10     # Max simultaneous embedding API calls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1.  Initialise clients (sync — used by Streamlit / app.py)
# ---------------------------------------------------------------------------


def init_gemini_client() -> genai.Client:
    """Return an authenticated Google GenAI client."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    log.info("Gemini client initialised.")
    return client


def init_qdrant_client() -> QdrantClient:
    """Return a synchronous Qdrant Cloud client (HTTPS + API-key auth)."""
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    log.info("Qdrant client initialised  →  %s", QDRANT_URL)
    return client


def init_async_qdrant_client() -> AsyncQdrantClient:
    """Return an async Qdrant Cloud client for use inside coroutines."""
    client = AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    log.info("Async Qdrant client initialised  →  %s", QDRANT_URL)
    return client


# ---------------------------------------------------------------------------
# 2.  Create / ensure collection exists
# ---------------------------------------------------------------------------


from qdrant_client.models import PayloadSchemaType

def ensure_collection(qdrant: QdrantClient) -> None:
    """
    Create the target collection if it doesn't already exist.
    Configured for Cosine similarity with EMBEDDING_DIMS dimensions.
    Also creates a keyword payload index on 'category' so that
    metadata filtering works without manual index creation.
    """
    if qdrant.collection_exists(COLLECTION_NAME):
        log.info("Collection '%s' already exists — skipping creation.", COLLECTION_NAME)
        return

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMS,
            distance=Distance.COSINE,
        ),
    )
    log.info(
        "Created collection '%s'  (dims=%d, distance=Cosine).",
        COLLECTION_NAME,
        EMBEDDING_DIMS,
    )

    # Create payload index on 'category' for filtered retrieval
    qdrant.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="category",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    log.info("Created keyword payload index on 'category'.")


# ---------------------------------------------------------------------------
# 3.  Synchronous embedding helper (kept for backward compatibility)
# ---------------------------------------------------------------------------


def embed_texts(
    gemini: genai.Client,
    texts: Sequence[str],
) -> list[list[float]]:
    """
    Embed each text string using gemini-embedding-2 (synchronous).

    Uses task_type = RETRIEVAL_DOCUMENT so that the vectors are optimised
    for document-side storage (pair with RETRIEVAL_QUERY at query time).
    output_dimensionality is set to 768 to match the collection.
    """
    vectors: list[list[float]] = []
    for text in texts:
        result = gemini.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBEDDING_DIMS,
            ),
        )
        vectors.append(result.embeddings[0].values)
    return vectors


# ---------------------------------------------------------------------------
# 4.  Async embedding helpers
# ---------------------------------------------------------------------------


async def _embed_single_async(
    gemini: genai.Client,
    text: str,
    semaphore: asyncio.Semaphore,
) -> list[float]:
    """Embed a single text chunk asynchronously with concurrency control."""
    async with semaphore:
        result = await gemini.aio.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBEDDING_DIMS,
            ),
        )
        return result.embeddings[0].values


async def embed_texts_async(
    gemini: genai.Client,
    texts: Sequence[str],
    max_concurrency: int = MAX_CONCURRENCY,
) -> list[list[float]]:
    """
    Embed all text chunks concurrently using client.aio.models.embed_content.

    A semaphore limits the number of simultaneous API calls to avoid
    rate-limit errors (default: 10 concurrent requests).
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [_embed_single_async(gemini, text, semaphore) for text in texts]
    vectors = await asyncio.gather(*tasks)
    return list(vectors)


# ---------------------------------------------------------------------------
# 5.  Async upsert pipeline
# ---------------------------------------------------------------------------


async def ingest_texts_async(
    gemini: genai.Client,
    texts: Sequence[str],
    document_name: str = "unknown",
    category: str = "general",
    qdrant_async: Optional[AsyncQdrantClient] = None,
    max_concurrency: int = MAX_CONCURRENCY,
    upsert_batch_size: int = UPSERT_BATCH_SIZE,
) -> int:
    """
    Fully asynchronous ingestion pipeline.

    1. Embeds all text chunks concurrently via gemini.aio.models.embed_content.
    2. Builds PointStruct objects with text, document_name, and category
       metadata in the payload.
    3. Batch-upserts points into Qdrant using AsyncQdrantClient to
       minimise network round-trips.

    Args:
        gemini: Authenticated GenAI client (provides both sync and async).
        texts: Sequence of text chunks to embed and upsert.
        document_name: Name of the source document (e.g. filename).
        category: Category tag for metadata filtering
                  (e.g. "HR", "Technical", "Legal").
        qdrant_async: Optional pre-initialised AsyncQdrantClient.
                      Created automatically if not provided.
        max_concurrency: Maximum simultaneous embedding API calls.
        upsert_batch_size: Maximum points per Qdrant upsert call.

    Returns the total number of points upserted.
    """
    # Initialise async Qdrant client if not provided
    close_client = False
    if qdrant_async is None:
        qdrant_async = init_async_qdrant_client()
        close_client = True

    try:
        # Step 1: Embed all chunks concurrently
        log.info(
            "Async embedding %d chunks (concurrency=%d)...",
            len(texts), max_concurrency,
        )
        vectors = await embed_texts_async(gemini, texts, max_concurrency)
        log.info("Embedding complete — %d vectors generated.", len(vectors))

        # Step 2: Build point structs with metadata payload
        all_points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": text,
                    "document_name": document_name,
                    "category": category,
                },
            )
            for text, vector in zip(texts, vectors)
        ]

        # Step 3: Batch upsert to minimise network requests
        total_upserted = 0
        for batch_start in range(0, len(all_points), upsert_batch_size):
            batch = all_points[batch_start : batch_start + upsert_batch_size]
            await qdrant_async.upsert(
                collection_name=COLLECTION_NAME,
                points=batch,
            )
            total_upserted += len(batch)
            log.info(
                "Upserted batch %d–%d  (%d points, doc=%s, category=%s)",
                batch_start,
                batch_start + len(batch) - 1,
                len(batch),
                document_name,
                category,
            )

        return total_upserted

    finally:
        if close_client:
            await qdrant_async.close()


# ---------------------------------------------------------------------------
# 6.  Synchronous wrapper (used by Streamlit app.py)
# ---------------------------------------------------------------------------


def ingest_texts(
    gemini: genai.Client,
    qdrant: QdrantClient,
    texts: Sequence[str],
    document_name: str = "unknown",
    category: str = "general",
) -> int:
    """
    Synchronous wrapper around the async ingestion pipeline.

    Streamlit runs its own event loop, so this function creates a new
    loop in a background thread to run the async pipeline without
    conflicting with Streamlit's loop.

    Args:
        gemini: Authenticated GenAI client.
        qdrant: Synchronous Qdrant client (used only to satisfy the
                existing call signature; the async pipeline creates
                its own AsyncQdrantClient internally).
        texts: Sequence of text chunks to embed and upsert.
        document_name: Name of the source document (e.g. filename).
        category: Category tag for metadata filtering.

    Returns the total number of points upserted.
    """
    import concurrent.futures

    def _run_in_new_loop() -> int:
        # Create a fresh genai.Client inside this thread so the internal
        # async httpx client is scoped to the new event loop and won't
        # conflict with any previously closed loop.
        fresh_gemini = genai.Client(api_key=GEMINI_API_KEY)
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                ingest_texts_async(
                    gemini=fresh_gemini,
                    texts=texts,
                    document_name=document_name,
                    category=category,
                )
            )
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_in_new_loop)
        return future.result()


# ---------------------------------------------------------------------------
# 7.  Main – demo with sample documents
# ---------------------------------------------------------------------------


SAMPLE_DOCUMENTS: list[str] = [
    "Qdrant is a high-performance vector search engine written in Rust.",
    "Retrieval-Augmented Generation combines a retriever with a language model.",
    "Cosine similarity measures the angle between two vectors in high-dimensional space.",
    "The Gemini API provides state-of-the-art embedding models for text and multimodal content.",
    "Vector databases store embeddings and support fast approximate nearest-neighbour search.",
]


async def async_main() -> None:
    """Async demo entry-point."""
    gemini = init_gemini_client()

    # Use sync client for collection setup (one-time operation)
    qdrant_sync = init_qdrant_client()
    ensure_collection(qdrant_sync)

    # Use async pipeline for ingestion
    count = await ingest_texts_async(
        gemini, SAMPLE_DOCUMENTS,
        document_name="sample_docs", category="Technical",
    )
    log.info("Ingestion complete — %d documents upserted into '%s'.", count, COLLECTION_NAME)

    # Quick sanity check
    info = qdrant_sync.get_collection(COLLECTION_NAME)
    log.info("Collection info: vectors_count=%s", info.points_count)


def main() -> None:
    """Sync entry-point that runs the async demo."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
