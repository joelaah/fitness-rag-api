"""
Ingest Curated Fitness Knowledge Documents into Qdrant
======================================================
Reads all .txt documents from fitness_knowledge/, chunks them into
paragraphs / semantic chunks, and ingests them into Qdrant with
proper per-chunk embeddings, rate-limit pacing, and retry.
"""

from __future__ import annotations

import glob
import logging
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from ingest import (
    COLLECTION_NAME,
    EMBEDDING_DIMS,
    EMBEDDING_MODEL,
    ensure_collection,
    init_gemini_client,
    init_qdrant_client,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("ingest_fitness_docs")

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "fitness_knowledge"


def chunk_document(text: str, chunk_size: int = 600, chunk_overlap: int = 60) -> list[str]:
    """Chunk document text into paragraph-aware chunks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []

    current_chunk: list[str] = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para)
        if para_len > chunk_size:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0
            start = 0
            while start < para_len:
                end = min(start + chunk_size, para_len)
                chunks.append(para[start:end].strip())
                if end == para_len:
                    break
                start += chunk_size - chunk_overlap
        elif current_length + para_len + 2 > chunk_size:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_length = para_len
        else:
            current_chunk.append(para)
            current_length += para_len + 2

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return [c for c in chunks if c.strip()]


def embed_text_with_retry(
    gemini: genai.Client,
    text: str,
    max_retries: int = 5,
) -> list[float]:
    """Embed a single text chunk with automatic rate-limit backoff."""
    for attempt in range(1, max_retries + 1):
        try:
            response = gemini.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBEDDING_DIMS,
                ),
            )
            return response.embeddings[0].values
        except ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 15 * attempt
                log.warning(
                    "Rate limit (429) hit on attempt %d/%d. Sleeping %ds before retry...",
                    attempt,
                    max_retries,
                    wait_time,
                )
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            log.error("Unexpected error during embedding: %s", e)
            if attempt == max_retries:
                raise
            time.sleep(3)

    raise RuntimeError(f"Failed to embed chunk after {max_retries} attempts.")


def ingest_all_fitness_docs() -> None:
    """Read all docs in fitness_knowledge/ and ingest into Qdrant."""
    gemini = init_gemini_client()
    qdrant = init_qdrant_client()
    ensure_collection(qdrant)

    files = sorted(glob.glob(str(KNOWLEDGE_DIR / "*.txt")))
    if not files:
        log.error("No .txt files found in %s", KNOWLEDGE_DIR)
        return

    log.info("Found %d fitness documents to ingest in %s", len(files), KNOWLEDGE_DIR)
    total_points = 0

    for file_path in files:
        doc_path = Path(file_path)
        doc_name = doc_path.name
        content = doc_path.read_text(encoding="utf-8")
        chunks = chunk_document(content)

        if not chunks:
            log.warning("File %s had no extractable chunks. Skipping.", doc_name)
            continue

        log.info("Embedding %s (%d chunks)...", doc_name, len(chunks))
        doc_points: list[PointStruct] = []

        for idx, text_chunk in enumerate(chunks, 1):
            vec = embed_text_with_retry(gemini, text_chunk)
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "text": text_chunk,
                    "document_name": doc_name,
                    "category": "fitness",
                },
            )
            doc_points.append(point)
            # Sleep 0.65s to stay reliably below 100 requests per minute limit
            time.sleep(0.65)

        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=doc_points,
        )
        total_points += len(doc_points)
        log.info("Upserted %d/%d chunks for %s", len(doc_points), len(chunks), doc_name)

    log.info(
        "Ingestion completed! Total %d chunks from %d files added to '%s'",
        total_points,
        len(files),
        COLLECTION_NAME,
    )

    info = qdrant.get_collection(COLLECTION_NAME)
    log.info("Current total collection vectors count: %s", info.points_count)


if __name__ == "__main__":
    ingest_all_fitness_docs()
