
"""Split page text into overlapping chunks with metadata."""
from __future__ import annotations

import re
import uuid
from typing import Any

from config import (
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP_CHARS,
)


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace while keeping the text readable."""
    return re.sub(r"\s+", " ", text).strip()


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs and normalize whitespace."""
    parts = re.split(r"\n\s*\n+", text)
    return [
        normalized
        for part in parts
        if (normalized := _normalize_whitespace(part))
    ]


def chunk_page(
    url: str,
    title: str,
    text: str,
    fetched_at: str,
) -> list[dict[str, Any]]:
    """Build chunks with a maximum length and controlled overlap."""
    full_text = " ".join(_split_paragraphs(text))

    if not full_text:
        return []

    max_chars = CHUNK_MAX_CHARS
    min_chars = max(1, min(CHUNK_MIN_CHARS, max_chars))
    overlap = max(0, min(CHUNK_OVERLAP_CHARS, max_chars - 1))

    if max_chars <= 0:
        raise ValueError("CHUNK_MAX_CHARS must be positive")

    chunks: list[dict[str, Any]] = []
    start = 0
    text_length = len(full_text)

    while start < text_length:
        end = min(start + max_chars, text_length)

        # If the final remainder would be too short, try to rebalance.
        if end < text_length and overlap > 0:
            remaining_after_overlap = text_length - (end - overlap)
            if remaining_after_overlap < min_chars:
                candidate = text_length - min_chars + overlap
                if start < candidate < end:
                    end = candidate

        # Prefer a word boundary, without making the chunk too short.
        if end < text_length:
            lower_bound = min(start + min_chars, end)
            cut = full_text.rfind(" ", lower_bound, end)
            if cut > start:
                end = cut

        chunk_text = full_text[start:end].strip()
        if chunk_text:
            chunks.append({
                "id": str(uuid.uuid4()),
                "url": url,
                "title": title,
                "text": chunk_text,
                "fetched_at": fetched_at,
            })

        if end >= text_length:
            break

        # Start the next chunk near the requested overlap.
        next_start = max(start + 1, end - overlap)

        # Move forward to a word boundary if needed.
        while next_start < end and full_text[next_start] != " ":
            next_start += 1

        while next_start < text_length and full_text[next_start] == " ":
            next_start += 1

        start = next_start

    return chunks


def pages_to_chunks(
    pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert crawled pages into chunks."""
    all_chunks: list[dict[str, Any]] = []

    for page in pages:
        all_chunks.extend(
            chunk_page(
                url=page["url"],
                title=page.get("title") or "",
                text=page.get("text") or "",
                fetched_at=page.get("fetched_at") or "",
            )
        )

    return all_chunks
