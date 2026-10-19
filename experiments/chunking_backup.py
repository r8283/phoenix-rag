
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
    """Split text into paragraphs before normalizing whitespace."""
    parts = re.split(r"\n\s*\n+", text)
    return [
        normalized
        for part in parts
        if (normalized := _normalize_whitespace(part))
    ]


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Split text into pieces without losing or duplicating characters."""
    text = _normalize_whitespace(text)

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    pieces = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        if end < len(text):
            cut = text.rfind(" ", start + max_chars // 2, end)

            if cut > start:
                end = cut

        piece = text[start:end].strip()

        if piece:
            pieces.append(piece)

        start = end

        while start < len(text) and text[start].isspace():
            start += 1

    return pieces


def chunk_page(
    url: str,
    title: str,
    text: str,
    fetched_at: str,
) -> list[dict[str, Any]]:
    """Build chunks with a maximum length and controlled overlap."""
    paragraphs = _split_paragraphs(text)

    if not paragraphs:
        return []

    full_text = " ".join(paragraphs)
    max_chars = CHUNK_MAX_CHARS
    overlap = min(
        max(CHUNK_OVERLAP_CHARS, 0),
        max_chars - 1,
    )

    if max_chars <= 0:
        raise ValueError("CHUNK_MAX_CHARS must be positive")

    chunks: list[dict[str, Any]] = []

    def flush_buffer(buf: str) -> None:
        buf = _normalize_whitespace(buf)

        if not buf:
            return

        chunks.append({
            "id": str(uuid.uuid4()),
            "url": url,
            "title": title,
            "text": buf,
            "fetched_at": fetched_at,
        })

    start = 0
    text_length = len(full_text)

    while start < text_length:
        end = min(start + max_chars, text_length)

        # Avoid leaving a very short final chunk when possible.
        if end < text_length and overlap > 0:
            remaining = text_length - (end - overlap)

            if remaining < CHUNK_MIN_CHARS:
                adjusted_end = text_length - CHUNK_MIN_CHARS + overlap

                if start < adjusted_end < end:
                    end = adjusted_end

        # Prefer cutting at a word boundary.
        if end < text_length:
            cut = full_text.rfind(" ", start + max_chars // 2, end)

            if cut > start:
                end = cut

        flush_buffer(full_text[start:end])

        if end >= text_length:
            break

        next_start = max(start + 1, end - overlap)

        # Move to a word boundary so chunks do not start mid-word.
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
