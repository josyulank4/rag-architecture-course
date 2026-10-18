"""Lesson: Chunking Strategies.

Implements three chunkers over the same source document and compares them:
fixed-size windows, recursive paragraph/sentence packing, and structure-aware
chunking that never crosses a Markdown heading boundary.

Run from the repo root:
    python -m chunking.chunkers
    python chunking/chunkers.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingestion.ingest import _sections, load_documents  # noqa: E402

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def fixed_chunk(text: str, size: int = 220, overlap: int = 40) -> list[str]:
    """Character windows with overlap. Simple, fast, blind to structure."""
    if size <= overlap:
        raise ValueError("size must be greater than overlap")
    chunks = []
    step = size - overlap
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
    return chunks


def _split_long_piece(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_RE.split(text):
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
        while len(current) > max_chars:  # single very long sentence
            pieces.append(current[:max_chars].strip())
            current = current[max_chars:].strip()
    if current:
        pieces.append(current)
    return pieces


def recursive_chunk(text: str, max_chars: int = 320) -> list[str]:
    """Pack paragraphs, then sentences, under ``max_chars`` (LangChain-style)."""
    chunks: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split()).strip()
        if not paragraph:
            continue
        candidate = f"{current} {paragraph}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
        if len(current) > max_chars:
            pieces = _split_long_piece(current, max_chars)
            chunks.extend(pieces[:-1])
            current = pieces[-1]
    if current:
        chunks.append(current)
    return chunks


def structure_aware_chunk(text: str, max_chars: int = 420) -> list[str]:
    """One chunk stream per heading section; every chunk keeps its heading."""
    chunks: list[str] = []
    for heading, paragraphs in _sections(text):
        buffer = ""
        for paragraph in paragraphs:
            candidate = f"{buffer} {paragraph}".strip()
            if buffer and len(candidate) > max_chars:
                chunks.append(f"{heading}\n{buffer}")
                buffer = paragraph
            else:
                buffer = candidate
            if len(buffer) > max_chars:
                pieces = _split_long_piece(buffer, max_chars)
                chunks.extend(f"{heading}\n{piece}" for piece in pieces[:-1])
                buffer = pieces[-1]
        if buffer:
            chunks.append(f"{heading}\n{buffer}")
    return chunks


def _stats(chunks: list[str]) -> dict:
    sizes = [len(c) for c in chunks]
    return {
        "count": len(chunks),
        "avg_chars": round(sum(sizes) / len(sizes), 1) if sizes else 0.0,
        "min_chars": min(sizes) if sizes else 0,
        "max_chars": max(sizes) if sizes else 0,
    }


def comparison_run(text: str) -> dict:
    """Run all three chunkers over ``text`` and return comparable stats."""
    fixed = fixed_chunk(text)
    recursive = recursive_chunk(text)
    structured = structure_aware_chunk(text)
    return {
        "fixed": _stats(fixed),
        "recursive": _stats(recursive),
        "structure_aware": _stats(structured),
        "structure_aware_chunks": structured,
    }


def main() -> None:
    doc = load_documents()[0]
    print(f"document: {doc['doc_id']} ({doc['title']})")
    result = comparison_run(doc["text"])
    for name in ("fixed", "recursive", "structure_aware"):
        stats = result[name]
        print(
            f"{name}: count={stats['count']} avg_chars={stats['avg_chars']} "
            f"min={stats['min_chars']} max={stats['max_chars']}"
        )
    structured = result["structure_aware_chunks"]
    headed = all("\n" in chunk for chunk in structured)
    print(f"all structure-aware chunks carry a heading: {headed}")


if __name__ == "__main__":
    main()
