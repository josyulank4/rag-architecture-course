"""Lesson: Document Ingestion and Preprocessing.

Loads the fictional sample corpus from ``data/``, normalizes the text, and
emits retrieval-ready chunks. Every chunk carries metadata (source document,
title, heading path, access label) because production RAG systems must be able
to filter and audit what they retrieve.

Run from the repo root:
    python -m ingestion.ingest
    python ingestion/ingest.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"

# Access labels are assigned at ingest time, never guessed at query time.
ACCESS_LABELS = {
    "deploy_policy": "internal",
    "incident_runbook": "internal",
    "security_policy": "restricted",
    "support_faq": "public",
}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def clean_text(text: str) -> str:
    """Normalize whitespace without changing meaning."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    out: list[str] = []
    blank = False
    for line in lines:
        if line:
            out.append(line)
            blank = False
        elif not blank and out:
            out.append("")
            blank = True
    return "\n".join(out).strip()


def _title_of(text: str, fallback: str) -> str:
    for line in text.split("\n"):
        match = _HEADING_RE.match(line)
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()
    return fallback


def load_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    """Load every Markdown document in ``data_dir`` with document metadata."""
    documents: list[dict] = []
    for path in sorted(data_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        doc_id = path.stem
        documents.append(
            {
                "doc_id": doc_id,
                "source": path.name,
                "title": _title_of(raw, doc_id),
                "access_label": ACCESS_LABELS.get(doc_id, "internal"),
                "text": clean_text(raw),
            }
        )
    return documents


def _sections(text: str) -> list[tuple[str, list[str]]]:
    """Split Markdown into (heading_path, paragraphs) sections."""
    sections: list[tuple[str, list[str]]] = []
    stack: list[str] = []
    heading_path = ""
    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush_paragraph() -> None:
        if buffer:
            paragraphs.append(" ".join(buffer).strip())
            buffer.clear()

    def flush_section() -> None:
        flush_paragraph()
        if paragraphs:
            sections.append((heading_path, list(paragraphs)))
            paragraphs.clear()

    for line in text.split("\n"):
        match = _HEADING_RE.match(line)
        if match:
            flush_section()
            level = len(match.group(1))
            stack = stack[: level - 1] + [match.group(2).strip()]
            heading_path = " > ".join(stack)
        elif line.strip():
            buffer.append(line.strip())
        else:
            flush_paragraph()
    flush_section()
    return sections


def emit_chunks(documents: list[dict], max_chars: int = 700) -> list[dict]:
    """Emit chunks with metadata. Chunks never cross a heading boundary."""
    chunks: list[dict] = []
    for doc in documents:
        index = 0
        for heading, paragraphs in _sections(doc["text"]):
            buffer = ""
            for paragraph in paragraphs:
                candidate = f"{buffer} {paragraph}".strip()
                if buffer and len(candidate) > max_chars:
                    chunks.append(_make_chunk(doc, heading, buffer, index))
                    index += 1
                    buffer = paragraph
                else:
                    buffer = candidate
            if buffer:
                chunks.append(_make_chunk(doc, heading, buffer, index))
                index += 1
    return chunks


def _make_chunk(doc: dict, heading: str, body: str, index: int) -> dict:
    text = f"{heading}\n{body}" if heading else body
    return {
        "chunk_id": f"{doc['doc_id']}-{index:03d}",
        "doc_id": doc["doc_id"],
        "source": doc["source"],
        "title": doc["title"],
        "heading": heading,
        "access_label": doc["access_label"],
        "text": text,
        "char_count": len(text),
    }


def main() -> None:
    documents = load_documents()
    chunks = emit_chunks(documents)
    print(f"documents loaded: {len(documents)}")
    for doc in documents:
        print(f"  {doc['doc_id']}: {doc['title']} [{doc['access_label']}]")
    print(f"chunks emitted: {len(chunks)}")
    print("sample chunk:")
    print(json.dumps(chunks[0], indent=2))


if __name__ == "__main__":
    main()
