"""Lesson: Embeddings and Vector Representations.

A deterministic, offline stand-in for a hosted embedding model: a hashing
vectorizer (tokens are hashed into a fixed-width vector with hashlib, never
Python's salted ``hash()``), TF weighting, corpus IDF, and cosine-similarity
search with numpy. No API keys, no network, same output on every run.

Run from the repo root:
    python -m embeddings.vectorizer
    python embeddings/vectorizer.py
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingestion.ingest import emit_chunks, load_documents  # noqa: E402

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "the", "is", "are", "to", "of", "for", "and", "or", "in",
    "on", "at", "by", "with", "from", "how", "what", "when", "do", "does",
    "i", "you", "we", "it", "this", "that", "be", "can", "should",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]


class HashingVectorizer:
    """Deterministic token-hashing vectorizer with TF-IDF weighting."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.idf_ = np.ones(dim, dtype=np.float64)

    def _bucket(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self.dim

    def _tf_vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float64)
        counts: dict[int, int] = {}
        for token in tokenize(text):
            bucket = self._bucket(token)
            counts[bucket] = counts.get(bucket, 0) + 1
        for bucket, count in counts.items():
            vector[bucket] = 1.0 + math.log(count)
        return vector

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        matrix = np.vstack([self._tf_vector(t) for t in texts])
        doc_freq = (matrix > 0).sum(axis=0)
        n_docs = len(texts)
        self.idf_ = np.log((1.0 + n_docs) / (1.0 + doc_freq)) + 1.0
        matrix = matrix * self.idf_
        return _normalize(matrix)

    def transform(self, text: str) -> np.ndarray:
        vector = self._tf_vector(text) * self.idf_
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_search(
    query: str,
    vectorizer: HashingVectorizer,
    matrix: np.ndarray,
    chunks: list[dict],
    k: int = 3,
) -> list[tuple[dict, float]]:
    """Return the top-k (chunk, cosine_score) pairs, deterministically ordered."""
    scores = matrix @ vectorizer.transform(query)
    order = np.argsort(-scores, kind="stable")[:k]
    return [(chunks[int(i)], float(scores[int(i)])) for i in order]


def main() -> None:
    chunks = emit_chunks(load_documents())
    vectorizer = HashingVectorizer()
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    print(f"embedded {len(chunks)} chunks into {vectorizer.dim}-dim vectors")
    queries = [
        "how do I roll back a failed production deployment",
        "what happens when the primary database fails",
        "how long is restricted data retained",
    ]
    for query in queries:
        print(f"\nquery: {query}")
        for chunk, score in cosine_search(query, vectorizer, matrix, chunks, k=3):
            print(f"  {score:.3f}  {chunk['doc_id']}  [{chunk['heading']}]")


if __name__ == "__main__":
    main()
