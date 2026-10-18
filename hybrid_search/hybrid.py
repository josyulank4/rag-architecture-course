"""Lesson: Hybrid Search — Dense + Sparse + Reranking (retrieval stage).

Combines BM25 keyword retrieval with dense vector retrieval and fuses the two
rankings with Reciprocal Rank Fusion (RRF). Hybrid retrieval is the production
default for enterprise corpora because users search with exact names, codes,
and clauses (BM25 strength) as well as paraphrases (dense strength).

Run from the repo root:
    python -m hybrid_search.hybrid
    python hybrid_search/hybrid.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embeddings.vectorizer import HashingVectorizer, tokenize  # noqa: E402
from ingestion.ingest import emit_chunks, load_documents  # noqa: E402


class BM25:
    """Okapi BM25 over tokenized chunks."""

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(c["text"]) for c in chunks]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        doc_freq: dict[str, int] = {}
        for tokens in self.doc_tokens:
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        n_docs = len(chunks)
        self.idf = {
            token: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freq.items()
        }

    def search(self, query: str, k: int = 5) -> list[tuple[dict, float]]:
        query_tokens = tokenize(query)
        scored: list[tuple[dict, float]] = []
        for chunk, tokens, length in zip(self.chunks, self.doc_tokens, self.doc_lengths):
            if length == 0:
                continue
            freq: dict[str, int] = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
            score = 0.0
            norm = 1.0 - self.b + self.b * (length / self.avgdl if self.avgdl else 1.0)
            for token in query_tokens:
                tf = freq.get(token, 0)
                if tf:
                    score += self.idf.get(token, 0.0) * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
            scored.append((chunk, score))
        scored.sort(key=lambda item: (-item[1], item[0]["chunk_id"]))
        return scored[:k]


class HybridIndex:
    """BM25 + dense index over the same chunks, fused with RRF."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.bm25 = BM25(chunks)
        self.vectorizer = HashingVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def dense_search(self, query: str, k: int = 5) -> list[tuple[dict, float]]:
        scores = self.matrix @ self.vectorizer.transform(query)
        order = sorted(range(len(self.chunks)), key=lambda i: (-float(scores[i]), self.chunks[i]["chunk_id"]))
        return [(self.chunks[i], float(scores[i])) for i in order[:k]]

    def hybrid_search(self, query: str, k: int = 5, pool: int = 10) -> list[tuple[dict, float]]:
        """Reciprocal Rank Fusion of the BM25 and dense candidate pools."""
        fused: dict[str, float] = {}
        by_id: dict[str, dict] = {}
        for ranking in (self.bm25.search(query, k=pool), self.dense_search(query, k=pool)):
            for rank, (chunk, _score) in enumerate(ranking, start=1):
                fused[chunk["chunk_id"]] = fused.get(chunk["chunk_id"], 0.0) + 1.0 / (60.0 + rank)
                by_id[chunk["chunk_id"]] = chunk
        ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        return [(by_id[chunk_id], score) for chunk_id, score in ordered[:k]]


def build_index(chunks: list[dict] | None = None) -> HybridIndex:
    return HybridIndex(chunks if chunks is not None else emit_chunks(load_documents()))


def main() -> None:
    index = build_index()
    print(f"hybrid index built over {len(index.chunks)} chunks")
    queries = [
        "how do I roll back a failed production deployment",
        "what happens when the primary database fails",
        "how does a user reset their password",
    ]
    for query in queries:
        print(f"\nquery: {query}")
        for chunk, score in index.hybrid_search(query, k=3):
            print(f"  {score:.4f}  {chunk['doc_id']}  [{chunk['heading']}]")


if __name__ == "__main__":
    main()
