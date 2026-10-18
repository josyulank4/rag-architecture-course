"""Lesson: Reranking for Precision.

First-stage retrieval optimizes recall; a reranker re-scores a small candidate
set more carefully. This demo shows the hybrid top-8 before and after a simple
feature-based reranker (token overlap, exact phrase matches, heading matches).

Run from the repo root:
    python -m reranking.rerank
    python reranking/rerank.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embeddings.vectorizer import tokenize  # noqa: E402
from hybrid_search.hybrid import build_index  # noqa: E402

QUERY = "roll back a failed deployment quickly"


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


def rerank_score(query: str, chunk: dict) -> float:
    """Cheap second-stage features; a cross-encoder would replace this in prod."""
    query_tokens = tokenize(query)
    chunk_tokens = tokenize(chunk["text"])
    query_set, chunk_set = set(query_tokens), set(chunk_tokens)
    union = query_set | chunk_set
    jaccard = len(query_set & chunk_set) / len(union) if union else 0.0
    phrase_hits = len(_bigrams(query_tokens) & _bigrams(chunk_tokens))
    heading_hits = len(query_set & set(tokenize(chunk["heading"])))
    return jaccard + 0.30 * phrase_hits + 0.10 * heading_hits


def rerank(query: str, candidates: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
    rescored = [(chunk, rerank_score(query, chunk)) for chunk, _ in candidates]
    rescored.sort(key=lambda item: (-item[1], item[0]["chunk_id"]))
    return rescored


def main() -> None:
    index = build_index()
    before = index.hybrid_search(QUERY, k=8)
    after = rerank(QUERY, before)
    before_rank = {chunk["chunk_id"]: rank for rank, (chunk, _) in enumerate(before, start=1)}

    print(f"query: {QUERY}")
    print("\nBEFORE (hybrid first stage):")
    for rank, (chunk, score) in enumerate(before, start=1):
        print(f"  {rank}. {score:.4f}  {chunk['doc_id']}  [{chunk['heading']}]")
    print("\nAFTER (reranked):")
    for rank, (chunk, score) in enumerate(after, start=1):
        moved = before_rank[chunk["chunk_id"]] - rank
        arrow = f"up {moved}" if moved > 0 else ("same" if moved == 0 else f"down {-moved}")
        print(f"  {rank}. {score:.4f}  {chunk['doc_id']}  [{chunk['heading']}]  ({arrow})")


if __name__ == "__main__":
    main()
