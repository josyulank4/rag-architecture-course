# RAG Architecture: Production-Grade Retrieval Systems — Course Code

Companion code for the Udemy course **"RAG Architecture: Production-Grade
Retrieval Systems"** (Course 2) by Naveen Kumar Josyula.

This folder lives in the shared course-code repo alongside Course 1
(*Enterprise AI Agents*) and serves both the Udemy and Skool programs.

Every demo is small, self-contained, and runs **offline with no API keys**:
Python standard library + numpy only. The embedding lesson uses a
deterministic local hashing vectorizer as a stand-in for a hosted embedding
model, so results are identical on every run and every machine.

## Requirements

- Python 3.11+
- `pip install -r requirements.txt` (numpy, pinned)

## Sample corpus

`data/` holds four short, fictional enterprise documents written for this
course (no real company content): a deployment policy, an incident runbook, a
security/access policy, and a support FAQ. Access labels (public / internal /
restricted) are assigned at ingestion time in `ingestion/ingest.py`.

## How to run each demo

Run from this folder's root:

| Lesson | Command | What it shows |
|---|---|---|
| L13 — Metadata, Permissions, and Freshness at Ingest | `python -m ingestion.ingest` | Loads `data/`, cleans text, emits chunks with metadata + access labels |
| L16–L17 — Chunking strategies / Chunking Lab | `python -m chunking.chunkers` | Fixed vs recursive vs structure-aware chunking, with a comparison run |
| L19–L22 — Embeddings / Embedding Evaluation Lab | `python -m embeddings.vectorizer` | Deterministic hashing embeddings + cosine-similarity search |
| L28–L29 — Hybrid search / Query understanding | `python -m hybrid_search.hybrid` | BM25 + dense retrieval fused with Reciprocal Rank Fusion |
| L30 — [DEMO] Reranking for Precision | `python -m reranking.rerank` | First-stage top-8 before/after a second-stage reranker |
| L39 — [DEMO] Evaluation Harness and Release Gates | `python -m evaluation.harness` | Golden-set recall@5 + groundedness checks with a pass/fail quality gate |

Each module also runs as a plain script, e.g. `python ingestion/ingest.py`.

## Validation

`python validate.py` runs every demo end-to-end as a subprocess, checks the
expected outputs, asserts retrieval quality on golden questions, and exits
non-zero on any failure. See `RESULTS.md` for the recorded validation run.

## Layout

```
data/            fictional sample corpus (Markdown)
ingestion/       load, clean, chunk, metadata + access labels
chunking/        fixed / recursive / structure-aware chunkers
embeddings/      deterministic local vectorizer + cosine search
hybrid_search/   BM25 + dense + RRF fusion
reranking/       second-stage reranker demo
evaluation/      golden-set harness (recall@k, groundedness, quality gate)
validate.py      end-to-end validation for all demos
```

## License

MIT — © 2026 Naveen Kumar Josyula. See `LICENSE`.
