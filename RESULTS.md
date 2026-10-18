# Validation Results — Course 2 Code Repo

Run date: 2026-10-10
Environment: Python 3.12.3, numpy 1.26.4 (matches `requirements.txt`)
Working directory: repo root (`code-repo/`)

## Commands run

```
python3 validate.py                      # exit 0
python3 -m ingestion.ingest              # exit 0
python3 -m chunking.chunkers             # exit 0
python3 -m embeddings.vectorizer         # exit 0
python3 -m hybrid_search.hybrid          # exit 0
python3 -m reranking.rerank              # exit 0
python3 -m evaluation.harness            # exit 0
```

## Validation outcome

`python3 validate.py` — **VALIDATION PASSED: all demos and assertions green**
(34/34 checks PASS, 0 failures, exit code 0). Checks covered: every module
running end-to-end as a subprocess with expected output markers, corpus and
metadata integrity, all three chunking strategies producing chunks, hybrid
top-1 correctness on four golden questions, reranker promoting the correct
chunk to #1, and the evaluation quality gate.

## Actual demo output highlights

- **Ingestion:** 4 documents loaded (deploy_policy [internal],
  incident_runbook [internal], security_policy [restricted], support_faq
  [public]); 16 chunks emitted, each with doc_id, heading path, and access
  label. Sample chunk `deploy_policy-000` = "Deployment Policy > Release
  windows", 261 chars.
- **Chunking comparison** (deploy_policy): fixed = 6 chunks (avg 203.2 chars),
  recursive = 4 chunks (avg 253.5), structure_aware = 4 chunks (avg 265.5);
  all structure-aware chunks carry a heading: True.
- **Embeddings:** 16 chunks embedded into 512-dim deterministic hashing
  vectors; dense-only top-1s: database-failover query → incident_runbook
  (0.339), retention query → security_policy (0.454).
- **Hybrid search (BM25 + dense + RRF):** top-1 correct on all demo queries —
  rollback → deploy_policy, database failure → incident_runbook, password
  reset → support_faq.
- **Reranking:** query "roll back a failed deployment quickly" — the
  deploy_policy Rollback procedure chunk stays #1 after reranking; the
  Approvals chunk moves up 3 places (6 → 3) and the support-FAQ deployment
  distractor moves down.
- **Evaluation harness:** Q1–Q5 all PASS, every expected document hit at
  rank 1, groundedness checks all "yes". **recall@5: 1.00 — OVERALL: PASS.**

## Fixes made during validation

None required. The first full `validate.py` run passed end-to-end; no demo
code changes were needed afterward. (`__pycache__` by-products of the runs
were removed and a `.gitignore` added.)

## Known simplifications (by design, for teaching)

- The embedding model is a deterministic local hashing/TF-IDF-style
  vectorizer so demos run offline with no API keys; production systems would
  swap in a hosted embedding model behind the same interface.
- The reranker uses transparent overlap/phrase/heading features; the lesson
  notes a cross-encoder or LLM reranker as the production replacement.
- The sample corpus is four short fictional documents written for this
  course; no real or third-party content is included.
