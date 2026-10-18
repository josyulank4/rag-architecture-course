"""Lesson: Evaluating RAG — golden set, recall@k, groundedness checks.

Runs a small golden set of questions with known source documents against the
hybrid retriever. Reports recall@5 per query and a groundedness check: the
top-ranked chunk must come from the expected document and contain the expected
evidence keyword. Exits non-zero when the quality gate fails, so the harness
can block a bad release.

Run from the repo root:
    python -m evaluation.harness
    python evaluation/harness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_search.hybrid import build_index  # noqa: E402

GOLDEN_SET = [
    {
        "query": "how do I roll back a failed production deployment",
        "expected_doc": "deploy_policy",
        "keyword": "rollback",
    },
    {
        "query": "what happens when the primary database fails",
        "expected_doc": "incident_runbook",
        "keyword": "replica",
    },
    {
        "query": "how long is restricted data retained",
        "expected_doc": "security_policy",
        "keyword": "90 days",
    },
    {
        "query": "how does a user reset their password",
        "expected_doc": "support_faq",
        "keyword": "reset",
    },
    {
        "query": "when are the production deployment windows",
        "expected_doc": "deploy_policy",
        "keyword": "tuesday",
    },
]


def evaluate() -> dict:
    index = build_index()
    rows = []
    hits = 0
    for item in GOLDEN_SET:
        results = index.hybrid_search(item["query"], k=5)
        retrieved_docs = [chunk["doc_id"] for chunk, _ in results]
        hit = item["expected_doc"] in retrieved_docs
        rank = retrieved_docs.index(item["expected_doc"]) + 1 if hit else None
        top_chunk = results[0][0] if results else {}
        grounded = (
            top_chunk.get("doc_id") == item["expected_doc"]
            and item["keyword"] in top_chunk.get("text", "").lower()
        )
        hits += int(hit)
        rows.append({**item, "hit": hit, "rank": rank, "grounded": grounded, "top_doc": top_chunk.get("doc_id")})
    recall = hits / len(GOLDEN_SET)
    overall = recall >= 0.8 and all(row["grounded"] for row in rows)
    return {"rows": rows, "recall_at_5": recall, "overall_pass": overall}


def main() -> None:
    report = evaluate()
    for i, row in enumerate(report["rows"], start=1):
        status = "PASS" if row["hit"] and row["grounded"] else "FAIL"
        print(
            f"Q{i}: {status}  expected={row['expected_doc']}  top={row['top_doc']}  "
            f"recall@5={'hit rank ' + str(row['rank']) if row['hit'] else 'miss'}  "
            f"grounded={'yes' if row['grounded'] else 'no'}  query={row['query']!r}"
        )
    print(f"recall@5: {report['recall_at_5']:.2f}")
    print(f"OVERALL: {'PASS' if report['overall_pass'] else 'FAIL'}")
    if not report["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
