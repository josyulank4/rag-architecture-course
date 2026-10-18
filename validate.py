"""End-to-end validation for every Course 2 demo.

Runs each lesson module as a subprocess (exactly how a student runs it),
checks the expected output markers, and adds direct retrieval assertions.
Exits non-zero if anything fails.

Run from the repo root:
    python validate.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def run_module(module: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def expect_module(name: str, module: str, markers: list[str]) -> str:
    proc = run_module(module)
    check(f"{name}: runs end-to-end (exit 0)", proc.returncode == 0, proc.stderr.strip()[-300:])
    for marker in markers:
        check(f"{name}: output contains {marker!r}", marker in proc.stdout)
    return proc.stdout


def main() -> None:
    print("== module demos (subprocess) ==")
    expect_module("ingestion", "ingestion.ingest", ["documents loaded: 4", "chunks emitted:", "access_label"])
    expect_module(
        "chunking",
        "chunking.chunkers",
        ["fixed:", "recursive:", "structure_aware:", "all structure-aware chunks carry a heading: True"],
    )
    expect_module("embeddings", "embeddings.vectorizer", ["embedded", "deploy_policy"])
    expect_module("hybrid_search", "hybrid_search.hybrid", ["hybrid index built over", "deploy_policy"])
    expect_module("reranking", "reranking.rerank", ["BEFORE (hybrid first stage):", "AFTER (reranked):"])
    expect_module("evaluation", "evaluation.harness", ["recall@5:", "OVERALL: PASS"])

    print("\n== direct assertions ==")
    from chunking.chunkers import comparison_run
    from evaluation.harness import evaluate
    from hybrid_search.hybrid import build_index
    from ingestion.ingest import emit_chunks, load_documents
    from reranking.rerank import QUERY, rerank

    documents = load_documents()
    chunks = emit_chunks(documents)
    check("corpus: 4 documents loaded", len(documents) == 4, f"got {len(documents)}")
    check("corpus: chunks emitted", len(chunks) >= 12, f"got {len(chunks)}")
    check(
        "corpus: every chunk has metadata + access label",
        all(c["doc_id"] and c["heading"] and c["access_label"] for c in chunks),
    )

    stats = comparison_run(documents[0]["text"])
    check("chunking: all strategies produce chunks", all(stats[s]["count"] > 0 for s in ("fixed", "recursive", "structure_aware")))
    check(
        "chunking: every structure-aware chunk carries its heading",
        all("\n" in c for c in stats["structure_aware_chunks"]),
    )

    index = build_index(chunks)
    for query, expected_doc in [
        ("how do I roll back a failed production deployment", "deploy_policy"),
        ("what happens when the primary database fails", "incident_runbook"),
        ("how long is restricted data retained", "security_policy"),
        ("how does a user reset their password", "support_faq"),
    ]:
        top_doc = index.hybrid_search(query, k=1)[0][0]["doc_id"]
        check(f"hybrid top-1 for {query!r}", top_doc == expected_doc, f"got {top_doc}")

    before = index.hybrid_search(QUERY, k=8)
    after = rerank(QUERY, before)
    before_rank = {c["chunk_id"]: r for r, (c, _) in enumerate(before, start=1)}
    target = after[0][0]
    check("reranking: deploy_policy chunk is #1 after rerank", target["doc_id"] == "deploy_policy", f"got {target['doc_id']}")
    check(
        "reranking: top chunk did not move down",
        before_rank[target["chunk_id"]] >= 1 and 1 <= before_rank[target["chunk_id"]],
        f"was rank {before_rank[target['chunk_id']]}",
    )

    report = evaluate()
    check("evaluation: recall@5 >= 0.80", report["recall_at_5"] >= 0.8, f"got {report['recall_at_5']:.2f}")
    check("evaluation: overall quality gate passes", report["overall_pass"] is True)

    print("\n== summary ==")
    if FAILURES:
        print(f"VALIDATION FAILED: {len(FAILURES)} check(s) failed")
        for failure in FAILURES:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("VALIDATION PASSED: all demos and assertions green")


if __name__ == "__main__":
    main()
