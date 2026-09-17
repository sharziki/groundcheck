"""End-to-end validation: does the SHIPPED library reproduce the benchmark?

The benchmark measured a research script. This replays held-out labeled examples
through the real `GroundCheck.check()` code path, including its policy logic, and
asserts the verdict distribution matches what the thresholds promise.

A library whose own thresholds do not deliver their advertised recall is a bug,
not a rounding difference. This is the check that catches it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundcheck import GroundCheck, Policy, Verdict  # noqa: E402

SHAPES = {
    "qa": "halueval.jsonl",
    "summarization": "halueval_summarization.jsonl",
    "dialogue": "halueval_dialogue.jsonl",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path.home() / "jev-judge-bench" / "data"))
    ap.add_argument("--shape", default="qa", choices=sorted(SHAPES))
    ap.add_argument("--sensitivity", default="balanced")
    ap.add_argument("--n", type=int, default=120, help="held-out examples to replay")
    ap.add_argument("--offset", type=int, default=400, help="skip rows used for calibration")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="bench/e2e_validation.json")
    a = ap.parse_args()

    if not os.environ.get("TYPESAFE_API_KEY"):
        raise SystemExit("TYPESAFE_API_KEY not set")

    path = Path(a.data_dir) / SHAPES[a.shape]
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    held = rows[a.offset : a.offset + a.n]
    if len(held) < 20:
        raise SystemExit(f"only {len(held)} held-out rows at offset {a.offset}")

    policy = Policy(sensitivity=a.sensitivity, task=a.shape)
    gc = GroundCheck()

    items = [{"source": r["knowledge"], "answer": r["answer"], "question": r.get("question", "")}
             for r in held]
    t0 = time.perf_counter()
    results = gc.check_many(items, policy=policy, workers=a.workers)
    wall = time.perf_counter() - t0

    hall = [r for r, ex in zip(results, held) if ex["label"] == 0]
    good = [r for r, ex in zip(results, held) if ex["label"] == 1]

    blocked_hall = sum(1 for r in hall if r.verdict is Verdict.BLOCK)
    flagged_hall = sum(1 for r in hall if r.verdict is not Verdict.PASS)
    blocked_good = sum(1 for r in good if r.verdict is Verdict.BLOCK)
    flagged_good = sum(1 for r in good if r.verdict is not Verdict.PASS)
    escalate = sum(1 for r in results if r.should_escalate)
    lat = sorted(r.latency_ms for r in results)

    target = {"permissive": 0.70, "balanced": 0.85, "strict": 0.95}[a.sensitivity]
    block_recall = blocked_hall / max(len(hall), 1)

    out = {
        "shape": a.shape,
        "sensitivity": a.sensitivity,
        "n": len(results),
        "n_hallucinated": len(hall),
        "n_grounded": len(good),
        "block_recall": round(block_recall, 4),
        "block_recall_target": target,
        "flag_recall_block_or_review": round(flagged_hall / max(len(hall), 1), 4),
        "false_block_rate": round(blocked_good / max(len(good), 1), 4),
        "false_flag_rate": round(flagged_good / max(len(good), 1), 4),
        "escalation_share": round(escalate / max(len(results), 1), 4),
        "latency_ms": {
            "median": round(lat[len(lat) // 2]),
            "p95": round(lat[int(len(lat) * 0.95)]),
        },
        "throughput_per_s": round(len(results) / wall, 1),
    }
    # the library must deliver within 12 points of its advertised recall
    out["PASS"] = bool(block_recall >= target - 0.12)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    if not out["PASS"]:
        raise SystemExit(
            f"FAIL: block recall {block_recall:.3f} misses target {target} by more than 12 points"
        )


if __name__ == "__main__":
    main()
