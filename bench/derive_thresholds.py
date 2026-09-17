"""Derive GroundCheck's thresholds from the measured ROC curves.

Run this whenever the benchmark data changes. It prints the _SENSITIVITY_TABLE
literal so the library's thresholds stay traceable to evidence instead of taste.

Definitions, per task shape:
  strict     block at ~95% hallucination recall
  balanced   block at ~85% recall
  permissive block at ~70% recall
`review_below` is the threshold one recall step above `block_below`, so the
REVIEW band covers the cases the block threshold is about to let through.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RECALL_TARGETS = {"permissive": (0.70, 0.85), "balanced": (0.85, 0.95), "strict": (0.95, 0.99)}


def thresholds_for(path: Path) -> dict[str, tuple[float, float]] | None:
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    ok = [r for r in rows if "p_grounded" in r]
    if not ok:
        return None
    p = np.array([r["p_grounded"] for r in ok])
    hall = np.array([1 - r["label"] for r in ok])
    p_hall = p[hall == 1]  # p_grounded values on known-hallucinated rows

    out = {}
    for name, (block_recall, review_recall) in RECALL_TARGETS.items():
        # to catch `recall` of hallucinations we must block everything with
        # p_grounded below that quantile of the hallucinated distribution
        block = float(np.quantile(p_hall, block_recall))
        review = float(np.quantile(p_hall, review_recall))
        out[name] = (round(block, 3), round(max(review, block), 3))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(Path.home() / "jev-judge-bench" / "results"))
    a = ap.parse_args()

    files = {
        "qa": "jev_judge.jsonl",
        "summarization": "jev_summarization.jsonl",
        "dialogue": "jev_dialogue.jsonl",
    }
    table: dict[str, dict[str, tuple[float, float]]] = {k: {} for k in RECALL_TARGETS}
    for shape, fname in files.items():
        path = Path(a.results_dir) / fname
        if not path.exists():
            print(f"missing {path}, skipping {shape}")
            continue
        got = thresholds_for(path)
        if not got:
            continue
        for sens, pair in got.items():
            table[sens][shape] = pair
        print(f"{shape}: {got}")

    print("\n_SENSITIVITY_TABLE = {")
    for sens in ("balanced", "strict", "permissive"):
        inner = ", ".join(f'"{s}": {v}' for s, v in sorted(table[sens].items()))
        print(f'    "{sens}": {{{inner}}},')
    print("}")


if __name__ == "__main__":
    main()
