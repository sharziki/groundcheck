"""Command-line interface for GroundCheck.

Two shapes, because two different people need this:

    groundcheck check --source-file ctx.txt --answer "..."   # one answer, human
    groundcheck batch answers.jsonl                          # a file, CI

Exit codes are the point of the CI path: 0 = pass, 1 = review, 2 = block. That
makes `groundcheck batch` usable as a build gate without parsing anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import GroundCheck, Policy, Verdict, __version__

EXIT = {Verdict.PASS: 0, Verdict.REVIEW: 1, Verdict.BLOCK: 2}


def _read(path: str | None, inline: str | None, what: str) -> str:
    if path:
        return Path(path).read_text()
    if inline:
        return inline
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit(f"no {what} given: pass --{what} or --{what}-file, or pipe it on stdin")


def cmd_check(a: argparse.Namespace) -> int:
    source = _read(a.source_file, a.source, "source")
    answer = _read(a.answer_file, a.answer, "answer")
    gc = GroundCheck()
    r = gc.check(
        source, answer, question=a.question,
        policy=Policy(sensitivity=a.sensitivity, task=a.task),
        explain=a.explain,
    )
    if a.json:
        print(json.dumps(r.as_dict()))
    else:
        icon = {Verdict.PASS: "PASS", Verdict.REVIEW: "REVIEW", Verdict.BLOCK: "BLOCK"}[r.verdict]
        print(f"{icon}  grounded={r.p_grounded:.2f}  confidence={r.confidence:.2f}  {r.latency_ms:.0f}ms")
        if r.unsupported_claim:
            print(f"  least supported: {r.unsupported_claim}")
        if r.should_escalate:
            print("  (uncertain: worth escalating to a larger judge)")
    return EXIT[r.verdict]


def cmd_batch(a: argparse.Namespace) -> int:
    rows = [json.loads(l) for l in Path(a.file).read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"{a.file} has no rows")
    missing = [i for i, r in enumerate(rows) if "source" not in r or "answer" not in r]
    if missing:
        raise SystemExit(f"rows {missing[:5]} are missing a 'source' or 'answer' field")

    gc = GroundCheck()
    results = gc.check_many(rows, policy=Policy(sensitivity=a.sensitivity, task=a.task),
                            workers=a.workers)

    blocked = review = 0
    for row, r in zip(rows, results):
        blocked += r.verdict is Verdict.BLOCK
        review += r.verdict is Verdict.REVIEW
        if a.json:
            print(json.dumps({"id": row.get("id"), **r.as_dict()}))

    total = len(results)
    if not a.json:
        print(f"{total} checked: {total - blocked - review} pass, {review} review, {blocked} block")
    if blocked:
        return 2
    if review and a.fail_on_review:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="groundcheck", description=__doc__.split("\n")[0])
    p.add_argument("--version", action="version", version=f"groundcheck {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--task", default="qa", choices=["qa", "summarization", "dialogue"])
    common.add_argument("--sensitivity", default="balanced",
                        choices=["permissive", "balanced", "strict"])
    common.add_argument("--json", action="store_true", help="machine-readable output")

    c = sub.add_parser("check", parents=[common], help="check one answer")
    c.add_argument("--source"); c.add_argument("--source-file")
    c.add_argument("--answer"); c.add_argument("--answer-file")
    c.add_argument("--question", default="")
    c.add_argument("--explain", action="store_true", help="name the least-supported span")
    c.set_defaults(func=cmd_check)

    b = sub.add_parser("batch", parents=[common], help="check a JSONL file (CI gate)")
    b.add_argument("file")
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--fail-on-review", action="store_true",
                   help="exit 1 when any answer lands in the review band")
    b.set_defaults(func=cmd_batch)

    a = p.parse_args(argv)
    try:
        return a.func(a)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
