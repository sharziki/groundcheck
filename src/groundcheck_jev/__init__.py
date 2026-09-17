"""GroundCheck: a calibrated grounding guardrail for RAG answers.

The thresholds in this module are not guesses. They are read off a measured ROC
curve (see bench/BENCHMARK.md): 900 labeled examples across three HaluEval task
shapes, each with a human-written hard negative.

    qa            AUC 0.952   (n=600)
    dialogue      AUC 0.912   (n=500)
    summarization AUC 0.875   (n=500)

The design decision that matters: a guardrail must let the caller choose its own
error tradeoff, because flagging a good answer and shipping a fabricated one cost
wildly different amounts in different products. So `Policy` is expressed in the
language of the buyer ("catch 90% of hallucinations") and converted to a
threshold here, rather than asking the caller to invent a magic number.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

__version__ = "0.1.0"

TaskShape = Literal["qa", "summarization", "dialogue"]


class Verdict(str, Enum):
    """What the caller should DO, not merely what the model thought."""

    PASS = "pass"          # ship it
    REVIEW = "review"      # ship with a caveat, or route to a human/bigger model
    BLOCK = "block"        # do not ship this answer as-is


# Thresholds on p_grounded, DERIVED from the measured ROC curve of each task
# shape by bench/derive_thresholds.py -- not hand-tuned. Regenerate that script's
# output whenever the benchmark data changes.
#   permissive  block at ~70% hallucination recall
#   balanced    block at ~85%
#   strict      block at ~95%
# `review_below` is the next recall step up, so the REVIEW band covers exactly
# the cases the block threshold is about to let through.
#
# NOTE: the summarization row is on the `support`/4 scale, not the noul scale,
# because that task decides on the score primitive (see `check`). The other two
# rows are on the noul scale. Changing one without the other silently breaks
# the task's operating point.
_SENSITIVITY_TABLE: dict[str, dict[TaskShape, tuple[float, float]]] = {
    # sensitivity: {shape: (block_below, review_below)}
    "balanced":   {"dialogue": (0.08, 0.135), "qa": (0.472, 0.82), "summarization": (0.808, 0.93)},
    "strict":     {"dialogue": (0.135, 0.581), "qa": (0.82, 0.92), "summarization": (0.93, 0.969)},
    "permissive": {"dialogue": (0.05, 0.08), "qa": (0.103, 0.472), "summarization": (0.666, 0.773)},
}


def _support_to_probability(score: float) -> float:
    """Map the 0-4 `support` rubric onto a 0-1 grounding probability.

    Linear rescale: the summarization thresholds below are derived from this
    same transform, so the two must change together.
    """
    return max(0.0, min(1.0, score / 4.0))


@dataclass(frozen=True)
class Policy:
    """How much risk the caller is willing to carry.

    sensitivity:
      permissive - block only near-certain fabrication (fewest false flags)
      balanced   - the ~80-90% recall operating point (default)
      strict     - catch almost everything, accept more false flags
    """

    sensitivity: Literal["permissive", "balanced", "strict"] = "balanced"
    task: TaskShape = "qa"

    def thresholds(self) -> tuple[float, float]:
        return _SENSITIVITY_TABLE[self.sensitivity][self.task]


@dataclass
class Result:
    verdict: Verdict
    p_grounded: float
    support: float
    confidence: float
    latency_ms: float
    failure_mode: str | None = None
    failure_confidence: float | None = None
    policy: Policy = field(default_factory=Policy)

    @property
    def ok(self) -> bool:
        return self.verdict is Verdict.PASS

    @property
    def should_escalate(self) -> bool:
        """True when this result is worth a second look by a HUMAN.

        Read the benchmark before wiring this to another model. Measured: Jev's
        low-confidence stratum scores AUC 0.847 while its high-confidence
        stratum scores 0.982, so the signal is real. But escalating those rows
        to Claude (Haiku *or* Sonnet) made accuracy WORSE, because both are
        weaker judges on this task. An oracle router that escalated exactly the
        rows Jev gets wrong still scored below Jev alone.

        So: use this to prioritize a human review queue, not to route to a
        bigger model, unless you have measured that your escalation target
        actually beats Jev on your data.
        """
        return self.verdict is Verdict.REVIEW or self.confidence < 0.5

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "p_grounded": round(self.p_grounded, 4),
            "support": self.support,
            "confidence": round(self.confidence, 4),
            "latency_ms": round(self.latency_ms, 1),
            "failure_mode": self.failure_mode,
            "failure_confidence": (
                round(self.failure_confidence, 4) if self.failure_confidence is not None else None
            ),
            "policy": {"sensitivity": self.policy.sensitivity, "task": self.policy.task},
        }


_SOURCE_LABEL: dict[TaskShape, tuple[str, str | None, str]] = {
    "qa": ("SOURCE PASSAGE", "QUESTION", "PROPOSED ANSWER"),
    "summarization": ("SOURCE DOCUMENT", None, "PROPOSED SUMMARY"),
    "dialogue": ("BACKGROUND KNOWLEDGE", "CONVERSATION SO FAR", "PROPOSED REPLY"),
}


def _build_state(source: str, answer: str, question: str, task: TaskShape) -> str:
    src_l, q_l, ans_l = _SOURCE_LABEL[task]
    parts = [f"{src_l}:", source, ""]
    if q_l and question:
        parts.append(f"{q_l}: {question}")
    parts.append(f"{ans_l}: {answer}")
    return "\n".join(parts)


_QUESTIONS = {
    "grounded": Noul(
        instructions=(
            "The proposed text is fully supported by the source material above. "
            "Answer NO if any part of it is invented, contradicted, or cannot be "
            "verified from the source alone."
        )
    ),
    "support": Score(
        instructions="How much of the proposed text is directly supported by the source material",
        criteria=[
            "Contradicted by the source",
            "Mostly unsupported",
            "Partly supported",
            "Nearly all supported",
            "Fully supported",
        ],
    ),
}


class GroundCheck:
    """Synchronous grounding checks. Safe to share across threads."""

    def __init__(self, client: TypeSafeClient | None = None, policy: Policy | None = None):
        if client is None and not os.environ.get("TYPESAFE_API_KEY"):
            raise RuntimeError(
                "TYPESAFE_API_KEY is not set. Export it, or pass an explicit client."
            )
        self._client = client or TypeSafeClient()
        self._policy = policy or Policy()

    def check(
        self,
        source: str,
        answer: str,
        *,
        question: str = "",
        policy: Policy | None = None,
        explain: bool = False,
    ) -> Result:
        """Judge whether `answer` is grounded in `source`.

        explain=True additionally classifies HOW the text fails (contradicted /
        unsupported / overstated). It is a second round trip, so it is off by
        default and fires only when the verdict is not PASS.
        """
        pol = policy or self._policy
        if not source.strip():
            raise ValueError("source is empty; nothing to check the answer against")
        if not answer.strip():
            raise ValueError("answer is empty; nothing to check")

        state = _build_state(source, answer, question, pol.task)
        t0 = time.perf_counter()
        resp = self._client.system_one(state=state, questions=_QUESTIONS)
        latency = (time.perf_counter() - t0) * 1000

        p = float(resp.answers["grounded"].noul)
        support = float(resp.answers["support"].score)
        conf = float(resp.answers["support"].confidence)

        # On SUMMARIZATION the `score` primitive is a better detector than the
        # `noul`, so the decision uses it there. Measured on four disjoint
        # slices of fresh HaluEval data (n=160-180 each), score beat noul every
        # time: +0.048 AUC on the first fresh slice, CI [+0.025, +0.073].
        #
        # This is deliberately NOT applied to qa or dialogue, where the same
        # test showed score LOSES (qa -0.008, dialogue -0.031, both CIs
        # excluding zero). A blanket switch would have made two tasks worse.
        decision_p = _support_to_probability(support) if pol.task == "summarization" else p

        block_below, review_below = pol.thresholds()
        if decision_p < block_below:
            verdict = Verdict.BLOCK
        elif decision_p < review_below:
            verdict = Verdict.REVIEW
        else:
            verdict = Verdict.PASS

        mode, mode_conf = None, None
        if explain and verdict is not Verdict.PASS:
            mode, mode_conf = self._diagnose(state)

        return Result(verdict, p, support, conf, latency, mode, mode_conf, pol)

    def _diagnose(self, state: str) -> tuple[str | None, float | None]:
        """Classify HOW the text fails its source. Separate call: only on failures.

        An earlier version asked for a quoted span of the offending text. That
        could never work: Jev's primitives return typed values (a float, a score,
        a labelled choice), never free text, so the call always yielded None.
        Classifying the failure mode is both achievable and more actionable,
        since it maps to different remediations: `contradicted` means retrieval
        found the wrong passage, `unsupported` means the model padded.
        """
        try:
            resp = self._client.system_one(
                state=state,
                questions={
                    "failure": Choice(
                        instructions=(
                            "What is the main problem with the proposed text "
                            "relative to the source?"
                        ),
                        criteria={
                            "contradicted": "it states something the source contradicts",
                            "unsupported": "it adds details absent from the source",
                            "overstated": "it overstates the confidence or scope of the source",
                            "none": "it is fully supported by the source",
                        },
                    )
                },
            )
            a = resp.answers["failure"]
            return str(a.choice), float(a.confidence)
        except Exception:
            return None, None

    def check_many(
        self, items: list[dict], *, policy: Policy | None = None, workers: int = 8
    ) -> list[Result]:
        """Check a batch concurrently. Measured throughput: ~50/s at 12 workers."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(
                pool.map(
                    lambda it: self.check(
                        it["source"], it["answer"],
                        question=it.get("question", ""), policy=policy,
                    ),
                    items,
                )
            )
