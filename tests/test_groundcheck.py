"""Tests for GroundCheck.

Split deliberately:
  - policy/threshold logic is tested with a fake client, so it runs offline and
    in CI with no key and no spend
  - a small live suite runs only when TYPESAFE_API_KEY is present

The threshold tests assert the actual contract of the library: the verdict
boundaries must match the derived table, and the sensitivity levels must be
correctly ordered. A silent threshold regression is the failure mode that would
make the whole product lie about its recall.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundcheck import (  # noqa: E402
    GroundCheck, Policy, Result, Verdict, _SENSITIVITY_TABLE, _build_state,
)

LIVE = bool(os.environ.get("TYPESAFE_API_KEY"))


class FakeAnswer:
    def __init__(self, noul=None, score=None, confidence=1.0):
        self.noul = noul
        self.score = score
        self.confidence = confidence


class FakeChoiceAnswer:
    def __init__(self, choice="contradicted", confidence=1.0):
        self.choice, self.confidence = choice, confidence


class FakeResponse:
    def __init__(self, p, score=4.0, conf=0.9):
        self.answers = {
            "grounded": FakeAnswer(noul=p),
            "support": FakeAnswer(score=score, confidence=conf),
        }


class FakeDiagnoseResponse:
    def __init__(self, choice="contradicted"):
        self.answers = {"failure": FakeChoiceAnswer(choice)}


class FakeClient:
    """Returns a scripted probability so threshold logic can be tested exactly."""

    def __init__(self, p=0.9, score=4.0, conf=0.9):
        self.p, self.score, self.conf = p, score, conf
        self.calls = 0

    def system_one(self, state, questions):
        self.calls += 1
        if "failure" in questions:          # the diagnose round trip
            return FakeDiagnoseResponse()
        return FakeResponse(self.p, self.score, self.conf)


def checker(p, conf=0.9):
    return GroundCheck(client=FakeClient(p=p, conf=conf))


# --- threshold / verdict contract -------------------------------------------

def _at(shape, value):
    """Build a checker whose DECISION variable equals `value` for this task.

    summarization decides on support/4; the others decide on the noul. The test
    must drive whichever one the library actually reads, or it silently tests
    nothing.
    """
    if shape == "summarization":
        return GroundCheck(client=FakeClient(p=0.5, score=value * 4.0))
    return GroundCheck(client=FakeClient(p=value))


@pytest.mark.parametrize("shape", ["qa", "summarization", "dialogue"])
def test_verdict_boundaries_match_table(shape):
    block, review = _SENSITIVITY_TABLE["balanced"][shape]
    pol = Policy(task=shape)

    # just below the block threshold -> BLOCK
    assert _at(shape, block - 0.01).check("src", "ans", policy=pol).verdict is Verdict.BLOCK
    # between the two -> REVIEW
    mid = (block + review) / 2
    if review > block:
        assert _at(shape, mid).check("src", "ans", policy=pol).verdict is Verdict.REVIEW
    # above review -> PASS
    assert _at(shape, min(review + 0.05, 1.0)).check("src", "ans", policy=pol).verdict is Verdict.PASS


def test_summarization_decides_on_score_not_noul():
    """Regression: summarization must read the score primitive.

    Measured on four disjoint fresh slices, `score` beat `noul` every time on
    summarization (+0.048 AUC, CI [+0.025, +0.073]) while LOSING on qa and
    dialogue. If someone reverts this to the noul, this test fails.
    """
    # noul says 'clearly grounded', score says 'contradicted' -> must BLOCK
    c = GroundCheck(client=FakeClient(p=0.99, score=0.0))
    assert c.check("src", "ans", policy=Policy(task="summarization")).verdict is Verdict.BLOCK
    # the same inputs on qa follow the noul instead -> must PASS
    assert c.check("src", "ans", policy=Policy(task="qa")).verdict is Verdict.PASS


@pytest.mark.parametrize("shape", ["qa", "summarization", "dialogue"])
def test_sensitivity_is_monotonic(shape):
    """Stricter policy must never block less than a looser one."""
    perm = _SENSITIVITY_TABLE["permissive"][shape][0]
    bal = _SENSITIVITY_TABLE["balanced"][shape][0]
    strict = _SENSITIVITY_TABLE["strict"][shape][0]
    assert perm <= bal <= strict


def test_review_threshold_is_never_below_block():
    for sens, shapes in _SENSITIVITY_TABLE.items():
        for shape, (block, review) in shapes.items():
            assert review >= block, f"{sens}/{shape} has an inverted band"


def test_all_shapes_present_for_every_sensitivity():
    shapes = {"qa", "summarization", "dialogue"}
    for sens, table in _SENSITIVITY_TABLE.items():
        assert set(table) == shapes, f"{sens} is missing a task shape"


# --- input validation --------------------------------------------------------

def test_empty_source_rejected():
    with pytest.raises(ValueError, match="source is empty"):
        checker(0.9).check("   ", "an answer")


def test_empty_answer_rejected():
    with pytest.raises(ValueError, match="answer is empty"):
        checker(0.9).check("a source", "")


def test_missing_key_without_client(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        GroundCheck()


# --- escalation / result surface ---------------------------------------------

def test_low_confidence_triggers_escalation_even_on_pass():
    r = checker(0.99, conf=0.2).check("src", "ans")
    assert r.verdict is Verdict.PASS
    assert r.should_escalate, "low confidence must escalate regardless of verdict"


def test_high_confidence_pass_does_not_escalate():
    assert not checker(0.99, conf=0.95).check("src", "ans").should_escalate


def test_ok_property_tracks_pass():
    assert checker(0.99).check("src", "ans").ok
    assert not checker(0.01).check("src", "ans").ok


def test_as_dict_is_json_serializable():
    import json
    d = checker(0.9).check("src", "ans").as_dict()
    json.loads(json.dumps(d))
    assert set(d) >= {"verdict", "p_grounded", "confidence", "latency_ms", "policy"}


def test_explain_skipped_on_pass():
    """The explain round trip must not be paid for when the answer passes."""
    c = FakeClient(p=0.99)
    r = GroundCheck(client=c).check("src", "ans", explain=True)
    assert c.calls == 1, "explain should not fire a second call on PASS"
    assert r.failure_mode is None


def test_explain_populates_failure_mode_on_block():
    """Regression: this shipped returning None for every input.

    The original implementation asked a Noul for a quoted span. Noul returns a
    float, so the field was always None while still costing a second API call.
    """
    c = FakeClient(p=0.01)
    r = GroundCheck(client=c).check("src", "ans", explain=True)
    assert c.calls == 2, "diagnose must fire on a non-PASS verdict"
    assert r.failure_mode == "contradicted"
    assert r.failure_confidence == 1.0


def test_diagnose_failure_is_non_fatal():
    """A failed diagnose call must not take down the primary verdict."""
    class Exploding(FakeClient):
        def system_one(self, state, questions):
            if "failure" in questions:
                raise RuntimeError("upstream down")
            return FakeResponse(self.p, self.score, self.conf)

    r = GroundCheck(client=Exploding(p=0.01)).check("src", "ans", explain=True)
    assert r.verdict is Verdict.BLOCK
    assert r.failure_mode is None


# --- prompt construction ------------------------------------------------------

def test_state_includes_question_for_qa():
    s = _build_state("SRC", "ANS", "Q?", "qa")
    assert "Q?" in s and "SRC" in s and "ANS" in s


def test_summarization_state_omits_question_label():
    s = _build_state("DOC", "SUM", "ignored", "summarization")
    assert "PROPOSED SUMMARY" in s and "QUESTION" not in s


# --- live suite ---------------------------------------------------------------

@pytest.mark.skipif(not LIVE, reason="TYPESAFE_API_KEY not set")
def test_live_catches_a_blatant_fabrication():
    gc = GroundCheck()
    r = gc.check(
        source="The Eiffel Tower is located in Paris, France. It was completed in 1889.",
        answer="The Eiffel Tower is in Berlin and was completed in 1975.",
        question="Where is the Eiffel Tower?",
    )
    assert r.verdict is Verdict.BLOCK
    assert r.p_grounded < 0.5


@pytest.mark.skipif(not LIVE, reason="TYPESAFE_API_KEY not set")
def test_live_passes_a_faithful_answer():
    gc = GroundCheck()
    r = gc.check(
        source="The Eiffel Tower is located in Paris, France. It was completed in 1889.",
        answer="The Eiffel Tower is in Paris and was finished in 1889.",
        question="Where is the Eiffel Tower?",
    )
    assert r.verdict is Verdict.PASS
    assert r.p_grounded > 0.5


@pytest.mark.skipif(not LIVE, reason="TYPESAFE_API_KEY not set")
def test_live_latency_is_interactive():
    """The whole product thesis is request-path latency. Guard it."""
    gc = GroundCheck()
    r = gc.check(source="Cats are mammals.", answer="Cats are mammals.")
    assert r.latency_ms < 3000, f"too slow for the request path: {r.latency_ms:.0f}ms"
