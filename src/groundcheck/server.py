"""HTTP service for GroundCheck.

Exists because the buyer is usually not a Python app. One POST, one JSON verdict,
p50 under 250ms, so it can sit in the request path of a RAG pipeline rather than
in a nightly eval job.

    uvicorn groundcheck.server:app --port 8099
"""

from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import GroundCheck, Policy, Verdict, __version__

app = FastAPI(
    title="GroundCheck",
    version=__version__,
    description="Calibrated grounding guardrail for RAG answers, powered by Jev.",
)

_checker: GroundCheck | None = None


def checker() -> GroundCheck:
    global _checker
    if _checker is None:
        if not os.environ.get("TYPESAFE_API_KEY"):
            raise HTTPException(503, "TYPESAFE_API_KEY is not configured on this server")
        _checker = GroundCheck()
    return _checker


class CheckRequest(BaseModel):
    source: str = Field(..., description="The retrieved context the answer must be grounded in")
    answer: str = Field(..., description="The generated text to verify")
    question: str = Field("", description="The user's question, when the task has one")
    task: Literal["qa", "summarization", "dialogue"] = "qa"
    sensitivity: Literal["permissive", "balanced", "strict"] = "balanced"
    explain: bool = Field(
        False, description="Also classify the failure mode (extra round trip)"
    )


class BatchRequest(BaseModel):
    items: list[CheckRequest]
    workers: int = 8


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "configured": bool(os.environ.get("TYPESAFE_API_KEY"))}


@app.post("/v1/check")
def check(req: CheckRequest) -> dict:
    try:
        result = checker().check(
            req.source, req.answer,
            question=req.question,
            policy=Policy(sensitivity=req.sensitivity, task=req.task),
            explain=req.explain,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return result.as_dict()


@app.post("/v1/check/batch")
def check_batch(req: BatchRequest) -> dict:
    if not req.items:
        raise HTTPException(422, "items is empty")
    if len(req.items) > 500:
        raise HTTPException(413, "batch limit is 500 items")
    gc = checker()
    # policy is per item, so map rather than using check_many's single policy
    from concurrent.futures import ThreadPoolExecutor

    def one(it: CheckRequest) -> dict:
        try:
            return gc.check(
                it.source, it.answer, question=it.question,
                policy=Policy(sensitivity=it.sensitivity, task=it.task),
            ).as_dict()
        except ValueError as exc:
            return {"error": str(exc)}

    with ThreadPoolExecutor(max_workers=min(req.workers, 16)) as pool:
        results = list(pool.map(one, req.items))

    blocked = sum(1 for r in results if r.get("verdict") == Verdict.BLOCK.value)
    return {"results": results, "n": len(results), "blocked": blocked}
