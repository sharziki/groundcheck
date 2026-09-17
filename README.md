# GroundCheck

**Catch RAG hallucinations in ~200ms for ~$23 per million checks.**

Your retrieval pipeline returns a passage. Your model writes an answer. GroundCheck
tells you whether the answer is actually supported by the passage, fast enough and
cheap enough to run on **every** response instead of a 1% sample.

```python
from groundcheck import GroundCheck

gc = GroundCheck()
result = gc.check(
    source="The Apollo 11 mission landed on the Moon on July 20, 1969.",
    answer="Buzz Aldrin was the first human on Mars, in 1972.",
    question="Who first walked on the Moon?",
)

result.verdict      # Verdict.BLOCK
result.p_grounded   # 0.01
result.latency_ms   # 242
```

## Why it exists

LLM-as-judge works, but it is slow and expensive enough that teams run it offline,
on a sample, after the bad answer already shipped. At $0.0000228 per check you can
move that judgment into the request path.

Measured on 1,600 labeled examples with human-written hard negatives:

| Task | AUC | Median latency | Cost / 1M |
|---|---|---|---|
| Question answering | **0.952** | 221 ms | $22.82 |
| Dialogue | **0.912** | 169 ms | $23.26 |
| Summarization | **0.875** | 172 ms | $49.87 |

Against Claude Haiku on the same examples, GroundCheck is **more accurate**
(+0.050 AUC, 95% CI [+0.018, +0.085]), about 20x faster, and about 25x cheaper.
Full methodology, baselines, and limitations: [bench/BENCHMARK.md](bench/BENCHMARK.md).

## Install

```bash
pip install groundcheck                # library
pip install 'groundcheck[server]'      # + HTTP service
export TYPESAFE_API_KEY='...'          # https://typesafe.ai
```

## Use it

### As a gate in your RAG pipeline

```python
from groundcheck import GroundCheck, Policy, Verdict

gc = GroundCheck(policy=Policy(sensitivity="strict", task="qa"))
result = gc.check(source=retrieved_context, answer=llm_answer, question=user_question)

if result.verdict is Verdict.BLOCK:
    return "I could not verify that from my sources."
if result.should_escalate:
    queue_for_human_review(answer, result.p_grounded)   # the uncertain slice
```

### Why did it fail?

```python
r = gc.check(source=ctx, answer=ans, explain=True)
r.failure_mode   # 'contradicted' | 'unsupported' | 'overstated' | None
```

The modes map to different fixes: `contradicted` means retrieval returned the
wrong passage; `unsupported` means the model padded. It is a second round trip
and fires only when the verdict is not PASS.

`sensitivity` is expressed in the language of your risk tolerance, not in magic
numbers. Thresholds are **derived from a measured ROC curve**, not hand-tuned:

| Sensitivity | Blocks at | Use when |
|---|---|---|
| `permissive` | ~70% of hallucinations | false alarms are costly |
| `balanced` | ~85% | default |
| `strict` | ~95% | shipping a fabrication is costly |

### As a CI gate

Exit codes are `0` pass, `1` review, `2` block, so it gates a build with no parsing:

```bash
groundcheck batch answers.jsonl --task summarization --sensitivity strict
echo $?   # 2 if anything was fabricated
```

### As a service

```bash
uvicorn groundcheck.server:app --port 8099
```

```bash
curl -X POST localhost:8099/v1/check -H 'content-type: application/json' -d '{
  "source": "Refunds are available within 30 days of purchase.",
  "answer": "You can get a refund any time, no limit.",
  "sensitivity": "strict"
}'
```

```json
{"verdict": "block", "p_grounded": 0.02, "confidence": 0.95, "latency_ms": 242}
```

`POST /v1/check/batch` takes up to 500 items and fans out concurrently
(~50 checks/sec measured).

## Confidence is a real signal (but do not build a cascade on it)

Jev reports how sure it is, and that number is honest:

| Confidence | Share of traffic | AUC in band |
|---|---|---|
| >= 0.8 | 51% | **0.982** |
| < 0.5 | 23% | 0.847 |

`result.should_escalate` surfaces the uncertain slice. **Use it to prioritize a
human review queue, not to route to a bigger model.** I tested that cascade
against both Claude Haiku and Claude Sonnet and it made accuracy *worse* both
times, because both are weaker judges on this task. Even an oracle router that
escalated exactly the rows Jev gets wrong scored below Jev alone. Details in
[bench/BENCHMARK.md](bench/BENCHMARK.md).

## Known limitations

Stated up front rather than discovered in production:

- **Summarization flags ~32% of faithful summaries** at the balanced setting. Use
  `permissive` for that task, or treat BLOCK as "route to a human".
- Thresholds are calibrated on HaluEval. They held up across three task shapes and
  on held-out data, but recalibrate on your own examples with
  `bench/derive_thresholds.py` before trusting the exact numbers.
- English only so far.
- It checks grounding against the source you give it. It cannot tell you that your
  **retrieval** returned the wrong passage.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev,server]'
.venv/bin/python -m pytest          # 21 tests; live tests need TYPESAFE_API_KEY
.venv/bin/python bench/validate_e2e.py --shape qa   # verify shipped thresholds
```

## License

MIT
