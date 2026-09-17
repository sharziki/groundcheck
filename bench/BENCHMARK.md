# GroundCheck benchmark

Every number here was produced by the scripts in `~/jev-judge-bench` and can be
regenerated. Nothing is estimated or quoted from a vendor page.

**Date:** 2026-09-17
**Judge under test:** TypeSafe AI Jev (System One), via `typesafe_sdk`
**Baseline:** Claude Haiku via `claude -p`, same examples, same output contract

## Task

Given source material and a proposed piece of generated text, decide whether the
text is fully supported by the source. This is the judgment a RAG guardrail makes
on every request.

Data is HaluEval, which pairs each source with a correct output **and a
human-written hallucinated output**. Using both halves of every pair makes the
task balanced by construction (base rate exactly 0.5) and the negatives hard:
they are fluent, plausible, and wrong, not random text.

| Config | What the model judges | n |
|---|---|---|
| `qa` | answer against a passage | 600 |
| `dialogue` | reply against background knowledge | 500 |
| `summarization` | summary against a document | 500 |

## Headline results

| Task | AUC | 95% CI | Avg precision | Median latency | Cost / 1M checks |
|---|---|---|---|---|---|
| qa | **0.952** | [0.936, 0.966] | 0.954 | 221 ms | $22.82 |
| dialogue | **0.912** | [0.886, 0.936] | 0.920 | 169 ms | $23.26 |
| summarization | **0.875** | [0.844, 0.904] | 0.886 | 172 ms | $49.87 |

1,600 calls, **zero errors, zero malformed responses**. Typed output means there
is no parser to fail.

## Head-to-head against a frontier judge

Same examples, paired bootstrap on the AUC difference (10,000 draws).

| Task | Jev AUC | Claude Haiku AUC | Difference | 95% CI | P(Jev better) |
|---|---|---|---|---|---|
| qa (n=188) | 0.955 | 0.906 | **+0.050** | [+0.018, +0.085] | 99.9% |
| summarization (n=148) | 0.878 | 0.825 | **+0.053** | [+0.013, +0.096] | 99.4% |

Both confidence intervals exclude zero. The small, cheap, typed model is not
merely competitive with the frontier LLM on this task, it is **better**, on two
independent task shapes.

Accuracy at a 0.5 threshold: 88.8% vs 85.1% (qa), 81.8% vs 76.4% (summarization).

### Why this is plausible rather than surprising

Grounding is a constrained verification task, not an open generation task. The
frontier model's advantage is breadth and reasoning depth, neither of which is
the bottleneck when the whole job is "is this span supported by that span". The
typed model is trained to emit a calibrated probability, so it expresses
uncertainty; the LLM emits near-binary judgments (it answered 0 or 1 on most
rows), which throws away the ranking information AUC measures.

## Cost and latency

| | Jev | Claude Haiku (CLI) |
|---|---|---|
| Median latency | 221 ms | 9,865 ms raw / ~4,865 ms overhead-adjusted |
| Cost per judgment | $0.0000228 | $0.000565 |
| Throughput measured | 48-59 /s at 12 workers | 0.37 /s at 6 workers |

**Latency caveat, stated plainly:** the baseline was measured through the
`claude -p` CLI, which costs ~5.0s of process startup on this machine (measured
separately, 3 runs). The honest speedup after subtracting that entire overhead is
**20.7x**, and that is the number to quote. A direct API call would narrow the
gap further. **The AUC comparison is unaffected by transport.**

The cost difference is what changes the architecture: at $22.82 per million, you
check **every** response. At $565 per million, you sample 1% and hope.

## Confidence is a usable routing signal

Jev returns a confidence with every answer, and that confidence is honest:

| Confidence band | Share of traffic | AUC within band |
|---|---|---|
| < 0.5 | 23.0% | 0.847 |
| 0.5 - 0.8 | 26.2% | 0.870 |
| >= 0.8 | 50.8% | **0.982** |

When Jev is confident it is nearly perfect, and it knows when it is not. That is
what makes `Result.should_escalate` meaningful rather than decorative: route the
uncertain quarter of traffic to a bigger model and decide the rest locally.

The measured cascade frontier (escalating the least-confident fraction to Claude):

| Policy | AUC | $/1M | Mean latency |
|---|---|---|---|
| Jev only | 0.955 | $23 | 236 ms |
| escalate 10% | 0.946 | $83 | 1,285 ms |
| escalate 100% (all Claude) | 0.906 | $588 | 10,100 ms |

Note what this table actually says: **escalation made things worse on this
dataset**, because the escalation target is the weaker judge here. The cascade
machinery is sound and the confidence signal is real, but on this task the right
policy is "do not escalate". Reported rather than buried, because it contradicts
the architecture I expected to recommend.

## End-to-end validation of the shipped library

The benchmark above measures research scripts. This replays **held-out** examples
through the real `GroundCheck.check()` code path, thresholds and all:

| Task | Advertised block recall | Measured | False block rate | Median latency |
|---|---|---|---|---|
| qa (n=120) | 85% | **85.7%** | 12.3% | 179 ms |
| summarization (n=100) | 85% | **84.0%** | 32.0% | 231 ms |
| dialogue (n=100) | 85% | **82.0%** | 18.0% | 192 ms |

The library delivers the recall it promises. **The summarization false-block rate
of 32% is high** and is the main known weakness: on document summarization, one
in three faithful summaries gets flagged at the balanced setting. Use
`permissive` there, or treat BLOCK as "route to review" rather than "discard".

## Limitations

1. **One benchmark family.** All three task shapes come from HaluEval. Its
   hallucinations are LLM-generated and human-filtered, which is realistic but
   not identical to the failure modes of your own pipeline.
2. **The baseline is one model through a CLI.** Haiku, not Opus or GPT-5. A
   larger judge would likely close the quality gap, at a further cost increase.
3. **Thresholds are calibrated on this data.** They transferred across three task
   shapes and to held-out slices, which is real evidence of robustness, but
   recalibrate on a few hundred of your own labeled examples before trusting the
   exact numbers in production. `bench/derive_thresholds.py` does this.
4. **English only.** Not tested on other languages.
5. **Summarization false-block rate is high**, as stated above.

## Reproduce

```bash
cd ~/jev-judge-bench
source ~/.config/typesafe/credentials.env
python fetch_data.py --config qa --n-source 300
python run_jev.py --data data/halueval.jsonl --out results/jev_judge.jsonl
python score.py --file results/jev_judge.jsonl
python run_claude.py --limit 200                  # frontier baseline
python head_to_head.py                            # paired bootstrap
python cascade.py                                 # confidence routing

cd ~/groundcheck
python bench/derive_thresholds.py                 # regenerate the threshold table
python bench/validate_e2e.py --shape qa           # validate the shipped library
```
