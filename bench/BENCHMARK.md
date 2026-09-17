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

## Head-to-head against frontier judges

Same examples, paired bootstrap on the AUC difference (10,000 draws).

| Task | Baseline | Jev AUC | Baseline AUC | Difference | 95% CI | P(Jev better) |
|---|---|---|---|---|---|---|
| qa (n=188) | Claude Haiku | 0.955 | 0.906 | **+0.050** | [+0.018, +0.085] | 99.9% |
| qa (n=200) | **Claude Sonnet** | 0.955 | 0.915 | **+0.040** | [+0.007, +0.077] | 99.3% |
| summarization (n=148) | Claude Haiku | 0.878 | 0.825 | **+0.053** | [+0.013, +0.096] | 99.4% |

All three confidence intervals exclude zero. **Moving up the model ladder did not
close the gap**: Sonnet costs roughly 3x Haiku per judgment and scored only
+0.009 AUC above it, still well below Jev.

**Not all three are equally strong, and the Sonnet result is the thinnest.** Its
CI lower bound is +0.006, close enough to zero to deserve stress-testing, so it
got some:

- re-run across **20 different bootstrap seeds**: the lower bound stayed in
  [+0.004, +0.008] and excluded zero in **20/20**
- an assumption-free **paired permutation test**: p = 0.022
- at a 0.5 threshold Jev is correct on 177/200 vs Sonnet's 173/200

So it holds, but treat it as "Jev is at least as good as Sonnet, probably
better" rather than the decisive margin the Haiku comparisons show.

Accuracy at a 0.5 threshold: 88.5% (Jev) vs 86.5% (Sonnet), 85.1% (Haiku).

### Baseline handling, stated explicitly

**Haiku produced 12 unparsable responses out of 200** (it replied "NO" instead
of a number, ignoring the output contract). Those rows were **dropped** from its
score. That is the treatment most charitable to the baseline: scoring them as
uninformative (0.5) instead would move Haiku from 0.906 to **0.898** and widen
Jev's margin. The reported gap is therefore conservative. Sonnet had zero
unparsable responses.

### Why this is plausible rather than surprising

Grounding is a constrained verification task, not open generation. The frontier
model's advantage is breadth and reasoning depth, neither of which is the
bottleneck when the whole job is "is this span supported by that span".

The mechanism differs between the two baselines, and an earlier draft of this
document got it wrong by generalizing from Haiku:

- **Haiku** collapses to the extremes: 70% of its answers were exactly 0.0 or
  1.0, discarding the ranking information AUC measures.
- **Sonnet** does spread its probabilities (only 11% at the extremes, 16
  distinct values) and still loses. So calibration granularity is not the whole
  story; it is simply less accurate at the underlying judgment.

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

The measured cascade frontier (escalating the least-confident fraction):

| Policy | AUC | $/1M | Mean latency |
|---|---|---|---|
| **Jev only** | **0.955** | **$23** | **236 ms** |
| escalate 10% to Sonnet | 0.949 | $192 | 935 ms |
| escalate 30% to Sonnet | 0.937 | $531 | 2,334 ms |
| escalate 100% (all Sonnet) | 0.915 | $1,718 | 7,230 ms |
| *oracle: escalate exactly Jev's errors* | *0.939* | - | - |

**Escalation makes things worse, and the oracle row proves it is not the
signal's fault.** Even a *perfect* router that escalated exactly the 11.5% of
rows Jev gets wrong would score 0.939, still below Jev alone at 0.955. When the
escalation target is worse than the source model, no routing policy can help.

This was tested against both Haiku and Sonnet, because the obvious objection to
the first negative result was "you escalated to a weak model". Sonnet is
3x the price and still loses. The conclusion holds:

> **On this task, do not build a cascade. Use Jev alone.**

The confidence signal itself is real and remains useful for *human* review
queues, where the goal is prioritizing which outputs a person looks at rather
than routing to another model.

## End-to-end validation of the shipped library

The benchmark above measures research scripts. This replays **held-out** examples
through the real `GroundCheck.check()` code path, thresholds and all:

| Task | Advertised block recall | Measured | False block rate | Median latency |
|---|---|---|---|---|
| qa (n=120) | 85% | **85.7%** | 12.3% | 179 ms |
| summarization (n=100) | 85% | **84.0%** | 32.0% | 231 ms |
| dialogue (n=100) | 85% | **82.0%** | 18.0% | 192 ms |

The library delivers the recall it promises **on slices of the same fetched
file the thresholds were derived from**. That is a weaker claim than it looks,
so it was re-run on genuinely fresh source rows (HaluEval offset 1000+, never
fetched during derivation):

| summarization | block recall | false block |
|---|---|---|
| same-file held-out slice | 0.840 | 0.320 |
| **fresh source rows (n=160)** | **0.775** | **0.375** |

Both degrade on truly unseen data.

### Summarization now decides on the `score` primitive, not the `noul`

Investigating that weakness produced a real improvement. Jev returns two signals
per call, and on **summarization only**, the `score` rubric is the better
detector. Measured across four disjoint fresh slices:

| slice | noul AUC | score AUC |
|---|---|---|
| derivation set | 0.875 | **0.890** |
| fresh @1000 | 0.838 | **0.858** |
| fresh @2000 | 0.740 | **0.787** |
| fresh @3000 | 0.787 | **0.835** |
| fresh @5000 | 0.794 | **0.805** |

`score` wins in every slice (+0.048, CI [+0.025, +0.073] on the first fresh
slice). **It is deliberately not applied to qa or dialogue**, where the same
test showed it *loses* (qa -0.008, dialogue -0.031, both CIs excluding zero).
A blanket switch would have degraded two tasks to improve one.

Prompt rewording was tried first and did **not** work: a summary-aware phrasing
that explicitly permits compression moved AUC 0.860 -> 0.872 but made
false-blocks *worse* (0.324 -> 0.382). The gain came from the primitive, not
the wording.

### Summarization after the change, on fresh data

| slice | level | recall | false block |
|---|---|---|---|
| @1000 | permissive | 0.675 | **0.188** |
| @1000 | balanced | 0.787 | 0.287 |
| @4000 | permissive | 0.637 | 0.275 |
| @4000 | balanced | 0.825 | **0.512** |

**Summarization is still the weak task and this did not fix it.** The false-block
rate swings from 0.19 to 0.51 across slices at the same setting. Use
`permissive` for summarization, treat BLOCK as "route to a human", and
recalibrate on your own documents. The honest headline is
**0.64-0.83 recall at 0.19-0.51 false blocks**, depending on the corpus.

Also note the benchmark slice was easier than typical data: the 0.875 AUC in the
headline table is the optimistic end of a 0.74-0.89 range.

**The summarization false-block rate is the main known weakness**: roughly three
in eight faithful summaries get flagged at the balanced setting. Use
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
