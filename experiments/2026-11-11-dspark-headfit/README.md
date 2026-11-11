# DSpark Markov head: what it is worth, and what a fit would need

Date: 2026-11-11. Image `spark-ds41f:1111-headfit` (646-patch vLLM series,
tree `dd00e7837930`), `tp4-ring`, temperature 0 unless stated.

## Why this experiment

The 2026-11-10 trace recorded drafts, confidences and the drafter's input
hidden states, but not the distribution the drafter is fitted against: the
target (backbone) logits *before* the Markov bias. Without them a head fit has
no residual to regress on, and a standalone bigram fit is known to backfire
(single-token top-1 13%, bigram ceiling 39%).

Two more questions had to be answered before writing a trainer: is the
confidence head mis-calibrated (it is not — see below), and how much does the
Markov head contribute at all (measured here).

## Instrumentation (series 0047-0049)

| patch | what it adds |
| --- | --- |
| 0047 | target top-k logits at every verification position: `l:{off,rows,k}` per JSON line, `<path>.topk.f32` / `.topk.i32`, k=32 by default (`SPARK3_DSPARK_TRACE_TOPK`). Greedy steps record `lx:1` instead. |
| 0048 | `MARKOV_SCALE` sweep knob: one float or one per draft position, read once at speculator init from `/cache/dspark-trace/MARKOV_SCALE`; missing/invalid keeps the historical head bit-for-bit. |
| 0049 | `TOPK` marker file, so the logits width can be chosen per capture run without rebuilding. |

The logits hook sits one line after `logits = self.model.compute_logits(...)` in
`GPUModelRunner.sample()`. `torch.topk` allocates its own outputs, so sampling
is unchanged; with tracing off the only cost is one falsy module-level check.

**Greedy verification carries no logits.** `VLLM_GREEDY_VERIFICATION` defaults
on and fires for all-`temperature==0` batches, which sample straight from the
vocabulary shards and never materialise a full-vocab tensor. A capture must
therefore run at `temperature > 0`; measured: 3,075 steps with logits, 0 greedy.

## Capture (temperature 0.9, k=64)

200 replayed prompts from `~/logs/vllm-spec/*.jsonl[.gz]` (own traffic,
stratified over prompt size 2k-50k chars), 6 concurrent:

- 200/200 ok, 50,698 generated tokens, 525.8 s, **96.4 gen tok/s**
- acceptance rate median **0.630**, acceptance length median **3.08**
- trace: 7,853 verification positions, `trace.jsonl` 5.0 MB, `.topk.*` 19 MB each,
  `.hidden.f16` 904 MB

## What the logits say (7,853 positions)

- the committed token is inside the top-64 in **7,852 of 7,853 positions (100%)**,
  median rank 0 — the backbone's top-k is wide enough for a bias to act on
- conditional acceptance per position: pos0 69.6%, pos1 65.2%, pos2 68.1%,
  pos3 66.6%, pos4 64.5%, pos5 10.6% (n=218, the tail of long draft blocks)
- rejected drafts whose committed token is inside the top-k, and the bias they
  would have needed: pos0 928 rejected, **78.0% within 1 nat, 88.8% within
  2 nats** (pos1 86.4/93.8, pos2 89.8/95.3, pos3 89.8/95.3)
- required bias median **-1.15 nats**: the committed token usually already
  scores *higher* than the draft in the backbone — the drafter chose a token the
  target ranks lower
- 6,124 distinct `(previous token, committed token)` pairs in this sample

## Markov-bias sweep (100 prompts, temperature 0, three-window replay)

| `MARKOV_SCALE` | acceptance (median) | acceptance length | gen tok/s |
| ---: | ---: | ---: | ---: |
| 0.0 (bias off) | 0.547 | 1.93 | 69.5 |
| 0.8 | 0.637 | 3.15 | 76.5 |
| 1.0 (shipped) | **0.639** | **3.14** | 76.9 |
| 1.2 | 0.634 | 3.12 | 74.1 |

The engine log confirms each point loaded its value
(`DSpark Markov scale: ... -> [1.2, ...] per draft position`), so this is a real
measurement, not a dead knob.

**The Markov head is worth +9.2 pt of acceptance and +1.21 tokens per step**
(0.547 -> 0.639). It is *not* a marginal component. But the shipped magnitude is
already at the optimum of that one-dimensional family: -20% costs 0.2 pt and
+20% costs 0.5 pt, so there is nothing to win by scaling alone.

## What this rules in and out

- **Confidence-head calibration: skip.** Across the 2026-11-10 capture the mean
  confidence per position matched the *conditional* acceptance within 0.03
  (pos0 0.801 vs 0.801, pos1 0.740 vs 0.760). Its unconditional counterpart is
  lower by design (0.609 at pos1); that is the head doing its job, not a defect.
- **A magnitude sweep: exhausted.** 1.0 is optimal within ±20%.
- **A refit is still plausible**, because the head demonstrably carries a large
  effect and 78-90% of rejections sit within 1-2 nats of flipping. But to fit it
  as a *residual* we need the drafter's own logits at the same positions — the
  target top-k alone cannot say how far the drafter's argmax has to move. That
  is the next instrumentation (a fourth patch, one more build).

## Operational notes

- Clear the markers after a sweep. The scale is read once at startup, so a
  cluster left running with `MARKOV_SCALE=0.0` serves at 0.547 acceptance — the
  live cluster was restored (markers removed, restarted, 0.641 measured).
- One trace directory per run: the side files are append-mode while the JSONL's
  offsets restart at 0, so appending to an old directory produces colliding
  offsets. The 2026-11-10 capture was moved to `cache/dspark-trace-1110-run`.
- `cluster sync --apply` is required before `start` after a new image tag: the
  image has to be loaded on every node (35 GB, 104-132 s each).
