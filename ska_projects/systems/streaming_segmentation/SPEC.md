# Project: Streaming Bounded-Memory Segmentation

## Summary

Transform the `GeometryLearner` from a batch algorithm that requires all
embeddings in memory into a streaming algorithm that processes sentences
in bounded-memory windows, producing segments incrementally.

**Area:** Systems
**GPU:** Minimal (one-time embedding pass)
**Duration:** 8 weeks
**Team size:** 1–2

## Motivation

The current `OfflinePipeline.process()` loads all sentences, embeds them all,
then runs DP segmentation over the full corpus. For a 10M-sentence corpus,
this means ~40GB of float64 embeddings in memory before segmentation even
starts. The DP itself is bounded by `lookback_k=50`, which means it already
has local structure, but the implementation doesn't exploit this.

The `lookback_k` parameter means that `dp[i]` only depends on
`dp[i-1], dp[i-2], ... dp[i-lookback_k]`. This is a textbook bounded-window
DP that can be computed in a single streaming pass with O(lookback_k) state.

## Deliverables

1. **`StreamingGeometryLearner`** class that:
   - Accepts one sentence (+ embedding) at a time
   - Maintains O(lookback_k) state
   - Emits finalized segments as soon as they're determined
   - Produces identical output to the batch algorithm

2. **`StreamingOfflinePipeline`** that:
   - Reads text line-by-line (or paragraph-by-paragraph)
   - Embeds in micro-batches
   - Feeds into StreamingGeometryLearner
   - Writes segments to disk incrementally

3. **Memory benchmarks** showing constant memory usage regardless of corpus size

4. **Correctness tests** showing identical segmentation to batch mode

## Where This Fits in the Codebase

```
ska_agent/core/geometry.py      ← GeometryLearner.learn_geometry()
                                   The batch DP. Your streaming version
                                   replaces this.

ska_agent/pipeline.py            ← OfflinePipeline.process()
                                   Currently loads everything into memory.
                                   Your StreamingOfflinePipeline replaces this.

ska_agent/utils/math_utils.py   ← MathUtils.compute_prefix_sums(),
                                   compute_pairwise_distances(),
                                   segment_internal_cost()
                                   These assume full arrays. You need
                                   streaming versions.
```

## Key Insight

The DP recurrence is:

```
dp[i] = min over j in [max(0, i-K), i) of:
    dp[j] + internal_cost(j, i) + λ
```

where K = lookback_k. This means:
- To compute dp[i], you need dp[i-K..i-1] (bounded window)
- internal_cost(j, i) needs prefix_dist[j..i] (also bounded window)
- Once dp[i] is computed and i > K, dp[i-K] is never accessed again

So you can maintain a circular buffer of size K for dp values and
prefix sums. When a segment boundary is determined to be "behind" the
window (i.e. no future dp[i'] can choose a split point within it),
that segment can be emitted.

The tricky part: a segment isn't finalized until you're sure no future
decision will split it differently. This requires tracking the "latest
possible start" for any future segment that could overlap.

## Milestones

| Week | Milestone |
|------|-----------|
| 1–2 | Understand batch DP. Implement circular-buffer prefix sums. |
| 3–4 | Implement streaming DP with bounded state. Correctness tests. |
| 5–6 | Implement segment finalization logic. StreamingOfflinePipeline. |
| 7–8 | Memory benchmarks on large synthetic corpora. Integration. |
