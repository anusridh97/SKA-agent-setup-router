# Background: Streaming Dynamic Programming for Segmentation

## 1. The Batch Segmentation Algorithm

The `GeometryLearner` solves a 1D segmentation problem via dynamic
programming. Given N sentences with embeddings, find boundaries
b₀ = 0 < b₁ < b₂ < ... < b_K = N that minimize:

    Φ = Σ_{k=1}^{K} internal_cost(b_{k-1}, b_k) + λ · K

where internal_cost(i, j) = Σ_{t=i}^{j-2} d(e_t, e_{t+1}) is the sum
of consecutive cosine distances within the segment [i, j).

The DP recurrence is:

    dp[0] = 0
    dp[i] = min_{j ∈ [max(0, i-K_max), i)}  dp[j] + cost(j, i) + λ

where K_max = lookback_k (default 50) and we additionally require
min_segment_size ≤ (i - j) ≤ max_segment_size.

**Time complexity:** O(N · K_max) = O(N · 50) = O(N).
**Space complexity:** O(N) for dp array, parent pointers, and prefix sums.

## 2. Prefix Sums for O(1) Cost Queries

The internal cost of segment [i, j) is:

    cost(i, j) = prefix_dist[j-1] - prefix_dist[i]

where prefix_dist[t] = Σ_{s=0}^{t-1} d(e_s, e_{s+1}) is precomputed.

In the batch setting, we compute prefix_dist for all N sentences upfront.
In the streaming setting, we maintain it incrementally:

    prefix_dist[t] = prefix_dist[t-1] + d(e_{t-1}, e_t)

We only need the last K_max + 1 values, so a circular buffer suffices.

## 3. Streaming DP with Bounded State

### State Required

To compute dp[i], we need:
1. dp[j] for j ∈ [i - K_max, i)  →  circular buffer of size K_max
2. prefix_dist[j..i]              →  circular buffer of size K_max + 1
3. parent[j] for j ∈ [i - K_max, i)  →  for backtracking

### Segment Finalization

A segment [a, b) is **finalized** when we know for certain that no future
dp[i] (with i > current position) will choose a split point j < b that
would create a segment overlapping [a, b).

**Claim:** Once we've computed dp[i] for i ≥ b + K_max, the segment [a, b)
is finalized.

**Proof:** Any future dp[i'] with i' > b + K_max can only look back to
j ≥ i' - K_max > b. So no future segment can start before b. The segment
[a, b) is determined by dp[b] and parent[b], which are already computed
and won't change.

This means we can emit segments with a delay of K_max positions. For
K_max = 50, this is a very small buffer.

## 4. Lambda Auto-Tuning in Streaming Mode

The batch algorithm auto-tunes λ based on the distance distribution:

```python
lambda_est = (p50 + p75) / 2
```

In streaming mode, we don't have all distances upfront. Options:

1. **Two-pass:** First pass computes distances and estimates λ, second
   pass runs the streaming DP. Still streaming for memory, but 2× I/O.

2. **Online quantile estimation:** Use a streaming quantile sketch
   (e.g. t-digest or GK sketch) to estimate p50 and p75 incrementally.
   Set λ after seeing the first W sentences (warmup window).

3. **Fixed λ:** User provides λ. Simplest, and appropriate when the
   corpus characteristics are known in advance.

Recommendation: implement option 3 first, option 1 as an enhancement.

## 5. Micro-Batch Embedding

Embedding models are most efficient in batches (GPU parallelism). The
streaming pipeline should accumulate sentences into micro-batches of
size B (e.g. 32), embed the batch, then feed embeddings one-at-a-time
into the streaming DP.

```
Sentence stream → [buffer of B sentences] → embed batch → stream to DP
```

This is a standard producer-consumer pattern. Memory usage is O(B · D)
for the embedding buffer plus O(K_max · D) for the DP state, where D is
the embedding dimension.

## 6. Correctness Argument

**Theorem:** The streaming DP produces identical segmentation to the
batch DP, provided:
1. The same λ is used
2. K_max, min_segment_size, and max_segment_size are the same
3. Distances are computed with the same numerical precision

**Proof:** The streaming DP computes the same recurrence. The circular
buffer contains exactly the values that the batch DP would access at
each step i (since the batch DP only looks back K_max positions). The
prefix sums are computed identically (same sequence of additions). By
induction on i, dp_streaming[i] = dp_batch[i] for all i.

## 7. References

1. Bellman. "Dynamic Programming." Princeton, 1957. The original DP framework.

2. Killick, Fearnhead, Eckley. "Optimal Detection of Changepoints with
   a Linear Computational Cost." JASA, 2012. PELT algorithm for changepoint detection with pruning. Relevant
   because our DP has similar structure and similar pruning opportunities.

3. Greenwald & Khanna. "Space-Efficient Online Computation of Quantile
   Summaries." SIGMOD, 2001. For streaming quantile estimation (option 2 in Section 4).
