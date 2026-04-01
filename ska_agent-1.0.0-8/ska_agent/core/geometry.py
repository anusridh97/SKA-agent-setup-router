"""
Stage I: Geometry Learner (corpus segmentation via dynamic programming).

This is the offline preprocessing step that runs before any model inference.
It takes raw corpus text (e.g., 89,000 pages of Treasury Bulletins) and
segments it into semantically coherent atomic units called Segments.

Unlike fixed-size chunking (every 512 tokens), this finds natural topic
boundaries by minimizing internal pairwise cosine distance within each
segment, subject to a sparsity penalty lambda per segment:

    min  sum_i internal_cost(segment_i) + lambda * num_segments

The dynamic programming runs in O(N * K) where N is sentence count and
K is the lookback window (default 50). Lambda is auto-tuned from the
pairwise distance distribution if not provided.

Output: a list of Segment objects (defined in core/structures.py), each
containing the segment text, centroid embedding, sentence indices, and
internal cost.

These segments are then fed to the pricing engine (core/pricing.py) for
retrieval, and their embeddings are used by the SKA modules for operator
construction.

Dependencies:
  - core/structures.py for the Segment dataclass
  - utils/math_utils.py for prefix sums and pairwise distance computation
"""

from __future__ import annotations

import numpy as np
from typing import List

from ..core.structures import Segment
from ..utils.math_utils import MathUtils


class GeometryLearner:
    """
    Stage I: Learn stable atomic units via pairwise distance minimization.

    Auto-tunes lambda based on the distance distribution so that cuts
    occur at above-median jumps (likely topic boundaries).
    """

    def __init__(
        self,
        lambda_seg: float = None,
        lookback_k: int = 50,
        min_segment_size: int = 2,
        max_segment_size: int = 15,
    ):
        self.lambda_seg = lambda_seg
        self.lookback_k = lookback_k
        self.min_segment_size = min_segment_size
        self.max_segment_size = max_segment_size

    def estimate_lambda(self, distances: np.ndarray, verbose: bool = True) -> float:
        """Auto-tune lambda based on pairwise distance distribution."""
        nonzero = distances[distances > 0]
        if len(nonzero) == 0:
            return 0.1

        p25 = np.percentile(nonzero, 25)
        p50 = np.percentile(nonzero, 50)
        p75 = np.percentile(nonzero, 75)
        p90 = np.percentile(nonzero, 90)

        if verbose:
            print(f" Pairwise distance percentiles: p25={p25:.3f}, p50={p50:.3f}, p75={p75:.3f}, p90={p90:.3f}")
            top_k = min(10, len(nonzero) // 3)
            if top_k > 0:
                top_indices = np.argsort(distances)[-top_k:][::-1]
                print(f" Top jump positions: {sorted(top_indices.tolist())}")

        lambda_est = (p50 + p75) / 2
        return np.clip(lambda_est, 0.05, 1.0)

    def learn_geometry(
        self,
        embeddings: np.ndarray,
        sentences: List[str],
        verbose: bool = True,
    ) -> List[Segment]:
        """Solve the segmentation problem via dynamic programming."""
        N = len(sentences)
        if N == 0:
            return []

        prefix_sum, _ = MathUtils.compute_prefix_sums(embeddings)
        distances, prefix_dist = MathUtils.compute_pairwise_distances(embeddings)

        if self.lambda_seg is None:
            lambda_val = self.estimate_lambda(distances, verbose=verbose)
        else:
            lambda_val = self.lambda_seg

        if verbose:
            print(f" lambda = {lambda_val:.4f}")
            print(f" Running DP (N={N}, K={self.lookback_k})...")

        dp = np.full(N + 1, float('inf'))
        dp[0] = 0.0
        parent = np.zeros(N + 1, dtype=int)
        seg_cost = np.zeros(N + 1)

        for i in range(1, N + 1):
            min_start = max(0, i - self.lookback_k)
            for j in range(min_start, i):
                seg_len = i - j
                if seg_len < self.min_segment_size or seg_len > self.max_segment_size:
                    continue

                internal = MathUtils.segment_internal_cost(j, i, prefix_dist)
                cost = dp[j] + internal + lambda_val

                if cost < dp[i]:
                    dp[i] = cost
                    parent[i] = j
                    seg_cost[i] = internal

        boundaries = []
        costs = []
        i = N
        while i > 0:
            boundaries.append(i)
            costs.append(seg_cost[i])
            i = parent[i]
        boundaries.append(0)
        boundaries.reverse()
        costs.reverse()

        if verbose:
            print(f" Boundaries: {boundaries}")

        segments = []
        for idx in range(len(boundaries) - 1):
            start, end = boundaries[idx], boundaries[idx + 1]
            centroid = MathUtils.segment_centroid(start, end, prefix_sum)

            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                centroid = centroid / norm

            segments.append(Segment(
                text=' '.join(sentences[start:end]),
                vector=centroid,
                start_idx=start,
                end_idx=end,
                sentences=sentences[start:end],
                internal_cost=costs[idx] if idx < len(costs) else 0.0,
            ))

        if verbose:
            avg_size = N / len(segments) if segments else 0
            print(f" Found {len(segments)} segments (avg {avg_size:.1f} sentences)")

        return segments
