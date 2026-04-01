"""
Stage II: Pricing Engine (retrieval via reduced-cost optimization).

Given a query and a set of Segments from the geometry learner
(core/geometry.py), this module selects the most informative segments
using a pricing-guided greedy algorithm.

The master objective being minimized:

    Phi(V) = phi(V) + lambda * |V| + eta * R(V)

where phi(V) is reconstruction error (how much of the query embedding
is unexplained), lambda * |V| is sparsity penalty (fewer segments is
cheaper), and R(V) is redundancy penalty (selected segments should not
overlap).

The reduced cost for adding segment j is:

    c_bar(j) = lambda + eta * delta_R - delta_phi

where delta_phi is the Schur complement information gain:

    delta_phi = (residual dot segment_j)^2 / ||segment_j||^2

If c_bar(j) < 0, adding segment j strictly improves the objective.
The algorithm greedily adds the segment with the most negative reduced
cost, updates the residual via orthogonal projection (removing the
explained component), and repeats until no segment has negative reduced
cost or the budget is exhausted.

This is the retrieval mechanism that feeds context to the LLM generator
(models/llm.py) and the Qwen coordinator (models/qwen_coordinator.py).

Dependencies:
  - core/structures.py for Segment, RetrievalResult
  - utils/math_utils.py for orthogonal_projection
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from ..core.structures import Segment, RetrievalResult
from ..utils.math_utils import MathUtils


class PricingEngine:
    """
    Pricing-guided discrete selection for retrieval.

    The information gain is computed via the Schur complement formula:
        δφ = (residual · segment)² / ||segment||²

    After selection, the residual is updated by orthogonal projection.
    """

    def __init__(
        self,
        segments: List[Segment],
        embed_fn,
        lambda_sparsity: float = 0.05,
        eta_redundancy: float = 0.0,
        max_segments: int = 5,
        stopping_threshold: float = 1e-6,
    ):
        self.segments = segments
        self.embed_fn = embed_fn
        self.lambda_sparsity = lambda_sparsity
        self.eta_redundancy = eta_redundancy
        self.max_segments = max_segments
        self.stopping_threshold = stopping_threshold

        self.U = np.array([s.vector for s in segments])
        self.norms_sq = np.sum(self.U ** 2, axis=1)

    def compute_information_gain(self, residual: np.ndarray, j: int) -> float:
        """Schur complement: δφ = (residual · segment)² / ||segment||²"""
        if self.norms_sq[j] < 1e-10:
            return 0.0
        inner = np.dot(self.U[j], residual)
        return (inner ** 2) / self.norms_sq[j]

    def compute_redundancy_penalty(self, j: int, selected: List[int]) -> float:
        """Max cosine similarity to already-selected segments."""
        if not selected or self.eta_redundancy == 0:
            return 0.0
        u_j = self.U[j]
        max_sim = max(np.dot(u_j, self.U[idx]) for idx in selected)
        return max_sim

    def compute_reduced_cost(
        self,
        residual: np.ndarray,
        j: int,
        selected: List[int],
    ) -> Tuple[float, float]:
        """c̄ = λ + η·δR - δφ. Negative certifies improvement."""
        delta_phi = self.compute_information_gain(residual, j)
        delta_R = self.compute_redundancy_penalty(j, selected)
        reduced_cost = self.lambda_sparsity + self.eta_redundancy * delta_R - delta_phi
        return reduced_cost, delta_phi

    def update_residual(self, residual: np.ndarray, j: int) -> np.ndarray:
        """Remove component explained by segment j."""
        return MathUtils.orthogonal_projection(residual, self.U[j])

    def retrieve(self, query: str, verbose: bool = True) -> RetrievalResult:
        """Run pricing-guided retrieval with monotone descent."""
        query_emb = self.embed_fn(query)
        query_norm = np.linalg.norm(query_emb)
        residual = query_emb / query_norm if query_norm > 1e-10 else query_emb

        selected_indices = []
        reduced_costs = []
        visited = np.zeros(len(self.segments), dtype=bool)

        if verbose:
            print(f"\nQuery: '{query}'")
            print(f" Retrieval (lambda={self.lambda_sparsity:.3f}):")

        for iteration in range(self.max_segments):
            best_j, best_rc, best_ig = -1, float('inf'), 0.0

            for j in range(len(self.segments)):
                if visited[j]:
                    continue
                rc, ig = self.compute_reduced_cost(residual, j, selected_indices)
                if rc < best_rc:
                    best_rc, best_j, best_ig = rc, j, ig

            if verbose:
                print(f" Iter {iteration+1}: reduced_cost={best_rc:.4f}, info_gain={best_ig:.4f}")

            if best_rc >= -self.stopping_threshold:
                if verbose:
                    print(f" Stopping: reduced_cost >= 0")
                break

            selected_indices.append(best_j)
            reduced_costs.append(best_rc)
            visited[best_j] = True

            if verbose:
                preview = self.segments[best_j].text[:60] + "..."
                print(f" Selected segment {best_j}: '{preview}'")

            residual = self.update_residual(residual, best_j)
            if np.linalg.norm(residual) < 1e-6:
                if verbose:
                    print(f" Stopping: residual depleted")
                break

        if verbose:
            print(f" Retrieved {len(selected_indices)} segments")

        return RetrievalResult(
            segments=[self.segments[i] for i in selected_indices],
            reduced_costs=reduced_costs,
            total_segments_considered=len(self.segments),
        )
