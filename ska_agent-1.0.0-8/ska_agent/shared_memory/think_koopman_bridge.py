"""
Think-to-Koopman Bridge.

This file connects the Qwen coordinator's <think> reasoning traces to
Koopman operator slot 2 (reasoning state, 0-indexed).

The problem this solves: when the coordinator reasons through a multi-step
problem, other agents (retriever, code executor) need to know what the
coordinator is looking for, without reading the full text of the <think>
trace. The bridge projects reasoning hidden states into the shared operator
space so other agents can query the coordinator's reasoning state through
the spectral read protocol, at O(r^2) cost instead of O(context_length).

The data flow:

  QwenCoordinator.reason() produces <think> text with numbered steps
  -> QwenCoordinator.extract_reasoning_state() extracts hidden states
     from the model's last 4 layers, mean-pooled to R^3584
  -> ThinkKoopmanBridge.project() maps R^3584 to R^48 via W_think
  -> ThinkKoopmanBridge.accumulate() updates Gram and transition matrices
  -> ThinkKoopmanBridge.build_operator() constructs A_w = L^{-1} M L^{-T}
  -> ThinkKoopmanBridge.write_to_shared_memory() calls inject_operator()
     on SharedSpectralMemory, making it readable by other agents

The temporal structure matters: each numbered reasoning step becomes a
time step in the Koopman operator. So the operator doesn't just capture
what the coordinator concluded, it captures how the reasoning evolved.
Directions where reasoning was consistent across steps (high eigenvalues)
are amplified by the power filter; directions that fluctuated randomly
(low eigenvalues) are suppressed.

The value readout B_v is computed via two triangular solves against the
Cholesky factor L, following the same no-explicit-inverse discipline as
everywhere else in the codebase.

Dependencies:
  - core/structures.py for SharedOperator, MultiKoopmanConfig
  - utils/math_utils.py for SpectralUtils
  - shared_memory/spectral_memory.py for inject_operator()
  - models/qwen_coordinator.py provides the reasoning embeddings
"""

from __future__ import annotations

from typing import Optional, List

import numpy as np

from ..core.structures import MultiKoopmanConfig, SKAConfig
from ..utils.math_utils import SpectralUtils


class ThinkKoopmanBridge:
    """
    Projects reasoning embeddings into Koopman operator key space.

    The bridge maintains a running sequence of reasoning state vectors
    and accumulates them into the Koopman operator for slot 3.

    Architecture:
        W_think ∈ R^{rank × hidden_size} (learned or SVD-initialized)
        z_t = W_think · h_t (project reasoning hidden state)
        G_3 += z_t z_t^T (accumulate into slot 3 Gram)
        M_3 += z_t z_{t-1}^T (accumulate into slot 3 transition)
    """

    def __init__(
        self,
        hidden_size: int = 3584, # Qwen3.5-27B hidden dim
        rank: int = 48, # Koopman slot rank
        ridge_eps: float = 1e-3,
    ):
        self.hidden_size = hidden_size
        self.rank = rank
        self.ridge_eps = ridge_eps

        # Projection matrix: initialized as random orthogonal
        W = np.random.randn(rank, hidden_size)
        Q, _ = np.linalg.qr(W.T) # (hidden_size, rank)
        self.W_think = Q[:, :rank].T.copy() # (rank, hidden_size)

        # Accumulated Koopman matrices for slot 3
        self.gram = ridge_eps * np.eye(rank)
        self.transition = np.zeros((rank, rank))
        self.value_cross = None
        self.prev_key = None
        self.num_steps = 0

        # Raw key buffer - kept so write_to_shared_memory can replay keys
        self._key_buffer: List[np.ndarray] = []

        # Cached operator
        self._operator_stale = True
        self._A_w = None
        self._L = None
        self._B_v = None

    def project(self, reasoning_embedding: np.ndarray) -> np.ndarray:
        """
        Project a reasoning state vector into the Koopman key space.

        Args:
            reasoning_embedding: (hidden_size,) from extract_reasoning_state()

        Returns:
            key: (rank,) projected key vector
        """
        key = self.W_think @ reasoning_embedding # (rank,)
        # L2 normalize for stable operator construction
        norm = np.linalg.norm(key)
        if norm > 1e-10:
            key = key / norm
        return key

    def accumulate(
        self,
        reasoning_embedding: np.ndarray,
        value_embedding: Optional[np.ndarray] = None,
    ):
        """
        Project and accumulate a reasoning state into slot 3's operator.

        Call this after each coordinator reasoning step. The sequence of
        accumulated states builds the Koopman operator that captures
        how the coordinator's reasoning evolves over time.

        Args:
            reasoning_embedding: (hidden_size,) from extract_reasoning_state()
            value_embedding: (d_v,) optional value vector for readout
        """
        key = self.project(reasoning_embedding)
        self._key_buffer.append(key.copy())

        # Accumulate Gram: G += z_t z_t^T
        self.gram += np.outer(key, key)

        # Accumulate transition: M += z_t z_{t-1}^T
        if self.prev_key is not None:
            self.transition += np.outer(key, self.prev_key)

        # Accumulate value cross-correlation if provided
        if value_embedding is not None:
            if self.value_cross is None:
                self.value_cross = np.zeros((len(value_embedding), self.rank))
            self.value_cross += np.outer(value_embedding, key)

        self.prev_key = key.copy()
        self.num_steps += 1
        self._operator_stale = True

    def accumulate_multi_step(
        self,
        thinking_steps: List[str],
        extract_fn,
    ):
        """
        Accumulate multiple reasoning steps from a single ThinkExtraction.

        This captures the *evolution* of reasoning within a single query - 
        each numbered step becomes a time step in the Koopman operator.

        Args:
            thinking_steps: List of reasoning step strings
            extract_fn: Callable that takes text -> (hidden_size,) embedding
                        (typically QwenCoordinator.extract_reasoning_state)
        """
        for step_text in thinking_steps:
            if step_text.strip():
                embedding = extract_fn(step_text)
                self.accumulate(embedding)

    def build_operator(self) -> dict:
        """
        Build the Koopman operator for slot 3 from accumulated matrices.

        Returns dict with A_w, L, B_v for injection into SharedSpectralMemory.
        """
        from scipy.linalg import solve_triangular

        if not self._operator_stale and self._A_w is not None:
            return {"A_w": self._A_w, "L": self._L, "B_v": self._B_v}

        G = self.gram.copy()
        M = self.transition.copy()

        # Cholesky
        try:
            L = np.linalg.cholesky(G)
        except np.linalg.LinAlgError:
            G += 1e-4 * np.eye(self.rank)
            L = np.linalg.cholesky(G)

        # A_w = L^{-1} M L^{-T} (similarity transform)
        A_w = SpectralUtils.whiten_operator(L, M)
        A_w = SpectralUtils.spectral_normalize(A_w, gamma=1.0)

        # Value readout via triangular solves (no explicit inverse)
        B_v = None
        if self.value_cross is not None:
            from scipy.linalg import solve_triangular
            # B_v = C_v G^{-1} = C_v L^{-T} L^{-1}
            Y = solve_triangular(L, self.value_cross.T, lower=True)
            Bv_T = solve_triangular(L.T, Y, lower=False)
            B_v = Bv_T.T

        self._A_w = A_w
        self._L = L
        self._B_v = B_v
        self._operator_stale = False

        return {"A_w": A_w, "L": L, "B_v": B_v}

    def write_to_shared_memory(self, shared_memory):
        """
        Inject the bridge's operator into SharedSpectralMemory so other
        agents can read slot 3 through the standard read protocol.

        Uses SharedSpectralMemory.inject_operator() - the public API for
        external operator injection with proper cache invalidation.
        """
        if self.num_steps == 0:
            return

        op = self.build_operator()

        shared_memory.inject_operator(
            A_w=op["A_w"],
            L=op["L"],
            B_v=op["B_v"],
            num_tokens_seen=self.num_steps,
        )

    def query_reasoning(
        self,
        query_embedding: np.ndarray,
        power_K: int = 2,
    ) -> np.ndarray:
        """
        Query the reasoning operator with a key vector.

        Used by other agents to access the coordinator's reasoning state:
            z_q = W_think · h_query
            ŷ = B_v · L · A_w^K · L^{-1} · z_q

        Args:
            query_embedding: (hidden_size,) query from another agent
            power_K: Number of power iterations

        Returns:
            result: (rank,) filtered output
        """
        from scipy.linalg import solve_triangular

        op = self.build_operator()
        z_q = self.project(query_embedding)

        # Whiten
        w_q = solve_triangular(op["L"], z_q, lower=True)

        # Power filter
        w_f = w_q.copy()
        for _ in range(power_K):
            w_f = op["A_w"] @ w_f

        # Unwhiten
        z_hat = op["L"] @ w_f

        return z_hat

    def reset(self):
        """Clear accumulated state."""
        self.gram = self.ridge_eps * np.eye(self.rank)
        self.transition = np.zeros((self.rank, self.rank))
        self.value_cross = None
        self.prev_key = None
        self.num_steps = 0
        self._key_buffer = []
        self._operator_stale = True
        self._A_w = None
        self._L = None
        self._B_v = None

    def get_stats(self) -> dict:
        """Return operator statistics."""
        op = self.build_operator()
        cond = SpectralUtils.condition_number(self.gram)
        sigma = np.linalg.norm(op["A_w"], ord=2)
        return {
            "num_steps": self.num_steps,
            "condition_number": cond,
            "spectral_radius": sigma,
            "rank": self.rank,
        }
