"""
Structured Kernel Attention (SKA) Module.

This file contains the core mathematical innovation of the system: a
drop-in replacement for standard attention that uses Koopman operators
for fixed-size spectral memory.

How it works:

  Standard attention stores per-token KV pairs that grow linearly with
  context length. SKA compresses a prefix of key/value pairs into an
  r x r operator (typically r=64) and applies it to query positions via
  power spectral filtering. The operator size is independent of context
  length, giving O(r^2) memory instead of O(T * d).

The math (all implemented via triangular solves, no explicit inverses):

  Given prefix keys Z in R^{L x r} and transition between consecutive keys:

  1. Gram matrix:     G = Z^T Z + epsilon * I_r
  2. Transition:      M = Z[1:]^T Z[:-1]  (how keys evolve over time)
  3. Cholesky:        LL^T = G
  4. Whitened operator: A_w = L^{-1} M L^{-T}
     Computed as: solve L U = M (forward), solve L V = U^T (forward), A_w = V^T
     This is a similarity transform: L A_w L^{-1} = M G^{-1}
     So the whiten-power-unwhiten pipeline computes: L A_w^K L^{-1} = (MG^{-1})^K
  5. Spectral normalization: A_w = gamma * A_w / max(1, sigma_max(A_w))
     Gamma is clamped to 1 to prevent blowup under power iteration.
  6. Value readout: B_v = C_v G^{-1} where C_v = V^T Z
     Computed as: solve L Y = C_v^T (forward), solve L^T B = Y (backward), B_v = B^T

  For each query z_q:
    w_q = L^{-1} z_q      (whiten via triangular solve)
    w_f = A_w^K w_q        (power spectral filtering, K=2 default)
    z_hat = L w_f           (unwhiten)
    y_hat = B_v z_hat       (read out values)

This file defines two classes:

  SKAModule: Single Koopman operator per attention head.
    Used by models/jamba_ska.py to replace Jamba's 4 attention layers.
    Configured by SKAConfig (core/structures.py).

  MultiHeadKoopmanModule: K=4 parallel rank-48 operators per head.
    Each operator has its own key projection, Gram matrix, and gate.
    Slot specialization (parser, reasoning, temporal) emerges during
    training. 16x cheaper Cholesky solves than a single rank-192 operator.
    Configured by MultiKoopmanConfig (core/structures.py).

Key property: fixed-size memory. The operator A_w is r x r regardless
of whether the prefix had 100 or 262,000 tokens. This is what enables
the shared memory protocol (shared_memory/spectral_memory.py) to give
agents O(r^2) communication instead of O(context_length).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.structures import SKAConfig


class SKAModule(nn.Module):
    """
    Structured Kernel Attention module.

    Replaces a standard attention layer with Koopman operator-based
    spectral filtering. Implements Equations 5-10 from the spec.

    Args:
        config: SKAConfig with d_model, n_heads, head_dim, rank, etc.
    """

    def __init__(self, config: SKAConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.rank = config.rank
        self.power_K = config.power_K
        self.ridge_eps = config.ridge_eps
        self.spectral_gamma = config.spectral_gamma

        # Key projection: d_model -> n_heads * rank
        # (lower rank than full head_dim for efficiency)
        self.W_K = nn.Linear(self.d_model, self.n_heads * self.rank, bias=False)

        # Query projection: d_model -> n_heads * rank
        self.W_Q = nn.Linear(self.d_model, self.n_heads * self.rank, bias=False)

        # Value projection: d_model -> n_heads * head_dim
        self.W_V = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)

        # Output projection: n_heads * head_dim -> d_model
        self.W_O = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        # Layer norm (SKA-internal)
        self.layer_norm = nn.LayerNorm(self.d_model)

        # Learnable gate parameter α, initialized so σ(α) = 0.5
        self.gate_alpha = nn.Parameter(torch.zeros(1))

        # Learnable spectral parameters
        self.log_ridge = nn.Parameter(torch.tensor(math.log(config.ridge_eps)))
        self.log_gamma = nn.Parameter(torch.tensor(math.log(config.spectral_gamma)))

    @property
    def ridge(self) -> torch.Tensor:
        return self.log_ridge.exp()

    @property
    def gamma(self) -> torch.Tensor:
        return self.log_gamma.exp()

    def _build_operator(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build the Koopman operator from prefix keys/values.

        Implements Equations 5-8 using triangular solves only:

            G = Σ z_t z_t^T + εI (Gram matrix with ridge)
            M = Σ z_t z_{t-1}^T (Transition matrix)
            LL^T = G (Cholesky factorization)

            A_w = L^{-1} M L^{-T} (Whitened Koopman operator)
                This is a SIMILARITY transform of A_nat = MG^{-1} via L:
                    L A_w L^{-1} = MG^{-1}
                    L A_w^K L^{-1} = (MG^{-1})^K
                Computed as:
                  solve L U = M for U = L^{-1} M (forward solve)
                  solve L V = U^T for V = (A_w)^T (forward solve)
                  A_w = V^T

            B_v = C_v G^{-1} = C_v L^{-T} L^{-1} (Value readout)
                Computed as:
                  solve L Y = C_v^T for Y = L^{-1} C_v^T (forward solve)
                  solve L^T B = Y for B = G^{-1} C_v^T (backward solve)
                  B_v = B^T

        No explicit matrix inverse is ever formed.

        Args:
            keys: (B, H, L, r) prefix key projections
            values: (B, H, L, d_h) prefix value projections

        Returns:
            A_w: (B, H, r, r) whitened operator
            L: (B, H, r, r) Cholesky factor
            B_v: (B, H, d_h, r) value readout matrix
        """
        B, H, L_len, r = keys.shape
        d_h = values.shape[-1]
        device = keys.device
        dtype = keys.dtype

        # Gram matrix: G = K^T K + εI (Eq. 5)
        G = torch.matmul(keys.transpose(-2, -1), keys) # (B, H, r, r)
        G = G + self.ridge * torch.eye(r, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)

        # Transition matrix: M = K[1:]^T K[:-1] (Eq. 6)
        if L_len > 1:
            M = torch.matmul(keys[:, :, 1:, :].transpose(-2, -1), keys[:, :, :-1, :])
        else:
            M = torch.zeros(B, H, r, r, device=device, dtype=dtype)

        # Cholesky factorization (Eq. 7) - in double for stability
        G_double = G.to(torch.float64)
        try:
            L_chol = torch.linalg.cholesky(G_double)
        except torch.linalg.LinAlgError:
            G_double = G_double + 1e-4 * torch.eye(r, device=device, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
            L_chol = torch.linalg.cholesky(G_double)

        # --- Whitened operator via two triangular solves ---
        # We need A_w = L^{-1} M L^{-T}, which is a similarity transform of
        # the natural operator A = G^{-1}M via L:
        # L A_w L^{-1} = L (L^{-1} M L^{-T}) L^{-1} = M L^{-T} L^{-1} = M G^{-1}
        # So the whiten->power->unwhiten pipeline computes:
        # L A_w^K L^{-1} = (M G^{-1})^K
        # which is the correct K-step Koopman propagator.
        #
        # Step 1: solve L U = M -> U = L^{-1} M
        # Step 2: solve L^T V^T = U^T -> V = U L^{-T} = L^{-1} M L^{-T}
        M_double = M.to(torch.float64)
        L_T = L_chol.transpose(-2, -1) # upper triangular
        U = torch.linalg.solve_triangular(L_chol, M_double, upper=False) # L^{-1} M
        V = torch.linalg.solve_triangular(L_chol, U.transpose(-2, -1), upper=False) # L^{-1} U^T
        A_w = V.transpose(-2, -1).to(dtype) # V^T = U L^{-T} = L^{-1} M L^{-T}

        L_chol = L_chol.to(dtype)

        # Spectral normalization (Eq. 10)
        # Clamp gamma ≤ 1 to prevent blowup under power iteration:
        # σ_max(A_w^K) ≤ γ^K, so γ > 1 with K > 1 is unstable.
        gamma_clamped = torch.clamp(self.gamma, max=1.0)
        sigma_max = torch.linalg.norm(A_w, ord=2, dim=(-2, -1), keepdim=True)
        sigma_max = sigma_max.clamp(min=1e-8)
        scale = gamma_clamped / torch.clamp(sigma_max, min=1.0)
        A_w = A_w * scale

        # --- Value readout via triangular solves ---
        # B_v = C_v G^{-1} = C_v (LL^T)^{-1} = C_v L^{-T} L^{-1}
        # Step 1: solve L Y^T = C_v^T -> Y^T = L^{-1} C_v^T, Y = C_v L^{-T}
        # Step 2: solve L^T B_v^T = Y^T -> B_v^T = L^{-T} L^{-1} C_v^T
        # B_v = C_v L^{-T} L^{-1} = C_v G^{-1} ✓
        C_v = torch.matmul(values.transpose(-2, -1), keys) # (B, H, d_h, r)
        C_v_double = C_v.to(torch.float64)
        L_chol_double = L_chol.to(torch.float64)

        Y_T = torch.linalg.solve_triangular(
            L_chol_double, C_v_double.transpose(-2, -1), upper=False
        ) # Y^T = L^{-1} C_v^T, shape (B, H, r, d_h)
        Bv_T = torch.linalg.solve_triangular(
            L_chol_double.transpose(-2, -1), Y_T, upper=True
        ) # B_v^T = L^{-T} L^{-1} C_v^T, shape (B, H, r, d_h)
        B_v = Bv_T.transpose(-2, -1).to(dtype) # (B, H, d_h, r)

        return A_w, L_chol, B_v

    def _spectral_filter(
        self,
        A_w: torch.Tensor,
        L_chol: torch.Tensor,
        B_v: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply power spectral filtering to query positions (Eq. 9).

        For each query z_q:
            w_q = L^{-1} z_q (whiten)
            w_f = A_w^K w_q (power filter)
            ẑ = L w_f (unwhiten)
            ŷ = B_v ẑ (readout)

        Args:
            A_w: (B, H, r, r) whitened operator
            L_chol: (B, H, r, r) Cholesky factor
            B_v: (B, H, d_h, r) value readout
            queries: (B, H, Q, r) query projections

        Returns:
            output: (B, H, Q, d_h) filtered output
        """
        # Whiten queries: w_q = L^{-1} z_q
        # Solve L w_q = z_q for w_q (more numerically stable than explicit inverse)
        # queries: (B, H, Q, r) -> need to transpose for triangular_solve
        Q_len = queries.shape[2]

        # (B, H, r, Q) = solve L X = queries^T
        queries_t = queries.transpose(-2, -1) # (B, H, r, Q)

        # Use double precision for solve
        L_double = L_chol.to(torch.float64)
        queries_double = queries_t.to(torch.float64)
        w_q = torch.linalg.solve_triangular(L_double, queries_double, upper=False)
        w_q = w_q.to(queries.dtype) # (B, H, r, Q)

        # Power filtering: w_f = A_w^K w_q
        w_f = w_q
        for _ in range(self.power_K):
            w_f = torch.matmul(A_w, w_f) # (B, H, r, Q)

        # Unwhiten: ẑ = L w_f
        z_hat = torch.matmul(L_chol, w_f) # (B, H, r, Q)

        # Value readout: ŷ = B_v ẑ
        output = torch.matmul(B_v, z_hat) # (B, H, d_h, Q)

        # Transpose back to (B, H, Q, d_h)
        output = output.transpose(-2, -1)

        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        prefix_len: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of SKA module.

        Args:
            hidden_states: (B, T, d_model) input tensor
            prefix_len: number of prefix positions to use for operator
                        construction. If None, uses first half.
            attention_mask: optional mask (not currently used, kept for
                           interface compatibility)

        Returns:
            output: (B, T, d_model) output tensor
        """
        B, T, D = hidden_states.shape

        # Default: first half is prefix, second half is query
        if prefix_len is None:
            prefix_len = T // 2
        prefix_len = max(1, min(prefix_len, T - 1))

        # Apply layer norm
        normed = self.layer_norm(hidden_states)

        # Project to key/query/value spaces
        keys_all = self.W_K(normed) # (B, T, n_heads * rank)
        queries_all = self.W_Q(normed) # (B, T, n_heads * rank)
        values_all = self.W_V(normed) # (B, T, n_heads * head_dim)

        # Reshape to multi-head format
        keys_all = keys_all.view(B, T, self.n_heads, self.rank).transpose(1, 2)
        queries_all = queries_all.view(B, T, self.n_heads, self.rank).transpose(1, 2)
        values_all = values_all.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # Now: (B, H, T, dim)

        # Split into prefix and query portions
        prefix_keys = keys_all[:, :, :prefix_len, :]
        prefix_values = values_all[:, :, :prefix_len, :]
        query_keys = queries_all[:, :, prefix_len:, :]

        # Build Koopman operator from prefix (Eqs. 5-8)
        A_w, L_chol, B_v = self._build_operator(prefix_keys, prefix_values)

        # Apply spectral filtering to query positions (Eq. 9)
        query_output = self._spectral_filter(A_w, L_chol, B_v, query_keys)
        # query_output: (B, H, Q, d_h)

        # For prefix positions, use direct attention-like readout
        # (they contribute to the operator but also need output)
        prefix_output = self._spectral_filter(A_w, L_chol, B_v, keys_all[:, :, :prefix_len, :])

        # Concatenate prefix and query outputs
        full_output = torch.cat([prefix_output, query_output], dim=2) # (B, H, T, d_h)

        # Reshape back to (B, T, n_heads * head_dim)
        full_output = full_output.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)

        # Output projection
        output = self.W_O(full_output)

        # Apply gate: σ(α) controls mixing
        gate = torch.sigmoid(self.gate_alpha)
        output = gate * output

        return output

    def get_operator_stats(self, hidden_states: torch.Tensor, prefix_len: int) -> dict:
        """
        Compute operator statistics for monitoring (§6.5).

        Returns condition numbers, spectral radii, etc.
        """
        with torch.no_grad():
            normed = self.layer_norm(hidden_states)
            keys = self.W_K(normed).view(
                hidden_states.shape[0], -1, self.n_heads, self.rank
            ).transpose(1, 2)
            values = self.W_V(normed).view(
                hidden_states.shape[0], -1, self.n_heads, self.head_dim
            ).transpose(1, 2)

            prefix_keys = keys[:, :, :prefix_len, :]
            prefix_values = values[:, :, :prefix_len, :]

            A_w, L_chol, _ = self._build_operator(prefix_keys, prefix_values)

            # Condition number of Gram matrix
            G = torch.matmul(prefix_keys.transpose(-2, -1), prefix_keys)
            G = G + self.ridge * torch.eye(self.rank, device=G.device).unsqueeze(0).unsqueeze(0)
            eigvals = torch.linalg.eigvalsh(G)
            cond = eigvals[:, :, -1] / eigvals[:, :, 0].clamp(min=1e-15)

            # Spectral radius of A_w
            spec_radius = torch.linalg.norm(A_w, ord=2, dim=(-2, -1))

            return {
                "condition_number_mean": cond.mean().item(),
                "condition_number_max": cond.max().item(),
                "spectral_radius_mean": spec_radius.mean().item(),
                "spectral_radius_max": spec_radius.max().item(),
                "gate_value": torch.sigmoid(self.gate_alpha).item(),
                "ridge_value": self.ridge.item(),
                "gamma_value": self.gamma.item(),
            }


class MultiHeadKoopmanModule(nn.Module):
    """
    Multi-Head Koopman operators (§7).

    Replaces each SKA head's single operator with K parallel rank-r_k operators,
    each with its own key projection. This gives K^3× cheaper Cholesky solves
    than a single rank-(Kr_k) operator.

    Architecture (Definition 7.1):
        For each attention head h and Koopman index k = 1, ..., K:
            z_t^{(k)} = W_K^{(k)} h_t (Independent key projections)
            G^{(k)} = Σ z_t^{(k)} (z_t^{(k)})^T + εI (Per-operator Gram)
            A_w^{(k)} = (L^{(k)})^{-T} M^{(k)} (L^{(k)})^{-1} (Per-operator Koopman)

        Final output combines via learned gating (Eq. 36):
            ŷ = Σ_k σ(α_k) · ŷ^{(k)}
    """

    def __init__(self, config: SKAConfig, num_operators: int = 4, rank_per_op: int = 48):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.num_operators = num_operators
        self.rank_per_op = rank_per_op
        self.power_K = config.power_K
        self.ridge_eps = config.ridge_eps

        # Independent key projections for each operator (Eq. 32)
        self.W_K_ops = nn.ModuleList([
            nn.Linear(self.d_model, self.n_heads * self.rank_per_op, bias=False)
            for _ in range(num_operators)
        ])

        # Shared query projection (uses operator-specific key to query)
        self.W_Q_ops = nn.ModuleList([
            nn.Linear(self.d_model, self.n_heads * self.rank_per_op, bias=False)
            for _ in range(num_operators)
        ])

        # Shared value projection
        self.W_V = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)

        # Output projection
        self.W_O = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        # Layer norm
        self.layer_norm = nn.LayerNorm(self.d_model)

        # Per-operator gate parameters α_k (Eq. 36)
        self.gate_alphas = nn.Parameter(torch.zeros(num_operators))

        # Overall gate
        self.gate_alpha = nn.Parameter(torch.zeros(1))

        # Learnable ridge and gamma per operator
        self.log_ridges = nn.Parameter(torch.full((num_operators,), math.log(config.ridge_eps)))
        self.log_gammas = nn.Parameter(torch.zeros(num_operators))

    def _build_single_operator(
        self,
        keys: torch.Tensor,
        ridge: torch.Tensor,
        gamma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build operator for one Koopman head via triangular solves."""
        B, H, L, r = keys.shape
        device = keys.device
        dtype = keys.dtype

        # Gram matrix
        G = torch.matmul(keys.transpose(-2, -1), keys)
        G = G + ridge * torch.eye(r, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)

        # Transition matrix
        if L > 1:
            M = torch.matmul(keys[:, :, 1:, :].transpose(-2, -1), keys[:, :, :-1, :])
        else:
            M = torch.zeros(B, H, r, r, device=device, dtype=dtype)

        # Cholesky
        G_double = G.to(torch.float64)
        try:
            L_chol = torch.linalg.cholesky(G_double)
        except torch.linalg.LinAlgError:
            G_double = G_double + 1e-4 * torch.eye(r, device=device, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
            L_chol = torch.linalg.cholesky(G_double)

        # Whitened operator via two triangular solves (similarity transform):
        # A_w = L^{-1} M L^{-T} so that L A_w^K L^{-1} = (M G^{-1})^K
        # Step 1: solve L U = M -> U = L^{-1} M
        # Step 2: solve L^T V^T = U^T -> V = U L^{-T} = L^{-1} M L^{-T}
        M_double = M.to(torch.float64)
        L_T = L_chol.transpose(-2, -1)
        U = torch.linalg.solve_triangular(L_chol, M_double, upper=False) # L^{-1} M
        V = torch.linalg.solve_triangular(L_chol, U.transpose(-2, -1), upper=False) # L^{-1} U^T
        A_w = V.transpose(-2, -1).to(dtype) # L^{-1} M L^{-T}
        L_chol = L_chol.to(dtype)

        # Spectral normalization - clamp gamma ≤ 1 for power-iteration stability
        gamma_clamped = torch.clamp(gamma, max=1.0)
        sigma_max = torch.linalg.norm(A_w, ord=2, dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        scale = gamma_clamped / torch.clamp(sigma_max, min=1.0)
        A_w = A_w * scale

        return A_w, L_chol

    def forward(
        self,
        hidden_states: torch.Tensor,
        prefix_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass with K parallel Koopman operators.

        Args:
            hidden_states: (B, T, d_model)
            prefix_len: prefix length for operator construction

        Returns:
            output: (B, T, d_model)
        """
        B, T, D = hidden_states.shape
        if prefix_len is None:
            prefix_len = T // 2
        prefix_len = max(1, min(prefix_len, T - 1))

        normed = self.layer_norm(hidden_states)

        # Shared values
        values_all = self.W_V(normed).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Accumulate gated outputs from each operator
        accumulated = torch.zeros(
            B, self.n_heads, T, self.head_dim,
            device=hidden_states.device, dtype=hidden_states.dtype
        )

        for k in range(self.num_operators):
            ridge_k = self.log_ridges[k].exp()
            gamma_k = self.log_gammas[k].exp()
            gate_k = torch.sigmoid(self.gate_alphas[k])

            # Independent key/query projections
            keys_k = self.W_K_ops[k](normed).view(B, T, self.n_heads, self.rank_per_op).transpose(1, 2)
            queries_k = self.W_Q_ops[k](normed).view(B, T, self.n_heads, self.rank_per_op).transpose(1, 2)

            # Build operator from prefix
            prefix_keys = keys_k[:, :, :prefix_len, :]
            A_w, L_chol = self._build_single_operator(prefix_keys, ridge_k, gamma_k)

            # Value readout via triangular solves (no explicit inverse):
            # B_v = C_v G^{-1} = C_v L^{-T} L^{-1}
            # Step 1: solve L Y^T = C_v^T -> Y = C_v L^{-T}
            # Step 2: solve L^T B_v^T = Y^T -> B_v = C_v L^{-T} L^{-1}
            prefix_values = values_all[:, :, :prefix_len, :]
            C_v = torch.matmul(prefix_values.transpose(-2, -1), prefix_keys) # (B, H, d_h, r_k)

            L_double = L_chol.to(torch.float64)
            C_v_double = C_v.to(torch.float64)
            Y_T = torch.linalg.solve_triangular(
                L_double, C_v_double.transpose(-2, -1), upper=False
            ) # Y^T = L^{-1} C_v^T, (B, H, r_k, d_h)
            Bv_T = torch.linalg.solve_triangular(
                L_double.transpose(-2, -1), Y_T, upper=True
            ) # B_v^T = L^{-T} L^{-1} C_v^T, (B, H, r_k, d_h)
            B_v = Bv_T.transpose(-2, -1).to(hidden_states.dtype) # (B, H, d_h, r_k)

            # Filter all positions via triangular solve for whitening
            all_queries = queries_k # (B, H, T, r_k)
            q_t = all_queries.transpose(-2, -1) # (B, H, r_k, T)

            # L_double already computed above
            q_double = q_t.to(torch.float64)
            w_q = torch.linalg.solve_triangular(L_double, q_double, upper=False).to(hidden_states.dtype)

            w_f = w_q
            for _ in range(self.power_K):
                w_f = torch.matmul(A_w, w_f)

            z_hat = torch.matmul(L_chol, w_f)
            output_k = torch.matmul(B_v, z_hat).transpose(-2, -1) # (B, H, T, d_h)

            accumulated = accumulated + gate_k * output_k

        # Reshape and project
        accumulated = accumulated.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        output = self.W_O(accumulated)

        # Overall gate
        gate = torch.sigmoid(self.gate_alpha)
        output = gate * output

        return output
