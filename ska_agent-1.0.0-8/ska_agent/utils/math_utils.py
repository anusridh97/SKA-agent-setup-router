"""
Mathematical utilities used throughout the SKA-Agent system.

This file provides three classes:

MathUtils: Efficient computation via prefix sums.
  Used by core/geometry.py for O(1) segment centroid and cost queries.
  Prefix sums enable the dynamic programming to run in O(N*K) instead
  of O(N^2).

  Key functions:
    compute_prefix_sums(embeddings) -> for O(1) centroid queries
    compute_pairwise_distances(embeddings) -> cosine distance between
      consecutive sentences (large distance = likely topic boundary)
    segment_internal_cost(start, end, prefix_dist) -> O(1) cost query
    orthogonal_projection(residual, segment_vector) -> removes the
      component of residual explained by segment_vector, used by
      core/pricing.py after each segment selection

SpectralUtils: Koopman operator construction and spectral operations.
  Used by core/ska_module.py, shared_memory/spectral_memory.py, and
  shared_memory/think_koopman_bridge.py.

  The math for building and using the operator:
    build_gram_matrix: G = Z^T Z + epsilon * I (positive definite)
    build_transition_matrix: M = Z[1:]^T Z[:-1] (temporal dynamics)
    cholesky_factor: LL^T = G
    whiten_operator: A_w = L^{-1} M L^{-T} via two forward triangular
      solves (scipy.linalg.solve_triangular). This is a similarity
      transform of the natural operator A = G^{-1}M, meaning
      L A_w L^{-1} = M G^{-1}, so the whiten-power-unwhiten pipeline
      correctly computes (MG^{-1})^K.
    spectral_normalize: clamp spectral radius to gamma (clamped <= 1)
    power_filter: A_w^K * w (K matrix-vector multiplies)
    build_value_readout: B_v = C_v G^{-1} via two triangular solves
      (forward then backward against L and L^T)
    condition_number: sigma_max / sigma_min for monitoring stability

TextPreprocessor: Sentence splitting with NLTK fallback.
  Used by the offline pipeline (pipeline.py) to split raw text before
  embedding and segmentation.

No explicit matrix inverses are formed anywhere in this file. All
G^{-1} and L^{-1} operations use scipy.linalg.solve_triangular.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List

try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            pass # Network may not be available
except ImportError:
    nltk = None


class MathUtils:
    """
    Efficient computation utilities using prefix sums.

    Many quantities (variance, pairwise distance sums) can be computed in O(1)
    per query if we precompute prefix sums in O(n). This makes the DP
    tractable for large corpora.
    """

    @staticmethod
    def compute_prefix_sums(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Precompute prefix sums for O(1) centroid and variance queries.

        For segment [i, j), the centroid is:
            mean = (prefix_sum[j] - prefix_sum[i]) / (j - i)
        """
        N, D = embeddings.shape
        prefix_sum = np.zeros((N + 1, D), dtype=np.float64)
        prefix_sum[1:] = np.cumsum(embeddings, axis=0)

        sq_norms = np.sum(embeddings ** 2, axis=1)
        prefix_sq_sum = np.zeros(N + 1, dtype=np.float64)
        prefix_sq_sum[1:] = np.cumsum(sq_norms)

        return prefix_sum, prefix_sq_sum

    @staticmethod
    def compute_pairwise_distances(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cosine distances between consecutive sentences.

        Large distance between sentence i and i+1 suggests a topic boundary.
        """
        N = embeddings.shape[0]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        X_norm = embeddings / norms

        distances = np.zeros(N)
        for i in range(N - 1):
            cos_sim = np.dot(X_norm[i], X_norm[i + 1])
            distances[i] = 1.0 - cos_sim

        prefix_dist = np.zeros(N + 1)
        prefix_dist[1:] = np.cumsum(distances)

        return distances, prefix_dist

    @staticmethod
    def segment_internal_cost(start: int, end: int, prefix_dist: np.ndarray) -> float:
        """Cost of segment [start, end) = sum of internal pairwise distances."""
        if end <= start + 1:
            return 0.0
        return prefix_dist[end - 1] - prefix_dist[start]

    @staticmethod
    def segment_centroid(start: int, end: int, prefix_sum: np.ndarray) -> np.ndarray:
        """Compute segment centroid in O(1) using prefix sums."""
        n = end - start
        if n <= 0:
            raise ValueError("Empty segment")
        vec_sum = prefix_sum[end] - prefix_sum[start]
        return vec_sum / n

    @staticmethod
    def orthogonal_projection(residual: np.ndarray, segment_vector: np.ndarray) -> np.ndarray:
        """
        Update residual by removing the component explained by segment_vector.

        Schur complement update:
            E_new = E_old - proj(E_old, segment)
        """
        norm_sq = np.dot(segment_vector, segment_vector)
        if norm_sq < 1e-10:
            return residual.copy()
        projection = (np.dot(residual, segment_vector) / norm_sq) * segment_vector
        return residual - projection


class SpectralUtils:
    """
    Spectral operations for SKA operators.

    Handles Cholesky factorization, whitening, spectral normalization,
    and condition number monitoring.
    """

    @staticmethod
    def build_gram_matrix(keys: np.ndarray, ridge_eps: float = 1e-3) -> np.ndarray:
        """
        Build Gram matrix G = Σ z_t z_t^T + εI_r (Eq. 5).

        Args:
            keys: (L, r) array of key projections z_t
            ridge_eps: ridge regularization ε

        Returns:
            G: (r, r) positive definite Gram matrix
        """
        r = keys.shape[1]
        G = keys.T @ keys + ridge_eps * np.eye(r)
        return G

    @staticmethod
    def build_transition_matrix(keys: np.ndarray) -> np.ndarray:
        """
        Build transition matrix M = Σ_{t=2}^L z_t z_{t-1}^T (Eq. 6).

        Args:
            keys: (L, r) array of key projections

        Returns:
            M: (r, r) transition matrix
        """
        # z_t z_{t-1}^T for t = 1, ..., L-1
        return keys[1:].T @ keys[:-1]

    @staticmethod
    def cholesky_factor(G: np.ndarray) -> np.ndarray:
        """
        Compute Cholesky factorization LL^T = G (Eq. 7).

        Returns lower triangular L.
        """
        return np.linalg.cholesky(G)

    @staticmethod
    def whiten_operator(L: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Compute whitened Koopman operator A_w = L^{-1} M L^{-T}.

        This is a similarity transform of the natural operator A = G^{-1}M
        via L, so that:
            L A_w L^{-1} = M G^{-1}
        and the whiten->power->unwhiten pipeline correctly computes:
            L A_w^K L^{-1} = (M G^{-1})^K

        Step 1: solve L U = M -> U = L^{-1} M
        Step 2: solve L^T V^T = U^T -> V = U L^{-T} = L^{-1} M L^{-T}

        Args:
            L: Lower-triangular Cholesky factor of Gram matrix
            M: Transition matrix

        Returns:
            A_w: Whitened operator (r × r)
        """
        from scipy.linalg import solve_triangular
        U = solve_triangular(L, M, lower=True) # L^{-1} M
        V = solve_triangular(L, U.T, lower=True) # L^{-1} (L^{-1} M)^T = (L^{-1} M L^{-T})^T
        return V.T # L^{-1} M L^{-T}

    @staticmethod
    def spectral_normalize(A_w: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Spectral normalization (Eq. 10):
            A_w ← γ · A_w / max(1, σ_max(A_w))

        Gamma is clamped to ≤ 1 to ensure stability under power iteration:
        σ_max(A_w^K) ≤ γ^K, so γ > 1 causes spectral blowup.
        """
        gamma = min(gamma, 1.0)
        sigma_max = np.linalg.norm(A_w, ord=2)
        if sigma_max > 1.0:
            A_w = gamma * A_w / sigma_max
        else:
            A_w = gamma * A_w
        return A_w

    @staticmethod
    def power_filter(A_w: np.ndarray, w: np.ndarray, K: int) -> np.ndarray:
        """
        Apply power spectral filtering: w_f = A_w^K · w (Eq. 9).

        Args:
            A_w: Whitened Koopman operator (r × r)
            w: Whitened query vector (r,)
            K: Number of power iterations

        Returns:
            w_f: Filtered vector (r,)
        """
        result = w.copy()
        for _ in range(K):
            result = A_w @ result
        return result

    @staticmethod
    def condition_number(G: np.ndarray) -> float:
        """
        Compute condition number κ(G) = σ_max / σ_min (Eq. 31).

        Alert threshold: κ > 10^4.
        """
        eigenvalues = np.linalg.eigvalsh(G)
        sigma_max = np.max(np.abs(eigenvalues))
        sigma_min = np.min(np.abs(eigenvalues))
        if sigma_min < 1e-15:
            return float('inf')
        return sigma_max / sigma_min

    @staticmethod
    def build_value_readout(
        values: np.ndarray,
        keys: np.ndarray,
        L_or_G_inv: np.ndarray,
        use_cholesky: bool = True,
    ) -> np.ndarray:
        """
        Build value readout matrix B_v = C_v G^{-1} (Eq. 9)
        via triangular solves when a Cholesky factor is provided.

        C_v = Σ_t v_t z_t^T

        B_v = C_v G^{-1} = C_v (LL^T)^{-1} = C_v L^{-T} L^{-1}
            Step 1: solve L Y = C_v^T -> Y = L^{-1} C_v^T (forward)
            Step 2: solve L^T B = Y -> B = L^{-T} Y
                                             = L^{-T} L^{-1} C_v^T
                                             = G^{-1} C_v^T (backward)
            B_v = B^T = C_v G^{-1}

        Note: the solve order matters. L^{-1} L^{-T} ≠ L^{-T} L^{-1} = G^{-1}.

        Args:
            values: (L_seq, d_v) value projections
            keys: (L_seq, r) key projections
            L_or_G_inv: (r, r) Cholesky factor L if use_cholesky=True,
                        or precomputed G^{-1} if use_cholesky=False
            use_cholesky: if True, treat L_or_G_inv as Cholesky factor

        Returns:
            B_v: (d_v, r) readout matrix
        """
        C_v = values.T @ keys # (d_v, r)
        if not use_cholesky:
            return C_v @ L_or_G_inv

        from scipy.linalg import solve_triangular
        L = L_or_G_inv
        # B_v = C_v L^{-T} L^{-1}
        # Step 1: solve L Y^T = C_v^T -> Y = C_v L^{-T}
        Y_T = solve_triangular(L, C_v.T, lower=True) # (r, d_v)
        # Step 2: solve L^T B_v^T = Y^T -> B_v = C_v L^{-T} L^{-1}
        Bv_T = solve_triangular(L.T, Y_T, lower=False) # (r, d_v)
        return Bv_T.T # (d_v, r)


class TextPreprocessor:
    """Split raw text into sentences."""

    def split_sentences(self, text: str, min_length: int = 15) -> List[str]:
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
        except (ImportError, LookupError):
            # Fallback: split on sentence-ending punctuation
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) >= min_length]
