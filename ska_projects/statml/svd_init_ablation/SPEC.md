# Project: SVD Initialization Ablation for SKA Surgery

## Summary

Empirically study the SVD-based weight initialization used in
`jamba_ska.svd_init_ska_weights()`. Compare against random initialization,
PCA-based alternatives, and truncated SVD variants. Measure the effect on
operator condition numbers, convergence speed, and downstream quality.

**Area:** Statistical ML
**GPU:** ~1 hour (one-time weight extraction from a small model)
**Duration:** 8 weeks
**Team size:** 1–2

## Motivation

The current SKA surgery (Algorithm 1 in the codebase) initializes each
head's key/query projections via:

    U, Σ, Vᵀ = SVD(W_K_head)
    W_K_SKA = diag(√Σ[:r]) · V[:r, :]

The sqrt-singular-value scaling distributes energy symmetrically between
keys and queries. But this is one of many possible warm-start strategies,
and it hasn't been ablated against alternatives. Questions:

1. How much does SVD init help vs. random init? (Convergence speed)
2. Is sqrt-scaling optimal, or would full-Σ or no-Σ be better?
3. Does the init quality depend on the rank r?
4. Does PCA (which differs from SVD for non-centered data) behave differently?

## Deliverables

1. **Weight extraction script**: Load a model (use a small one like
   Qwen2.5-1.5B or any model with GQA), extract attention weights,
   save to disk. One-time GPU use.

2. **Initialization strategies** (all CPU, operating on saved weights):
   - Random orthogonal (baseline)
   - Random Gaussian (scaled)
   - SVD with sqrt-Σ scaling (current)
   - SVD with full-Σ scaling
   - SVD with no scaling (just V[:r, :])
   - PCA (center data first, then SVD)
   - NMF (non-negative matrix factorization, for comparison)

3. **Analysis metrics** (all computed on the initialized SKA module, no training):
   - Condition number of the resulting Gram matrix G
   - Spectral gap of A_w (ratio of first to second singular value)
   - Reconstruction error: ||W_K_original - W_K_SKA_expanded||_F
   - Frobenius distance to the original attention output on test inputs

4. **Training curves** (requires small GPU budget):
   - Initialize SKA modules with each strategy
   - Train for 1000 steps on a small text dataset
   - Measure loss convergence speed and final perplexity

## Where This Fits

```
ska_agent/models/jamba_ska.py
    svd_init_ska_weights()      ← The function you're studying
    initialize_ska_from_gqa()   ← Wrapper that calls SVD init
    repeat_kv()                 ← KV head expansion (prerequisite)

ska_agent/core/structures.py
    SKAConfig                   ← rank, n_heads, head_dim parameters
```

## Key Experiment: Gram Condition vs. Init Strategy

```python
import numpy as np
from ska_agent.utils.math_utils import SpectralUtils

def analyze_init_strategy(W_K_ska, rank, n_heads):
    """
    Given initialized SKA key weights, analyze operator quality.
    """
    # Simulate a batch of random inputs
    np.random.seed(42)
    batch_size = 32
    d_model = W_K_ska.shape[1]
    x = np.random.randn(batch_size, d_model) * 0.1

    # Project to keys
    keys = x @ W_K_ska.T  # (batch_size, n_heads * rank)
    keys = keys.reshape(batch_size, n_heads, rank)

    # Build Gram for each head
    conditions = []
    for h in range(n_heads):
        k_h = keys[:, h, :]  # (batch_size, rank)
        G = SpectralUtils.build_gram_matrix(k_h, ridge_eps=1e-3)
        cond = SpectralUtils.condition_number(G)
        conditions.append(cond)

    return {
        'mean_condition': np.mean(conditions),
        'max_condition': np.max(conditions),
        'std_condition': np.std(conditions),
    }
```

## Grading

| Component | Weight |
|-----------|--------|
| Extraction script works | 10% |
| All 7 init strategies implemented | 25% |
| Analysis metrics computed and plotted | 25% |
| Training convergence comparison | 25% |
| Writeup with clear conclusions | 15% |
