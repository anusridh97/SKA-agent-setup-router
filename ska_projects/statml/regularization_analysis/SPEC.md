# Project: Spectral and Orthogonal Regularization Interaction

## Summary

Empirically characterize how the spectral regularization (λ_spec) and
orthogonal regularization (λ_ortho) interact during SKA training.
Map the (λ_spec, λ_ortho) landscape and identify Pareto-optimal settings.

**Area:** Statistical ML
**GPU:** ~2 hours (small-scale SKA modules, short training runs)
**Duration:** 8 weeks
**Team size:** 1

## Motivation

The codebase defines two regularization losses in `training/trainers.py`:

**Spectral regularization** (§8.1.1):
    L_spec = λ_spec · Σ_i σ_i(A_w)²

Penalizes large singular values of the Koopman operator. Keeps the
operator "tame", prevents blowup under power iteration.

**Orthogonal regularization** (§8.1.1):
    L_ortho = λ_ortho · ||WᵀW - I||_F²

Encourages projection matrices (W_K, W_Q) to preserve distances.
Prevents collapse of the key/query space into a low-dimensional subspace.

These serve different purposes but interact: spectral reg pushes A_w
toward smaller singular values, while orthogonal reg pushes projections
toward isometry. The combined effect on operator quality (condition
number, retrieval accuracy) hasn't been studied.

## Deliverables

1. **Grid sweep**: Train small SKA modules (d_model=256, rank=16, n_heads=4)
   for 2000 steps across a grid of (λ_spec, λ_ortho) values:
   λ_spec ∈ {0, 0.001, 0.01, 0.1, 1.0}
   λ_ortho ∈ {0, 0.001, 0.01, 0.1, 1.0}
   That's 25 configurations.

2. **Metrics at each configuration** (measured every 100 steps):
   - Training loss (LM loss + reg losses)
   - Gram condition number κ(G)
   - Spectral radius ρ(A_w)
   - Projection orthogonality ||W_K^T W_K - I||_F
   - Gate value σ(α)
   - Retrieval precision on synthetic retrieval task

3. **Visualization**:
   - Heatmaps of final metrics over the (λ_spec, λ_ortho) grid
   - Training curves for selected configurations (corners + best)
   - Pareto frontier plot: quality vs. condition number

4. **Recommendations**: Which (λ_spec, λ_ortho) to use as defaults.

## Where This Fits

```
ska_agent/training/trainers.py
    SpectralRegularization.forward()   ← Builds A_w, computes SVD, penalizes
    OrthogonalRegularization.forward() ← Computes WᵀW - I penalty
    SKATrainer.__init__()              ← Creates both reg objects
    SKATrainer.train_lm_recovery()     ← Adds reg losses to total loss

ska_agent/core/ska_module.py
    SKAModule.get_operator_stats()     ← Computes condition number, spectral radius
```

## Background

### Why Spectral Regularization?

The power filter applies A_w^K to queries. If σ_max(A_w) > 1, then
||A_w^K|| grows exponentially with K. Even with spectral normalization
(which clamps σ_max ≤ γ ≤ 1 at inference time), during training the
gradients through A_w^K can explode if A_w has large singular values
before normalization.

L_spec = λ · Σ σ_i² directly penalizes the squared Frobenius norm of A_w
(since ||A||_F² = Σ σ_i²). This is a softer version of spectral
normalization, it encourages small singular values everywhere, not just
clamping the largest one.

### Why Orthogonal Regularization?

If W_K collapses (all rows become nearly parallel), the Gram matrix
G = K^T K becomes ill-conditioned (one large eigenvalue, rest near zero).
This makes the Cholesky factorization unstable and the operator A_w
degenerate. Orthogonal regularization prevents this by pushing W_K toward
an isometry.

### The Interaction

Consider what happens when both are strong:
- Strong λ_ortho → W_K ≈ partial isometry → keys have unit norm,
  G ≈ L·I + εI, well-conditioned
- Strong λ_spec → σ_i(A_w) all small → A_w ≈ 0 → power filter does nothing

If A_w ≈ 0, the SKA module becomes a no-op (gate starts at 0.5, but the
signal through A_w^K vanishes). So there's a tension: you want A_w to have
non-trivial spectral structure (to be a useful filter) but not too large
(for stability). The Pareto frontier captures this tradeoff.

## Starter Experiment

```python
import torch
import torch.nn as nn
import numpy as np
from ska_agent.core.structures import SKAConfig
from ska_agent.core.ska_module import SKAModule

# Small-scale SKA for fast experiments
config = SKAConfig(d_model=256, n_heads=4, head_dim=64, rank=16,
                   power_K=2, ridge_eps=1e-3)
ska = SKAModule(config)

# Random input
B, T = 4, 128
x = torch.randn(B, T, 256)
prefix_len = 64

# Forward pass
out = ska(x, prefix_len=prefix_len)
print(f"Output shape: {out.shape}")

# Get operator stats
stats = ska.get_operator_stats(x, prefix_len)
print(f"Condition number: {stats['condition_number_mean']:.1f}")
print(f"Spectral radius: {stats['spectral_radius_mean']:.4f}")
print(f"Gate: {stats['gate_value']:.4f}")

# Compute reg losses
from ska_agent.training.trainers import SpectralRegularization, OrthogonalRegularization
spec_reg = SpectralRegularization(lambda_spec=0.01)
ortho_reg = OrthogonalRegularization(lambda_ortho=0.01)

modules = nn.ModuleDict({"test": ska})
l_spec = spec_reg(modules, x, prefix_len)
l_ortho = ortho_reg(modules)
print(f"Spectral loss: {l_spec.item():.4f}")
print(f"Orthogonal loss: {l_ortho.item():.4f}")
```

Run this first to verify everything works before launching the grid sweep.

## Grading

| Component | Weight |
|-----------|--------|
| Grid sweep runs cleanly (25 configs) | 20% |
| Metrics computed correctly at each checkpoint | 20% |
| Heatmap visualizations | 20% |
| Pareto frontier analysis | 20% |
| Clear recommendations with justification | 20% |
