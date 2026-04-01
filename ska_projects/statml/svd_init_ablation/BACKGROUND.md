# Background: Matrix Factorization for Weight Initialization

## 1. Why Initialization Matters

When replacing an attention layer with an SKA module, the initial weights
determine:
- Whether the model produces coherent outputs immediately after surgery
  (important for the `verify_logits()` check)
- How far the optimizer needs to travel to recover pretrained quality
- The condition number of the initial Gram matrix (affects numerical stability)

A good initialization should:
1. Preserve as much of the original attention behavior as possible
2. Produce well-conditioned Gram matrices
3. Distribute energy across the rank-r subspace (not concentrate in one direction)

## 2. SVD Review

The singular value decomposition of A ∈ R^{m×n} is:

    A = U Σ Vᵀ

where U ∈ R^{m×m} is orthogonal, Σ ∈ R^{m×n} is diagonal with
non-negative entries σ₁ ≥ σ₂ ≥ ... ≥ 0, and V ∈ R^{n×n} is orthogonal.

**Truncated SVD:** Keep only the top r singular values/vectors:

    A ≈ U[:, :r] · diag(σ₁, ... σ_r) · V[:, :r]ᵀ

This is the best rank-r approximation in both Frobenius and operator norm
(Eckart-Young theorem).

## 3. The Current SKA Initialization (Algorithm 1)

For each attention head h (d_h × d_model weight matrix):

```
K_h = W_K[h*d_h : (h+1)*d_h, :]     # per-head key weights
U_K, Σ_K, V_Kᵀ = SVD(K_h)
W_K_SKA[h*r : (h+1)*r, :] = diag(√Σ_K[:r]) · V_K[:r, :]
```

The sqrt-scaling is motivated by symmetry: if you think of the key-query
inner product as z_K · z_Q = (√Σ V x) · (√Σ V x) = x V Σ Vᵀ x, then
the full singular value product is split equally between keys and queries.

## 4. Alternative Strategies to Implement

### Random Orthogonal
```python
W = np.random.randn(n_heads * rank, d_model)
Q, _ = np.linalg.qr(W.T)
W_K_ska = Q[:, :n_heads * rank].T
```
Preserves norms but has no relationship to original weights.

### SVD with Full-Σ Scaling
```python
W_K_SKA[h*r:(h+1)*r, :] = diag(Σ_K[:r]) · V_K[:r, :]
```
Concentrates all singular value energy in keys (queries get just V).

### SVD with No Scaling
```python
W_K_SKA[h*r:(h+1)*r, :] = V_K[:r, :]
```
Purely directional, the r principal directions of K_h.

### PCA-Based
```python
K_centered = K_h - K_h.mean(axis=0)
U, Σ, Vᵀ = SVD(K_centered)
# ... same extraction but on centered data
```
PCA = SVD of centered data. For weight matrices (which aren't "data"),
centering may or may not help, this is an empirical question.

## 5. Measuring Reconstruction Error

Given original per-head key matrix K_h ∈ R^{d_h × d_model} and
SKA key matrix W_K_ska_h ∈ R^{r × d_model}, the reconstruction
error is:

    err = ||K_h - expand(W_K_ska_h)||_F / ||K_h||_F

where expand maps from rank r back to d_h dimensions. For SVD-based
methods, the Eckart-Young theorem tells us this equals:

    err = √(Σ σ_i² for i > r) / √(Σ σ_i² for all i)

For non-SVD methods, this can be larger.

## 6. References

1. Eckart & Young. "The Approximation of One Matrix by Another of Lower
   Rank." Psychometrika, 1936.
2. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR, 2022. Uses SVD-like initialization for low-rank adapters. Directly relevant.
3. Li et al. "Measuring the Intrinsic Dimension of Objective Landscapes."
   ICLR, 2018. Motivates why low-rank projections can capture most of
   the variation in weight matrices.
