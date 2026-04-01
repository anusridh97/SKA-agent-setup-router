# Background: Regularization in Spectral Operator Learning

## 1. The Training Objective

The total loss for SKA training is:

    L = L_LM + λ_spec · L_spec + λ_ortho · L_ortho

where L_LM is the language modeling loss (cross-entropy on next token).

## 2. Spectral Regularization

L_spec = Σ_i σ_i(A_w)² = ||A_w||_F²

This is the squared Frobenius norm of the Koopman operator. It's
differentiable (computed via SVD in the forward pass) and penalizes
operators with large singular values.

**Effect on training:** Pushes A_w toward zero. If λ_spec is too large,
A_w ≈ 0 and the SKA module becomes a no-op (the power filter A_w^K
produces near-zero output).

## 3. Orthogonal Regularization

L_ortho = ||W_K^T W_K - I||_F² + ||W_Q^T W_Q - I||_F²

Penalizes deviation from isometry. If W_K is exactly orthogonal, then
keys have unit norm and the Gram matrix G is well-conditioned.

**Effect on training:** Prevents projection collapse. If λ_ortho is too
large, the projections are constrained to be nearly orthogonal, which
limits their representational capacity.

## 4. The Interaction Space

Think of the (λ_spec, λ_ortho) space as a 2D landscape:

- (0, 0): No regularization. Risk of ill-conditioned operators.
- (high, 0): Operators damped toward zero. Stable but weak.
- (0, high): Projections orthogonal but operator unconstrained.
- (high, high): Everything constrained. Very stable but underfitting.

The interesting region is the "Pareto frontier" where you can't improve
stability without sacrificing quality or vice versa.

## 5. References

1. Bhojanapalli et al. "Orthogonal Regularization." NeurIPS 2016.
2. Miyato et al. "Spectral Normalization for GANs." ICLR 2018.
