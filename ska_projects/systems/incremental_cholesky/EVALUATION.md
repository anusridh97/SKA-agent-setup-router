# Evaluation: Incremental Cholesky Operator Updates

## Grading Breakdown

| Component | Weight | Description |
|-----------|--------|-------------|
| Correctness | 30% | Incremental matches from-scratch to floating-point tolerance |
| Implementation quality | 25% | Clean code, good abstractions, handles edge cases |
| Benchmarks | 20% | Rigorous timing at multiple ranks, roofline analysis |
| Stability analysis | 15% | Drift measurement, refactorization scheduling |
| Integration | 10% | Drop-in replacement works with SharedSpectralMemory |

## What "Done" Looks Like

### A-level work
- Rank-1 Cholesky update passes all correctness tests
- Full incremental A_w update via Givens rotation tracking works
- Benchmarks show measurable speedup at r ≥ 128
- Numerical stability characterized over 100K+ steps
- Refactorization schedule justified empirically
- Integrated into SharedSpectralMemory with config flag
- Clear writeup explaining the math and design decisions

### B-level work
- Rank-1 Cholesky update works correctly
- A_w update works but may use O(r³) triangular solves instead of
  full Givens-based O(r²) path
- Benchmarks present but limited to one or two ranks
- Integration present but minimal testing

### C-level work
- Cholesky update implemented but correctness issues at scale
- No A_w update or only from-scratch fallback
- Benchmarks incomplete

## Common Failure Modes

1. **Not accounting for L' ≠ L in the A_w update.** The most common
   mistake. If you just update M and recompute A_w = L⁻¹ M L⁻ᵀ using
   the *old* L, you get the wrong answer.

2. **Not handling the ridge regularization.** G starts as εI, not zero.
   The initial L = √ε · I. Make sure your first update handles this.

3. **Forgetting spectral normalization.** After updating A_w, you must
   re-normalize. The spectral radius can drift above γ after an update.

4. **Testing with too-large keys.** If ||z|| >> 1, the rank-1 update
   can be poorly conditioned. Use ||z|| ~ 0.1 for initial testing.
