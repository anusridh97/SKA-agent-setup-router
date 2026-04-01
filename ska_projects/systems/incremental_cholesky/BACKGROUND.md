# Background: Incremental Cholesky and Operator Updates

This document is self-contained. Read it before touching the codebase.

## 1. The Cholesky Factorization

Given a symmetric positive definite (SPD) matrix G ∈ R^{r×r}, the
Cholesky factorization is:

    G = L Lᵀ

where L is lower triangular with positive diagonal entries. This exists
and is unique for any SPD matrix.

**Cost:** O(r³/3) flops, about half the cost of a general LU.

**Why we use it:** The SKA operator construction needs to solve systems
of the form Gx = b. With Cholesky, this is two triangular solves:

    Ly = b     (forward substitution, O(r²))
    Lᵀx = y   (back substitution, O(r²))

We never form G⁻¹ explicitly. This is numerically superior and the
codebase is consistent about this, search for `solve_triangular` and
you'll see this pattern everywhere.

## 2. Rank-1 Cholesky Updates

### The Problem

Given L such that LLᵀ = G, and a vector z ∈ R^r, compute L' such that:

    L' L'ᵀ = G + z zᵀ = G'

without computing the full Cholesky of G' from scratch.

### The Algorithm (Gill-Golub-Murray-Saunders)

The idea: absorb the rank-1 perturbation into L using a sequence of
Givens rotations.

```
Algorithm: CHOLESKY_RANK1_UPDATE(L, z)
Input:  L ∈ R^{r×r} lower triangular (Cholesky factor of G)
        z ∈ R^r (the update vector)
Output: L' ∈ R^{r×r} (Cholesky factor of G + zzᵀ)
        rotations: list of (i, c, s) Givens parameters

1.  w ← z              // working copy
2.  rotations ← []
3.  for i = 0, 1, ... r-1:
4.      // Givens rotation to zero out w[i]
5.      a ← L[i, i]
6.      b ← w[i]
7.      ρ ← sqrt(a² + b²)
8.      c ← a / ρ       // cosine
9.      s ← b / ρ       // sine
10.     L'[i, i] ← ρ
11.     rotations.append((i, c, s))
12.
13.     // Update remaining elements in row i of L and w
14.     for j = i+1, ... r-1:
15.         L'[j, i] ← c * L[j, i] + s * w[j]
16.         w[j]     ← -s * L[j, i] + c * w[j]
17.
18. return L', rotations
```

**Cost:** O(r²), one pass through the lower triangle.

**Key property:** The rotations list encodes an orthogonal matrix Q
(product of Givens rotations) such that L' = L Q in a certain sense.
More precisely, if we define the full orthogonal transformation, it
satisfies L' = the result of applying the rotations to the columns of
[L | z]. We will need these rotations for the operator update.

### Correctness Proof Sketch

At each step i, we have the invariant:

    (partial L')[:i,:i] @ (partial L')[:i,:i]ᵀ = G[:i,:i] + z[:i] z[:i]ᵀ

The Givens rotation (c, s) at step i is chosen to make the (i,i)
element of L' satisfy:

    L'[i,i]² = L[i,i]² + w[i]²

which accounts for the diagonal contribution of the rank-1 update.
The off-diagonal updates propagate the remaining perturbation downward
through w, maintaining the invariant at each step.

### Rank-1 Downdate

For completeness: if you need L' such that L'L'ᵀ = G - zzᵀ, the
algorithm is similar but uses hyperbolic rotations instead of Givens.
This is numerically trickier (can fail if G - zzᵀ is not SPD) and
we don't need it for this project, our Gram matrix only accumulates
(G ← G + z zᵀ), never subtracts.

## 3. Givens Rotations

A Givens rotation G(i, j, θ) is an identity matrix with four modified
entries:

    G[i,i] = cos θ = c
    G[j,j] = cos θ = c
    G[i,j] = sin θ = s
    G[j,i] = -sin θ = -s

Multiplying x' = G(i,j,θ) x rotates the vector x in the (i,j) plane
by angle θ. It changes only components i and j:

    x'[i] =  c * x[i] + s * x[j]
    x'[j] = -s * x[i] + c * x[j]

**Key properties:**
- Orthogonal: Gᵀ G = I, so G⁻¹ = Gᵀ
- Applying G to a matrix (left or right multiply) costs O(r) per
  rotation (only two rows/columns change)
- A product of k Givens rotations can be applied to an r×r matrix
  in O(kr) time

In the Cholesky update, we produce r Givens rotations (one per diagonal
entry), so applying the full product Q = G₁ G₂ ... Gᵣ to a matrix
costs O(r²).

## 4. The SKA Operator Construction

Here's the pipeline from the codebase, written in math notation:

### Inputs
- Keys z₁, z₂, ... z_L ∈ R^r (from prefix of length L)
- Values v₁, v₂, ... v_L ∈ R^{d_v}

### Step 1: Gram Matrix (Eq. 5)

    G = Σ_{t=1}^{L} zₜ zₜᵀ + ε I_r

This is SPD by construction (sum of PSD matrices + positive diagonal).

### Step 2: Transition Matrix (Eq. 6)

    M = Σ_{t=2}^{L} zₜ z_{t-1}ᵀ

This is NOT symmetric in general.

### Step 3: Cholesky Factor (Eq. 7)

    G = L Lᵀ

### Step 4: Whitened Operator (Eq. 8)

    A_w = L⁻¹ M L⁻ᵀ

This is a **similarity transform** of the natural operator A_nat = M G⁻¹.
The relationship is:

    A_nat = L A_w L⁻¹

So A_w and A_nat have the same eigenvalues. The whitened form is
preferred because the power filter (step 6) is numerically stable in
the whitened basis.

**Computation via triangular solves (no explicit inverse):**

    Solve L U = M       for U = L⁻¹ M       (forward substitution)
    Solve L V = Uᵀ      for V = L⁻¹ (L⁻¹ M)ᵀ
    A_w = Vᵀ                                  (= L⁻¹ M L⁻ᵀ)

### Step 5: Spectral Normalization (Eq. 10)

    A_w ← γ · A_w / max(1, σ_max(A_w))

where γ ≤ 1 and σ_max is the largest singular value (operator 2-norm).
This ensures ||A_w||₂ ≤ γ ≤ 1, which guarantees that A_w^K → 0 as
K → ∞ (stability under power iteration).

### Step 6: Value Readout (Eq. 9)

    C_v = Σ_{t=1}^{L} vₜ zₜᵀ     (value-key cross-correlation)
    B_v = C_v G⁻¹                  (readout matrix)

Computed via:

    Solve L Yᵀ = C_vᵀ    for Y = C_v L⁻ᵀ
    Solve Lᵀ B_vᵀ = Yᵀ   for B_v = C_v L⁻ᵀ L⁻¹ = C_v G⁻¹

### Step 7: Query (at read time)

For a query z_q:

    w_q = L⁻¹ z_q              (whiten)
    w_f = A_w^K w_q             (power filter, K iterations)
    ẑ = L w_f                   (unwhiten)
    ŷ = B_v ẑ                   (readout)

## 5. The Incremental Update Derivation

Now we derive the O(r²) incremental pipeline. At time t, a new key
zₜ arrives (and optionally a new value vₜ and the previous key z_{t-1}).

### Updates to Accumulated Matrices

    G' = G + zₜ zₜᵀ                    (rank-1 PSD update)
    M' = M + zₜ z_{t-1}ᵀ               (rank-1 update)
    C_v' = C_v + vₜ zₜᵀ                (rank-1 update if values present)

### Step A: Update L → L'

Use the rank-1 Cholesky update (Section 2). This gives us:
- L' such that L' L'ᵀ = G'  (exact)
- The Givens rotations Q₁, Q₂, ... Qᵣ

**Cost:** O(r²)

### Step B: Update A_w → A_w'

We want A_w' = L'⁻¹ M' L'⁻ᵀ. Expand:

    A_w' = L'⁻¹ (M + zₜ z_{t-1}ᵀ) L'⁻ᵀ
         = L'⁻¹ M L'⁻ᵀ + (L'⁻¹ zₜ)(L'⁻¹ z_{t-1})ᵀ

The second term is a rank-1 matrix computable in O(r²): two forward
solves with L' (each O(r²)) plus an outer product (O(r²)).

For the first term, L'⁻¹ M L'⁻ᵀ, we use the relationship between
L and L'. The Cholesky rank-1 update produces L' from L via a known
sequence of Givens rotations. Abstractly, there exists an orthogonal
matrix Q (the product of all Givens rotations) such that the update
can be expressed as a similarity transform:

    L'⁻¹ M L'⁻ᵀ = Qᵀ (L⁻¹ M L⁻ᵀ) Q = Qᵀ A_w^{old-M} Q

Wait, this isn't quite right, because L' ≠ L Q in general (the Givens
rotations act on a combined [L | z] structure, not just L).

**Correct approach:** Don't try to relate L'⁻¹ to L⁻¹ algebraically.
Instead, compute L'⁻¹ M L'⁻ᵀ directly using the *new* L', which we
already have from Step A. This requires two triangular solves with L'
(same as from-scratch), which is O(r² · r) = O(r³).

**This means the A_w update is O(r³) in the worst case.**

However, there's an important practical optimization: if we maintain
L⁻¹ M as a cached matrix (call it U = L⁻¹ M), then after the rank-1
Cholesky update, we can update U incrementally:

    U' = L'⁻¹ M' = L'⁻¹ (M + z wᵀ)

We can compute L'⁻¹ M by applying the inverse Givens rotations to U:

    L'⁻¹ = (sequence of Givens rotations applied to L⁻¹)

Each Givens rotation modifies two rows of U, costing O(r) per rotation,
and we have r rotations, so updating U → U' via Givens is O(r²), plus
the rank-1 correction L'⁻¹ z wᵀ which is O(r²).

Then A_w' = U' L'⁻ᵀ requires another round of Givens applications,
also O(r²).

**Total cost of Step B: O(r²)** (amortized, maintaining U as state).

### Step C: Update B_v

    B_v' = C_v' G'⁻¹

Using Woodbury:

    G'⁻¹ = (G + z zᵀ)⁻¹ = G⁻¹ - G⁻¹ z zᵀ G⁻¹ / (1 + zᵀ G⁻¹ z)

If we maintain g = G⁻¹ z (one forward+back solve, O(r²)), then:

    G'⁻¹ = G⁻¹ - g gᵀ / (1 + zᵀ g)

And:

    B_v' = (C_v + v zᵀ)(G⁻¹ - g gᵀ / (1 + zᵀ g))
         = B_v + v zᵀ G⁻¹ - (C_v g)(gᵀ / (1 + zᵀ g)) - v zᵀ g gᵀ / (1 + zᵀ g)

All terms are O(r · d_v) or O(r²) to compute.

**Caveat:** Maintaining G⁻¹ explicitly defeats the purpose of using
Cholesky (numerical stability). An alternative is to just redo the two
triangular solves with L' for B_v, which is O(r² · d_v), still better
than full rebuild if d_v < r.

**Recommendation for the project:** Don't use Woodbury for B_v. Just
redo the triangular solves with L'. It's simpler, more stable, and
O(r² · d_v) is fine for our use case.

### Step D: Spectral Normalization

Computing σ_max(A_w') requires O(r²) per power iteration. Typically
3–5 iterations suffice (we know σ_max(A_w) ≈ γ from the previous step,
so we have a warm start). This is O(r²) and not the bottleneck.

### Summary

| Operation | From-scratch | Incremental |
|-----------|-------------|-------------|
| Cholesky | O(r³) | O(r²) via rank-1 update |
| A_w | O(r³) | O(r²) via Givens + cached U |
| B_v | O(r² d_v) | O(r² d_v) (redo solves with L') |
| σ_max | O(r²) | O(r²) with warm start |
| **Total** | **O(r³)** | **O(r²)** (assuming d_v ≤ r) |

## 6. Numerical Stability

### The Problem

Floating-point arithmetic is not associative. After N rank-1 Cholesky
updates, the accumulated rounding errors mean L' may not be the exact
Cholesky factor of G' = G₀ + Σ zₜ zₜᵀ.

### Quantifying Drift

The standard error bound for rank-1 Cholesky updates is:

    ||L'L'ᵀ - G'||_F ≤ O(N · u · ||G'||_F)

where u ≈ 10⁻¹⁶ (double precision unit roundoff) and N is the number
of updates. For N = 10⁴ and ||G'|| ~ 10², this gives drift of order
10⁻¹⁰, which is fine.

For N = 10⁶, drift is order 10⁻⁸. Still probably fine for our purposes,
but worth monitoring.

### Monitoring

The codebase already computes condition numbers:

```python
# In spectral_memory.py
cond = SpectralUtils.condition_number(G)
if cond > self.condition_alert:  # default 1e4
    print("WARNING: Operator condition number ...")
```

We can additionally monitor the Cholesky residual:

    residual = ||L L^T - G||_F / ||G||_F

If this exceeds a threshold (say 10⁻¹²), trigger a full refactorization.

### Refactorization Schedule

**Strategy:** Maintain a counter. Every P steps (or when residual exceeds
threshold), recompute L from scratch via full Cholesky. This gives:

    Amortized cost = (P-1) · O(r²) + 1 · O(r³)) / P
                   = O(r²) + O(r³/P)
                   ≈ O(r²)  for P >> r

With r = 64, setting P = 1000 means one O(r³) ≈ 260K flop refactorization
per 1000 O(r²) ≈ 4K flop updates. The amortized overhead is <0.1%.

## 7. References

1. Gill, Golub, Murray, Saunders. "Methods for Modifying Matrix
   Factorizations." Mathematics of Computation, 1974. The original rank-1 Cholesky update algorithm.

2. Golub and Van Loan. "Matrix Computations," 4th edition, §6.5.4. Textbook treatment of rank-1 updates.

3. Stewart. "Matrix Algorithms Vol. I: Basic Decompositions," §4.3. Givens rotations and their numerical properties.

4. Seeger. "Low Rank Updates for the Cholesky Decomposition." 2004. Practical implementation notes and numerical experiments.

5. Osborne. "Finite Algorithms in Optimization and Data Analysis," Ch. 5. Connections to recursive least squares and Kalman filtering.
   (The Gram + transition accumulation pattern in SKA is essentially
   a Koopman-flavored Kalman filter, this reference makes the
   connection explicit.)
