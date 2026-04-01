# Starter: Incremental Cholesky Operator Updates

## Step 0: Set Up Your Environment

```bash
# Clone or extract the SKA-Agent codebase
tar xf ska_agent-1_0_0-7.tar
pip install numpy scipy matplotlib pytest

# Verify you can import the math utilities
python -c "from ska_agent.utils.math_utils import SpectralUtils; print('OK')"
```

You do NOT need torch for this project. Everything is numpy/scipy.

## Step 1: Implement Rank-1 Cholesky Update

Start here. This is the foundation of everything else.

```python
# file: incremental_cholesky.py

import numpy as np
from typing import Tuple, List

def cholesky_rank1_update(
    L: np.ndarray,
    z: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[int, float, float]]]:
    """
    Rank-1 update: given L such that LLᵀ = G, compute L' such that
    L'L'ᵀ = G + zzᵀ.

    Returns:
        L_new: updated Cholesky factor (r, r)
        rotations: list of (index, cosine, sine) for each Givens rotation

    TODO: Implement the algorithm from BACKGROUND.md Section 2.
    """
    r = L.shape[0]
    L_new = L.copy()
    w = z.copy().astype(np.float64)
    rotations = []

    for i in range(r):
        a = L_new[i, i]
        b = w[i]
        rho = np.sqrt(a**2 + b**2)

        if rho < 1e-15:
            # Degenerate case, skip this rotation
            rotations.append((i, 1.0, 0.0))
            continue

        c = a / rho
        s = b / rho
        L_new[i, i] = rho
        rotations.append((i, c, s))

        # Update remaining elements
        for j in range(i + 1, r):
            L_new_ji = c * L_new[j, i] + s * w[j]
            w[j] = -s * L_new[j, i] + c * w[j]
            L_new[j, i] = L_new_ji

    return L_new, rotations

def verify_cholesky_update():
    """
    Your first experiment: verify the update is correct.
    Run this before anything else.
    """
    np.random.seed(42)
    r = 64

    # Build a random SPD matrix
    A = np.random.randn(r, r)
    G = A @ A.T + 0.001 * np.eye(r)
    L = np.linalg.cholesky(G)

    # Random update vector
    z = np.random.randn(r)

    # Ground truth
    G_new = G + np.outer(z, z)
    L_true = np.linalg.cholesky(G_new)

    # Our incremental update
    L_inc, rotations = cholesky_rank1_update(L, z)

    # Check
    residual_true = np.linalg.norm(L_true @ L_true.T - G_new) / np.linalg.norm(G_new)
    residual_inc = np.linalg.norm(L_inc @ L_inc.T - G_new) / np.linalg.norm(G_new)

    print(f"Full Cholesky residual:        {residual_true:.2e}")
    print(f"Incremental Cholesky residual:  {residual_inc:.2e}")
    print(f"L agreement (Frobenius):        {np.linalg.norm(L_inc - L_true):.2e}")
    print(f"Number of Givens rotations:     {len(rotations)}")

    assert residual_inc < 1e-12, "Incremental update is not accurate!"
    print("\nPASSED: Rank-1 Cholesky update is correct.")

if __name__ == "__main__":
    verify_cholesky_update()
```

**Run this first.** If it passes, you have the foundation. If not,
debug before moving on.

## Step 2: Apply Givens Rotations to a Matrix

You need a helper that applies the Givens rotations from the Cholesky
update to an arbitrary matrix. This is how you'll update A_w.

```python
def apply_givens_left(
    rotations: List[Tuple[int, float, float]],
    M: np.ndarray,
    transpose: bool = False,
) -> np.ndarray:
    """
    Apply Givens rotations to M from the left: Q @ M or Qᵀ @ M.

    Each rotation (i, c, s) acts on row i and the "virtual" row
    that was the z vector. But since we're applying to a separate
    matrix, we need to track how the rotations compose.

    TODO: This is the tricky part. The Givens rotations from the
    Cholesky update operate on a (r+1)-dimensional space (the
    columns of [L | z]). When applying to a separate matrix M,
    you need to understand which rows the rotation touches.

    Hint: Read Golub & Van Loan §6.5.4 carefully. The rotation
    at step i mixes row i of L with element i of w. When applied
    to a separate matrix, it mixes row i with a "carry" vector
    that starts as some function of z.
    """
    # YOUR IMPLEMENTATION HERE
    # This is the core intellectual challenge of the project.
    pass
```

**This is intentionally left incomplete.** Working out the correct
formulation is the main systems challenge. The BACKGROUND.md gives
you the mathematical framework; you need to figure out how to apply
the rotations that were computed in the [L | z] context to the
separate matrix M.

Hint: think about what happens if you augment M with an extra row
(the "z contribution to M"), apply the Givens rotations to the
augmented matrix, then extract the result.

## Step 3: The Full Incremental Operator Update

Once you have Steps 1 and 2, assemble the full pipeline:

```python
class IncrementalOperatorBuilder:
    """
    Maintains L, A_w, B_v incrementally.

    Usage:
        builder = IncrementalOperatorBuilder(rank=64, ridge_eps=1e-3)

        for z_t, z_prev, v_t in stream_of_keys:
            builder.update(z_t, z_prev, v_t)
            A_w = builder.A_w
            L = builder.L
            B_v = builder.B_v
    """

    def __init__(self, rank: int, ridge_eps: float = 1e-3):
        self.rank = rank
        self.ridge_eps = ridge_eps

        # Initialize G = εI, L = sqrt(ε) I
        self.G = ridge_eps * np.eye(rank)
        self.L = np.sqrt(ridge_eps) * np.eye(rank)
        self.M = np.zeros((rank, rank))
        self.C_v = None  # (d_v, rank), initialized on first value

        # Cached intermediate: U = L⁻¹ M
        self._U = np.zeros((rank, rank))

        # Operator state
        self.A_w = np.zeros((rank, rank))
        self.B_v = None
        self.num_updates = 0
        self._refac_interval = 1000

    def update(
        self,
        z: np.ndarray,
        z_prev: np.ndarray = None,
        value: np.ndarray = None,
    ):
        """
        Incorporate a new key (and optionally transition + value).

        TODO: Implement the O(r²) update pipeline from BACKGROUND.md §5.
        """
        # Step A: Rank-1 Cholesky update
        self.L, rotations = cholesky_rank1_update(self.L, z)
        self.G += np.outer(z, z)

        # Step B: Update M and A_w
        if z_prev is not None:
            self.M += np.outer(z, z_prev)

        # TODO: Update self._U and self.A_w using the Givens rotations
        # instead of recomputing from scratch.

        # Step C: Update B_v
        if value is not None:
            if self.C_v is None:
                self.C_v = np.zeros((len(value), self.rank))
            self.C_v += np.outer(value, z)
            # Recompute B_v via triangular solves with new L
            # (This is O(r² · d_v), acceptable)

        # Step D: Spectral normalization
        # TODO: warm-started power iteration for σ_max

        self.num_updates += 1

        # Periodic full refactorization for numerical hygiene
        if self.num_updates % self._refac_interval == 0:
            self._full_refactorize()

    def _full_refactorize(self):
        """Recompute L from G from scratch. O(r³) but rare."""
        self.L = np.linalg.cholesky(self.G)
        # Recompute A_w from scratch too
        from ska_agent.utils.math_utils import SpectralUtils
        self.A_w = SpectralUtils.whiten_operator(self.L, self.M)
        self.A_w = SpectralUtils.spectral_normalize(self.A_w, gamma=1.0)
        # Recompute B_v
        if self.C_v is not None:
            self.B_v = SpectralUtils.build_value_readout(
                # need values and keys, or use C_v directly
                # TODO: implement B_v rebuild
            )
```

## Step 4: Correctness Test Suite

Write tests that compare incremental vs. from-scratch at every step:

```python
def test_incremental_vs_full(num_steps=500, rank=64):
    """
    Generate a random key sequence, build the operator both ways,
    verify agreement at every step.
    """
    np.random.seed(123)

    builder = IncrementalOperatorBuilder(rank=rank)

    # Also accumulate for from-scratch comparison
    from ska_agent.utils.math_utils import SpectralUtils
    G = 1e-3 * np.eye(rank)
    M = np.zeros((rank, rank))
    prev_z = None

    max_error = 0.0

    for t in range(num_steps):
        z = np.random.randn(rank) * 0.1  # small keys for stability

        # Update incremental builder
        builder.update(z, z_prev=prev_z)

        # Update from-scratch accumulators
        G += np.outer(z, z)
        if prev_z is not None:
            M += np.outer(z, prev_z)

        # From-scratch operator
        L_full = np.linalg.cholesky(G)
        A_w_full = SpectralUtils.whiten_operator(L_full, M)
        A_w_full = SpectralUtils.spectral_normalize(A_w_full, gamma=1.0)

        # Compare
        L_err = np.linalg.norm(builder.L @ builder.L.T - G) / np.linalg.norm(G)
        A_err = np.linalg.norm(builder.A_w - A_w_full) / max(np.linalg.norm(A_w_full), 1e-10)

        max_error = max(max_error, L_err, A_err)

        if t % 100 == 0:
            print(f"Step {t}: L_err={L_err:.2e}, A_err={A_err:.2e}")

        prev_z = z.copy()

    print(f"\nMax error over {num_steps} steps: {max_error:.2e}")
    assert max_error < 1e-10, f"Incremental error too large: {max_error}"
    print("PASSED")
```

## Step 5: Benchmarks

```python
import time

def benchmark(ranks=[32, 64, 128, 256, 512], num_steps=1000):
    """Compare wall-clock time of incremental vs. from-scratch."""
    for r in ranks:
        np.random.seed(42)
        keys = [np.random.randn(r) * 0.1 for _ in range(num_steps)]

        # --- From scratch at every step ---
        G = 1e-3 * np.eye(r)
        M = np.zeros((r, r))
        t0 = time.perf_counter()
        for t in range(num_steps):
            G += np.outer(keys[t], keys[t])
            if t > 0:
                M += np.outer(keys[t], keys[t-1])
            L = np.linalg.cholesky(G)
            # Would compute A_w here
        scratch_time = time.perf_counter() - t0

        # --- Incremental ---
        builder = IncrementalOperatorBuilder(rank=r)
        t0 = time.perf_counter()
        for t in range(num_steps):
            builder.update(keys[t], z_prev=keys[t-1] if t > 0 else None)
        inc_time = time.perf_counter() - t0

        speedup = scratch_time / inc_time
        print(f"r={r:4d}: scratch={scratch_time:.3f}s, "
              f"incremental={inc_time:.3f}s, speedup={speedup:.2f}x")
```

**Expected result:** Speedup should be ~r/const for large r (since
you're replacing O(r³) with O(r²)). At r=64 the constant factors
may dominate; at r=256+ you should see clear wins.

## Step 6: Integration

Once everything works, integrate into the codebase:

1. Add `IncrementalOperatorBuilder` to `ska_agent/utils/math_utils.py`
2. Modify `SharedSpectralMemory._rebuild_operator()` to use the
   incremental path when `self._incremental_builder` is available
3. Add a `use_incremental: bool = False` flag to `SharedSpectralMemory.__init__`
4. Write an integration test using the actual `SharedSpectralMemory` API

```python
def test_shared_memory_incremental():
    """
    Verify that incremental mode produces the same read() results
    as from-scratch mode.
    """
    from ska_agent.shared_memory.spectral_memory import SharedSpectralMemory

    np.random.seed(42)
    rank = 64
    num_writes = 200

    mem_scratch = SharedSpectralMemory(rank=rank)
    mem_inc = SharedSpectralMemory(rank=rank, use_incremental=True)  # new flag

    for _ in range(num_writes):
        keys = np.random.randn(5, rank) * 0.1
        mem_scratch.write(keys)
        mem_inc.write(keys)

    # Compare read results
    queries = np.random.randn(10, rank) * 0.1
    out_scratch = mem_scratch.read(queries)
    out_inc = mem_inc.read(queries)

    error = np.linalg.norm(out_scratch - out_inc) / np.linalg.norm(out_scratch)
    print(f"Read output error: {error:.2e}")
    assert error < 1e-10
    print("PASSED")
```
