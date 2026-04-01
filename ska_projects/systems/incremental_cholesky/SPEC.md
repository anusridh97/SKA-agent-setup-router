# Project: Incremental Cholesky Operator Updates

## Summary

Replace the current O(r³) from-scratch operator reconstruction in
`SharedSpectralMemory._rebuild_operator()` with an O(r²)-per-step
incremental update pipeline that maintains exact numerical guarantees.

**Area:** Systems
**GPU:** None
**Duration:** 8 weeks
**Team size:** 1–2

## Motivation

Every time a new key arrives in `SharedSpectralMemory.write()`, the
`_stale` flag is set, and the next `.operator` access triggers a full
rebuild: Cholesky factorization of G (O(r³)), two triangular solves for
A_w (O(r³)), and two more for B_v (O(r² · d_v)). At r=64 this is fast
in absolute terms, but in a streaming multi-agent setting where keys
arrive at every token, this becomes the bottleneck, especially when
multiple agents are writing and reading concurrently.

The algebra supports an O(r²) incremental path, but it's non-trivial
because the whitened operator A_w = L⁻¹ M L⁻ᵀ couples G, M, and L
through a similarity transform. A naive rank-1 update to L doesn't
propagate correctly into A_w without also accounting for how the change
in L affects the *existing* M contribution.

## Deliverables

1. **Implementation** of `IncrementalOperatorBuilder` class that maintains
   L, A_w, B_v incrementally as keys/values arrive one at a time.

2. **Correctness tests** showing bitwise agreement (up to floating point)
   between incremental and from-scratch reconstruction at every step.

3. **Numerical stability analysis**: empirical measurement of drift over
   1K, 10K, 100K updates, with condition number monitoring and
   automatic refactorization trigger.

4. **Benchmarks**: wall-clock comparison of incremental vs. from-scratch
   at r ∈ {32, 64, 128, 256, 512}, with roofline analysis.

5. **Integration**: drop-in replacement for the `_rebuild_operator` path
   in `SharedSpectralMemory`, gated by a config flag.

## Milestones

| Week | Milestone |
|------|-----------|
| 1–2 | Read background material. Implement standalone rank-1 Cholesky update. Verify against scipy. |
| 3–4 | Implement the full incremental A_w update (Givens rotation tracking). |
| 5–6 | Implement incremental B_v update. Correctness test suite. |
| 7 | Benchmarks, numerical stability experiments, refactorization scheduling. |
| 8 | Integration into SharedSpectralMemory. Final writeup. |

## Where This Fits in the Codebase

```
ska_agent/
├── shared_memory/
│   ├── spectral_memory.py      ← SharedSpectralMemory._rebuild_operator()
│   │                              This is what you're replacing.
│   └── think_koopman_bridge.py ← ThinkKoopmanBridge.build_operator()
│                                  Also rebuilds from scratch; same pattern.
├── utils/
│   └── math_utils.py           ← SpectralUtils.cholesky_factor(),
│                                  whiten_operator(), build_value_readout()
│                                  These are the O(r³) primitives.
└── core/
    └── ska_module.py           ← SKAModule._build_operator()
                                   Torch version of the same pipeline.
                                   Incremental version would go here too
                                   (but is a stretch goal, torch Givens
                                   rotations are fiddly).
```

Your new code goes primarily in `utils/math_utils.py` (the incremental
primitives) and `shared_memory/spectral_memory.py` (the integration).
