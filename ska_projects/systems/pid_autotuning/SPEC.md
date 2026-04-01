# Project: PID Controller Auto-Tuning

## Summary

Build an automated system that optimizes the PID cost controller gains
(Kp, Ki, Kd) and budget parameters by running the router on held-out
queries and measuring cost-quality tradeoffs.

**Area:** Systems
**GPU:** Minimal (for running router on queries)
**Duration:** 8 weeks
**Team size:** 1

## Motivation

The `PIDController` in `router/adaptive_router.py` has hardcoded gains:
Kp=0.3, Ki=0.01, Kd=0.05, with budget_rate and lambda_max also fixed.
These were chosen by hand. The controller operates on a 5-dimensional
cost vector (input tokens, output tokens, latency, dollar cost, meta
overhead), and the optimal gains depend on the query distribution and
the cost characteristics of the available specialists.

Auto-tuning PID controllers is a well-studied problem in control theory
(Ziegler-Nichols, relay feedback, etc.). This project adapts these
methods to the discrete, stochastic setting of the SKA router.

> **Dependency:** This project requires trained router checkpoints from
> the **S5 Router Training** project. S5 delivers v1 checkpoints at week 3.
> You can start your Phase 1 work (UI, infrastructure, data generation)
> immediately, and integrate the trained router starting week 4. If S5
> checkpoints are delayed, use mock data / random weights for your Phase 1
> and swap in trained weights when available.

## Deliverables

1. **`PIDTuner` class** that:
   - Takes a set of queries + ground truth answers
   - Runs the full router pipeline with different PID configurations
   - Measures quality (accuracy) and cost (total 5-dim cost vector)
   - Optimizes gains to maximize quality subject to cost budget

2. **Tuning methods:**
   - Grid search over (Kp, Ki, Kd) space
   - Bayesian optimization (using e.g. scikit-optimize)
   - Ziegler-Nichols adaptation for the discrete setting

3. **Budget scheduling:** policies like "spend freely on first N queries,
   then tighten" and analysis of their effect on quality.

4. **Evaluation** on the OfficeQA benchmark showing improvement over
   default gains.

## Where This Fits in the Codebase

```
ska_agent/router/
├── adaptive_router.py  ← PIDController class (also in pid_controller.py)
│                          ActionScorer.score_action() uses pid.lambda_vec
│                          AdaptiveRouter.route() calls pid.update()
├── pid_controller.py   ← Standalone PID (same algorithm, no torch dep)
│
ska_agent/core/structures.py
    PIDConfig            ← Kp, Ki, Kd, lambda_max, budget_rate, window_size
```

You create a new file `ska_agent/tuning/pid_tuner.py`.

## Milestones

| Week | Milestone |
|------|-----------|
| 1–2 | Read PID background. Implement grid search tuner. |
| 3–4 | Implement Bayesian optimization. Compare methods. |
| 5–6 | Budget scheduling policies. Experiments. |
| 7–8 | Evaluation on OfficeQA subset. Final writeup. |
