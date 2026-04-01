# Project: Neural Sparsity Parameter Prediction

## Summary

Replace the fixed `lambda_sparsity` in `PricingEngine` with a learned MLP
that predicts the optimal λ per query. Train it, evaluate it against fixed-λ
baselines, and integrate it into the retrieval pipeline.

**Area:** Statistical ML
**Tier:** Intermediate
**GPU:** Minimal (tiny MLP, retrieval is numpy)
**Duration:** 8 weeks
**Team size:** 1

## Motivation

The `PricingEngine` uses a fixed λ that controls the sparsity-quality
tradeoff. The optimal λ depends on the query: a simple factual lookup
needs λ=0.1, while a multi-hop comparison needs λ=0.001. Learning λ
per query is equivalent to learning a query-dependent constraint
relaxation, a form of amortized optimization.

## Deliverables

### Phase 1 (Weeks 1–4): Data Generation + Training

1. **Training data pipeline:** For each query in a dev set, sweep
   λ ∈ {0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5} and record:
   - Number of segments retrieved
   - Answer quality (score_answer or proxy)
   - Label: best λ per query

2. **LambdaPredictor MLP:** e_Q → λ*(e_Q) with Softplus output.
   Train with MSE on log(λ).

3. **Training and validation:** Proper train/val split. Learning curves.
   Report: MSE on held-out queries, distribution of predicted vs. actual λ.

### Phase 2 (Weeks 5–8): Evaluation + Integration

4. **End-to-end retrieval comparison:** On 100+ test queries:
   - Fixed λ = 0.05 (current default)
   - Fixed λ = best single λ found by grid search on dev set
   - Learned λ from LambdaPredictor
   - Oracle λ (best λ per query, not achievable in practice)

   For each, measure: retrieval precision, answer accuracy, segments
   retrieved per query, total tokens consumed.

5. **λ analysis:** What does the predictor learn?
   - Histogram of predicted λ values: is the distribution unimodal or
     does it cluster by query type?
   - Scatter plot: predicted λ vs. query complexity (measured by number
     of required reasoning steps, or mode selector confidence)
   - Do simple queries get high λ and complex queries get low λ?

6. **Integration:** Drop-in replacement for `PricingEngine.lambda_sparsity`:
   ```python
   engine = PricingEngine(
       segments=segments,
       embed_fn=embed_fn,
       lambda_fn=lambda q: predictor(embed_fn(q)),  # new parameter
   )
   ```

7. **Robustness:** Does the predictor generalize across documents?
   Train on document A, test on document B. Report: how much does
   performance degrade? Does fine-tuning on 10 examples from B recover it?

## Where This Fits

```
ska_agent/core/pricing.py
    PricingEngine.__init__()     ← lambda_sparsity is set here
    PricingEngine.retrieve()     ← uses self.lambda_sparsity
    PricingEngine.compute_reduced_cost() ← λ appears in c̄ = λ + η·δR - δφ
```

## Milestones

| Week | Milestone |
|------|-----------|
| 1 | Training data pipeline: sweep λ for 100 queries. |
| 2 | LambdaPredictor MLP. Train on dev set. |
| 3 | Validation. Learning curves. Predicted vs. actual λ. |
| 4 | End-to-end retrieval comparison setup. |
| 5 | Run comparison: fixed vs. learned vs. oracle. |
| 6 | λ analysis: what does the predictor learn? |
| 7 | Integration into PricingEngine. Cross-document robustness. |
| 8 | Written analysis. Final benchmarks. |
