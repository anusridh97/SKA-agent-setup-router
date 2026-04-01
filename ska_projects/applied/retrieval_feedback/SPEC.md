# Project: Retrieval Quality Feedback Loop

## Summary

Build a system where users rate retrieval quality (thumbs up/down on
segments, star ratings on answers), store that feedback, generate training
data from it, and demonstrate a full train→improve→measure cycle.

**Area:** Applied
**Tier:** Intermediate
**GPU:** Minimal
**Duration:** 8 weeks
**Team size:** 1–2

## Motivation

The reward predictor (`RewardPredictor`) and mode selector both need
training data. Currently the only way to generate it is by running the
full OfficeQA evaluation with ground-truth answers. A feedback loop lets
users provide this signal naturally, closing the loop between deployment
and training.

> **Dependency:** This project requires trained router checkpoints from
> the **S5 Router Training** project. S5 delivers v1 checkpoints at week 3.
> You can start your Phase 1 work (UI, infrastructure, data generation)
> immediately, and integrate the trained router starting week 4. If S5
> checkpoints are delayed, use mock data / random weights for your Phase 1
> and swap in trained weights when available.

## Deliverables

### Phase 1 (Weeks 1–4): Feedback Collection

1. **Feedback UI:** Extend the OfficeQA demo (or standalone app):
   - Thumbs up/down on each retrieved segment
   - 1–5 star rating on the overall answer
   - "Which segment was most helpful?" selection
   - "Was anything missing?" free text

2. **Feedback storage:** SQLite database storing:
   - Query, query embedding, mode selected, λ used
   - Retrieved segments with individual ratings
   - Answer text and rating, timestamp

3. **Feedback dashboard:** Summary view:
   - Average rating over time
   - Rating distribution per mode (are COMPUTE queries rated lower?)
   - Segment relevance rate per retrieval method

### Phase 2 (Weeks 5–8): Training Pipeline + Improvement Cycle

4. **Training data generation:** Convert feedback into formats consumed by:
   - `RouterTrainer.train_reward_predictor()`: {query_embedding, model_idx,
     base_model_idx, delta_r}
   - `RouterTrainer.train_mode_selector()`: {query_embedding, mode_idx}
   - LambdaPredictor (from M2 project): {query_embedding, optimal_lambda}

5. **Training round:** Collect 100+ feedback entries (from classmates,
   synthetic generation, or self-annotation). Run one full training cycle:
   - Train reward predictor on the collected data
   - Re-run the same queries with the updated predictor
   - Measure whether routing decisions change and quality improves

6. **A/B comparison:** Run the same query set with:
   - Original (untrained) reward predictor
   - Feedback-trained reward predictor
   Measure: routing agreement, cost difference, quality difference (using
   the feedback ratings as ground truth).

7. **Lambda tuning from feedback:** For queries where users rated most
   segments as irrelevant, infer that λ was too low. For queries where
   users said context was missing, infer λ was too high. Build a per-query-
   type λ recommendation from the feedback data.

## Where This Fits

```
ska_agent/router/adaptive_router.py
    RewardPredictor               ← you train this
    ModeSelector                  ← you train this
    AdaptiveRouter.route()        ← you collect feedback on results

ska_agent/training/trainers.py
    RouterTrainer.train_reward_predictor()  ← consumes your data
    RouterTrainer.train_mode_selector()     ← consumes your data

ska_agent/core/pricing.py
    PricingEngine                 ← λ you help tune from feedback
```

## Milestones

| Week | Milestone |
|------|-----------|
| 1 | Feedback UI: thumbs on segments, stars on answer. |
| 2 | SQLite storage. Record all metadata. |
| 3 | Feedback dashboard. Summary statistics. |
| 4 | Collect 50+ feedback entries (self-annotation or classmates). |
| 5 | Training data generation (reward predictor + mode selector format). |
| 6 | Run training cycle. Re-evaluate with updated predictor. |
| 7 | A/B comparison: before vs. after training. |
| 8 | Lambda tuning from feedback. Written analysis. |
