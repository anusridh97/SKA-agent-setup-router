# Project: Router Training Pipeline

## Summary

Build the data generation and training pipeline for the router's two
learned components, the ModeSelector (4-way query classifier) and the
RewardPredictor (marginal quality estimator). Produce trained checkpoints
that the rest of the system depends on.

**Area:** Systems / ML
**Tier:** Intermediate
**GPU:** Minimal (tiny MLPs, CPU training)
**Duration:** 8 weeks
**Team size:** 2 (one person per component, shared infrastructure)

## Why This Is a Critical-Path Project

Four other projects depend on the router producing non-trivial output:

- **A1 Router Dashboard:** needs meaningful mode probabilities and
  action scores to visualize
- **A4 Retrieval Feedback:** needs the router making real decisions
  to collect useful feedback
- **S4 PID Autotuning:** needs the router responding to PID changes
  to tune gains
- **M4 Mode Calibration:** needs the ModeSelector producing
  non-uniform predictions to study calibration

With untrained (random) weights, the ModeSelector outputs ~uniform
probabilities over all 4 modes, and the RewardPredictor outputs near-zero
scores for every action. The router stops immediately because no action
scores positive. Nothing interesting happens.

**Your trained checkpoints unblock everyone else.** The first milestone
(week 3) should produce usable checkpoints, even if rough, so downstream
projects can start integrating.

## Team Split

### Person A: ModeSelector

The ModeSelector is a 2-layer MLP (384→256→4) that classifies queries
into one of four collaboration modes:

| Mode | When to use | Example query |
|------|------------|---------------|
| LOOKUP | Single fact extraction | "What was the total federal debt in 2023?" |
| MULTI_DOC | Cross-document comparison | "How did spending change from 2022 to 2023?" |
| COMPUTE | Numerical calculation needed | "What percentage of GDP was military spending?" |
| MULTI_STEP | Complex multi-hop reasoning | "Which department had the highest growth after inflation?" |

**Your job:**
1. Build a labeled dataset of 1000+ (query, mode) pairs
2. Train the ModeSelector to classify accurately
3. Evaluate: accuracy, confusion matrix, per-mode precision/recall
4. Ship a checkpoint that the M4 (Mode Calibration) project can study

### Person B: RewardPredictor

The RewardPredictor is a 3-layer MLP (448→512→256→1) that predicts how
much quality improvement a specialist will provide over a baseline:

    Δr̂ = F_θr(e_Q, specialist, baseline)

**Your job:**
1. Build a training dataset by running the retrieval pipeline at different
   configurations and measuring answer quality
2. Train the RewardPredictor to predict quality deltas
3. Evaluate: MSE, correlation with actual quality, does the ranking of
   specialists match ground truth?
4. Ship a checkpoint that produces positive scores for good actions
   (so the router actually executes them instead of stopping immediately)

### Shared Infrastructure (both people)

- Checkpoint save/load utilities
- A `load_trained_router()` function that other projects can call
- Integration test: load both checkpoints, run 50 queries, verify
  the router makes non-trivial decisions

## Deliverables

### Week 3 Checkpoint (early release for downstream projects)

- ModeSelector: trained on ≥500 queries, accuracy ≥60%
- RewardPredictor: trained, produces positive scores for at least
  some actions (so the router doesn't always stop immediately)
- `load_trained_router()` function works
- Announce to other teams: "checkpoints available, here's how to load"

### Final Deliverables (Week 8)

1. **ModeSelector** (Person A):
   - Dataset of 1000+ labeled queries with methodology writeup
   - Trained model with accuracy ≥75%
   - Confusion matrix analysis: which modes get confused?
   - Ablation: accuracy vs. dataset size curve (how much data is enough?)
   - Error analysis: 20 misclassified queries examined by hand

2. **RewardPredictor** (Person B):
   - Dataset of quality measurements across specialists
   - Trained model with meaningful predictions (positive for good actions,
     negative for bad ones)
   - Correlation analysis: predicted vs. actual quality
   - Integration test: router with trained predictor makes non-trivial
     multi-step routing decisions

3. **Shared** (both):
   - `ska_agent/checkpoints/` directory with saved state_dicts
   - `ska_agent/router/pretrained.py` with `load_trained_router()`
   - README documenting: how the data was generated, how to retrain,
     how to load checkpoints
   - Verification that the 4 downstream projects work with the checkpoints

## Where This Fits in the Codebase

```
ska_agent/router/adaptive_router.py
    QueryEncoder            ← frozen pretrained MiniLM (no training needed)
    ModeSelector            ← Person A trains this
    RewardPredictor         ← Person B trains this
    PIDController           ← no learned params, no training needed
    ActionScorer            ← uses RewardPredictor + PID, not trained itself
    AdaptiveRouter          ← orchestrates everything

ska_agent/training/trainers.py
    RouterTrainer           ← has train_reward_predictor() and
                              train_mode_selector() already implemented
                              (you'll use or extend these)

ska_agent/core/structures.py
    RouterTrainingConfig    ← hyperparameters
    CollaborationMode       ← the 4 modes
    MODE_TEMPLATES          ← DAG template per mode
```

## Milestones

| Week | Person A (ModeSelector) | Person B (RewardPredictor) |
|------|------------------------|---------------------------|
| 1 | Design query templates for all 4 modes. Generate 200 queries. | Understand the scoring function S(a) = Δr̂ - λᵀΔĉ. Set up evaluation harness with mock specialists. |
| 2 | Expand dataset to 500+. Paraphrase and diversify. First training run. | Run retrieval pipeline on 100 queries at multiple configs. Measure quality with score_answer(). |
| 3 | **Ship v1 checkpoint** (≥60% accuracy). Announce to downstream teams. | **Ship v1 checkpoint** (positive scores for retrieval actions). Announce to downstream teams. |
| 4 | Error analysis on v1. Identify gaps. Generate targeted examples for confused modes. | Expand training data. Include code_executor and reasoner quality measurements. |
| 5 | Expand to 1000+ queries. Retrain. Confusion matrix analysis. | Train on expanded data. Correlation analysis: predicted vs. actual. |
| 6 | Ablation: accuracy vs. dataset size. How much data is enough? | Integration: does the router make multi-step decisions now? |
| 7 | Final error analysis (20 misclassified queries). | Comparison: routing with trained vs. untrained predictor on 100 queries. |
| 8 | Final checkpoint. Writeup. Verify downstream projects work. | Final checkpoint. Writeup. Verify downstream projects work. |
