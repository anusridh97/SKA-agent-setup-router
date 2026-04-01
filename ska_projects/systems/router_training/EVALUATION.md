# Evaluation: Router Training Pipeline

## Grading Breakdown

### Person A: ModeSelector

| Component | Weight |
|-----------|--------|
| Dataset quality (1000+ queries, diverse, balanced) | 25% |
| Training + validation accuracy ≥ 75% | 25% |
| Confusion matrix + per-mode precision/recall | 15% |
| Ablation: accuracy vs. dataset size | 15% |
| Error analysis (20 misclassified queries examined) | 10% |
| Week 3 checkpoint delivered on time | 10% |

### Person B: RewardPredictor

| Component | Weight |
|-----------|--------|
| Training data with meaningful quality signal | 25% |
| Trained model predicts positive for appropriate actions | 25% |
| Correlation analysis: predicted vs. actual quality | 15% |
| Router integration: makes multi-step decisions | 15% |
| Comparison: trained vs. untrained router behavior | 10% |
| Week 3 checkpoint delivered on time | 10% |

### Shared

| Component | Weight |
|-----------|--------|
| load_trained_router() function works | 30% |
| Downstream projects verified with checkpoints | 40% |
| Documentation: how to retrain, data format, loading | 30% |

## What "Done" Looks Like

### A-level
- Person A: 1000+ queries, ≥75% accuracy, clear confusion matrix showing
  which mode pairs are hardest, ablation curve, 20 error examples with
  analysis of *why* they're hard
- Person B: Router makes 2+ actions on multi-step queries, predicted
  rewards correlate with actual quality, clear before/after comparison
- Shared: load_trained_router() works, all 4 downstream projects tested
  and verified, clean documentation

### B-level
- Person A: 500+ queries, ≥65% accuracy, confusion matrix
- Person B: Router makes at least 1 action on most queries
- Shared: Checkpoints exist and load correctly

### C-level
- Person A: < 500 queries or < 60% accuracy
- Person B: Router still stops immediately on most queries
- Shared: No load_trained_router() function

## Critical-Path Note

The **week 3 checkpoint** is graded separately (10% for each person)
because it's a dependency for 4 other teams. Late delivery of v1
checkpoints blocks A1, A4, S4, and M4. Even a rough checkpoint that
makes the router non-trivial is more valuable on time than a perfect
checkpoint two weeks late.
