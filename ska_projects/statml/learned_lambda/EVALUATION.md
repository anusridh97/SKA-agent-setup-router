# Evaluation: Learned Lambda

## Grading

| Component | Weight |
|-----------|--------|
| Training data generation pipeline | 25% |
| LambdaPredictor MLP implemented and trained | 20% |
| Comparison: learned λ vs fixed λ on retrieval quality | 25% |
| Comparison: learned λ vs fixed λ on segment count (efficiency) | 20% |
| Writeup | 10% |

### A-level
- Clean pipeline. Learned λ demonstrably improves quality or efficiency
  (or ideally both) over the best fixed λ on a held-out test set.
- Error bars / significance testing.

### B-level
- Pipeline works. Some improvement shown on at least one metric.

### C-level
- MLP implemented but no clear improvement.
