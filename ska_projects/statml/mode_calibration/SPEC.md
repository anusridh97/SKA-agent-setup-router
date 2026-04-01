# Project: Mode Selector Calibration + Calibrated Routing

## Summary

Study whether the `ModeSelector`'s softmax probabilities are well-calibrated,
implement post-hoc calibration methods, and measure whether better-calibrated
probabilities actually improve downstream routing decisions.

**Area:** Statistical ML
**Tier:** Intermediate
**GPU:** Minimal (tiny MLP, calibration is numpy)
**Duration:** 8 weeks
**Team size:** 1

## Motivation

The `ModeSelector` is a 4-way softmax classifier over collaboration modes.
The router uses the predicted probabilities to decide confidence in the mode
selection. Modern neural networks are notoriously overconfident, if the
selector says 95% LOOKUP but is only right 60% of the time, the router trusts
it too much and never explores alternative modes.

The first half studies calibration. The second half measures whether fixing
calibration actually changes routing behavior and quality.

> **Dependency:** This project requires trained router checkpoints from
> the **S5 Router Training** project. S5 delivers v1 checkpoints at week 3.
> You can start your Phase 1 work (UI, infrastructure, data generation)
> immediately, and integrate the trained router starting week 4. If S5
> checkpoints are delayed, use mock data / random weights for your Phase 1
> and swap in trained weights when available.

## Deliverables

### Phase 1 (Weeks 1–4): Calibration Study

1. **Calibration dataset:** 300+ queries manually labeled with ground-truth
   modes (use templates, paraphrase, recruit classmates to annotate).

2. **Calibration metrics:** ECE, MCE, Brier score, reliability diagrams
   for the untrained and trained mode selector.

3. **Calibration methods:** Implement and compare:
   - Temperature scaling (single T)
   - Platt scaling (per-class affine)
   - Histogram binning
   - Isotonic regression

4. **Calibration comparison:** Table + plots: ECE/MCE before and after
   each method. Reliability diagrams side-by-side.

### Phase 2 (Weeks 5–8): Calibrated Routing Study

5. **Route with calibrated probabilities:** Modify the router to use
   calibrated mode probabilities. When the calibrated confidence drops
   below a threshold, explore: run the top-2 modes and compare outputs.

6. **Exploration policy:** Implement a simple exploration strategy:
   - If max calibrated prob < 0.6: run top-2 modes, pick better result
   - If max calibrated prob < 0.4: run top-3 modes
   - Otherwise: use the top mode as before

7. **Routing comparison:** On the calibration dataset:
   - Uncalibrated routing (original) vs. calibrated routing (yours)
   - Measure: mode accuracy, exploration frequency, total cost, answer quality
   - Measure: does exploration actually find better modes when it fires?

8. **Written analysis:** When does calibration-driven exploration help?
   Characterize the query types where exploration discovers a better mode
   vs. where it just wastes budget. Recommend exploration threshold.

## Where This Fits

```
ska_agent/router/adaptive_router.py
    ModeSelector.predict()       ← returns (mode, probs)
    ModeSelector.__init__()      ← has self.temperature
    AdaptiveRouter.route()       ← uses mode_selector.predict()
```

## Milestones

| Week | Milestone |
|------|-----------|
| 1 | Generate calibration dataset (300+ queries). |
| 2 | Compute ECE/MCE on untrained selector. Reliability diagrams. |
| 3 | Implement temperature scaling and Platt scaling. |
| 4 | Implement histogram binning. Full comparison table. |
| 5 | Implement calibrated routing with exploration policy. |
| 6 | Run routing comparison: original vs. calibrated. |
| 7 | Analyze exploration behavior: when does it help? |
| 8 | Written analysis with threshold recommendation. |
