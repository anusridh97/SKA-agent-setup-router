# Evaluation: Router Dashboard + Comparative Analysis

## Grading Breakdown

| Component | Weight | Description |
|-----------|--------|-------------|
| Dashboard core (mode, actions, trace) | 20% | Weeks 1–3 deliverables |
| PID dynamics + DAG + history | 10% | Week 3–4 deliverables |
| Batch evaluation mode | 15% | Load queries, run all, export traces |
| Mode confusion analysis | 15% | Confidence vs. quality, confusion matrix |
| Cost-quality Pareto frontier | 15% | Budget sweep, knee identification |
| Written failure analysis | 15% | 2–3 page report with concrete findings |
| Polish and usability | 10% | Layout, labeling, a non-expert could use it |

## What "Done" Looks Like

### A-level
- Dashboard runs end-to-end with real AdaptiveRouter
- Batch mode processes 100+ queries with full trace capture
- Mode confusion analysis reveals specific failure patterns (e.g. "COMPUTE
  queries are misclassified as LOOKUP 40% of the time when they contain
  the word 'percentage'")
- Pareto frontier plotted with clear knee identification
- Written report includes 3+ concrete, actionable recommendations
- Dashboard polished enough to screenshot for a paper

### B-level
- Dashboard works for interactive single-query use
- Batch mode runs but analysis is surface-level
- Written report exists but recommendations are vague

### C-level
- Dashboard shows some router output
- No batch mode or systematic analysis
