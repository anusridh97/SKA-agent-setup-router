# Project: Router Decision Dashboard + Comparative Analysis

## Summary

Build a real-time web dashboard that visualizes the `AdaptiveRouter`'s
decision-making process, then use it as an analysis tool to systematically
compare routing strategies and identify failure patterns.

**Area:** Applied
**Tier:** Starter
**GPU:** None
**Duration:** 8 weeks
**Team size:** 1–2

## Motivation

The router is the "brain" of SKA-Agent. Right now, the only way to see
what it's doing is `verbose=True` print statements. A dashboard makes the
system's intelligence visible, and once it's visible, you can analyze it.

The first 4 weeks build the dashboard. The second 4 weeks use it to run
a systematic study of routing behavior across query types, producing
a written analysis with actionable findings.

> **Dependency:** This project requires trained router checkpoints from
> the **S5 Router Training** project. S5 delivers v1 checkpoints at week 3.
> You can start your Phase 1 work (UI, infrastructure, data generation)
> immediately, and integrate the trained router starting week 4. If S5
> checkpoints are delayed, use mock data / random weights for your Phase 1
> and swap in trained weights when available.

## Deliverables

### Phase 1 (Weeks 1–4): The Dashboard

1. **Live routing visualization:** Submit a query, see decisions unfold:
   - Mode selection bar chart (LOOKUP, MULTI_DOC, COMPUTE, MULTI_STEP)
   - Action scoring table with score breakdowns
   - Execution trace with timing

2. **PID dynamics panel:** Real-time chart of the 5-dimensional λ vector
   across a query session.

3. **DAG visualization:** Collaboration template with execution status.

4. **Query history:** Table of past queries with mode, actions, cost, answer.

### Phase 2 (Weeks 5–8): Comparative Analysis

5. **Batch evaluation mode:** Load a set of queries (JSON), run all of
   them through the router, collect full traces. Export as CSV/JSON.

6. **Mode confusion analysis:** For queries where the mode selector is
   uncertain (max probability < 0.5), what happens? Do uncertain queries
   lead to worse routing decisions? Build a confusion-matrix view.

7. **Cost-quality Pareto analysis:** For the same set of queries, sweep
   the PID budget_rate and plot the Pareto frontier: quality vs. total cost.
   Identify the "knee" where quality stops improving.

8. **Failure pattern report:** Written analysis (2–3 pages) identifying:
   - Query types where the router consistently chooses the wrong mode
   - Queries where the PID controller over-constrains (stops too early)
   - Recommendations for router improvement (concrete, actionable)

## Where This Fits in the Codebase

You're building a UI and analysis layer, you call these components and
display/analyze the results:

```
ska_agent/router/adaptive_router.py
    AdaptiveRouter.route()          ← returns List[ActionResult]
    ModeSelector.predict()          ← returns (mode, probs)
    ActionScorer.score_action()     ← returns score float
    PIDController                   ← lambda_vec, cost_history

ska_agent/core/structures.py
    CollaborationMode, ActionCandidate, ActionResult, CostVector,
    MODE_TEMPLATES
```

## Milestones

| Week | Milestone |
|------|-----------|
| 1 | Streamlit app running. Mode selection bar chart for a hardcoded query. |
| 2 | Wired to real AdaptiveRouter. Full routing trace for user queries. |
| 3 | PID dynamics panel. DAG visualization. |
| 4 | Query history table. Batch evaluation mode (load JSON, run all). |
| 5 | Mode confusion analysis. Scatter plot: confidence vs. outcome quality. |
| 6 | Cost-quality Pareto frontier across budget_rate sweep. |
| 7 | Failure pattern identification. Export results. |
| 8 | Written analysis report. Polish dashboard. |
