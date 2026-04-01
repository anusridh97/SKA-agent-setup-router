# Project: Spectral Memory Inspector + Anomaly Detection

## Summary

Build a visual debugger for `SharedSpectralMemory`, then extend it with
automated anomaly detection that identifies operator degradation and
triggers alerts or rebuilds.

**Area:** Applied
**Tier:** Starter
**GPU:** None
**Duration:** 8 weeks
**Team size:** 1

## Motivation

The shared memory is the communication backbone between agents. When it
works, agents share information through Koopman operators. When it breaks,
condition numbers blow up and retrieval returns garbage, all silently.

The first 4 weeks build the visual inspector. The second 4 weeks build an
automated anomaly detection layer that monitors operator health in real time
and can trigger corrective actions (rebuild, ridge adjustment, alerts).

## Deliverables

### Phase 1 (Weeks 1–4): Visual Inspector

1. **Operator spectrum view:** Singular value bar chart of A_w.
2. **Condition number timeline:** κ(G) over write count, with alert line.
3. **Write activity log:** Source agent, key count, post-write condition.
4. **Read test panel:** Random query through operator, compare K values.

### Phase 2 (Weeks 5–8): Anomaly Detection + Auto-Recovery

5. **Anomaly detector:** Monitor operator health metrics over time and
   flag anomalies:
   - Condition number spike (κ jumps by >10× in a single write)
   - Spectral collapse (σ₁/σ₂ ratio exceeds threshold)
   - Stale operator (no writes for N seconds while queries arrive)
   - Ridge insufficiency (condition number rising despite ridge)

6. **Auto-recovery policies:** When an anomaly is detected:
   - **Rebuild from scratch:** Call `rebuild_from_scratch()` with buffered keys
   - **Ridge boost:** Temporarily increase ε to stabilize, then decay
   - **Selective forget:** Discard oldest keys and rebuild from recent ones

7. **Policy comparison:** Run the same agent workload with each recovery
   policy. Measure: downtime (reads during rebuild), condition number after
   recovery, retrieval quality before/after.

8. **Integration:** Add `HealthMonitor` class to `SharedSpectralMemory` with
   configurable thresholds and callbacks. The inspector dashboard shows
   detected anomalies and recovery events in the timeline.

## Where This Fits

```
ska_agent/shared_memory/spectral_memory.py
    SharedSpectralMemory.write()          ← you instrument this
    SharedSpectralMemory.operator         ← you read this for health checks
    SharedSpectralMemory.should_rebuild() ← currently just checks threshold
    SharedSpectralMemory.reset()          ← one recovery option
    SharedSpectralMemory.rebuild_from_scratch() ← another recovery option

ska_agent/core/structures.py
    SharedOperator.condition_number, .num_tokens_seen
```

## Milestones

| Week | Milestone |
|------|-----------|
| 1 | Streamlit app. Write random keys, show singular value bar chart. |
| 2 | Condition number timeline. Write activity log. |
| 3 | Read test panel. Alert visualization when κ crosses threshold. |
| 4 | Multi-slot view if ThinkKoopmanBridge active. |
| 5 | Anomaly detector: spike detection, spectral collapse detection. |
| 6 | Auto-recovery policies: rebuild, ridge boost, selective forget. |
| 7 | Policy comparison experiment on synthetic workload. |
| 8 | HealthMonitor class integrated. Dashboard shows anomaly events. |
