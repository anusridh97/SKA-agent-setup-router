# Evaluation: Memory Inspector + Anomaly Detection

## Grading

| Component | Weight |
|-----------|--------|
| Visual inspector (spectrum, κ timeline, log, read test) | 25% |
| Anomaly detector (spike, collapse, stale, ridge insufficiency) | 20% |
| Auto-recovery policies (rebuild, ridge boost, selective forget) | 20% |
| Policy comparison experiment | 20% |
| HealthMonitor integration + dashboard events | 15% |

### A-level
- Inspector visually distinguishes healthy vs. degenerate operators
- Anomaly detector catches all 4 anomaly types on synthetic workloads
- 3 recovery policies implemented and quantitatively compared
- HealthMonitor class is clean, configurable, and integrated
- Dashboard shows anomaly timeline with recovery events marked

### B-level
- Inspector works. Anomaly detector catches spikes.
- 1–2 recovery policies. Some comparison.

### C-level
- Inspector shows raw numbers only. No anomaly detection.
