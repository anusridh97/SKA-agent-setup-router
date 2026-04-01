# Evaluation: TypeScript Orchestrator

## Grading Breakdown

| Component | Weight | Description |
|-----------|--------|-------------|
| Scheduler correctness | 25% | Topological sort, parallel groups, cycle detection |
| Agent pool | 20% | HTTP dispatch, retries, timeouts, error propagation |
| End-to-end demo | 25% | Full pipeline works with mock and/or real server |
| Shared memory sync | 15% | Text + spectral memory, bidirectional |
| Code quality | 15% | TypeScript idioms, types, tests, documentation |

## What "Done" Looks Like

### A-level
- Scheduler handles arbitrary DAGs correctly (including diamonds, forks)
- Parallel dispatch works and measurably reduces latency vs. sequential
- Error handling is robust: retries, timeouts, partial result synthesis
- Shared memory syncs with Python spectral memory
- End-to-end demo on a multi-hop question with real Python ToolServer
- Comprehensive test suite with mock server

### B-level
- Scheduler works on simple DAGs
- Sequential dispatch (no parallelism)
- Basic error handling
- Demo works with mock server only

### C-level
- Scheduler has bugs on complex DAGs
- No shared memory
- Limited testing
