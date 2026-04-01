# Project: TypeScript Orchestrator Integration

## Summary

Build the TypeScript side of the multi-agent orchestrator that communicates
with the Python `ToolServer` via HTTP. Implement task DAG execution, parallel
agent dispatch, dependency resolution, and shared memory synchronization.

**Area:** Systems
**GPU:** None
**Duration:** 8 weeks
**Team size:** 2–3

## Motivation

The Python codebase defines a complete `ToolServer` (in `orchestration/__init__.py`)
with endpoints for reasoning, retrieval, code execution, task decomposition,
and shared memory read/write. But there's no client. The TS orchestrator is
the "brain" that coordinates which agents to call, in what order, and how
to combine results.

The existing Python `AdaptiveRouter` does sequential action selection (one
action at a time, greedy). The TS orchestrator should support **parallel
execution** of independent tasks in a DAG, which is strictly more powerful.

## Deliverables

1. **`orchestrator.ts`**: Main coordinator that accepts a goal, calls the
   Python `/tools/decompose` endpoint to get a task DAG, then executes it.

2. **`scheduler.ts`**: Dependency-aware task scheduler. Given a DAG of tasks,
   determines which can run in parallel and dispatches them.

3. **`agent_pool.ts`**: Pool of agent workers, each configured to call specific
   Python tool endpoints. Handles retries, timeouts, and error propagation.

4. **`shared_memory.ts`**: Client-side shared memory that syncs with Python's
   `SharedSpectralMemory` via the `/memory/read` and `/memory/write` endpoints.
   Also maintains a text-based summary (like `TSMemoryBridge`).

5. **End-to-end demo**: Given a multi-hop question, the orchestrator decomposes
   it into subtasks, dispatches them (some in parallel), collects results, and
   synthesizes a final answer.

6. **Test suite**: Including mock Python server for unit testing without GPU.

## Architecture

```
┌─────────────────────────────────────────────┐
│  TypeScript Orchestrator (Node.js)          │
│                                             │
│  orchestrator.ts                            │
│    ├── Receives goal                        │
│    ├── POST /tools/decompose → task DAG     │
│    ├── scheduler.ts                         │
│    │   ├── Topological sort                 │
│    │   ├── Identify parallel groups         │
│    │   └── Dispatch to agent_pool           │
│    ├── agent_pool.ts                        │
│    │   ├── AgentWorker("retriever")         │
│    │   │   └── POST /tools/retrieve         │
│    │   ├── AgentWorker("coder")             │
│    │   │   └── POST /tools/execute          │
│    │   └── AgentWorker("reasoner")          │
│    │       └── POST /tools/reason           │
│    └── shared_memory.ts                     │
│        ├── POST /memory/write               │
│        ├── POST /memory/read                │
│        └── Local text store                 │
└─────────────┬───────────────────────────────┘
              │ HTTP (localhost:8741)
┌─────────────▼───────────────────────────────┐
│  Python ToolServer (FastAPI)                │
│  (Already implemented in ska_agent)         │
└─────────────────────────────────────────────┘
```

## Where This Fits in the Codebase

```
ska_agent/orchestration/__init__.py
    ToolServer          ← the Python HTTP server (already done)
    TSMemoryBridge      ← Python-side memory sync (already done)
    roster_to_ts_format ← converts Python roster to TS config (already done)
    DEFAULT_ROSTER      ← agent configurations (already done)

NEW files you create (in a separate ts/ directory):
    ts/src/orchestrator.ts
    ts/src/scheduler.ts
    ts/src/agent_pool.ts
    ts/src/shared_memory.ts
    ts/src/types.ts         ← TypeScript interfaces matching Python models
    ts/tests/mock_server.ts ← mock Python server for testing
    ts/tests/*.test.ts
```

## Key Design Decisions

### Task DAG Format

The Python `/tools/decompose` endpoint returns:
```json
[
  {"title": "Retrieve FY2023 data", "description": "...",
   "assignee": "retriever", "dependsOn": []},
  {"title": "Retrieve FY2022 data", "description": "...",
   "assignee": "retriever", "dependsOn": []},
  {"title": "Compare years", "description": "...",
   "assignee": "reasoner", "dependsOn": ["Retrieve FY2023 data", "Retrieve FY2022 data"]}
]
```

Tasks 1 and 2 have no dependencies → dispatch in parallel.
Task 3 depends on both → wait, then dispatch.

### Shared Memory Protocol

The TS orchestrator maintains a text-based shared memory (key-value store
of agent findings). After each task completes, the result is:
1. Stored locally in TS shared memory (text)
2. Embedded and written to Python spectral memory via `/memory/write`

When an agent needs context from another agent's work, it can:
1. Read the text summary (fast, local)
2. Query the spectral memory via `/memory/read` (slower, but captures
   semantic relationships)

### Error Handling

- If a tool call fails, retry up to 3 times with exponential backoff
- If a task fails permanently, mark it as failed and propagate to dependents
- Dependent tasks that can't run should be skipped with an error message
- The orchestrator should still synthesize a partial answer from successful tasks

## Milestones

| Week | Milestone |
|------|-----------|
| 1–2 | TypeScript project setup. Types matching Python models. Mock server. |
| 3–4 | scheduler.ts: topological sort, parallel group identification. |
| 5–6 | agent_pool.ts: HTTP client, retries, timeouts. Basic orchestrator. |
| 7–8 | shared_memory.ts. End-to-end demo with mock server. |
| 8 | Integration test with real Python ToolServer. Error handling. |
