# Background: Task DAG Orchestration

## 1. Directed Acyclic Graphs (DAGs) for Task Scheduling

A task DAG G = (V, E) has:
- Vertices V = {t₁, t₂, ... tₙ}: tasks to execute
- Edges E: (tᵢ → tⱼ) means tⱼ depends on tᵢ (tᵢ must complete first)
- No cycles (otherwise tasks can't be ordered)

A **topological sort** of G is a linear ordering of vertices such that
for every edge (tᵢ → tⱼ), tᵢ appears before tⱼ. This ordering exists
iff G is acyclic.

**Kahn's algorithm** (BFS-based topological sort):
```
1. Compute in-degree of each vertex
2. Initialize queue Q with all vertices of in-degree 0
3. While Q is not empty:
   a. Remove vertex v from Q
   b. Output v
   c. For each neighbor w of v:
      - Decrease in-degree of w by 1
      - If in-degree of w becomes 0, add w to Q
```

For parallel execution, we modify step 3a: instead of removing one
vertex at a time, we remove ALL vertices with in-degree 0 and dispatch
them as a parallel batch.

## 2. The SKA-Agent Collaboration Modes

The Python codebase defines 4 collaboration modes (in `structures.py`),
each with a DAG template:

**LOOKUP:** Parse → Retrieve → Extract
Simple single-document questions.

**MULTI_DOC:** Parse → Retrieve_A → Retrieve_B → Compare
Questions requiring comparison across documents.

**COMPUTE:** Parse → Retrieve → Extract → Code → Answer
Questions requiring numerical computation.

**MULTI_STEP:** Decompose → [Retrieve/Code]* → Synthesize
Complex multi-hop reasoning with iterative retrieval/computation.

The TS orchestrator generalizes these templates. Instead of hardcoded
DAGs, the coordinator generates task-specific DAGs via `/tools/decompose`.

## 3. HTTP Communication Pattern

The Python ToolServer uses FastAPI with these endpoints:

| Endpoint | Method | Request Body | Response |
|----------|--------|-------------|----------|
| /tools/reason | POST | {query, context, temperature} | {answer, thinking, thinking_steps} |
| /tools/retrieve | POST | {query, max_segments} | {segments: [{text, score}]} |
| /tools/execute | POST | {code, timeout} | {output, success} |
| /tools/decompose | POST | {goal, agents} | {tasks: [{title, description, assignee, dependsOn}]} |
| /memory/write | POST | {keys, values, source} | {status, tokens_seen} |
| /memory/read | POST | {queries} | {outputs} |
| /roster | GET |, | [{name, model, systemPrompt, tools}] |
| /health | GET |, | {status, coordinator_loaded, ...} |

All communication is JSON over HTTP. The TS side should use `fetch` or
a thin HTTP client wrapper.

## 4. Concurrency in Node.js / TypeScript

Node.js uses an event-loop model with async/await for concurrency.
Multiple HTTP requests can be in-flight simultaneously:

```typescript
// Parallel dispatch
const results = await Promise.all([
    fetch('/tools/retrieve', { body: task1 }),
    fetch('/tools/retrieve', { body: task2 }),
]);

// Sequential dispatch (for dependent tasks)
const result1 = await fetch('/tools/retrieve', { body: task1 });
const result2 = await fetch('/tools/reason', {
    body: { query: task2.description, context: result1.answer }
});
```

The scheduler's job is to determine which pattern to use for each group
of tasks, based on the DAG structure.

## 5. Shared Memory Design Pattern

The dual-mode shared memory (text + spectral) follows a standard
pattern in multi-agent systems:

**Text store** (fast, local, human-readable):
```typescript
interface SharedMemory {
    entries: Map<string, { value: string; agent: string; timestamp: number }>;
    get(key: string): string | undefined;
    set(key: string, value: string, agent: string): void;
    getSummary(): string;
}
```

**Spectral store** (Python-backed, semantic retrieval):
```typescript
interface SpectralMemory {
    write(keys: number[][], source: string): Promise<void>;
    read(queries: number[][]): Promise<number[][]>;
}
```

The TS orchestrator writes to both after each task completes. Agents
read from whichever is more appropriate (text for exact lookups,
spectral for semantic similarity).

## 6. References

1. Kahn. "Topological Sorting of Large Networks." CACM, 1962.
2. Coffman & Graham. "Optimal Scheduling for Two-Processor Systems." 1972. Classic parallel scheduling on DAGs.
3. The Python `orchestration/__init__.py` source code, read every line.
