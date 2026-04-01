# Starter: TypeScript Orchestrator

## Step 0: Project Setup

```bash
mkdir ska-orchestrator && cd ska-orchestrator
npm init -y
npm install typescript ts-node @types/node
npx tsc --init
```

tsconfig.json should have:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "commonjs",
    "strict": true,
    "esModuleInterop": true,
    "outDir": "./dist",
    "rootDir": "./src"
  }
}
```

## Step 1: Type Definitions

```typescript
// src/types.ts

export interface TaskSpec {
    title: string;
    description: string;
    assignee: string;        // "retriever" | "coder" | "reasoner" | "coordinator"
    dependsOn: string[];     // titles of tasks this depends on
}

export interface TaskResult {
    task: TaskSpec;
    output: string;
    success: boolean;
    latencyMs: number;
    error?: string;
}

export interface ReasonResponse {
    answer: string;
    thinking: string;
    thinking_steps: string[];
    thinking_tokens: number;
    answer_tokens: number;
    latency_ms: number;
}

export interface RetrieveResponse {
    segments: Array<{ text: string; score: number }>;
    total_considered: number;
}

export interface ExecuteResponse {
    output: string;
    success: boolean;
}

export interface DecomposeResponse {
    tasks: TaskSpec[];
}

export interface AgentConfig {
    name: string;
    model: string;
    systemPrompt: string;
    tools: string[];
}
```

## Step 2: Scheduler (the core algorithm)

```typescript
// src/scheduler.ts
import { TaskSpec, TaskResult } from './types';

export interface ScheduleLevel {
    /** Tasks at this level can all run in parallel */
    tasks: TaskSpec[];
}

export function buildSchedule(tasks: TaskSpec[]): ScheduleLevel[] {
    // TODO: Implement Kahn's algorithm to produce parallel levels.
    //
    // 1. Build adjacency list: task title → list of dependent task titles
    // 2. Compute in-degrees
    // 3. BFS: collect all in-degree-0 tasks as one level
    // 4. "Remove" them (decrement dependents' in-degrees)
    // 5. Repeat until empty
    //
    // If tasks remain but none have in-degree 0: cycle detected → error

    const titleToTask = new Map<string, TaskSpec>();
    const inDegree = new Map<string, number>();
    const dependents = new Map<string, string[]>(); // task → tasks that depend on it

    for (const task of tasks) {
        titleToTask.set(task.title, task);
        inDegree.set(task.title, task.dependsOn.length);
        for (const dep of task.dependsOn) {
            if (!dependents.has(dep)) dependents.set(dep, []);
            dependents.get(dep)!.push(task.title);
        }
    }

    const levels: ScheduleLevel[] = [];

    // YOUR IMPLEMENTATION HERE

    return levels;
}
```

## Step 3: Agent Pool

```typescript
// src/agent_pool.ts
import { TaskSpec, TaskResult } from './types';

const BASE_URL = 'http://localhost:8741';

export class AgentPool {
    private maxRetries = 3;
    private timeoutMs = 30000;

    async dispatch(task: TaskSpec, context: string): Promise<TaskResult> {
        const startTime = Date.now();

        // Map assignee to endpoint
        const endpointMap: Record<string, string> = {
            retriever: '/tools/retrieve',
            coder: '/tools/execute',
            reasoner: '/tools/reason',
            coordinator: '/tools/reason',
        };

        const endpoint = endpointMap[task.assignee];
        if (!endpoint) {
            return {
                task, output: '', success: false,
                latencyMs: 0, error: `Unknown assignee: ${task.assignee}`,
            };
        }

        // Build request body based on endpoint
        // TODO: format request body appropriately for each endpoint type

        for (let attempt = 0; attempt < this.maxRetries; attempt++) {
            try {
                const response = await this.fetchWithTimeout(
                    `${BASE_URL}${endpoint}`,
                    { /* request body */ },
                );

                return {
                    task,
                    output: JSON.stringify(response),
                    success: true,
                    latencyMs: Date.now() - startTime,
                };
            } catch (error) {
                if (attempt === this.maxRetries - 1) {
                    return {
                        task, output: '', success: false,
                        latencyMs: Date.now() - startTime,
                        error: String(error),
                    };
                }
                // Exponential backoff
                await new Promise(r => setTimeout(r, 1000 * 2 ** attempt));
            }
        }

        // Unreachable, but TypeScript wants it
        throw new Error('Unreachable');
    }

    private async fetchWithTimeout(url: string, body: any): Promise<any> {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), this.timeoutMs);

        try {
            const res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
                signal: controller.signal,
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return res.json();
        } finally {
            clearTimeout(timeout);
        }
    }
}
```

## Step 4: Mock Server for Testing

```typescript
// tests/mock_server.ts
import http from 'http';

export function createMockServer(port = 8741): http.Server {
    const server = http.createServer((req, res) => {
        let body = '';
        req.on('data', chunk => body += chunk);
        req.on('end', () => {
            res.setHeader('Content-Type', 'application/json');

            if (req.url === '/tools/decompose') {
                res.end(JSON.stringify({
                    tasks: [
                        { title: "Retrieve data", description: "Get the numbers",
                          assignee: "retriever", dependsOn: [] },
                        { title: "Compute result", description: "Calculate answer",
                          assignee: "coder", dependsOn: ["Retrieve data"] },
                    ]
                }));
            } else if (req.url === '/tools/retrieve') {
                res.end(JSON.stringify({
                    segments: [{ text: "The value was $1.2 trillion", score: -0.5 }],
                    total_considered: 100,
                }));
            } else if (req.url === '/tools/reason') {
                const parsed = JSON.parse(body);
                res.end(JSON.stringify({
                    answer: `Reasoned about: ${parsed.query?.substring(0, 50)}`,
                    thinking: "Let me analyze this...",
                    thinking_steps: ["Step 1", "Step 2"],
                    thinking_tokens: 100,
                    answer_tokens: 50,
                    latency_ms: 200,
                }));
            } else if (req.url === '/tools/execute') {
                res.end(JSON.stringify({ output: "42", success: true }));
            } else if (req.url === '/health') {
                res.end(JSON.stringify({ status: "ok" }));
            } else {
                res.statusCode = 404;
                res.end(JSON.stringify({ error: "Not found" }));
            }
        });
    });

    return server;
}
```

Test against this before connecting to the real Python server.

## Step 5: Orchestrator

```typescript
// src/orchestrator.ts
import { buildSchedule } from './scheduler';
import { AgentPool } from './agent_pool';
import { TaskSpec, TaskResult } from './types';

export class Orchestrator {
    private pool = new AgentPool();
    private results = new Map<string, TaskResult>();

    async run(goal: string): Promise<string> {
        // 1. Decompose goal into task DAG
        const tasks = await this.decompose(goal);
        console.log(`Decomposed into ${tasks.length} tasks`);

        // 2. Build parallel schedule
        const levels = buildSchedule(tasks);
        console.log(`Schedule has ${levels.length} levels`);

        // 3. Execute level by level
        for (const level of levels) {
            console.log(`Executing level: ${level.tasks.map(t => t.title)}`);

            // Gather context from completed dependencies
            const dispatches = level.tasks.map(task => {
                const context = this.gatherContext(task);
                return this.pool.dispatch(task, context);
            });

            // Parallel execution within level
            const results = await Promise.all(dispatches);

            for (const result of results) {
                this.results.set(result.task.title, result);
                if (!result.success) {
                    console.error(`Task failed: ${result.task.title}: ${result.error}`);
                }
            }
        }

        // 4. Synthesize final answer from results
        return this.synthesize(goal);
    }

    private gatherContext(task: TaskSpec): string {
        // Collect outputs from dependency tasks
        return task.dependsOn
            .map(dep => this.results.get(dep)?.output ?? '')
            .filter(Boolean)
            .join('\n\n');
    }

    private async decompose(goal: string): Promise<TaskSpec[]> {
        // TODO: call /tools/decompose
        return [];
    }

    private async synthesize(goal: string): Promise<string> {
        // TODO: call /tools/reason with all results as context
        return '';
    }
}
```
