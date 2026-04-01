"""
TypeScript Orchestrator Integration Layer.

This file bridges the TypeScript multi-agent orchestrator (the .ts files
in the project: orchestrator.ts, team.ts, scheduler.ts, agent.ts) with
the Python SKA-Agent system.

The TS orchestrator provides:
  - Coordinator pattern: an LLM decomposes goals into task DAGs
  - Dependency-aware parallel execution via AgentPool + Scheduler
  - Four scheduling strategies: round-robin, least-busy, capability-match,
    dependency-first
  - Shared text memory with agent namespacing

The Python system provides:
  - Koopman spectral retrieval (Jamba+SKA)
  - Structured reasoning (Qwen coordinator with <think>)
  - Spectral shared memory (operators, not text)
  - Cost-aware routing with PID control

This file connects them via HTTP:

  ToolServer (FastAPI)
    POST /tools/reason      -> QwenCoordinator.reason()
    POST /tools/retrieve    -> SKA pricing-guided retrieval
    POST /tools/execute     -> Sandboxed Python code
    POST /tools/decompose   -> QwenCoordinator.decompose()
    GET  /memory/summary    -> Spectral + text memory stats
    POST /memory/write      -> Write to spectral shared memory
    POST /memory/read       -> Read from spectral shared memory
    GET  /roster            -> Agent configs for TS scheduler
    GET  /health            -> Health check

  TSMemoryBridge
    Synchronizes the TS orchestrator's text-based SharedMemory with
    Python's SharedSpectralMemory. Text entries are embedded, then
    projected to rank-r via a deterministic random projection (Johnson-
    Lindenstrauss lemma). The projection seed is configurable so multiple
    bridge instances use compatible projections.

  roster_to_ts_format(roster)
    Converts Python SpecialistConfig objects to the TS orchestrator's
    AgentConfig JSON format.

  DEFAULT_ROSTER
    The four specialists: coordinator (Qwen 27B), retriever (Jamba+SKA),
    coder (sandbox), heavy_reasoner (DeepSeek V3).

Dependencies:
  - models/qwen_coordinator.py for the coordinator
  - shared_memory/spectral_memory.py for SharedSpectralMemory
  - core/structures.py for CostVector, SystemConfig
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np

from ..core.structures import CostVector, SystemConfig, CollaborationMode


# Agent Roster (maps TS agents to Python specialists)

@dataclass
class SpecialistConfig:
    """Configuration for a Python specialist callable from the TS orchestrator."""
    name: str
    role: str
    model: str
    description: str
    capabilities: List[str] = field(default_factory=list)
    cost_estimate: CostVector = field(default_factory=CostVector)


# Default three-tier roster matching the TS orchestrator's team pattern
DEFAULT_ROSTER: List[SpecialistConfig] = [
    SpecialistConfig(
        name="coordinator",
        role="Task decomposition, reasoning, synthesis",
        model="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled",
        description="Primary reasoner with structured <think> chain-of-thought. "
                    "Decomposes goals, coordinates specialists, synthesizes results.",
        capabilities=["reasoning", "planning", "synthesis", "tool_calling"],
        cost_estimate=CostVector(2000, 1000, 500, 0.005, 0.0),
    ),
    SpecialistConfig(
        name="retriever",
        role="Document retrieval via Koopman spectral filtering",
        model="ai21labs/Jamba-v0.1 + SKA",
        description="Structured Kernel Attention retriever. Finds relevant segments "
                    "from parsed document corpus using pricing-guided selection.",
        capabilities=["retrieval", "table_qa", "extraction"],
        cost_estimate=CostVector(500, 200, 100, 0.01, 0.0),
    ),
    SpecialistConfig(
        name="coder",
        role="Sandboxed Python execution for computation",
        model="code_executor",
        description="Executes Python code for numerical computation, "
                    "statistical analysis, and data processing.",
        capabilities=["code", "computation", "analysis"],
        cost_estimate=CostVector(100, 50, 20, 0.001, 0.0),
    ),
    SpecialistConfig(
        name="heavy_reasoner",
        role="Complex multi-hop reasoning (cost-gated)",
        model="deepseek-ai/DeepSeek-V3",
        description="Heavy-weight reasoner for multi-hop questions. Only invoked "
                    "when Qwen coordinator can't solve alone (PID cost-gated).",
        capabilities=["multi_hop", "complex_reasoning", "long_context"],
        cost_estimate=CostVector(5000, 2000, 2000, 0.10, 0.5),
    ),
]


def roster_to_ts_format(roster: List[SpecialistConfig] = None) -> List[Dict]:
    """Convert Python roster to TS orchestrator's AgentConfig format."""
    roster = roster or DEFAULT_ROSTER
    return [
        {
            "name": s.name,
            "model": s.model,
            "systemPrompt": s.description,
            "tools": s.capabilities,
        }
        for s in roster
    ]


# Tool Server (HTTP API for TS agents)

class ToolServer:
    """
    HTTP API server exposing SKA-Agent tools to the TS orchestrator.

    Each endpoint corresponds to a specialist capability:
      POST /tools/reason -> QwenCoordinator.reason()
      POST /tools/retrieve -> JambaSKA retrieval pipeline
      POST /tools/execute -> CodeExecutor.execute()
      POST /tools/decompose -> QwenCoordinator.decompose()
      GET /memory/summary -> SharedSpectralMemory + ThinkBridge stats
      POST /memory/write -> Write to spectral shared memory
      POST /memory/read -> Read from spectral shared memory

    Usage:
        server = ToolServer(coordinator, retriever, executor, shared_memory)
        server.start(port=8741)
    """

    def __init__(
        self,
        coordinator=None,
        retriever=None,
        code_executor=None,
        shared_memory=None,
        think_bridge=None,
    ):
        self.coordinator = coordinator
        self.retriever = retriever
        self.code_executor = code_executor
        self.shared_memory = shared_memory
        self.think_bridge = think_bridge
        self._app = None

    def build_app(self):
        """Build the FastAPI application."""
        try:
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
        except ImportError:
            raise ImportError(
                "FastAPI required for ToolServer. Install with: "
                "pip install fastapi uvicorn"
            )

        app = FastAPI(
            title="SKA-Agent Tool Server",
            description="Python specialist tools for the TS orchestrator",
            version="1.0.0",
        )

        class ReasonRequest(BaseModel):
            query: str
            context: str = ""
            system_prompt: str = None
            temperature: float = 0.6
            max_tokens: int = 4096

        class RetrieveRequest(BaseModel):
            query: str
            max_segments: int = 5
            prefix_len: int = None

        class ExecuteRequest(BaseModel):
            code: str
            timeout: int = 30

        class DecomposeRequest(BaseModel):
            goal: str
            agents: List[Dict] = None

        class MemoryWriteRequest(BaseModel):
            keys: List[List[float]]
            values: List[List[float]] = None
            source: str = "unknown"

        class MemoryReadRequest(BaseModel):
            queries: List[List[float]]

        @app.post("/tools/reason")
        async def reason(req: ReasonRequest):
            if not self.coordinator:
                raise HTTPException(500, "Coordinator not loaded")

            result = self.coordinator.reason(
                query=req.query,
                context=req.context,
                system_prompt=req.system_prompt,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )

            # Extract reasoning state into Koopman slot 3
            if self.think_bridge and result.thinking:
                embedding = self.coordinator.extract_reasoning_state(result.thinking)
                self.think_bridge.accumulate(embedding)

                # Also accumulate individual steps if available
                if result.thinking_steps and len(result.thinking_steps) > 1:
                    self.think_bridge.accumulate_multi_step(
                        result.thinking_steps,
                        self.coordinator.extract_reasoning_state,
                    )

            return {
                "answer": result.answer,
                "thinking": result.thinking,
                "thinking_steps": result.thinking_steps,
                "thinking_tokens": result.thinking_tokens,
                "answer_tokens": result.answer_tokens,
                "latency_ms": result.latency_ms,
            }

        @app.post("/tools/retrieve")
        async def retrieve(req: RetrieveRequest):
            if not self.retriever:
                raise HTTPException(500, "Retriever not loaded")

            result = self.retriever.retrieve(req.query, verbose=False)
            return {
                "segments": [
                    {"text": s.text, "score": rc}
                    for s, rc in zip(result.segments, result.reduced_costs)
                ],
                "total_considered": result.total_segments_considered,
            }

        @app.post("/tools/execute")
        async def execute(req: ExecuteRequest):
            if not self.code_executor:
                raise HTTPException(500, "Code executor not loaded")

            output, success = self.code_executor.execute(req.code)
            return {"output": output, "success": success}

        @app.post("/tools/decompose")
        async def decompose(req: DecomposeRequest):
            if not self.coordinator:
                raise HTTPException(500, "Coordinator not loaded")

            agents = req.agents or roster_to_ts_format()
            tasks = self.coordinator.decompose(req.goal, agents)
            return {"tasks": tasks}

        @app.get("/memory/summary")
        async def memory_summary():
            summary = {}
            if self.shared_memory and self.shared_memory.operator:
                op = self.shared_memory.operator
                summary["spectral"] = {
                    "tokens_seen": op.num_tokens_seen,
                    "condition_number": op.condition_number,
                    "rank": op.rank,
                }
            if self.think_bridge:
                summary["reasoning_slot"] = self.think_bridge.get_stats()
            return summary

        @app.post("/memory/write")
        async def memory_write(req: MemoryWriteRequest):
            if not self.shared_memory:
                raise HTTPException(500, "Shared memory not initialized")

            keys = np.array(req.keys)
            values = np.array(req.values) if req.values else None
            self.shared_memory.write(keys, values, source_agent=req.source)
            return {"status": "ok", "tokens_seen": self.shared_memory.num_tokens}

        @app.post("/memory/read")
        async def memory_read(req: MemoryReadRequest):
            if not self.shared_memory:
                raise HTTPException(500, "Shared memory not initialized")

            queries = np.array(req.queries)
            outputs = self.shared_memory.read(queries)
            return {"outputs": outputs.tolist()}

        @app.get("/roster")
        async def get_roster():
            return roster_to_ts_format()

        @app.get("/health")
        async def health():
            return {
                "status": "ok",
                "coordinator_loaded": self.coordinator is not None and self.coordinator.model is not None,
                "retriever_loaded": self.retriever is not None,
                "code_executor_loaded": self.code_executor is not None,
                "shared_memory_active": self.shared_memory is not None,
                "think_bridge_active": self.think_bridge is not None,
            }

        self._app = app
        return app

    def start(self, host: str = "0.0.0.0", port: int = 8741):
        """Start the tool server."""
        import uvicorn

        if self._app is None:
            self.build_app()

        print(f"\nStarting SKA-Agent Tool Server on {host}:{port}")
        print(f" Endpoints:")
        print(f" POST /tools/reason - Qwen coordinator reasoning")
        print(f" POST /tools/retrieve - SKA document retrieval")
        print(f" POST /tools/execute - Sandboxed code execution")
        print(f" POST /tools/decompose - Task DAG decomposition")
        print(f" GET /memory/summary - Shared memory stats")
        print(f" POST /memory/write - Write to spectral memory")
        print(f" POST /memory/read - Read from spectral memory")
        print(f" GET /roster - Agent roster for TS orchestrator")
        print(f" GET /health - Health check")

        uvicorn.run(self._app, host=host, port=port)


# TS SharedMemory Bridge

class TSMemoryBridge:
    """
    Synchronizes TS orchestrator's text-based SharedMemory with
    Python's SharedSpectralMemory.

    The TS side stores agent findings as text (researcher/findings -> string).
    The Python side maintains Koopman operators for spectral retrieval.

    This bridge:
      1. Receives text entries from TS SharedMemory
      2. Embeds them into dense vectors
      3. Writes the vectors to SharedSpectralMemory
      4. Returns spectral read results as text summaries

    This gives the TS orchestrator the benefits of both:
      - Text-based shared memory for human-readable coordination
      - Spectral shared memory for efficient cross-agent retrieval
    """

    def __init__(
        self,
        spectral_memory,
        embedder=None,
        rank: int = 64,
        embed_dim: int = 384,
        projection_seed: int = 42,
        projection_matrix: np.ndarray = None,
    ):
        """
        Args:
            spectral_memory: SharedSpectralMemory instance to write into
            embedder: Sentence embedding model
            rank: Target dimension for projection
            embed_dim: Source embedding dimension
            projection_seed: Fixed seed for JL projection matrix. All
                TSMemoryBridge instances that need compatible projections
                must use the same seed (or share the same projection_matrix).
            projection_matrix: Explicit (rank, embed_dim) projection matrix.
                If provided, overrides seed-based generation. Use this when
                multiple bridges must share the exact same projection.
        """
        self.spectral_memory = spectral_memory
        self.embedder = embedder
        self.rank = rank
        self._text_store: Dict[str, str] = {}

        if projection_matrix is not None:
            self._projection = projection_matrix.copy()
        else:
            # Deterministic JL projection - same seed -> same matrix
            rng = np.random.RandomState(projection_seed)
            self._projection = rng.randn(rank, embed_dim) / np.sqrt(rank)

    def sync_from_ts(self, entries: List[Dict[str, str]]):
        """
        Sync text entries from TS SharedMemory into spectral memory.

        Embeds each entry, projects to rank dimensions via random projection
        (JL lemma - preserves distances, unlike naive truncation), then
        writes to spectral memory.

        Args:
            entries: List of {key, value, agent} dicts from TS
        """
        if not self.embedder:
            return

        for entry in entries:
            key = entry.get("key", "")
            value = entry.get("value", "")
            agent = entry.get("agent", "unknown")

            self._text_store[key] = value

            # Embed and project to rank dimensions
            embedding = self.embedder.embed_single(value)

            # Resize projection matrix if embed_dim doesn't match
            if self._projection.shape[1] != len(embedding):
                self._projection = np.random.randn(self.rank, len(embedding)) / np.sqrt(self.rank)

            projected = self._projection @ embedding # (rank,)

            self.spectral_memory.write(
                projected.reshape(1, -1),
                source_agent=agent,
            )

    def get_summary(self) -> str:
        """Generate a combined text + spectral summary."""
        lines = ["## Shared Memory (Text + Spectral)", ""]

        # Text entries
        if self._text_store:
            lines.append("### Text Entries")
            for key, value in self._text_store.items():
                display = value[:200] + "..." if len(value) > 200 else value
                lines.append(f"- {key}: {display}")
            lines.append("")

        # Spectral stats
        if self.spectral_memory and self.spectral_memory.operator:
            op = self.spectral_memory.operator
            lines.append("### Spectral Memory")
            lines.append(f"- Tokens seen: {op.num_tokens_seen}")
            lines.append(f"- Operator condition: {op.condition_number:.1f}")
            lines.append(f"- Rank: {op.rank}")

        return "\n".join(lines)
