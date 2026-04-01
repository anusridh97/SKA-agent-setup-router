"""
SKA-Agent: Adaptive Multi-Model Orchestration with Structured Kernel
Attention and Spectral Shared Memory.

This package implements a three-tier multi-agent system for enterprise
document reasoning. The tiers are:

  Tier 1 (coordinator): Qwen3.5-27B-Claude-Distilled
    Handles task decomposition, structured reasoning with <think> tags,
    and final answer synthesis. Runs on ~16.5GB VRAM.

  Tier 2 (retriever): Jamba-v0.1 with SKA attention replacement
    Performs document retrieval using Koopman spectral filtering.
    The 4 attention layers at indices {4, 12, 20, 28} are replaced
    with Structured Kernel Attention modules. Runs on ~52GB INT8.

  Tier 3 (heavy reasoner): DeepSeek V3 (cost-gated)
    Only invoked for hard multi-hop questions when the PID controller
    determines the marginal reward justifies the cost.

Agents communicate through Koopman operators (fixed-size r x r matrices)
instead of serializing text, giving O(r^2) memory independent of context
length. The adaptive router scores each candidate action as:

    S(a) = predicted_reward - lambda^T * marginal_cost

and only executes actions with positive score.

Package structure:
    ska_agent/
        core/           Data structures, SKA module, geometry, pricing
        models/         Model wrappers (Qwen, Jamba+SKA, LLM, embedder)
        router/         Adaptive router, PID controller
        shared_memory/  Spectral memory protocol, think-Koopman bridge
        training/       Training loops for SKA, router, bridge
        evaluation/     OfficeQA benchmark evaluation
        orchestration/  TS orchestrator integration (FastAPI tool server)
        pipeline.py     End-to-end pipeline wiring all components
        cli.py          Command-line interface

Usage:
    from ska_agent import SKAAgentPipeline, SystemConfig
    agent = SKAAgentPipeline(SystemConfig())
    agent.build_coordinator()
    answer = agent.run("What was total federal debt in 1945?")
"""

__version__ = "1.0.0"
