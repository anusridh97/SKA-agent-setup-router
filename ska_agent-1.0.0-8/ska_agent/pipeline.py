"""
SKA-Agent Pipeline: end-to-end wiring of all components.

This file provides three pipeline classes at increasing levels of
complexity:

  OfflinePipeline (Stage I)
    Preprocesses raw corpus text into Segments:
    text -> sentences -> embeddings -> geometry learning -> segments
    Uses core/geometry.py for DP segmentation.

  RetrievalPipeline (Stage II)
    Combines pricing-guided retrieval with LLM generation:
    query -> embed -> pricing engine selects segments -> LLM generates answer
    Uses core/pricing.py and models/llm.py.

  SKAAgentPipeline (full system)
    Wires all six build phases plus the Qwen coordinator and TS integration.

    Build phases:
      Phase 1: Router infrastructure (encoder, mode selector, PID, sandbox)
      Phase 2: Reward predictor training on Phase 1 baseline scores
      Phase 3: Jamba+SKA surgery, LM recovery, table QA fine-tuning
      Phase 4: DeepSeek V3 hooks (MLA latent extraction)
      Phase 5: Shared spectral memory + think-Koopman bridge
      Phase 6: Multi-head Koopman (K=4 parallel operators)

    Additional build methods:
      build_coordinator(): Loads Qwen3.5-27B, sets up think bridge
      build_tool_server(): Creates FastAPI HTTP server for TS orchestrator

    The run() method implements three-tier dispatch:
      - If only coordinator is available, use it directly (fast path)
      - If router is available, use sequential marginal evaluation
      - Specialists are wired as callables that accept prefix_len

    Specialist lambdas accept prefix_len from the router so the SKA
    modules know how much of the input is context vs. query.

Dependencies:
  - Imports from every other subpackage
  - core/structures.py for SystemConfig
  - All model wrappers are imported lazily inside build methods
"""

from __future__ import annotations

import pickle
from typing import List, Optional, Tuple

import numpy as np

from .core.structures import (
    Segment, RetrievalResult, SystemConfig, SKAConfig,
    MultiKoopmanConfig,
)
from .core.geometry import GeometryLearner
from .core.pricing import PricingEngine
from .utils.math_utils import TextPreprocessor

# Torch-dependent model wrappers - imported lazily in methods that need them
# from .models.embedding import Embedder
# from .models.llm import LLMGenerator


# Offline Pipeline (Stage I)

class OfflinePipeline:
    """
    Stage I: Structure Learning.

    Transforms raw corpus text into stable, semantically coherent segments:
      1. Preprocessing: Split text into sentences
      2. Embedding: Dense vectorization
      3. Geometry Learning: DP-based optimal segmentation
    """

    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        lambda_seg: float = None,
        max_segment_size: int = 12,
        embedder=None,
    ):
        from .models.embedding import Embedder

        self.preprocessor = TextPreprocessor()
        self.embedder = embedder if embedder is not None else Embedder(embedding_model)
        self.geometry_learner = GeometryLearner(
            lambda_seg=lambda_seg,
            max_segment_size=max_segment_size,
        )

    def process(
        self,
        text: str,
        verbose: bool = True,
    ) -> Tuple[List[Segment], np.ndarray, List[str]]:
        """Process raw text into segments."""
        if verbose:
            print("Splitting text into sentences...")
        sentences = self.preprocessor.split_sentences(text)
        if verbose:
            print(f" Found {len(sentences)} sentences")

        if len(sentences) == 0:
            return [], np.array([]), []

        if verbose:
            print("Embedding sentences...")
        embeddings = self.embedder.embed(sentences, show_progress=verbose)

        if verbose:
            print("Learning geometry (segmentation)...")
        segments = self.geometry_learner.learn_geometry(embeddings, sentences)

        return segments, embeddings, sentences

    def save(self, segments: List[Segment], filepath: str):
        """Save segments to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(segments, f)
        print(f"Saved {len(segments)} segments to {filepath}")

    def load(self, filepath: str) -> List[Segment]:
        """Load segments from file."""
        with open(filepath, 'rb') as f:
            segments = pickle.load(f)
        print(f"Loaded {len(segments)} segments from {filepath}")
        return segments


# Retrieval Pipeline (Stage II)

class RetrievalPipeline:
    """
    Stage II: Pricing-Guided RAG.

    Integrates PricingEngine with LLM generation:
      1. Embed query
      2. Select segments via reduced-cost optimization
      3. Generate answer from selected context
    """

    def __init__(
        self,
        segments: List[Segment],
        embedder,
        generator,
        lambda_sparsity: float = 0.05,
        max_segments: int = 5,
    ):
        self.embedder = embedder
        self.generator = generator
        self.pricing_engine = PricingEngine(
            segments=segments,
            embed_fn=lambda q: embedder.embed_single(q),
            lambda_sparsity=lambda_sparsity,
            max_segments=max_segments,
        )

    def __call__(
        self,
        query: str,
        verbose: bool = True,
    ) -> Tuple[str, RetrievalResult]:
        """Run retrieval and generation."""
        result = self.pricing_engine.retrieve(query, verbose=verbose)

        if len(result.segments) == 0:
            return "No relevant information found.", result

        if verbose:
            print("Generating answer...")

        context = result.get_context(separator="\n\n")
        answer = self.generator.generate(query, context)

        return answer, result


# Full SKA-Agent Pipeline

class SKAAgentPipeline:
    """
    Full SKA-Agent system integrating all 6 phases.

    Components:
      Phase 1: Router infrastructure (encoder, mode selector, PID, sandbox)
      Phase 2: Reward predictor (trained on baseline scores)
      Phase 3: Jamba+SKA (attention surgery, LM recovery, table QA)
      Phase 4: DeepSeek V3 hooks (MLA extraction, interrupt trigger)
      Phase 5: Shared memory (operator write/read, bridge projection)
      Phase 6: Multi-head Koopman (K=4 parallel operators, slot specialization)

    Pipeline flow:
      Parser -> Router -> Retrieval -> Code -> Reasoning -> Answer
    """

    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()

        # Components (initialized on build)
        self.router = None
        self.jamba_ska = None
        self.deepseek = None
        self.shared_memory = None
        self.code_executor = None
        self.evaluator = None

        # Tier 1: Qwen coordinator
        self.coordinator = None
        self.think_bridge = None

        # TS orchestrator integration
        self.tool_server = None

        self._built = False

    @classmethod
    def from_config(cls, config: SystemConfig = None) -> SKAAgentPipeline:
        """Create pipeline from configuration."""
        pipeline = cls(config)
        return pipeline

    def build_phase1(self):
        """
        Phase 1: Router Infrastructure.

        Sets up: encoder, mode selector, PID controller, code sandbox,
        LLM API, OfficeQA eval harness.
        """
        from .router.adaptive_router import AdaptiveRouter
        from .evaluation.officeqa import CodeExecutor

        print("\n=== Building Phase 1: Router Infrastructure ===")
        self.router = AdaptiveRouter(config=self.config)
        self.code_executor = CodeExecutor()
        print("Phase 1 complete.")

    def build_phase2(self, training_data: Optional[list] = None):
        """
        Phase 2: Reward Predictor Training.

        Trains F_θr on Phase 1 baseline scores.
        Requires Phase 1 complete.
        """
        from .training.trainers import RouterTrainer

        if self.router is None:
            raise RuntimeError("Phase 1 must be built first")

        print("\n=== Building Phase 2: Reward Predictor ===")
        trainer = RouterTrainer(
            reward_predictor=self.router.reward_predictor,
            mode_selector=self.router.mode_selector,
            config=self.config.router_training,
        )

        if training_data:
            trainer.train_reward_predictor(training_data)
            trainer.train_mode_selector(training_data)

        print("Phase 2 complete.")

    def build_phase3(
        self,
        lm_dataset: str = "cerebras/SlimPajama-627B",
        num_tokens: int = None,
        use_multi_koopman: bool = False,
    ):
        """
        Phase 3: Jamba+SKA.

        Steps:
          1. Load Jamba-v0.1
          2. Attention->SKA surgery at layers {4, 12, 20, 28}
          3. SVD weight initialization
          4. Freeze non-SKA parameters
          5. LM recovery on SlimPajama
          6. Table QA fine-tuning
        """
        from .models.jamba_ska import JambaSKAModel, load_jamba_model
        from .training.trainers import SKATrainer

        print("\n=== Building Phase 3: Jamba+SKA ===")

        # Load Jamba
        jamba_model, tokenizer = load_jamba_model(self.config.jamba_model_name)

        # Surgery
        self.jamba_ska = JambaSKAModel(
            jamba_model=jamba_model,
            tokenizer=tokenizer,
            config=self.config,
            use_multi_koopman=use_multi_koopman,
        )

        # Verify
        self.jamba_ska.verify_logits()

        # Training
        trainer = SKATrainer(
            model=self.jamba_ska,
            tokenizer=tokenizer,
            config=self.config.ska_training,
        )

        # Stage 1: LM Recovery
        if num_tokens or self.config.ska_training.lm_recovery_tokens > 0:
            trainer.train_lm_recovery(
                dataset_name=lm_dataset,
                num_tokens=num_tokens,
            )

        print("Phase 3 complete.")

    def build_phase4(self):
        """
        Phase 4: DeepSeek V3 Hooks.

        Local DS-V3 deployment with MLA latent extraction,
        interrupt trigger, PID verbosity control.
        """
        from .shared_memory.spectral_memory import DeepSeekV3Integration

        print("\n=== Building Phase 4: DeepSeek V3 Hooks ===")
        self.deepseek = DeepSeekV3Integration(
            model_name=self.config.deepseek_model_name,
            latent_dim=self.config.deepseek_latent_dim,
            ska_rank=self.config.ska.rank,
        )
        self.deepseek.load_model()
        print("Phase 4 complete.")

    def build_phase5(self):
        """
        Phase 5: Shared Spectral Memory + Think-Koopman Bridge.

        Sets up:
          - SharedSpectralMemory for operator-level cross-agent comm
          - ThinkKoopmanBridge for coordinator reasoning -> slot 3
        No longer requires Phase 4 (DeepSeek is Tier 3, optional).
        """
        from .shared_memory.spectral_memory import SharedSpectralMemory
        from .shared_memory.think_koopman_bridge import ThinkKoopmanBridge

        print("\n=== Building Phase 5: Shared Memory + Think Bridge ===")
        self.shared_memory = SharedSpectralMemory(
            rank=self.config.ska.rank,
            ridge_eps=self.config.ska.ridge_eps,
            power_K=self.config.ska.power_K,
        )

        # Think-Koopman bridge: projects coordinator reasoning into slot 3
        mk = self.config.multi_koopman
        self.think_bridge = ThinkKoopmanBridge(
            hidden_size=3584, # Qwen3.5-27B hidden dim (updated on coordinator load)
            rank=mk.rank_per_operator,
            ridge_eps=mk.ridge_eps,
        )
        print("Phase 5 complete.")

    def build_phase6(self):
        """
        Phase 6: Multi-Head Koopman.

        K=4 parallel operators, agent-slot specialization,
        operator composition, dynamic K.
        """
        print("\n=== Building Phase 6: Multi-Head Koopman ===")

        if self.jamba_ska is not None:
            for key, module in self.jamba_ska.ska_modules.items():
                from .core.ska_module import MultiHeadKoopmanModule
                if isinstance(module, MultiHeadKoopmanModule):
                    print(f" {key}: {module.num_operators} operators, rank={module.rank_per_op}")
                else:
                    print(f" {key}: single operator (consider rebuilding with use_multi_koopman=True)")

        print("Phase 6 complete.")

    def build_coordinator(self):
        """
        Tier 1: Qwen3.5-27B-Claude-Distilled Coordinator.

        Primary reasoner with structured <think> chain-of-thought.
        Handles task decomposition, reasoning, and synthesis.
        ~16.5GB VRAM (Q4_K_M), 262K context.
        """
        from .models.qwen_coordinator import QwenCoordinator

        print("\n=== Building Tier 1: Qwen Coordinator ===")
        self.coordinator = QwenCoordinator(
            model_name=self.config.qwen_coordinator_name,
            quantization=self.config.qwen_quantization,
            think_layer_indices=list(self.config.qwen_think_layers),
        )
        self.coordinator.load()

        # Update think bridge hidden size to match loaded model
        if self.think_bridge:
            self.think_bridge.hidden_size = self.coordinator.hidden_size
            # Re-initialize projection matrix with correct dimensions
            W = np.random.randn(self.think_bridge.rank, self.coordinator.hidden_size)
            Q, _ = np.linalg.qr(W.T)
            self.think_bridge.W_think = Q[:, :self.think_bridge.rank].T.copy()
            # Reset accumulated state - old Gram/transition are inconsistent
            # with the new projection matrix
            self.think_bridge.reset()

        print("Tier 1 complete.")

    def build_tool_server(self):
        """
        Build the HTTP tool server for TS orchestrator integration.

        Exposes all Python specialists as HTTP endpoints that the
        TypeScript orchestrator's agents can call.
        """
        from .orchestration import ToolServer

        print("\n=== Building Tool Server ===")
        self.tool_server = ToolServer(
            coordinator=self.coordinator,
            retriever=None, # Set after phase 3 if retriever is built
            code_executor=self.code_executor,
            shared_memory=self.shared_memory,
            think_bridge=self.think_bridge,
        )
        self.tool_server.build_app()
        print("Tool server built (not yet serving).")

    def build_all(self, skip_training: bool = False):
        """Build all phases sequentially."""
        self.build_phase1()
        if not skip_training:
            self.build_phase2()
        self.build_phase3()
        self.build_phase4()
        self.build_phase5()
        self.build_phase6()
        self.build_coordinator()
        self.build_tool_server()
        self._built = True
        print("\n=== All phases built successfully ===")

    def run(
        self,
        query: str,
        verbose: bool = True,
    ) -> str:
        """
        Run the full pipeline on a query.

        Three-tier execution:
          Tier 1: Qwen coordinator reasons with <think> CoT
          Tier 2: Jamba+SKA retrieves relevant context
          Tier 3: DeepSeek V3 for hard multi-hop (cost-gated)

        The router decides which specialists to invoke based on
        S(a) = Δr̂ - λᵀΔĉ scoring.
        """
        if not self._built and self.router is None and self.coordinator is None:
            raise RuntimeError("Pipeline not built. Call build_coordinator() or build_all() first.")

        # Fast path: if only coordinator is available, use it directly
        if self.coordinator and self.router is None:
            result = self.coordinator.reason(query)
            if self.think_bridge and result.thinking:
                embedding = self.coordinator.extract_reasoning_state(result.thinking)
                self.think_bridge.accumulate(embedding)
            return result.answer

        specialists = {}

        # Tier 1: Qwen coordinator as primary reasoner
        if self.coordinator:
            def _coordinate(q, prefix_len=None):
                result = self.coordinator.reason(q)
                # Feed reasoning state into Koopman slot 3
                if self.think_bridge and result.thinking:
                    emb = self.coordinator.extract_reasoning_state(result.thinking)
                    self.think_bridge.accumulate(emb)
                return result.answer
            specialists['reasoner'] = _coordinate
            specialists['parser'] = lambda q: self.coordinator.reason(q, system_prompt="Extract structure from this document.").answer

        # Tier 2: SKA retriever
        if self.jamba_ska:
            specialists['ska_retriever'] = lambda q, prefix_len=None: self._ska_retrieve(q, prefix_len)

        # Code executor
        if self.code_executor:
            specialists['code_executor'] = lambda q: self.code_executor.execute(q)[0]

        # Tier 3: DeepSeek V3 heavy reasoner (only if available)
        if self.deepseek and self.shared_memory:
            specialists['heavy_reasoner'] = lambda q: self._reason_with_memory(q)

        # Route and execute
        if self.router:
            results = self.router.route(query, specialists=specialists, verbose=verbose)
            if results:
                return results[-1].output
            return "No actions taken."
        elif self.coordinator:
            return specialists['reasoner'](query)
        else:
            return "Pipeline not fully configured."

    def _ska_retrieve(self, query: str, prefix_len: int = None) -> str:
        """Retrieve using Jamba+SKA, forwarding prefix_len to the model."""
        # TODO: full implementation would tokenize query, run through
        # self.jamba_ska(input_ids, prefix_len=prefix_len), then decode.
        return f"[SKA Retrieved context for: {query}] (prefix_len={prefix_len})"

    def _reason_with_memory(self, query: str) -> str:
        """Reason using DeepSeek V3 with shared memory."""
        if self.deepseek and self.shared_memory:
            try:
                text, latents = self.deepseek.generate_with_memory(
                    query, self.shared_memory,
                )
                return text
            except Exception as e:
                return f"[Reasoning error: {e}]"
        return f"[Reasoning: {query}]"
