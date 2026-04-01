"""
Core data structures for the entire SKA-Agent system.

This file is imported by nearly every other module. It defines the typed
containers that flow through the system:

  Segment, RetrievalResult
    Output of the offline geometry learning pipeline (core/geometry.py)
    and the pricing-guided retrieval engine (core/pricing.py).
    A Segment is a contiguous block of sentences whose internal cosine
    distance is minimized by dynamic programming.

  CostVector
    The 5-dimensional cost representation used by the router:
    (input_tokens, output_tokens, latency_ms, dollar_cost, meta_overhead).
    The PID controller (router/pid_controller.py) adjusts a price vector
    lambda in R^5 that weights these dimensions.

  CollaborationMode, MODE_TEMPLATES, NODE_MODEL_MAPPING
    The four DAG templates that the router's mode selector classifies
    queries into: lookup, multi_doc, compute, multi_step. Each template
    defines valid transitions between node types (parse, retrieve, code,
    answer, etc.) and which specialist models can execute each node.

  ActionCandidate, ActionResult
    The router (router/adaptive_router.py) enumerates ActionCandidates
    from the current DAG node, scores them, and produces ActionResults.
    The prefix_len_hint field lets the router communicate to SKA modules
    how much of the input is context vs. query.

  SKAConfig, MultiKoopmanConfig
    Hyperparameters for the Structured Kernel Attention modules
    (core/ska_module.py). SKAConfig covers a single operator per head
    (rank=64, power_K=2, ridge=1e-3). MultiKoopmanConfig covers K=4
    parallel rank-48 operators with slot assignments:
      slots 0,1 = document structure (parser writes)
      slot 2    = reasoning state (coordinator writes via think bridge)
      slot 3    = temporal patterns
    Slot indices are 0-indexed throughout.

  SharedOperator
    The output of building a Koopman operator: the whitened operator
    A_w (r x r), Cholesky factor L (r x r), value readout B_v (d_v x r),
    plus monitoring metadata (condition number, tokens seen).
    Used by shared_memory/spectral_memory.py.

  SKATrainingConfig, RouterTrainingConfig, PIDConfig
    Hyperparameters for training (training/trainers.py).

  SystemConfig
    Top-level config aggregating all sub-configs plus model names,
    layer indices, and deployment settings. This is the single object
    you pass to SKAAgentPipeline to configure the entire system.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import numpy as np


# Stage I: Geometry / Segmentation

@dataclass
class Segment:
    """
    A semantic atomic unit derived from Stage I (Structure Learning).

    In the geometric framework, a Segment acts as a basis vector in the factor
    space. Unlike fixed-size chunks, these segments are learned by minimizing
    the incoherence (internal cost) within the window [start_idx, end_idx).

    Attributes:
        text: The full, concatenated text of the segment.
        vector: Centroid embedding μ_j of the constituent sentences.
        start_idx: Index of the first sentence in the original corpus.
        end_idx: Index of the last sentence (exclusive).
        sentences: Individual sentences contained in this unit.
        internal_cost: Sum of pairwise cosine distances within the segment.
    """
    text: str
    vector: np.ndarray
    start_idx: int
    end_idx: int
    sentences: List[str] = field(default_factory=list)
    internal_cost: float = 0.0

    def __len__(self) -> int:
        return self.end_idx - self.start_idx


@dataclass
class RetrievalResult:
    """
    Solution to the pricing-guided retrieval optimization (Stage II).

    Contains selected segments and their reduced-cost certificates.
    """
    segments: List[Segment]
    reduced_costs: List[float]
    total_segments_considered: int

    def get_context(self, separator: str = "\n\n") -> str:
        """Concatenate retrieved segments into context string."""
        return separator.join(seg.text for seg in self.segments)


# Cost Vector (§3.3 of the spec)

@dataclass
class CostVector:
    """
    5-dimensional cost vector ĉ(a) = (ĉ_in, ĉ_out, ĉ_lat, ĉ_$, ĉ_meta).

    Captures input tokens, output tokens, latency, dollar cost, and
    meta-overhead (interrupt cost for shared memory operations).
    """
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    latency_ms: float = 0.0
    dollar_cost: float = 0.0
    meta_overhead: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.input_tokens,
            self.output_tokens,
            self.latency_ms,
            self.dollar_cost,
            self.meta_overhead,
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> CostVector:
        return cls(
            input_tokens=float(arr[0]),
            output_tokens=float(arr[1]),
            latency_ms=float(arr[2]),
            dollar_cost=float(arr[3]),
            meta_overhead=float(arr[4]),
        )

    def __add__(self, other: CostVector) -> CostVector:
        return CostVector.from_array(self.to_array() + other.to_array())

    def __sub__(self, other: CostVector) -> CostVector:
        return CostVector.from_array(self.to_array() - other.to_array())


# Collaboration Modes (§3.1 Definition 3.1)

class CollaborationMode(enum.Enum):
    """
    Collaboration mode templates T ∈ T.

    Each mode defines a DAG template G_T = (V_T, E_T) specifying
    valid transitions between node types.
    """
    LOOKUP = "lookup" # Parse -> Retrieve -> Extract
    MULTI_DOC = "multi_doc" # Parse -> RetrieveA -> RetrieveB -> Compare
    COMPUTE = "compute" # Parse -> Retrieve -> Extract -> Code -> Answer
    MULTI_STEP = "multi_step" # Decompose -> [Retrieve/Code]* -> Synthesize


# DAG templates for each mode
MODE_TEMPLATES: Dict[CollaborationMode, Dict[str, List[str]]] = {
    CollaborationMode.LOOKUP: {
        "start": ["parse"],
        "parse": ["retrieve"],
        "retrieve": ["extract"],
        "extract": [], # terminal
    },
    CollaborationMode.MULTI_DOC: {
        "start": ["parse"],
        "parse": ["retrieve_a"],
        "retrieve_a": ["retrieve_b"],
        "retrieve_b": ["compare"],
        "compare": [],
    },
    CollaborationMode.COMPUTE: {
        "start": ["parse"],
        "parse": ["retrieve"],
        "retrieve": ["extract"],
        "extract": ["code"],
        "code": ["answer"],
        "answer": [],
    },
    CollaborationMode.MULTI_STEP: {
        "start": ["decompose"],
        "decompose": ["retrieve", "code"],
        "retrieve": ["retrieve", "code", "synthesize"],
        "code": ["retrieve", "code", "synthesize"],
        "synthesize": [],
    },
}


# Models valid for each node type
NODE_MODEL_MAPPING: Dict[str, List[str]] = {
    "parse": ["parser"],
    "retrieve": ["ska_retriever"],
    "retrieve_a": ["ska_retriever"],
    "retrieve_b": ["ska_retriever"],
    "extract": ["ska_retriever", "reasoner"],
    "compare": ["reasoner"],
    "code": ["code_executor"],
    "answer": ["reasoner"],
    "decompose": ["reasoner"],
    "synthesize": ["reasoner"],
}


# Router Action Types

@dataclass
class ActionCandidate:
    """
    A candidate action a = (v_t, v', m) in the router's decision space.

    source: current node v_t
    target: next node v'
    model: which specialist model to use
    prefix_len_hint: suggested prefix length for SKA modules (None = auto)
    """
    source: str
    target: str
    model: str
    estimated_cost: Optional[CostVector] = None
    predicted_reward: float = 0.0
    score: float = 0.0
    prefix_len_hint: Optional[int] = None


@dataclass
class ActionResult:
    """Result of executing an action."""
    action: ActionCandidate
    output: str
    actual_cost: CostVector
    quality: float = 0.0
    success: bool = True


# SKA Configuration (Table 1 in the spec)

@dataclass
class SKAConfig:
    """
    Configuration for an SKA module replacing a Jamba attention layer.

    Table 1 from the spec:
        d_model=4096, n_heads=32, head_dim=128, rank=64,
        power_K=2, chunk_size=128, ridge_eps=1e-3
    """
    d_model: int = 4096
    n_heads: int = 32
    head_dim: int = 128
    rank: int = 64
    power_K: int = 2
    chunk_size: int = 128
    ridge_eps: float = 1e-3
    spectral_gamma: float = 1.0 # γ ∈ [1.0, 1.5] for normalization


@dataclass
class MultiKoopmanConfig:
    """
    Configuration for multi-head Koopman operators (§7).

    K=4 parallel rank-48 operators per attention head.
    """
    num_operators: int = 4 # K parallel operators
    rank_per_operator: int = 48 # r per operator
    power_K: int = 2
    ridge_eps: float = 1e-3
    # Slot assignments (§7.4) - 0-indexed
    # Slot 0,1 -> parser (structure + values)
    # Slot 2 -> reasoning state (coordinator's <think> trace)
    # Slot 3 -> temporal patterns
    parser_slots: Tuple[int, ...] = (0, 1)
    reasoning_slot: int = 2
    temporal_slot: int = 3


# Shared Memory Types (§6)

@dataclass
class SharedOperator:
    """
    A shared Koopman operator for cross-agent communication.

    Contains the whitened operator A_w, value readout B_v,
    and Cholesky factor L from the Gram matrix.
    """
    A_w: np.ndarray # Whitened Koopman operator (r × r)
    B_v: np.ndarray # Value readout matrix
    L: np.ndarray # Cholesky factor of Gram matrix
    rank: int
    num_tokens_seen: int = 0
    condition_number: float = 1.0


# Training Configuration

@dataclass
class SKATrainingConfig:
    """Training hyperparameters for SKA modules (§8.1)."""
    # Stage 1: LM Recovery
    lm_recovery_tokens: int = 10_000_000_000 # 10B tokens
    lm_lr: float = 3e-4
    lm_warmup_steps: int = 1000
    lm_batch_size: int = 4
    lm_seq_length: int = 2048
    lm_target_ppl_ratio: float = 1.05 # within 5% of vanilla

    # Stage 2: Table QA
    tableqa_lr: float = 1e-4
    tableqa_epochs: int = 5

    # Regularization
    spectral_lambda: float = 0.01
    ortho_lambda: float = 0.01

    # Optimizer
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8


@dataclass
class RouterTrainingConfig:
    """Training hyperparameters for the router (§8.2)."""
    # Reward predictor
    reward_hidden_dim: int = 512
    reward_lr: float = 1e-3
    reward_epochs: int = 500
    reward_dropout: float = 0.1

    # Mode selector
    mode_hidden_dim: int = 256
    mode_lr: float = 1e-3
    mode_epochs: int = 200

    # Model embedding dimension
    model_embed_dim: int = 64


@dataclass
class PIDConfig:
    """PID controller gains (Eq. 25)."""
    Kp: float = 0.3
    Ki: float = 0.01
    Kd: float = 0.05
    lambda_max: np.ndarray = field(
        default_factory=lambda: np.array([5.0, 5.0, 2.0, 10.0, 3.0])
    )
    budget_rate: np.ndarray = field(
        default_factory=lambda: np.array([1000.0, 500.0, 100.0, 0.1, 1.0])
    )
    window_size: int = 10


# System-Level Config

@dataclass
class SystemConfig:
    """Top-level configuration aggregating all sub-configs."""
    ska: SKAConfig = field(default_factory=SKAConfig)
    multi_koopman: MultiKoopmanConfig = field(default_factory=MultiKoopmanConfig)
    ska_training: SKATrainingConfig = field(default_factory=SKATrainingConfig)
    router_training: RouterTrainingConfig = field(default_factory=RouterTrainingConfig)
    pid: PIDConfig = field(default_factory=PIDConfig)

    # Jamba surgery targets (§4.1)
    jamba_ska_layer_indices: Tuple[int, ...] = (4, 12, 20, 28)
    jamba_model_name: str = "ai21labs/Jamba-v0.1"

    # DeepSeek V3 (Tier 3 - heavy reasoning, cost-gated)
    deepseek_model_name: str = "deepseek-ai/DeepSeek-V3"
    deepseek_latent_dim: int = 512 # MLA compression dimension d_c

    # Qwen3.5 Coordinator (Tier 1 - primary reasoner)
    qwen_coordinator_name: str = "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"
    qwen_quantization: str = "auto" # "auto", "4bit", "8bit", "none"
    qwen_max_context: int = 262144 # 262K context window
    qwen_think_layers: Tuple[int, ...] = (-4, -3, -2, -1) # layers for reasoning extraction

    # Router encoder
    encoder_model_name: str = "all-MiniLM-L6-v2"
    encoder_dim: int = 384 # MiniLM-L6-v2 output dim

    # Bridge projection
    bridge_rank: int = 64 # r for W_bridge (r × 512)

    # TS Orchestrator integration
    tool_server_port: int = 8741
    tool_server_host: str = "0.0.0.0"
