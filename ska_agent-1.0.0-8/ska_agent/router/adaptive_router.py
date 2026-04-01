"""
Adaptive Inference Router.

This file implements the sequential marginal evaluation algorithm that
decides which specialist handles each part of a query. It is the
decision-making brain of the multi-agent system.

The router has four components:

  QueryEncoder (frozen MiniLM-L6-v2, 22M params)
    Encodes the input query into a 384-dim dense vector e_Q.
    Not trained. ~5ms latency.

  ModeSelector (2-layer MLP, ~100K params)
    Classifies the query into one of four collaboration modes:
    lookup, multi_doc, compute, multi_step. Each mode defines a DAG
    template of valid transitions between node types.
    Trained on cross-entropy over best-mode labels from Phase 1 baselines.

  RewardPredictor (3-layer MLP, ~400K params)
    Predicts the quality improvement from choosing model m over baseline:
    delta_r_hat(a) = g(e_Q, e_m - e_m_base)
    where e_m are learned 64-dim model embeddings.
    Trained on MSE regression over observed quality differences.

  PIDController (no learned params, defined in router/pid_controller.py)
    Adjusts the 5-dim price vector lambda based on spending rate.

The decision rule at each step:
  1. Enumerate candidate actions from the current DAG node
  2. Score each: S(a) = delta_r_hat(a) - lambda^T * delta_c_hat(a)
  3. If max score > 0, execute that action and advance the DAG
  4. If max score <= 0, terminate and return accumulated output

The execution graph is NOT predicted up front. It emerges from a
sequence of local positive-score decisions. Simple queries terminate
after 1-2 actions; complex multi-hop queries chain 4-5 specialists.

ModeSelector and RewardPredictor require torch (they are nn.Module
subclasses). When torch is not available, stub classes are defined
that raise ImportError on instantiation. The module remains importable
either way.

Dependencies:
  - core/structures.py for CostVector, CollaborationMode, MODE_TEMPLATES
  - router/pid_controller.py for the PID cost controller
  - pipeline.py wires this router with specialist callables
"""

from __future__ import annotations

import math
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING
from collections import deque

import numpy as np

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

def _import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F

from ..core.structures import (
    CostVector, CollaborationMode, ActionCandidate, ActionResult,
    MODE_TEMPLATES, NODE_MODEL_MAPPING,
    PIDConfig, RouterTrainingConfig, SystemConfig,
)


# Query Encoder (§5.1.1)

class QueryEncoder:
    """
    Frozen Sentence-BERT encoder (§5.1.1, Eq. 20).

    F_θe: Q -> e_Q ∈ R^D

    Uses all-MiniLM-L6-v2 (22M params, ~5ms latency).
    This is NOT trained - frozen pretrained weights.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"QueryEncoder ready: {model_name} (dim={self.embedding_dim})")

    def encode(self, query: str) -> np.ndarray:
        """Encode a single query to a dense vector."""
        import torch
        with torch.no_grad():
            embedding = self.model.encode(
                [query],
                show_progress_bar=False,
                convert_to_numpy=True,
            )[0]
        return embedding.astype(np.float64)

    def encode_batch(self, queries: List[str]) -> np.ndarray:
        """Encode a batch of queries."""
        import torch
        with torch.no_grad():
            embeddings = self.model.encode(
                queries,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return embeddings.astype(np.float64)


# Mode Selector (§5.1.2)
# nn.Module classes (ModeSelector, RewardPredictor) require torch.
# Defined inside try/except so the module is importable without torch.

try:
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    _HAS_TORCH = True

    class ModeSelector(_nn.Module):
        """4-way softmax classifier (§5.1.2, Eq. 21)."""

        def __init__(self, input_dim: int = 384, hidden_dim: int = 256,
                     num_modes: int = 4, temperature: float = 1.0):
            super().__init__()
            self.temperature = temperature
            self.num_modes = num_modes
            self.mlp = _nn.Sequential(
                _nn.Linear(input_dim, hidden_dim), _nn.ReLU(), _nn.Dropout(0.1),
                _nn.Linear(hidden_dim, num_modes),
            )
            self.index_to_mode = {
                0: CollaborationMode.LOOKUP, 1: CollaborationMode.MULTI_DOC,
                2: CollaborationMode.COMPUTE, 3: CollaborationMode.MULTI_STEP,
            }
            self.mode_to_index = {v: k for k, v in self.index_to_mode.items()}

        def forward(self, query_embedding):
            return self.mlp(query_embedding)

        def predict(self, query_embedding: np.ndarray) -> Tuple[CollaborationMode, np.ndarray]:
            self.eval()
            with _torch.no_grad():
                x = _torch.tensor(query_embedding, dtype=_torch.float32).unsqueeze(0)
                logits = self.forward(x)
                probs = _F.softmax(logits / self.temperature, dim=-1).squeeze(0).numpy()
            mode_idx = int(np.argmax(probs))
            return self.index_to_mode[mode_idx], probs

    class RewardPredictor(_nn.Module):
        """Marginal reward predictor (§5.1.3, Eq. 22-24)."""

        def __init__(self, query_dim: int = 384, model_embed_dim: int = 64,
                     hidden_dim: int = 512, num_models: int = 4, dropout: float = 0.1):
            super().__init__()
            self.model_embed_dim = model_embed_dim
            self.model_names = ["ska_retriever", "code_executor", "reasoner", "parser"]
            self.model_embeddings = _nn.Embedding(num_models, model_embed_dim)
            self.model_to_idx = {name: i for i, name in enumerate(self.model_names)}
            self.mlp = _nn.Sequential(
                _nn.Linear(query_dim + model_embed_dim, hidden_dim), _nn.ReLU(), _nn.Dropout(dropout),
                _nn.Linear(hidden_dim, hidden_dim // 2), _nn.ReLU(), _nn.Dropout(dropout),
                _nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, query_embedding, model_idx, base_model_idx):
            e_m = self.model_embeddings(model_idx)
            e_base = self.model_embeddings(base_model_idx)
            x = _torch.cat([query_embedding, e_m - e_base], dim=-1)
            return self.mlp(x)

        def predict(self, query_embedding: np.ndarray, model_name: str, base_model_name: str) -> float:
            self.eval()
            with _torch.no_grad():
                e_q = _torch.tensor(query_embedding, dtype=_torch.float32).unsqueeze(0)
                m_idx = _torch.tensor([self.model_to_idx.get(model_name, 0)])
                b_idx = _torch.tensor([self.model_to_idx.get(base_model_name, 0)])
                delta_r = self.forward(e_q, m_idx, b_idx)
            return delta_r.item()

except ImportError:
    _HAS_TORCH = False

    class ModeSelector: # type: ignore[no-redef]
        """Stub - torch not available."""
        def __init__(self, *a, **kw):
            raise ImportError("ModeSelector requires torch")

    class RewardPredictor: # type: ignore[no-redef]
        """Stub - torch not available."""
        def __init__(self, *a, **kw):
            raise ImportError("RewardPredictor requires torch")


# PID Controller (§5.1.4)

class PIDController:
    """
    PID cost controller (§5.1.4, Eq. 18, 25).

    No learned parameters - pure feedback control.

    λ_{t+1} = Clip(λ_t + K_p·e_t + K_i·Σe_τ + K_d·(e_t - e_{t-1}), 0, λ_max)

    where e_t = (1/ΔT)·Σc_τ - b is the burn rate error against budget b.

    Default gains (Eq. 25):
        K_p = 0.3, K_i = 0.01, K_d = 0.05
        λ_max = (5, 5, 2, 10, 3)
    """

    def __init__(self, config: PIDConfig = None):
        self.config = config or PIDConfig()

        # Current price vector λ ∈ R^5
        self.lambda_vec = np.ones(5) * 0.1 # Start with small prices

        # Error history for integral and derivative terms
        self.cost_history: deque = deque(maxlen=self.config.window_size)
        self.integral_error = np.zeros(5)
        self.prev_error = np.zeros(5)

    def update(self, cost: CostVector) -> np.ndarray:
        """
        Update price vector based on observed cost.

        Args:
            cost: Observed cost from the last action

        Returns:
            lambda_vec: Updated 5-dimensional price vector
        """
        cost_arr = cost.to_array()
        self.cost_history.append(cost_arr)

        # Compute burn rate error
        if len(self.cost_history) > 0:
            window = np.array(list(self.cost_history))
            avg_cost = window.mean(axis=0)
            error = avg_cost - self.config.budget_rate
        else:
            error = np.zeros(5)

        # PID update (Eq. 18)
        self.integral_error += error
        derivative = error - self.prev_error

        delta = (
            self.config.Kp * error
            + self.config.Ki * self.integral_error
            + self.config.Kd * derivative
        )

        self.lambda_vec = np.clip(
            self.lambda_vec + delta,
            0.0,
            self.config.lambda_max,
        )

        self.prev_error = error.copy()
        return self.lambda_vec

    def reset(self):
        """Reset controller state."""
        self.lambda_vec = np.ones(5) * 0.1
        self.cost_history.clear()
        self.integral_error = np.zeros(5)
        self.prev_error = np.zeros(5)


# Action Scorer

class ActionScorer:
    """
    Computes action scores (Definition 2, Eq. 13).

    S(a) = F_θr(e_Q, a.model, a_base.model) [predicted marginal reward]
           - λᵀ (ĉ(a) - ĉ(a_base)) [priced marginal cost]
    """

    # Historical cost statistics per model (initialized with estimates)
    DEFAULT_COSTS: Dict[str, CostVector] = {
        "ska_retriever": CostVector(500, 200, 50, 0.01, 0.0),
        "code_executor": CostVector(100, 50, 10, 0.001, 0.0),
        "reasoner": CostVector(2000, 1000, 200, 0.05, 0.5),
        "parser": CostVector(300, 100, 100, 0.005, 0.0),
    }

    def __init__(
        self,
        reward_predictor: RewardPredictor,
        pid_controller: PIDController,
        cost_stats: Optional[Dict[str, CostVector]] = None,
    ):
        self.reward_predictor = reward_predictor
        self.pid = pid_controller
        self.cost_stats = cost_stats or dict(self.DEFAULT_COSTS)

    def score_action(
        self,
        query_embedding: np.ndarray,
        action: ActionCandidate,
        base_model: str = "ska_retriever",
    ) -> float:
        """
        Score a candidate action (Eq. 13).

        S(a) = Δr̂(a) - λᵀ · Δĉ(a)
        """
        # Predicted marginal reward
        delta_r = self.reward_predictor.predict(
            query_embedding, action.model, base_model,
        )

        # Estimated marginal cost
        action_cost = self.cost_stats.get(action.model, CostVector()).to_array()
        base_cost = self.cost_stats.get(base_model, CostVector()).to_array()
        delta_c = action_cost - base_cost

        # Priced marginal cost
        priced_cost = np.dot(self.pid.lambda_vec, delta_c)

        score = delta_r - priced_cost
        action.predicted_reward = delta_r
        action.estimated_cost = CostVector.from_array(action_cost)
        action.score = score

        return score


# Router (Main Orchestrator)

class AdaptiveRouter:
    """
    Adaptive Inference Router - orchestrates sequential marginal evaluation.

    Algorithm 2: Sequential Marginal Evaluation
    1. Encode query
    2. Select collaboration mode
    3. For each step:
       a. Enumerate candidate actions from DAG template
       b. Score each action: S(a) = Δr̂ - λᵀ·Δĉ
       c. Execute best if S(a*) > 0, else terminate
    4. Update statistics
    """

    def __init__(
        self,
        config: SystemConfig = None,
        encoder: Optional[QueryEncoder] = None,
        mode_selector: Optional[ModeSelector] = None,
        reward_predictor: Optional[RewardPredictor] = None,
        pid_controller: Optional[PIDController] = None,
    ):
        self.config = config or SystemConfig()

        # Initialize components
        self.encoder = encoder or QueryEncoder(self.config.encoder_model_name)
        self.mode_selector = mode_selector or ModeSelector(
            input_dim=self.encoder.embedding_dim,
            hidden_dim=self.config.router_training.mode_hidden_dim,
        )
        self.reward_predictor = reward_predictor or RewardPredictor(
            query_dim=self.encoder.embedding_dim,
            model_embed_dim=self.config.router_training.model_embed_dim,
            hidden_dim=self.config.router_training.reward_hidden_dim,
        )
        self.pid = pid_controller or PIDController(self.config.pid)
        self.scorer = ActionScorer(self.reward_predictor, self.pid)

        # MCTS-style statistics for future policy improvement (§5.4)
        self.visit_counts: Dict[str, int] = {}
        self.win_counts: Dict[str, float] = {}

    def route(
        self,
        query: str,
        specialists: Optional[Dict[str, object]] = None,
        verbose: bool = True,
        max_steps: int = 10,
    ) -> List[ActionResult]:
        """
        Route a query through the multi-agent pipeline.

        Args:
            query: Input query text
            specialists: Dict mapping model names to callable specialists
            verbose: Print routing decisions
            max_steps: Maximum number of actions to execute

        Returns:
            List of ActionResult from executed actions
        """
        # Phase 1: Encode query (Eq. 15)
        e_Q = self.encoder.encode(query)
        if verbose:
            print(f"\nRouting query: '{query}'")
            print(f" Encoded: dim={len(e_Q)}")

        # Phase 2: Select mode (Eq. 16-17)
        mode, mode_probs = self.mode_selector.predict(e_Q)
        template = MODE_TEMPLATES[mode]
        if verbose:
            print(f" Mode: {mode.value} (probs: {mode_probs.round(3)})")

        # Phase 3: Iterative action selection (Algorithm 2)
        current_node = "start"
        accumulated_cost = CostVector()
        results: List[ActionResult] = []

        for step in range(max_steps):
            # Step 3a: Enumerate candidates
            if current_node not in template or not template[current_node]:
                if verbose:
                    print(f" Step {step}: Terminal node '{current_node}'")
                break

            candidates = self._enumerate_candidates(current_node, template)
            if not candidates:
                if verbose:
                    print(f" Step {step}: No candidates from '{current_node}'")
                break

            # Step 3b-3d: Score all candidates
            best_action = None
            best_score = float('-inf')

            for action in candidates:
                score = self.scorer.score_action(e_Q, action)
                if verbose:
                    print(f" Step {step}: {action.model}@{action.target} -> "
                          f"score={score:.4f} (reward={action.predicted_reward:.4f})")
                if score > best_score:
                    best_score = score
                    best_action = action

            # Step 3e-3f: Execute or terminate
            if best_score <= 0:
                if verbose:
                    print(f" Step {step}: Stopping (best score {best_score:.4f} ≤ 0)")
                break

            # Execute
            output = self._execute_action(best_action, query, specialists)
            actual_cost = best_action.estimated_cost or CostVector()

            result = ActionResult(
                action=best_action,
                output=output,
                actual_cost=actual_cost,
            )
            results.append(result)

            # Update PID controller
            self.pid.update(actual_cost)
            accumulated_cost = accumulated_cost + actual_cost
            current_node = best_action.target

            if verbose:
                print(f" Step {step}: Executed {best_action.model}@{best_action.target}")

        # Phase 4: Update MCTS statistics
        self._update_statistics(results)

        return results

    def _enumerate_candidates(
        self,
        current_node: str,
        template: Dict[str, List[str]],
    ) -> List[ActionCandidate]:
        """Enumerate valid actions from current node (Eq. 3)."""
        candidates = []
        for target in template.get(current_node, []):
            for model in NODE_MODEL_MAPPING.get(target, []):
                candidates.append(ActionCandidate(
                    source=current_node,
                    target=target,
                    model=model,
                ))
        return candidates

    def _execute_action(
        self,
        action: ActionCandidate,
        query: str,
        specialists: Optional[Dict[str, object]] = None,
    ) -> str:
        """Execute an action, passing prefix_len_hint to SKA specialists."""
        if specialists and action.model in specialists:
            specialist = specialists[action.model]
            if callable(specialist):
                # Pass prefix_len_hint as kwarg if the specialist accepts it
                import inspect
                sig = inspect.signature(specialist)
                if 'prefix_len' in sig.parameters:
                    return specialist(query, prefix_len=action.prefix_len_hint)
                return specialist(query)

        # Placeholder execution
        return f"[{action.model}] processed query at node '{action.target}'"

    def _update_statistics(
        self,
        results: List[ActionResult],
        gamma: float = 0.9,
    ):
        """
        Update MCTS-style statistics (§5.4, Eqs. 23-24).

        N(k) ← N(k) + 1
        W(k) ← W(k) + γ^{T-1-t} · r_final
        """
        if not results:
            return

        # Simple quality estimate: count of successful actions
        r_final = sum(1.0 for r in results if r.success) / len(results)
        T = len(results)

        for t, result in enumerate(results):
            key = f"{result.action.source}|{result.action.target}|{result.action.model}"
            self.visit_counts[key] = self.visit_counts.get(key, 0) + 1
            self.win_counts[key] = self.win_counts.get(key, 0.0) + gamma ** (T - 1 - t) * r_final
