"""
Shared Spectral Memory Protocol.

This file implements cross-agent communication through Koopman operators
instead of serialized text. This is the mechanism that allows a big model
and a small model to share the same operator.

The key insight: instead of Agent A writing 2K tokens of findings into
Agent B's context window (growing KV cache by ~32MB per handoff), Agent A
writes a rank-r operator update (32KB) and Agent B queries it. The
operator captures what Agent A learned in a fixed-size matrix.

Three classes:

  SharedSpectralMemory
    The central shared memory object. Agents write keys/values via
    write(), which accumulates Gram and transition matrices. The operator
    is rebuilt lazily on read. Other agents query via read(), which runs
    the whiten-power-unwhiten pipeline.

    The inject_operator() method allows external components (like the
    ThinkKoopmanBridge) to directly set the operator without going
    through the raw key accumulation path.

    compose() chains two agents' operators through proper basis change:
    unwhiten both to natural space, multiply, re-whiten into the second
    agent's basis.

  BridgeProjection (requires torch)
    Learned linear map W_bridge (r x d_c) that projects DeepSeek V3's
    MLA latents (512-dim) into the shared rank-r space.

  MLALatentExtractor (requires torch)
    Forward hooks on DeepSeek V3's attention modules to extract
    compressed KV latents during thinking-token generation.

  DeepSeekV3Integration
    Full Tier 3 integration: model loading, latent extraction, bridge
    projection, write-back of retrieval results as synthetic MLA latents.

The write/read protocols use the same similarity-transform operator
as core/ska_module.py: A_w = L^{-1} M L^{-T}, so that
L A_w^K L^{-1} = (MG^{-1})^K.

Dependencies:
  - core/structures.py for SharedOperator
  - utils/math_utils.py for SpectralUtils (triangular solves, etc.)
  - think_koopman_bridge.py uses inject_operator() to write slot 2
"""

from __future__ import annotations

import math
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING

import numpy as np

from ..core.structures import SharedOperator, SKAConfig, SystemConfig
from ..utils.math_utils import SpectralUtils

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

# Lazy torch import for classes that need it
def _import_torch():
    import torch
    import torch.nn as nn
    return torch, nn


# Bridge Projection (§6.2, Eq. 28)

class BridgeProjection:
    """
    Learned bridge matrix W_bridge ∈ R^{r × d_c} (Eq. 28).

    Projects MLA latents c_KV ∈ R^{d_c} into SKA's rank-r space:
        z_t = W_bridge · c_KV_t ∈ R^r

    Trained end-to-end on multi-hop QA accuracy with both
    backbone models frozen (§8.3).

    NOTE: This is a torch.nn.Module subclass. It requires torch to be
    installed. The actual inheritance is set up at __init__ time.
    """

    _nn_module_class = None

    @classmethod
    def _ensure_base(cls):
        if cls._nn_module_class is None:
            torch, nn = _import_torch()
            cls._nn_module_class = nn.Module
            # Rebuild class hierarchy
            cls.__bases__ = (nn.Module,)

    def __init__(
        self,
        latent_dim: int = 512,
        rank: int = 64,
    ):
        self._ensure_base()
        torch, nn = _import_torch()
        super().__init__()
        self.latent_dim = latent_dim
        self.rank = rank

        self.W_bridge = nn.Linear(latent_dim, rank, bias=False)
        self.W_bridge_inv = nn.Linear(rank, latent_dim, bias=False)
        self.injection_gate = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        """Initialize with orthogonal matrices for stable projection."""
        _, nn = _import_torch()
        nn.init.orthogonal_(self.W_bridge.weight)
        nn.init.orthogonal_(self.W_bridge_inv.weight)

    def forward(self, mla_latents):
        """Project MLA latents to SKA key space (Eq. 28)."""
        return self.W_bridge(mla_latents)

    def inverse_project(self, ska_output):
        """Project SKA output back to MLA latent space (Eq. 30)."""
        torch, _ = _import_torch()
        gate = torch.sigmoid(self.injection_gate)
        return gate * self.W_bridge_inv(ska_output)


# Shared Spectral Memory

class SharedSpectralMemory:
    """
    Shared spectral memory for cross-agent communication (§6).

    Maintains Koopman operators that agents can write to and read from.
    Operators are built from key projections and support:
      - Write: accumulate keys into Gram/transition matrices
      - Read: query the operator to retrieve associated values
      - Write-back: inject results into another model's KV cache
      - Monitoring: track condition numbers for stability (§6.5)

    Design principles:
      1. Fixed-size memory: operators are r×r regardless of context
      2. Subspace communication: share representations, not tokens
      3. Composability: operators multiply for multi-agent chaining (§7.5)
    """

    def __init__(
        self,
        rank: int = 64,
        ridge_eps: float = 1e-3,
        power_K: int = 2,
        spectral_gamma: float = 1.0,
        condition_alert_threshold: float = 1e4,
    ):
        self.rank = rank
        self.ridge_eps = ridge_eps
        self.power_K = power_K
        self.spectral_gamma = spectral_gamma
        self.condition_alert = condition_alert_threshold

        # Accumulated matrices
        self.gram: Optional[np.ndarray] = None # G = Σ z_t z_t^T + εI
        self.transition: Optional[np.ndarray] = None # M = Σ z_t z_{t-1}^T
        self.value_cross: Optional[np.ndarray] = None # C_v = Σ v_t z_t^T
        self.prev_key: Optional[np.ndarray] = None # z_{t-1} for transition update
        self.num_tokens: int = 0

        # Cached operator (rebuilt when stale)
        self._operator: Optional[SharedOperator] = None
        self._stale: bool = True

    def reset(self):
        """Clear all accumulated state."""
        self.gram = None
        self.transition = None
        self.value_cross = None
        self.prev_key = None
        self.num_tokens = 0
        self._operator = None
        self._stale = True
        # Invalidate cached torch tensors
        self._torch_cache = None
        self._torch_cache_key = None

    def write(
        self,
        keys: np.ndarray,
        values: Optional[np.ndarray] = None,
        source_agent: str = "unknown",
    ):
        """
        Write protocol (Definition 6.1, Eq. 27).

        Accumulate key projections into Gram and transition matrices.

        Args:
            keys: (L, r) key projections from agent's hidden states
            values: (L, d_v) optional value projections
            source_agent: name of writing agent (for logging)
        """
        L, r = keys.shape
        assert r == self.rank, f"Key rank {r} != memory rank {self.rank}"

        # Initialize matrices if first write
        if self.gram is None:
            self.gram = self.ridge_eps * np.eye(r)
            self.transition = np.zeros((r, r))

        # Accumulate Gram matrix: G += Σ z_t z_t^T
        self.gram += keys.T @ keys

        # Accumulate transition matrix: M += Σ z_t z_{t-1}^T
        if self.prev_key is not None:
            # First transition: from prev_key to keys[0]
            self.transition += keys[0:1].T @ self.prev_key.reshape(1, -1)
        if L > 1:
            self.transition += keys[1:].T @ keys[:-1]

        # Accumulate value cross-correlation if provided
        if values is not None:
            if self.value_cross is None:
                self.value_cross = np.zeros((values.shape[1], r))
            self.value_cross += values.T @ keys

        # Update state
        self.prev_key = keys[-1].copy()
        self.num_tokens += L
        self._stale = True

    def write_mla_latents(
        self,
        latents: np.ndarray,
        bridge: BridgeProjection,
        values: Optional[np.ndarray] = None,
    ):
        """
        Write from MLA latents (§6.2, Eq. 28).

        Projects DeepSeek V3's compressed KV latents through the bridge
        matrix before accumulating into the shared operator.

        Args:
            latents: (L, d_c) MLA compressed latents
            bridge: Trained bridge projection W_bridge
            values: Optional value projections
        """
        torch, _ = _import_torch()
        with torch.no_grad():
            latents_t = torch.tensor(latents, dtype=torch.float32)
            keys_t = bridge(latents_t)
            keys = keys_t.numpy()

        self.write(keys, values, source_agent="deepseek_v3")

    def _rebuild_operator(self):
        """Rebuild the Koopman operator from accumulated matrices."""
        if self.gram is None:
            return

        G = self.gram.copy()
        M = self.transition.copy()

        # Cholesky factorization
        try:
            L = np.linalg.cholesky(G)
        except np.linalg.LinAlgError:
            # Add more ridge if needed
            G += 1e-4 * np.eye(self.rank)
            L = np.linalg.cholesky(G)

        # Whitened operator
        A_w = SpectralUtils.whiten_operator(L, M)

        # Spectral normalization
        A_w = SpectralUtils.spectral_normalize(A_w, self.spectral_gamma)

        # Value readout via triangular solves (no explicit inverse)
        # B_v = C_v G^{-1} = C_v L^{-T} L^{-1}
        B_v = None
        if self.value_cross is not None:
            from scipy.linalg import solve_triangular
            C_v = self.value_cross # (d_v, r)
            # Step 1: solve L Y^T = C_v^T -> Y = C_v L^{-T}
            Y_T = solve_triangular(L, C_v.T, lower=True)
            # Step 2: solve L^T B_v^T = Y^T -> B_v = C_v L^{-T} L^{-1}
            Bv_T = solve_triangular(L.T, Y_T, lower=False)
            B_v = Bv_T.T

        # Condition number monitoring (§6.5, Eq. 31)
        cond = SpectralUtils.condition_number(G)

        self._operator = SharedOperator(
            A_w=A_w,
            B_v=B_v if B_v is not None else np.zeros((1, self.rank)),
            L=L,
            rank=self.rank,
            num_tokens_seen=self.num_tokens,
            condition_number=cond,
        )
        self._stale = False

        if cond > self.condition_alert:
            print(f" WARNING: Operator condition number {cond:.1f} > {self.condition_alert}")
            print(f" Consider rebuilding operator from scratch")

    @property
    def operator(self) -> Optional[SharedOperator]:
        """Get the current operator, rebuilding if stale."""
        if self._stale:
            self._rebuild_operator()
        return self._operator

    def inject_operator(
        self,
        A_w: np.ndarray,
        L: np.ndarray,
        B_v: Optional[np.ndarray] = None,
        num_tokens_seen: int = 0,
    ):
        """
        Inject a pre-built operator into this shared memory.

        Use this when an external component (e.g. ThinkKoopmanBridge) has
        already constructed an operator and wants to make it readable by
        other agents through the standard read() protocol.

        Handles all internal cache invalidation.

        Args:
            A_w: (r, r) whitened Koopman operator
            L: (r, r) Cholesky factor
            B_v: (d_v, r) value readout matrix, or None
            num_tokens_seen: number of tokens that built this operator
        """
        r = A_w.shape[0]
        cond = SpectralUtils.condition_number(L.T @ L)

        self._operator = SharedOperator(
            A_w=A_w,
            B_v=B_v if B_v is not None else np.zeros((1, r)),
            L=L,
            rank=r,
            num_tokens_seen=num_tokens_seen,
            condition_number=cond,
        )
        self._stale = False
        # Invalidate torch tensor cache
        self._torch_cache = None
        self._torch_cache_key = None

    def read(self, queries: np.ndarray) -> np.ndarray:
        """
        Read protocol (Definition 6.2, Eq. 29).

        ŷ_q = B_v · L · A_w^K · L^{-1} · z_q

        With the corrected similarity transform A_w = L^{-1} M L^{-T},
        this computes: B_v · (M G^{-1})^K · z_q (after whiten/unwhiten cancel).

        Agent B obtains retrieved values without seeing Agent A's tokens.

        Args:
            queries: (Q, r) query vectors from reading agent

        Returns:
            outputs: (Q, d_v) retrieved values
        """
        from scipy.linalg import solve_triangular

        op = self.operator
        if op is None:
            raise ValueError("No operator available. Call write() first.")

        # Batched triangular solve: W = L^{-1} Z_q where Z_q is (r, Q)
        W = solve_triangular(op.L, queries.T, lower=True) # (r, Q)

        # Power filter: W_f = A_w^K W
        W_f = W
        for _ in range(self.power_K):
            W_f = op.A_w @ W_f # (r, Q)

        # Unwhiten: Ẑ = L W_f
        Z_hat = op.L @ W_f # (r, Q)

        # Readout: Ŷ = B_v Ẑ
        outputs = (op.B_v @ Z_hat).T # (Q, d_v)

        return outputs

    def read_torch(
        self,
        queries,
    ):
        """
        Read protocol using PyTorch tensors (for backprop through bridge).

        Caches operator tensors on the query's device to avoid
        re-allocation on every call.

        Args:
            queries: (B, Q, r) query tensors

        Returns:
            outputs: (B, Q, d_v) retrieved values
        """
        torch, _ = _import_torch()
        op = self.operator
        if op is None:
            raise ValueError("No operator available")

        device = queries.device
        dtype = queries.dtype
        cache_key = (id(op), device, dtype)

        # Rebuild cached tensors only when operator or device changes
        if not hasattr(self, '_torch_cache') or self._torch_cache_key != cache_key:
            self._torch_cache = {
                'A_w': torch.tensor(op.A_w, dtype=dtype, device=device),
                'L': torch.tensor(op.L, dtype=dtype, device=device),
                'B_v': torch.tensor(op.B_v, dtype=dtype, device=device),
            }
            self._torch_cache_key = cache_key

        A_w = self._torch_cache['A_w']
        L = self._torch_cache['L']
        B_v = self._torch_cache['B_v']

        # Whiten
        L_double = L.to(torch.float64)
        q_double = queries.to(torch.float64).transpose(-2, -1)
        w_q = torch.linalg.solve_triangular(L_double, q_double, upper=False)
        w_q = w_q.to(dtype)

        # Power filter
        w_f = w_q
        for _ in range(self.power_K):
            w_f = A_w @ w_f

        # Unwhiten
        z_hat = L @ w_f

        # Readout
        output = B_v @ z_hat
        return output.transpose(-2, -1)

    def compose(
        self,
        other: SharedSpectralMemory,
    ) -> SharedSpectralMemory:
        """
        Operator composition for multi-agent chaining (§7.5, Eq. 37).

        Since each agent's whitened operator lives in its own Cholesky basis,
        naive multiplication A_w^B @ A_w^A is incoherent. We compose through
        the unwhitened (natural) operator space:

            A_nat^X = L_X A_w^X L_X^{-1} (unwhiten to natural space)
            A_chain_nat = A_nat^B @ A_nat^A (compose in shared natural space)

        Then re-whiten into B's basis for the output:
            A_chain_w = L_B^{-1} A_chain_nat L_B

        This uses triangular solves throughout.

        Args:
            other: Another SharedSpectralMemory (agent B)

        Returns:
            New SharedSpectralMemory with composed operator
        """
        from scipy.linalg import solve_triangular

        op_a = self.operator
        op_b = other.operator
        if op_a is None or op_b is None:
            raise ValueError("Both operators must be initialized")

        # Unwhiten A's operator: A_nat^A = L_A A_w^A L_A^{-1}
        # = L_A @ A_w^A @ L_A^{-1}
        # Compute via: temp = A_w^A @ L_A^{-1} (solve L_A^T X = (A_w^A)^T -> X^T)
        temp_T = solve_triangular(op_a.L.T, op_a.A_w.T, lower=False)
        A_nat_a = op_a.L @ temp_T.T # L_A @ A_w^A L_A^{-1}

        # Unwhiten B's operator: A_nat^B = L_B A_w^B L_B^{-1}
        temp_T = solve_triangular(op_b.L.T, op_b.A_w.T, lower=False)
        A_nat_b = op_b.L @ temp_T.T

        # Compose in natural space
        A_chain_nat = A_nat_b @ A_nat_a

        # Re-whiten into B's basis.
        # Our convention: A_nat = L A_w L^{-1}, so A_w = L^{-1} A_nat L.
        # Step 1: solve L_B U = A_chain_nat -> U = L_B^{-1} A_chain_nat
        U = solve_triangular(op_b.L, A_chain_nat, lower=True)
        # Step 2: A_chain_w = U @ L_B = L_B^{-1} A_chain_nat L_B
        A_chain_w = U @ op_b.L

        # Re-apply spectral normalization (gamma clamped ≤ 1 inside spectral_normalize)
        A_chain_w = SpectralUtils.spectral_normalize(A_chain_w, self.spectral_gamma)

        composed = SharedSpectralMemory(
            rank=self.rank,
            ridge_eps=self.ridge_eps,
            power_K=self.power_K,
            spectral_gamma=self.spectral_gamma,
        )

        composed._operator = SharedOperator(
            A_w=A_chain_w,
            B_v=op_b.B_v,
            L=op_b.L,
            rank=self.rank,
            num_tokens_seen=op_a.num_tokens_seen + op_b.num_tokens_seen,
            condition_number=SpectralUtils.condition_number(
                op_b.L.T @ op_b.L # condition of B's Gram
            ),
        )
        composed._stale = False

        return composed

    def should_rebuild(self) -> bool:
        """Check if operator should be rebuilt (§6.5)."""
        op = self.operator
        if op is None:
            return False
        return op.condition_number > self.condition_alert

    def rebuild_from_scratch(self, keys: np.ndarray, values: Optional[np.ndarray] = None):
        """Rebuild operator from fresh data."""
        self.reset()
        self.write(keys, values, source_agent="rebuild")


# DeepSeek V3 MLA Latent Extractor

class MLALatentExtractor:
    """
    DeepSeek V3 MLA latent extraction via forward hooks (§11.2.2, Listing 3).

    Hooks into the attention module's KV compression to extract
    c_KV_t ∈ R^{512} during thinking-token generation.

    Usage:
        extractor = MLALatentExtractor(model, hook_layers=[8, 16, 24])
        # Run model forward pass
        output = model(input_ids)
        # Get extracted latents
        latents = extractor.get_latents()
    """

    def __init__(
        self,
        model,
        hook_layers: List[int] = None,
        latent_dim: int = 512,
    ):
        self.model = model
        self.hook_layers = hook_layers or [8, 16, 24]
        self.latent_dim = latent_dim

        self._latent_buffer: Dict[int, List] = {}
        self._handles: List = []

        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on designated layers (Listing 3)."""
        for layer_idx in self.hook_layers:
            try:
                layer = self._get_attention_module(layer_idx)
                handle = layer.register_forward_hook(
                    self._make_hook(layer_idx)
                )
                self._handles.append(handle)
                self._latent_buffer[layer_idx] = []
            except (AttributeError, IndexError) as e:
                print(f"WARNING: Could not hook layer {layer_idx}: {e}")

    def _get_attention_module(self, layer_idx: int) -> nn.Module:
        """Get the attention module at a specific layer."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers[layer_idx].self_attn
        elif hasattr(self.model, 'layers'):
            return self.model.layers[layer_idx].self_attn
        raise AttributeError(f"Cannot find attention at layer {layer_idx}")

    def _make_hook(self, layer_idx: int):
        """Create extraction hook for a specific layer."""
        def hook_fn(module, input, output):
            # Extract compressed KV latent
            # The exact attribute depends on DeepSeek V3 implementation
            if hasattr(module, 'kv_proj'):
                # Standard DeepSeek MLA: c_kv = kv_proj(input)
                x = input[0] if isinstance(input, tuple) else input
                c_kv = module.kv_proj(x) # (B, L, d_c)
                self._latent_buffer[layer_idx].append(c_kv.detach())
            elif hasattr(module, 'kv_a_proj_with_mqa'):
                # Alternative naming in some implementations
                x = input[0] if isinstance(input, tuple) else input
                c_kv = module.kv_a_proj_with_mqa(x)
                self._latent_buffer[layer_idx].append(c_kv.detach())
        return hook_fn

    def get_latents(self, layer_idx: Optional[int] = None):
        """
        Get extracted latents.

        Args:
            layer_idx: Specific layer, or None for all layers concatenated

        Returns:
            Tensor of extracted MLA latents, or None if no latents captured
        """
        if layer_idx is not None:
            buffers = self._latent_buffer.get(layer_idx, [])
            if not buffers:
                return None
            import torch
            return torch.cat(buffers, dim=1)

        # Concatenate from all layers
        import torch
        all_latents = []
        for idx in sorted(self._latent_buffer.keys()):
            buffers = self._latent_buffer[idx]
            if buffers:
                all_latents.append(torch.cat(buffers, dim=1))

        if not all_latents:
            return None
        return torch.cat(all_latents, dim=1)

    def clear(self):
        """Clear latent buffers."""
        for key in self._latent_buffer:
            self._latent_buffer[key] = []

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# DeepSeek V3 Integration

class DeepSeekV3Integration:
    """
    Full DeepSeek V3 integration for shared memory protocol.

    Handles:
      1. Local deployment in INT8 on B200 with vLLM
      2. MLA latent extraction during thinking-token generation
      3. Projection into SKA's rank-r space via bridge matrix
      4. Write-back of retrieval results as synthetic MLA latents
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-V3",
        latent_dim: int = 512,
        ska_rank: int = 64,
        hook_layers: List[int] = None,
        use_vllm: bool = True,
    ):
        self.model_name = model_name
        self.latent_dim = latent_dim
        self.ska_rank = ska_rank
        self.hook_layers = hook_layers or [8, 16, 24]
        self.use_vllm = use_vllm

        # Bridge projection (trained end-to-end)
        self.bridge = BridgeProjection(latent_dim=latent_dim, rank=ska_rank)

        # Shared memory
        self.shared_memory = SharedSpectralMemory(rank=ska_rank)

        # Model and extractor (initialized on load)
        self.model = None
        self.tokenizer = None
        self.extractor = None

    def load_model(self):
        """Load DeepSeek V3 locally."""
        if self.use_vllm:
            self._load_vllm()
        else:
            self._load_transformers()

    def _load_vllm(self):
        """Load with vLLM for efficient serving."""
        try:
            from vllm import LLM, SamplingParams

            print(f"Loading {self.model_name} with vLLM (INT8)...")
            self.model = LLM(
                model=self.model_name,
                quantization="awq", # or "gptq" depending on available weights
                dtype="float16",
                gpu_memory_utilization=0.8,
                max_model_len=4096,
                trust_remote_code=True,
            )
            print("DeepSeek V3 loaded via vLLM")
        except ImportError:
            print("vLLM not available, falling back to Transformers")
            self._load_transformers()

    def _load_transformers(self):
        """Load with HuggingFace Transformers."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.model_name} with Transformers...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            load_in_8bit=True,
        )
        self.model.eval()

        # Set up latent extraction
        self.extractor = MLALatentExtractor(
            self.model,
            hook_layers=self.hook_layers,
            latent_dim=self.latent_dim,
        )
        print("DeepSeek V3 loaded via Transformers with MLA hooks")

    def generate_with_memory(
        self,
        query: str,
        shared_memory: SharedSpectralMemory,
        max_tokens: int = 512,
        extract_interval: int = 32,
    ) -> Tuple[str, List[np.ndarray]]:
        """
        Generate with thinking tokens, extracting MLA latents at intervals.

        At every `extract_interval` tokens:
          1. Extract MLA latents from hook layers
          2. Project through bridge into SKA key space
          3. Write to shared memory operator
          4. (Optionally) read from shared memory and inject back

        Args:
            query: Input query
            shared_memory: Shared spectral memory instance
            max_tokens: Maximum tokens to generate
            extract_interval: How often to extract latents

        Returns:
            generated_text: Full generated output
            extracted_latents: List of extracted latent arrays
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available (vLLM mode doesn't support hooks)")

        # Prepare input
        import torch
        inputs = self.tokenizer(query, return_tensors="pt").to(self.model.device)
        all_latents = []

        # Generate token by token for latent extraction
        generated_ids = inputs['input_ids'].clone()
        self.extractor.clear()

        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated_ids,
                    use_cache=True,
                )

            # Get next token
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Extract latents at intervals
            if (step + 1) % extract_interval == 0:
                latents = self.extractor.get_latents()
                if latents is not None:
                    latents_np = latents.cpu().numpy()
                    all_latents.append(latents_np)

                    # Write to shared memory
                    for batch_latents in latents_np:
                        shared_memory.write_mla_latents(
                            batch_latents, self.bridge,
                        )

                self.extractor.clear()

        # Decode
        generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True,
        )

        return generated_text, all_latents

    def inject_retrieval_results(
        self,
        retrieval_output,
        kv_cache=None,
    ):
        """
        Write-back to DeepSeek V3 (§6.4, Eq. 30).

        c_KV_synth = W_bridge^{-1} · W_K^{SKA T} · ŷ

        Injects retrieval results as synthetic MLA latents into
        DeepSeek's KV cache.

        Args:
            retrieval_output: (B, Q, n_heads * head_dim) from SKA
            kv_cache: DeepSeek V3's KV cache to inject into

        Returns:
            synthetic_latents: (B, Q, d_c) synthetic MLA latents
        """
        # Project retrieval output to SKA key space (simplified)
        # In full implementation, would use W_K^{SKA T}
        # Here we use the bridge inverse directly
        synthetic = self.bridge.inverse_project(retrieval_output)
        return synthetic
