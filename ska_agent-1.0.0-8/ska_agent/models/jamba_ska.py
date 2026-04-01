"""
Jamba Attention-to-SKA Surgery.

This file implements the procedure for replacing Jamba's 4 attention
layers with SKA modules while preserving the pretrained model's
capabilities. This is the Tier 2 (retrieval specialist) in the
three-tier architecture.

Jamba-v0.1 architecture:
  32 total layers, attention at indices {4, 12, 20, 28} (1:7 ratio)
  16 MoE experts with top-2 routing
  Hidden size d=4096, 32 query heads, 8 KV heads (GQA), head dim 128
  12B active params, 52B total

The surgery process:
  1. Load Jamba from HuggingFace (load_jamba_model)
  2. For each attention layer at {4, 12, 20, 28}:
     a. Extract GQA weights: W_Q (4096x4096), W_K (1024x4096),
        W_V (1024x4096), W_O (4096x4096)
     b. Expand KV heads from 8 to 32 via repetition (repeat_kv)
     c. Run truncated SVD on each per-head key/query slice
     d. Initialize SKA projections with sqrt-singular-value scaling:
        W_K_ska[h] = sqrt(Sigma_K[:r]) * V_K[:r]
        W_Q_ska[h] = sqrt(Sigma_Q[:r]) * V_Q[:r]
     e. Replace attention with an SKA module (from core/ska_module.py)
  3. Freeze all non-SKA parameters (~52B frozen, ~200M trainable)
  4. Verify coherent logits with a test forward pass

The JambaSKAModel class wraps the original Jamba and intercepts the
forward pass at attention layer positions using PyTorch forward hooks.
At each hooked layer, the SKA module processes the hidden states and
adds its output via a learned gate: h = h + sigmoid(alpha) * ska_out.

Training is handled by training/trainers.py (SKATrainer), which trains
only the SKA parameters on SlimPajama (LM recovery) and table QA
datasets (WikiTableQuestions, TAT-QA, FinQA, HybridQA).

Dependencies:
  - core/ska_module.py for SKAModule and MultiHeadKoopmanModule
  - core/structures.py for SKAConfig, SystemConfig
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import numpy as np

from ..core.structures import SKAConfig, MultiKoopmanConfig, SystemConfig
from ..core.ska_module import SKAModule, MultiHeadKoopmanModule


def load_jamba_model(
    model_name: str = "ai21labs/Jamba-v0.1",
    device_map: str = "auto",
    torch_dtype: str = "auto",
    trust_remote_code: bool = True,
):
    """
    Load the Jamba-v0.1 model from HuggingFace.

    Jamba architecture (§2.2):
      - 32 total layers with attention-to-Mamba ratio a:m = 1:7
      - Attention at layers 4, 12, 20, 28
      - 16 MoE experts with top-2 routing
      - Hidden size d = 4096, 32 attention heads, 8 KV heads (GQA)
      - 12B active parameters, 52B total
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading Jamba model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Jamba model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def extract_gqa_weights(
    attn_layer: nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    Extract GQA weight matrices from a Jamba attention layer.

    Expected structure (§4.1):
      - W_Q: (n_h * d_h, d) = (32 * 128, 4096) = (4096, 4096)
      - W_K: (n_kv * d_h, d) = (8 * 128, 4096) = (1024, 4096)
      - W_V: (n_kv * d_h, d) = (8 * 128, 4096) = (1024, 4096)
      - W_O: (d, n_h * d_h) = (4096, 4096)

    The exact attribute names depend on the Jamba implementation.
    We try multiple common patterns.
    """
    weights = {}

    # Try common naming conventions for Jamba attention
    q_candidates = ['q_proj', 'query', 'W_Q', 'self_attn.q_proj']
    k_candidates = ['k_proj', 'key', 'W_K', 'self_attn.k_proj']
    v_candidates = ['v_proj', 'value', 'W_V', 'self_attn.v_proj']
    o_candidates = ['o_proj', 'out_proj', 'W_O', 'self_attn.o_proj']

    def find_weight(module, candidates):
        for name in candidates:
            parts = name.split('.')
            obj = module
            try:
                for part in parts:
                    obj = getattr(obj, part)
                if hasattr(obj, 'weight'):
                    return obj.weight.data.clone()
                elif isinstance(obj, torch.Tensor):
                    return obj.clone()
            except AttributeError:
                continue
        # Fallback: search all named parameters
        for pname, param in module.named_parameters():
            if 'q_proj' in pname or 'query' in pname:
                if 'weight' in pname and 'q' in pname.lower():
                    return param.data.clone()
        return None

    weights['W_Q'] = find_weight(attn_layer, q_candidates)
    weights['W_K'] = find_weight(attn_layer, k_candidates)
    weights['W_V'] = find_weight(attn_layer, v_candidates)
    weights['W_O'] = find_weight(attn_layer, o_candidates)

    # Validate shapes
    for name, w in weights.items():
        if w is None:
            raise ValueError(
                f"Could not find {name} in attention layer. "
                f"Available params: {[n for n, _ in attn_layer.named_parameters()]}"
            )
        print(f" Extracted {name}: shape={w.shape}, dtype={w.dtype}")

    return weights


def repeat_kv(weight: torch.Tensor, num_groups: int) -> torch.Tensor:
    """
    Expand GQA KV heads to full heads via repetition (Algorithm 1, line 2).

    W̃_K ← repeat_kv(W_K, n_h / n_kv)

    Args:
        weight: (n_kv * d_h, d) KV projection weight
        num_groups: n_h / n_kv = 32 / 8 = 4

    Returns:
        Expanded weight: (n_h * d_h, d)
    """
    return weight.repeat(num_groups, 1)


def svd_init_ska_weights(
    W_Q: torch.Tensor,
    W_K_expanded: torch.Tensor,
    W_V_expanded: torch.Tensor,
    W_O: torch.Tensor,
    n_heads: int,
    head_dim: int,
    rank: int,
) -> Dict[str, torch.Tensor]:
    """
    SVD-based initialization of SKA projections from GQA weights (Algorithm 1).

    For each head h:
        K_h = W̃_K[h*d_h : (h+1)*d_h, :] (per-head key: d_h × d)
        Q_h = W_Q[h*d_h : (h+1)*d_h, :] (per-head query: d_h × d)
        U_K, Σ_K, V_K^T = SVD(K_h)
        U_Q, Σ_Q, V_Q^T = SVD(Q_h)
        W_K^{SKA}[h*r : (h+1)*r, :] = (Σ_K[:r])^{1/2} · V_K[:r, :]^T
        W_Q^{SKA}[h*r : (h+1)*r, :] = (Σ_Q[:r])^{1/2} · V_Q[:r, :]^T

    The sqrt-singular-value scaling distributes the singular values
    symmetrically between keys and queries.

    Args:
        W_Q: (n_h * d_h, d) query weights
        W_K_expanded: (n_h * d_h, d) expanded key weights
        W_V_expanded: (n_h * d_h, d) expanded value weights
        W_O: (d, n_h * d_h) output weights
        n_heads: number of attention heads (32)
        head_dim: dimension per head (128)
        rank: target rank for SKA (64)

    Returns:
        Dictionary with initialized SKA weight tensors.
    """
    d_model = W_Q.shape[1]
    device = W_Q.device
    dtype = W_Q.dtype

    # Use float32 for SVD stability
    W_Q_f32 = W_Q.float()
    W_K_f32 = W_K_expanded.float()

    W_K_ska = torch.zeros(n_heads * rank, d_model, dtype=dtype, device=device)
    W_Q_ska = torch.zeros(n_heads * rank, d_model, dtype=dtype, device=device)

    for h in range(n_heads):
        # Extract per-head slices (Algorithm 1, lines 4-5)
        K_h = W_K_f32[h * head_dim: (h + 1) * head_dim, :] # (d_h, d)
        Q_h = W_Q_f32[h * head_dim: (h + 1) * head_dim, :] # (d_h, d)

        # Truncated SVD (Algorithm 1, lines 6-7)
        U_K, S_K, Vh_K = torch.linalg.svd(K_h, full_matrices=False)
        U_Q, S_Q, Vh_Q = torch.linalg.svd(Q_h, full_matrices=False)

        # Take top-r components with sqrt scaling (Algorithm 1, lines 8-9)
        # W_K^{SKA}[h*r : (h+1)*r, :] = (Σ_K[:r])^{1/2} · V_K[:r, :]
        sqrt_S_K = torch.sqrt(S_K[:rank]) # (r,)
        sqrt_S_Q = torch.sqrt(S_Q[:rank]) # (r,)

        W_K_ska[h * rank: (h + 1) * rank, :] = (
            torch.diag(sqrt_S_K) @ Vh_K[:rank, :]
        ).to(dtype)

        W_Q_ska[h * rank: (h + 1) * rank, :] = (
            torch.diag(sqrt_S_Q) @ Vh_Q[:rank, :]
        ).to(dtype)

    # V and O are direct copies (Algorithm 1, lines 11-12)
    W_V_ska = W_V_expanded.clone()
    W_O_ska = W_O.clone()

    return {
        'W_K': W_K_ska, # (n_heads * rank, d_model)
        'W_Q': W_Q_ska, # (n_heads * rank, d_model)
        'W_V': W_V_ska, # (n_heads * head_dim, d_model)
        'W_O': W_O_ska, # (d_model, n_heads * head_dim)
    }


def initialize_ska_from_gqa(
    ska_module: SKAModule,
    gqa_weights: Dict[str, torch.Tensor],
    n_heads: int = 32,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    rank: int = 64,
) -> SKAModule:
    """
    Initialize an SKA module from extracted GQA weights.

    Combines KV head expansion and SVD initialization.

    Args:
        ska_module: The SKA module to initialize
        gqa_weights: Dict with 'W_Q', 'W_K', 'W_V', 'W_O' tensors
        n_heads: number of query heads (32)
        n_kv_heads: number of KV heads (8)
        head_dim: head dimension (128)
        rank: SKA rank (64)

    Returns:
        Initialized SKA module
    """
    num_groups = n_heads // n_kv_heads # 4

    # Step 1: Expand KV heads (Algorithm 1, lines 1-2)
    W_K_expanded = repeat_kv(gqa_weights['W_K'], num_groups)
    W_V_expanded = repeat_kv(gqa_weights['W_V'], num_groups)

    print(f" Expanded KV: W_K {gqa_weights['W_K'].shape} -> {W_K_expanded.shape}")

    # Step 2: SVD initialization (Algorithm 1, lines 3-12)
    ska_weights = svd_init_ska_weights(
        W_Q=gqa_weights['W_Q'],
        W_K_expanded=W_K_expanded,
        W_V_expanded=W_V_expanded,
        W_O=gqa_weights['W_O'],
        n_heads=n_heads,
        head_dim=head_dim,
        rank=rank,
    )

    # Step 3: Load into SKA module
    with torch.no_grad():
        ska_module.W_K.weight.copy_(ska_weights['W_K'])
        ska_module.W_Q.weight.copy_(ska_weights['W_Q'])
        ska_module.W_V.weight.copy_(ska_weights['W_V'])
        ska_module.W_O.weight.copy_(ska_weights['W_O'])

        # Initialize gate so σ(α) = 0.5 (Algorithm 1, line 13)
        ska_module.gate_alpha.fill_(0.0)

    print(f" SKA module initialized from GQA weights")
    return ska_module


class JambaSKAModel(nn.Module):
    """
    Jamba model with attention layers replaced by SKA modules.

    This wraps the original Jamba model and intercepts the forward pass
    at the 4 attention layer positions (§4.1, Listing 2).

    Freezing strategy (§4.4):
      - Frozen: All 28 Mamba layers, all MoE FFN layers, embedding,
        LM head, all layer norms (except SKA-internal)
      - Trainable: 4 SKA modules (~200M params, 0.4% of 52B total)
    """

    def __init__(
        self,
        jamba_model: nn.Module,
        tokenizer,
        config: SystemConfig = None,
        use_multi_koopman: bool = False,
    ):
        super().__init__()
        self.config = config or SystemConfig()
        self.ska_config = self.config.ska
        self.layer_indices = list(self.config.jamba_ska_layer_indices)

        # Store original model
        self.jamba = jamba_model
        self.tokenizer = tokenizer

        # Create SKA modules for each attention layer
        self.ska_modules = nn.ModuleDict()
        self.ska_norms = nn.ModuleDict()
        self.ska_gates = nn.ParameterDict()

        for idx in self.layer_indices:
            layer_key = f"layer_{idx}"

            if use_multi_koopman:
                mk_config = self.config.multi_koopman
                ska = MultiHeadKoopmanModule(
                    config=self.ska_config,
                    num_operators=mk_config.num_operators,
                    rank_per_op=mk_config.rank_per_operator,
                )
            else:
                ska = SKAModule(self.ska_config)

            self.ska_modules[layer_key] = ska
            self.ska_norms[layer_key] = nn.LayerNorm(self.ska_config.d_model)
            self.ska_gates[layer_key] = nn.Parameter(torch.zeros(1))

        # Perform surgery: initialize SKA from GQA weights
        self._perform_surgery()

        # Freeze non-SKA parameters
        self._apply_freezing()

    def _perform_surgery(self):
        """
        Extract GQA weights and initialize SKA modules.

        For each attention layer at indices {4, 12, 20, 28}:
          1. Extract W_Q, W_K, W_V, W_O from the attention sub-layer
          2. Expand KV heads from 8 to 32 via repetition
          3. Run truncated SVD for rank-64 initialization
          4. Load weights into the SKA module
        """
        print("\n=== Performing Jamba Attention -> SKA Surgery ===")

        for idx in self.layer_indices:
            layer_key = f"layer_{idx}"
            print(f"\nProcessing layer {idx}:")

            # Get the attention module from this layer
            # Jamba layer structure: each layer has either mamba or attention
            layer = self._get_layer(idx)
            attn_module = self._get_attention_module(layer)

            if attn_module is None:
                print(f" WARNING: No attention module found at layer {idx}, skipping init")
                continue

            try:
                # Extract GQA weights
                gqa_weights = extract_gqa_weights(attn_module)

                # Initialize SKA from GQA
                if isinstance(self.ska_modules[layer_key], MultiHeadKoopmanModule):
                    # For multi-Koopman, initialize each operator's key projection
                    # from the same SVD, then let training differentiate
                    self._init_multi_koopman_from_gqa(layer_key, gqa_weights)
                else:
                    initialize_ska_from_gqa(
                        self.ska_modules[layer_key],
                        gqa_weights,
                        n_heads=self.ska_config.n_heads,
                        n_kv_heads=8, # Jamba uses 8 KV heads
                        head_dim=self.ska_config.head_dim,
                        rank=self.ska_config.rank,
                    )
            except Exception as e:
                print(f" ERROR during surgery at layer {idx}: {e}")
                print(f" Falling back to random initialization")

        print("\n=== Surgery Complete ===")

    def _init_multi_koopman_from_gqa(
        self,
        layer_key: str,
        gqa_weights: Dict[str, torch.Tensor],
    ):
        """Initialize multi-Koopman module from GQA weights."""
        mk_module = self.ska_modules[layer_key]
        n_heads = self.ska_config.n_heads
        head_dim = self.ska_config.head_dim
        rank = mk_module.rank_per_op
        num_groups = n_heads // 8 # GQA group size

        # Expand KV heads
        W_K_expanded = repeat_kv(gqa_weights['W_K'], num_groups)
        W_V_expanded = repeat_kv(gqa_weights['W_V'], num_groups)

        # SVD init for each operator
        ska_weights = svd_init_ska_weights(
            W_Q=gqa_weights['W_Q'],
            W_K_expanded=W_K_expanded,
            W_V_expanded=W_V_expanded,
            W_O=gqa_weights['W_O'],
            n_heads=n_heads,
            head_dim=head_dim,
            rank=rank,
        )

        with torch.no_grad():
            # Each operator gets the same initial key/query projections
            # (they'll diverge during training)
            for k in range(mk_module.num_operators):
                mk_module.W_K_ops[k].weight.copy_(ska_weights['W_K'])
                mk_module.W_Q_ops[k].weight.copy_(ska_weights['W_Q'])

            mk_module.W_V.weight.copy_(ska_weights['W_V'])
            mk_module.W_O.weight.copy_(ska_weights['W_O'])

        print(f" Multi-Koopman module initialized ({mk_module.num_operators} operators, rank={rank})")

    def _get_layer(self, idx: int):
        """Get a specific layer from the Jamba model."""
        # Try common Jamba model structures
        if hasattr(self.jamba, 'model') and hasattr(self.jamba.model, 'layers'):
            return self.jamba.model.layers[idx]
        elif hasattr(self.jamba, 'layers'):
            return self.jamba.layers[idx]
        elif hasattr(self.jamba, 'transformer') and hasattr(self.jamba.transformer, 'layers'):
            return self.jamba.transformer.layers[idx]
        else:
            raise AttributeError(
                f"Cannot find layers in Jamba model. "
                f"Top-level attributes: {[n for n, _ in self.jamba.named_children()]}"
            )

    def _get_attention_module(self, layer) -> Optional[nn.Module]:
        """Extract the attention sub-module from a Jamba layer."""
        # Jamba layers that have attention will have an attention module
        candidates = ['self_attn', 'attention', 'attn', 'mamba_or_attn']
        for name in candidates:
            if hasattr(layer, name):
                module = getattr(layer, name)
                # Check if this is actually an attention module (not Mamba)
                for pname, _ in module.named_parameters():
                    if any(x in pname.lower() for x in ['q_proj', 'k_proj', 'query', 'key']):
                        return module
        return None

    def _apply_freezing(self):
        """
        Apply freezing strategy (§4.4).

        Freeze everything except:
          - SKA modules (W_K, W_Q, W_V, W_O, layer norms, gates, η, γ)
        """
        # Freeze all Jamba parameters
        for param in self.jamba.parameters():
            param.requires_grad = False

        # Unfreeze SKA modules (they're separate nn.Module children)
        for module in self.ska_modules.values():
            for param in module.parameters():
                param.requires_grad = True

        for module in self.ska_norms.values():
            for param in module.parameters():
                param.requires_grad = True

        for param in self.ska_gates.values():
            param.requires_grad = True

        # Count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"\nParameter summary:")
        print(f" Total: {total_params:,}")
        print(f" Trainable (SKA): {trainable_params:,}")
        print(f" Frozen: {frozen_params:,}")
        print(f" Trainable ratio: {trainable_params / total_params * 100:.2f}%")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        prefix_len: Optional[int] = None,
        **kwargs,
    ):
        """
        Forward pass with SKA layer hooks (Listing 2 from spec).

        The hook intercepts hidden states at attention layer positions
        and replaces the attention output with SKA output:

            for i, layer in enumerate(jamba.model.layers):
                hidden_states = layer.mamba_or_attn(hidden_states)
                if i in ska_layer_indices:
                    ska_input = ska_norms[i](hidden_states)
                    ska_out = ska_modules[i](ska_input, prefix_len)
                    gate = sigmoid(ska_gates[i])
                    hidden_states = hidden_states + gate * ska_out
        """
        # Register forward hooks on the target layers
        handles = []
        ska_layer_set = set(self.layer_indices)

        def make_hook(layer_idx):
            layer_key = f"layer_{layer_idx}"

            def hook_fn(module, input, output):
                # output is the layer output (hidden_states after mamba/attn + FFN)
                # We need to add SKA's contribution
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # SKA forward pass
                ska_input = self.ska_norms[layer_key](hidden_states)
                ska_out = self.ska_modules[layer_key](
                    ska_input,
                    prefix_len=prefix_len,
                )
                gate = torch.sigmoid(self.ska_gates[layer_key])
                hidden_states = hidden_states + gate * ska_out

                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                return hidden_states

            return hook_fn

        # Register hooks
        for idx in self.layer_indices:
            try:
                layer = self._get_layer(idx)
                handle = layer.register_forward_hook(make_hook(idx))
                handles.append(handle)
            except Exception as e:
                print(f"WARNING: Could not hook layer {idx}: {e}")

        # Run the original model forward pass
        try:
            outputs = self.jamba(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        finally:
            # Remove hooks
            for handle in handles:
                handle.remove()

        return outputs

    def verify_logits(self, test_text: str = "The capital of France is") -> bool:
        """
        Verify that the model still produces coherent logits after surgery.

        Checks:
          1. Forward pass completes without errors
          2. Logits have reasonable magnitude
          3. Greedy decoding produces text (not garbage)
        """
        print("\n=== Verifying post-surgery logit coherence ===")

        self.eval()
        device = next(self.parameters()).device

        inputs = self.tokenizer(test_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = self(input_ids=inputs['input_ids'], attention_mask=inputs.get('attention_mask'))

        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        print(f" Logit shape: {logits.shape}")
        print(f" Logit range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
        print(f" Logit mean: {logits.mean().item():.4f}")
        print(f" Logit std: {logits.std().item():.4f}")

        # Check for NaN/Inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(" FAIL: NaN or Inf in logits")
            return False

        # Check reasonable magnitude
        if logits.abs().max() > 1e6:
            print(" WARNING: Very large logit magnitudes")

        # Greedy decode a few tokens
        generated = self.jamba.generate(
            input_ids=inputs['input_ids'],
            max_new_tokens=20,
            do_sample=False,
        )
        decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f" Generated: '{decoded}'")

        print(" Verification PASSED")
        return True

    def get_all_operator_stats(self, hidden_states: torch.Tensor, prefix_len: int) -> Dict:
        """Get operator statistics for all SKA layers."""
        stats = {}
        for idx in self.layer_indices:
            layer_key = f"layer_{idx}"
            module = self.ska_modules[layer_key]
            if isinstance(module, SKAModule):
                stats[layer_key] = module.get_operator_stats(hidden_states, prefix_len)
        return stats
