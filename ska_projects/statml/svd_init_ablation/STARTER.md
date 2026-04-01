# Starter: SVD Initialization Ablation

## Step 0: Extract Weights (One-Time GPU)

```python
# Use a small model to avoid GPU requirements
# Qwen2.5-1.5B-Instruct has GQA and is small enough for CPU
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float32,
    trust_remote_code=True,
)

# Extract attention weights from layer 4 (or whichever has attention)
# Save to disk for offline analysis
weights = {}
for name, param in model.named_parameters():
    if any(k in name for k in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
        weights[name] = param.data.cpu().numpy()
        print(f"{name}: {param.shape}")

import pickle
with open("attention_weights.pkl", "wb") as f:
    pickle.dump(weights, f)
print(f"Saved {len(weights)} weight matrices")
```

After this, everything else is CPU-only numpy work on saved weights.

## Step 1: Implement Strategies

```python
import numpy as np

def init_random_orthogonal(d_model, n_heads, rank):
    W = np.random.randn(n_heads * rank, d_model)
    Q, _ = np.linalg.qr(W.T)
    return Q[:, :n_heads * rank].T

def init_svd_sqrt(W_K_head, rank):
    """Current codebase strategy."""
    U, S, Vt = np.linalg.svd(W_K_head, full_matrices=False)
    return np.diag(np.sqrt(S[:rank])) @ Vt[:rank, :]

def init_svd_full(W_K_head, rank):
    U, S, Vt = np.linalg.svd(W_K_head, full_matrices=False)
    return np.diag(S[:rank]) @ Vt[:rank, :]

def init_svd_noscale(W_K_head, rank):
    U, S, Vt = np.linalg.svd(W_K_head, full_matrices=False)
    return Vt[:rank, :]

def init_pca(W_K_head, rank):
    centered = W_K_head - W_K_head.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return np.diag(np.sqrt(S[:rank])) @ Vt[:rank, :]

# NMF requires non-negative data, use absolute values or skip
```

## Step 2: Measure and Compare

See the `analyze_init_strategy()` function in SPEC.md.
