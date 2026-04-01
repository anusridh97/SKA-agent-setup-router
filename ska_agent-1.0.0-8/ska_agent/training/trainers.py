"""
Training procedures for all learned components.

This file contains five classes:

SpectralRegularization
  Penalizes large singular values of the actual Koopman operator A_w
  (not the Gram matrix). Builds the full operator via Cholesky and
  triangular solves, then uses torch.linalg.svdvals(A_w) to compute
  the penalty: L_spec = lambda_spec * sum(sigma_i^2).
  This keeps operator eigenvalues decaying for stability.

OrthogonalRegularization
  Penalizes deviation of key/query projection matrices from orthogonality:
  L_ortho = lambda_ortho * ||W^T W - I||_F^2.
  Prevents collapse of the key/query spaces during training.

SKATrainer
  Trains the 4 SKA modules (200M params) in two stages:
  Stage 1 (LM recovery): 10B tokens of SlimPajama, AdamW lr=3e-4,
    cosine decay, targeting perplexity within 5% of vanilla Jamba.
  Stage 2 (table QA): WikiTableQuestions + TAT-QA + FinQA + HybridQA,
    loss on answer tokens only, tables serialized as graph text.

RouterTrainer
  Trains the reward predictor (MSE on quality differences) and mode
  selector (cross-entropy on best-mode labels) from Phase 1 baseline
  scores on OfficeQA.

BridgeTrainer
  Trains the bridge projection W_bridge end-to-end. The entire operator
  construction (Cholesky, triangular solves, spectral normalization,
  power filtering) is done in torch so gradients flow through:
  W_bridge -> ska_keys -> G,M -> L -> A_w -> w_f -> output -> loss.
  No detach() anywhere in the forward path.

All classes require torch at runtime but the module is importable without
torch (dummy nn.Module base class is provided so class definitions parse).

Dependencies:
  - core/structures.py for training configs
  - core/ska_module.py classes are used indirectly through the model
  - models/jamba_ska.py (JambaSKAModel) is the training target for SKATrainer
"""

from __future__ import annotations

import math
import os
from typing import Optional, List, Dict, Tuple, Callable

import numpy as np

from ..core.structures import SKATrainingConfig, RouterTrainingConfig, SystemConfig

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    # When torch is unavailable, provide a dummy nn.Module base class
    # so class definitions parse. Methods will fail with NameError at call time.
    import types
    nn = types.SimpleNamespace(Module=object, ModuleDict=dict, Parameter=None)
    F = None
    torch = None


# Regularization Losses (§8.1.1)
# All classes below require torch at runtime but are importable without it.

class SpectralRegularization(nn.Module):
    """
    Spectral regularization: penalize large eigenvalues of the Koopman operator.

    L_spec = λ_spec · Σ_i σ_i(A_w)^2

    Keeps operator eigenvalues decaying, ensuring stability (§8.1.1).
    """

    def __init__(self, lambda_spec: float = 0.01):
        super().__init__()
        self.lambda_spec = lambda_spec

    def forward(self, ska_modules: nn.ModuleDict, hidden_states: torch.Tensor, prefix_len: int) -> torch.Tensor:
        """
        Compute spectral regularization loss.

        Builds the actual Koopman operator A_w for each SKA module and
        penalizes its singular values: L_spec = λ · Σ σ_i(A_w)².
        """
        total_loss = torch.tensor(0.0, device=hidden_states.device, requires_grad=True)

        for name, module in ska_modules.items():
            if not hasattr(module, 'W_K'):
                continue

            normed = module.layer_norm(hidden_states)
            keys = module.W_K(normed)
            B, T, _ = keys.shape
            keys = keys.view(B, T, module.n_heads, module.rank).transpose(1, 2)
            prefix_keys = keys[:, :, :prefix_len, :]

            # Build Gram and transition matrices
            r = module.rank
            G = torch.matmul(prefix_keys.transpose(-2, -1), prefix_keys)
            ridge = module.log_ridge.exp()
            G = G + ridge * torch.eye(r, device=G.device).unsqueeze(0).unsqueeze(0)

            if prefix_len > 1:
                M = torch.matmul(
                    prefix_keys[:, :, 1:, :].transpose(-2, -1),
                    prefix_keys[:, :, :-1, :],
                )
            else:
                M = torch.zeros_like(G)

            # Build A_w = L^{-1} M L^{-T} via Cholesky + triangular solves
            G_d = G.to(torch.float64)
            try:
                L_chol = torch.linalg.cholesky(G_d)
            except torch.linalg.LinAlgError:
                G_d = G_d + 1e-4 * torch.eye(r, device=G.device, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
                L_chol = torch.linalg.cholesky(G_d)

            M_d = M.to(torch.float64)
            U = torch.linalg.solve_triangular(L_chol, M_d, upper=False)
            V = torch.linalg.solve_triangular(L_chol, U.transpose(-2, -1), upper=False)
            A_w = V.transpose(-2, -1) # (B, H, r, r)

            # Penalize singular values of A_w (the actual operator)
            sv = torch.linalg.svdvals(A_w) # (B, H, r)
            spec_loss = self.lambda_spec * (sv ** 2).sum()
            total_loss = total_loss + spec_loss

        return total_loss


class OrthogonalRegularization(nn.Module):
    """
    Orthogonal regularization on key/query projections (§8.1.1).

    L_ortho = λ_ortho · ||W^T W - I||_F^2

    Encourages the projection matrices to preserve distances,
    preventing collapse of the key/query spaces.
    """

    def __init__(self, lambda_ortho: float = 0.01):
        super().__init__()
        self.lambda_ortho = lambda_ortho

    def forward(self, ska_modules: nn.ModuleDict) -> torch.Tensor:
        """Compute orthogonal regularization loss."""
        total_loss = torch.tensor(0.0, requires_grad=True)

        for name, module in ska_modules.items():
            if not hasattr(module, 'W_K'):
                continue

            for proj_name, proj in [('W_K', module.W_K), ('W_Q', module.W_Q)]:
                W = proj.weight # (out_features, in_features)
                WtW = W @ W.T
                eye = torch.eye(W.shape[0], device=W.device)
                ortho_loss = self.lambda_ortho * ((WtW - eye) ** 2).sum()
                total_loss = total_loss + ortho_loss

        return total_loss


# SKA Training (Phase 3)

class SKATrainer:
    """
    SKA module trainer for Phase 3 (§8.1).

    Two stages:
      1. LM Recovery: Train on SlimPajama to recover perplexity
      2. Table QA Fine-tuning: Train on structured table datasets
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: SKATrainingConfig = None,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SKATrainingConfig()
        self.device = device

        # Regularization losses
        self.spectral_reg = SpectralRegularization(self.config.spectral_lambda)
        self.ortho_reg = OrthogonalRegularization(self.config.ortho_lambda)

        # Optimizer: only SKA parameters
        self.ska_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.ska_params,
            lr=self.config.lm_lr,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps,
            weight_decay=self.config.weight_decay,
        )

        # Cosine scheduler
        self.scheduler = None # Set up when total steps are known

    def _setup_scheduler(self, total_steps: int):
        """Create cosine decay scheduler with warmup."""
        warmup = self.config.lm_warmup_steps

        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / (total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda,
        )

    def train_lm_recovery(
        self,
        dataset_name: str = "cerebras/SlimPajama-627B",
        num_tokens: int = None,
        eval_dataset=None,
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_dir: str = "./checkpoints/ska_lm_recovery",
    ):
        """
        Stage 1: Language Modeling Recovery (§8.1.1).

        Train on general text to recover pretrained capabilities.

        Args:
            dataset_name: HuggingFace dataset for training
            num_tokens: Total tokens to train on (default: 10B)
            eval_dataset: Optional validation dataset for perplexity
            log_interval: Log every N steps
            eval_interval: Evaluate every N steps
            save_dir: Checkpoint directory
        """
        from datasets import load_dataset

        num_tokens = num_tokens or self.config.lm_recovery_tokens
        seq_len = self.config.lm_seq_length
        batch_size = self.config.lm_batch_size
        tokens_per_step = batch_size * seq_len
        total_steps = num_tokens // tokens_per_step

        print(f"\n=== Stage 1: LM Recovery ===")
        print(f" Dataset: {dataset_name}")
        print(f" Total tokens: {num_tokens:,}")
        print(f" Batch size: {batch_size}")
        print(f" Sequence length: {seq_len}")
        print(f" Total steps: {total_steps:,}")
        print(f" LR: {self.config.lm_lr}")

        self._setup_scheduler(total_steps)

        # Load dataset (streaming for large datasets)
        print(f" Loading dataset...")
        dataset = load_dataset(
            dataset_name,
            split="train",
            streaming=True,
        )

        os.makedirs(save_dir, exist_ok=True)
        self.model.train()

        step = 0
        total_loss = 0.0
        token_buffer = []

        for example in dataset:
            # Tokenize and buffer
            text = example.get('text', '')
            if not text:
                continue

            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=seq_len + 1,
                return_tensors="pt",
            )
            token_buffer.append(tokens['input_ids'].squeeze(0))

            if len(token_buffer) < batch_size:
                continue

            # Form batch
            batch_ids = torch.nn.utils.rnn.pad_sequence(
                token_buffer[:batch_size],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ).to(self.device)

            token_buffer = token_buffer[batch_size:]

            # Truncate to seq_len + 1
            if batch_ids.shape[1] > seq_len + 1:
                batch_ids = batch_ids[:, :seq_len + 1]

            input_ids = batch_ids[:, :-1]
            labels = batch_ids[:, 1:]

            # Forward pass
            outputs = self.model(input_ids=input_ids, labels=labels)
            lm_loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

            # Regularization
            # (Simplified: would need hidden states from hooks for full implementation)
            reg_loss = self.ortho_reg(self.model.ska_modules) if hasattr(self.model, 'ska_modules') else 0.0

            loss = lm_loss + reg_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ska_params, 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            step += 1

            if step % log_interval == 0:
                avg_loss = total_loss / log_interval
                lr = self.scheduler.get_last_lr()[0]
                ppl = math.exp(min(avg_loss, 20))
                print(f" Step {step}/{total_steps}: loss={avg_loss:.4f}, ppl={ppl:.2f}, lr={lr:.6f}")
                total_loss = 0.0

            if step % eval_interval == 0 and eval_dataset is not None:
                eval_ppl = self.evaluate_perplexity(eval_dataset)
                print(f" Eval perplexity: {eval_ppl:.2f}")

                # Save checkpoint
                ckpt_path = os.path.join(save_dir, f"step_{step}.pt")
                self._save_checkpoint(ckpt_path, step)

            if step >= total_steps:
                break

        print(f"\n LM Recovery complete. Final step: {step}")

    def train_table_qa(
        self,
        datasets: List[str] = None,
        num_epochs: int = None,
        save_dir: str = "./checkpoints/ska_table_qa",
    ):
        """
        Stage 2: Table QA Fine-tuning (§8.1.2).

        Fine-tune on structured table understanding:
          - WikiTableQuestions (22K)
          - TAT-QA (16K)
          - FinQA (8K)
          - HybridQA (70K)

        Tables serialized as graph text. Loss on answer tokens only.
        """
        from datasets import load_dataset, concatenate_datasets

        datasets = datasets or [
            "wikitablequestions",
            "next-tat-qa",
            "dreamerdeo/finqa",
            "hybridqa",
        ]
        num_epochs = num_epochs or self.config.tableqa_epochs

        print(f"\n=== Stage 2: Table QA Fine-tuning ===")
        print(f" Datasets: {datasets}")
        print(f" Epochs: {num_epochs}")

        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.tableqa_lr

        os.makedirs(save_dir, exist_ok=True)
        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_examples = 0

            for ds_name in datasets:
                try:
                    ds = load_dataset(ds_name, split="train", streaming=True)
                except Exception as e:
                    print(f" Could not load {ds_name}: {e}")
                    continue

                for example in ds:
                    # Format as graph-text (§9.2)
                    formatted = self._format_table_qa(example, ds_name)
                    if formatted is None:
                        continue

                    question, context, answer = formatted

                    # Tokenize
                    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                    full_text = f"{prompt} {answer}"

                    tokens = self.tokenizer(
                        full_text,
                        truncation=True,
                        max_length=self.config.lm_seq_length,
                        return_tensors="pt",
                    ).to(self.device)

                    # Create labels: -100 for non-answer tokens
                    prompt_tokens = self.tokenizer(
                        prompt,
                        truncation=True,
                        max_length=self.config.lm_seq_length,
                        return_tensors="pt",
                    )
                    prompt_len = prompt_tokens['input_ids'].shape[1]

                    labels = tokens['input_ids'].clone()
                    labels[0, :prompt_len] = -100 # Loss only on answer tokens

                    # Forward + backward
                    outputs = self.model(
                        input_ids=tokens['input_ids'],
                        labels=labels,
                    )
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.ska_params, 1.0)
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    num_examples += 1

                    if num_examples % 500 == 0:
                        avg = epoch_loss / num_examples
                        print(f" Epoch {epoch+1}, examples={num_examples}: avg_loss={avg:.4f}")

            if num_examples > 0:
                avg_loss = epoch_loss / num_examples
                print(f" Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}")

            # Save checkpoint
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}.pt")
            self._save_checkpoint(ckpt_path, epoch)

    def _format_table_qa(self, example: dict, dataset_name: str) -> Optional[Tuple[str, str, str]]:
        """
        Format a table QA example into graph text format (§9.2).

        Returns (question, context, answer) or None if invalid.
        """
        try:
            if "wikitable" in dataset_name.lower():
                question = example.get('question', '')
                table = example.get('table', {})
                answer = example.get('answers', [''])[0] if isinstance(example.get('answers'), list) else str(example.get('answers', ''))
                context = self._table_to_graph(table)
            elif "tat" in dataset_name.lower():
                question = example.get('question', '')
                context = example.get('context', '')
                answer = str(example.get('answer', ''))
            elif "finqa" in dataset_name.lower():
                question = example.get('question', '')
                context = example.get('context', '')
                answer = str(example.get('answer', ''))
            elif "hybrid" in dataset_name.lower():
                question = example.get('question', '')
                context = example.get('context', '')
                answer = str(example.get('answer', ''))
            else:
                return None

            if not question or not answer:
                return None

            return question, context, answer
        except Exception:
            return None

    def _table_to_graph(self, table: dict) -> str:
        """
        Convert table dict to graph serialization format (Listing 1, §9.2).

        [NODE: table_0_header] type=header | col1 | col2 | ...
        [NODE: table_0_row_i] type=row | val1 | val2 | ...
        [EDGE] table_0_header --has_row--> table_0_row_i
        """
        lines = []
        headers = table.get('header', table.get('columns', []))
        rows = table.get('rows', table.get('data', []))

        if headers:
            lines.append(f"[NODE: table_0_header] type=header | {' | '.join(str(h) for h in headers)}")

        for i, row in enumerate(rows):
            if isinstance(row, (list, tuple)):
                row_text = ' | '.join(str(v) for v in row)
            else:
                row_text = str(row)
            lines.append(f"[NODE: table_0_row_{i}] type=row | {row_text}")
            lines.append(f"[EDGE] table_0_header --has_row--> table_0_row_{i}")

        return '\n'.join(lines)

    def evaluate_perplexity(self, eval_dataset, max_examples: int = 100) -> float:
        """Evaluate perplexity on a validation dataset."""
        self.model.eval()
        total_loss = 0.0
        num_examples = 0

        with torch.no_grad():
            for example in eval_dataset:
                if num_examples >= max_examples:
                    break

                text = example.get('text', '')
                if not text:
                    continue

                tokens = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.config.lm_seq_length,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(
                    input_ids=tokens['input_ids'],
                    labels=tokens['input_ids'],
                )
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                total_loss += loss.item()
                num_examples += 1

        self.model.train()
        avg_loss = total_loss / max(num_examples, 1)
        return math.exp(min(avg_loss, 20))

    def _save_checkpoint(self, path: str, step_or_epoch: int):
        """Save trainable parameters only."""
        state = {
            'step': step_or_epoch,
            'model_state': {
                name: param.data
                for name, param in self.model.named_parameters()
                if param.requires_grad
            },
            'optimizer_state': self.optimizer.state_dict(),
        }
        if self.scheduler:
            state['scheduler_state'] = self.scheduler.state_dict()
        torch.save(state, path)
        print(f" Saved checkpoint: {path}")


# Router Training (Phase 2)

class RouterTrainer:
    """
    Router component training (§8.2).

    Trains:
      1. Reward predictor: MSE on quality differences (§8.2.1)
      2. Mode selector: Cross-entropy on best-mode labels (§8.2.2)
    """

    def __init__(
        self,
        reward_predictor: nn.Module,
        mode_selector: nn.Module,
        config: RouterTrainingConfig = None,
        device: str = "cuda",
    ):
        self.reward_predictor = reward_predictor
        self.mode_selector = mode_selector
        self.config = config or RouterTrainingConfig()
        self.device = device

    def train_reward_predictor(
        self,
        training_data: List[Dict],
        log_interval: int = 50,
    ):
        """
        Train the reward predictor (§8.2.1).

        Data format:
          {'query_embedding': np.ndarray,
           'model_idx': int,
           'base_model_idx': int,
           'delta_r': float}

        Trains MSE regression on observed quality differences (Eq. 24):
            L_r = E[(F_θr(e_Q, m1, m2) - Δr)²]
        """
        print(f"\n=== Training Reward Predictor ===")
        print(f" Samples: {len(training_data)}")
        print(f" Epochs: {self.config.reward_epochs}")

        self.reward_predictor.train()
        self.reward_predictor.to(self.device)

        optimizer = torch.optim.AdamW(
            self.reward_predictor.parameters(),
            lr=self.config.reward_lr,
        )

        for epoch in range(self.config.reward_epochs):
            np.random.shuffle(training_data)
            total_loss = 0.0

            for sample in training_data:
                e_q = torch.tensor(sample['query_embedding'], dtype=torch.float32).unsqueeze(0).to(self.device)
                m_idx = torch.tensor([sample['model_idx']]).to(self.device)
                b_idx = torch.tensor([sample['base_model_idx']]).to(self.device)
                target = torch.tensor([[sample['delta_r']]], dtype=torch.float32).to(self.device)

                pred = self.reward_predictor(e_q, m_idx, b_idx)
                loss = F.mse_loss(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % log_interval == 0:
                avg = total_loss / len(training_data)
                print(f" Epoch {epoch+1}/{self.config.reward_epochs}: MSE={avg:.6f}")

    def train_mode_selector(
        self,
        training_data: List[Dict],
        log_interval: int = 20,
    ):
        """
        Train the mode selector (§8.2.2).

        Data format:
          {'query_embedding': np.ndarray, 'mode_idx': int}

        Cross-entropy classification on best-mode labels.
        """
        print(f"\n=== Training Mode Selector ===")
        print(f" Samples: {len(training_data)}")
        print(f" Epochs: {self.config.mode_epochs}")

        self.mode_selector.train()
        self.mode_selector.to(self.device)

        optimizer = torch.optim.AdamW(
            self.mode_selector.parameters(),
            lr=self.config.mode_lr,
        )

        for epoch in range(self.config.mode_epochs):
            np.random.shuffle(training_data)
            total_loss = 0.0
            correct = 0

            for sample in training_data:
                e_q = torch.tensor(sample['query_embedding'], dtype=torch.float32).unsqueeze(0).to(self.device)
                target = torch.tensor([sample['mode_idx']]).to(self.device)

                logits = self.mode_selector(e_q)
                loss = F.cross_entropy(logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred_idx = logits.argmax(dim=-1).item()
                correct += int(pred_idx == sample['mode_idx'])

            if (epoch + 1) % log_interval == 0:
                avg = total_loss / len(training_data)
                acc = correct / len(training_data) * 100
                print(f" Epoch {epoch+1}/{self.config.mode_epochs}: loss={avg:.4f}, acc={acc:.1f}%")


# Bridge Training (Phase 5)

class BridgeTrainer:
    """
    Bridge projection training (§8.3).

    Trains W_bridge end-to-end on multi-agent reasoning tasks.

    Frozen: DeepSeek V3 weights, SKA weights (from Phase 3)
    Trainable: W_bridge, gate parameters for injection strength

    Objective: Final answer quality (QA accuracy) backpropagated
    through the bridge projection.
    """

    def __init__(
        self,
        bridge: nn.Module,
        shared_memory,
        config: SystemConfig = None,
        device: str = "cuda",
    ):
        self.bridge = bridge
        self.shared_memory = shared_memory
        self.config = config or SystemConfig()
        self.device = device

        self.optimizer = torch.optim.AdamW(
            bridge.parameters(),
            lr=1e-4,
        )

    def train_step(
        self,
        mla_latents: torch.Tensor,
        query_keys: torch.Tensor,
        target_values: torch.Tensor,
    ) -> float:
        """
        Single training step for bridge projection.

        Builds the Koopman operator entirely in torch so gradients flow:
            W_bridge -> ska_keys -> G,M -> L -> A_w,B_v -> output -> loss

        Args:
            mla_latents: (B, L, d_c) MLA latents from DeepSeek V3
            query_keys: (B, Q, r) query keys from retrieval agent
            target_values: (B, Q, d_v) expected retrieval values

        Returns:
            loss value
        """
        self.bridge.train()

        B, L_len, d_c = mla_latents.shape
        _, Q, r = query_keys.shape
        device = mla_latents.device

        # Project latents through bridge (differentiable)
        ska_keys = self.bridge(mla_latents) # (B, L, r)

        ridge = self.shared_memory.ridge_eps
        power_K = self.shared_memory.power_K

        # Build operator in torch (differentiable) - no numpy detour
        # G = K^T K + εI
        G = torch.bmm(ska_keys.transpose(-2, -1), ska_keys) # (B, r, r)
        G = G + ridge * torch.eye(r, device=device).unsqueeze(0)

        # M = K[1:]^T K[:-1]
        if L_len > 1:
            M = torch.bmm(ska_keys[:, 1:, :].transpose(-2, -1), ska_keys[:, :-1, :])
        else:
            M = torch.zeros(B, r, r, device=device)

        # Cholesky in double precision
        G_d = G.to(torch.float64)
        try:
            L_chol = torch.linalg.cholesky(G_d)
        except torch.linalg.LinAlgError:
            G_d = G_d + 1e-4 * torch.eye(r, device=device, dtype=torch.float64).unsqueeze(0)
            L_chol = torch.linalg.cholesky(G_d)

        # A_w = L^{-1} M L^{-T} (similarity transform)
        M_d = M.to(torch.float64)
        U = torch.linalg.solve_triangular(L_chol, M_d, upper=False)
        V = torch.linalg.solve_triangular(L_chol, U.transpose(-2, -1), upper=False)
        A_w = V.transpose(-2, -1) # (B, r, r) in float64

        # Spectral normalization (gamma ≤ 1)
        sigma = torch.linalg.norm(A_w, ord=2, dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        A_w = A_w / torch.clamp(sigma, min=1.0)

        # B_v = C_v G^{-1} via two triangular solves
        # Need values - use ska_keys as both keys and pseudo-values for bridge training
        # (In production, values come from the SKA value projection; here we train
        # the bridge to produce keys that retrieve target_values.)
        # C_v = target_values' contribution is implicit - B_v maps operator output to value space.
        # For bridge training, we directly evaluate the filter pipeline:
        # output = B_v · L · A_w^K · L^{-1} · query
        # But B_v depends on values we don't have from the bridge.
        # Solution: bypass B_v and train the bridge so the operator's
        # whiten->power->unwhiten pipeline aligns query_keys with target_values.

        # Whiten queries: w_q = L^{-1} z_q
        q_d = query_keys.to(torch.float64).transpose(-2, -1) # (B, r, Q)
        w_q = torch.linalg.solve_triangular(L_chol, q_d, upper=False)

        # Power filter
        w_f = w_q
        for _ in range(power_K):
            w_f = torch.bmm(A_w, w_f)

        # Unwhiten
        z_hat = torch.bmm(L_chol, w_f) # (B, r, Q)

        # Project to value dimension via a learned linear (part of bridge)
        # z_hat is (B, r, Q) - transpose to (B, Q, r)
        z_hat = z_hat.transpose(-2, -1).to(mla_latents.dtype)

        # Use inverse bridge as the value readout (maps r -> d_v)
        retrieved = self.bridge.W_bridge_inv(z_hat) # (B, Q, d_c)

        # Trim or pad to match target_values dimension
        d_v = target_values.shape[-1]
        if retrieved.shape[-1] > d_v:
            retrieved = retrieved[..., :d_v]
        elif retrieved.shape[-1] < d_v:
            pad = torch.zeros(*retrieved.shape[:-1], d_v - retrieved.shape[-1], device=device)
            retrieved = torch.cat([retrieved, pad], dim=-1)

        loss = F.mse_loss(retrieved, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
