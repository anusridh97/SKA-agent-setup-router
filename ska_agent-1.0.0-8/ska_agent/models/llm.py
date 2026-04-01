"""
LLM generator for answer synthesis.

Wraps a causal language model for generating final answers from
retrieved context. This is a simpler alternative to the Qwen coordinator
for cases where structured <think> reasoning is not needed.

Default model is Qwen2.5-7B-Instruct (ungated, no approval needed).
Falls back to 1.5B variant on CPU.

Used by pipeline.py (RetrievalPipeline) to generate answers after
the pricing engine selects relevant segments.

All torch and transformers imports are lazy so the module is importable
without torch installed.
"""

from __future__ import annotations


class LLMGenerator:
    """LLM generator for answer synthesis."""

    def __init__(
        self,
        model_name: str = None,
        max_new_tokens: int = 512,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            self.device = "cuda"
            default_model = "Qwen/Qwen2.5-7B-Instruct"
            print(f"GPU detected. Using high-performance model.")
        else:
            self.device = "cpu"
            default_model = "Qwen/Qwen2.5-1.5B-Instruct"
            print(f"No GPU found. Falling back to lightweight CPU model.")

        self.model_name = model_name or default_model
        self.max_new_tokens = max_new_tokens

        print(f"Loading {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"LLM ready on {self.model.device}")

    def generate(self, query: str, context: str, max_tokens: int = None) -> str:
        """Generate an answer given query and retrieved context."""
        tokens_to_generate = max_tokens or self.max_new_tokens

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer based only on the provided context. "
                    "Be concise and direct. If the answer is not in the context, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with __import__('torch').no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=tokens_to_generate,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()
