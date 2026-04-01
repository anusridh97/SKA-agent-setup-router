"""
Sentence embedding model wrapper.

Provides dense vector embeddings of text using sentence-transformers.
Used by:
  - core/geometry.py (via pipeline.py) for embedding sentences before
    segmentation
  - core/pricing.py (via pipeline.py) for embedding queries during
    retrieval
  - orchestration/__init__.py (TSMemoryBridge) for embedding text entries
    from the TS orchestrator's shared memory before projecting into
    spectral memory

Auto-detects GPU availability and selects an appropriate model size.
All torch imports are lazy so the module is importable without torch
installed (the Embedder class requires torch only at instantiation).
"""

from __future__ import annotations

from typing import List

import numpy as np


class Embedder:
    """Wrapper for sentence-transformers embedding models."""

    def __init__(self, model_name: str = None):
        import torch
        from sentence_transformers import SentenceTransformer

        if torch.cuda.is_available():
            self.device = "cuda"
            default_model = 'Alibaba-NLP/gte-Qwen2-7B-instruct'
            print(f"GPU detected. Using high-fidelity embedder: {default_model}")
        else:
            self.device = "cpu"
            default_model = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
            print(f"No GPU found. Using lightweight CPU embedder: {default_model}")

        self.model_name = model_name or default_model
        print(f"Loading embedding model: {self.model_name}")

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            model_kwargs={"torch_dtype": torch.float16},
        )

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedder ready (dim={self.embedding_dim})")

    def embed(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Embed sentences to dense vectors."""
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            device=self.device,
        )
        return embeddings.astype(np.float64)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed([text], show_progress=False)[0]
