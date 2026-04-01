"""
SKA-Agent: Adaptive Multi-Model Orchestration with Structured Kernel
Attention and Spectral Shared Memory.

Install:
    pip install -e .

    # With all dependencies (GPU):
    pip install -e ".[all]"
"""

from setuptools import setup, find_packages

setup(
    name="ska-agent",
    version="1.0.0",
    description=(
        "Adaptive multi-model orchestration with Structured Kernel Attention "
        "and Spectral Shared Memory for enterprise document reasoning"
    ),
    author="SKA-Agent Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        # Core
        "numpy>=1.24",
        "torch>=2.0",
        "transformers>=4.40",
        "sentence-transformers>=2.2",
        "nltk>=3.8",
        # Data
        "datasets>=2.14",
        # PDF processing
        "pdfplumber>=0.9",
    ],
    extras_require={
        "all": [
            # Full model support
            "mamba-ssm>=1.2",
            "causal-conv1d>=1.2",
            "vllm>=0.5.5",
            "bitsandbytes>=0.41",
            # Table QA datasets
            "rank_bm25>=0.2",
            # OCR
            "pytesseract>=0.3",
            # Evaluation
            "tqdm>=4.65",
            "wandb>=0.15",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "ruff>=0.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "ska-agent=ska_agent.cli:main",
        ],
    },
)
