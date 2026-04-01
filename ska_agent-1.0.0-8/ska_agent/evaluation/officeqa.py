"""
OfficeQA Evaluation Pipeline.

OfficeQA is an enterprise document reasoning benchmark:
  696 U.S. Treasury Bulletin PDFs (1939-2025), ~89,000 pages
  246 questions requiring parsing, retrieval, table understanding,
  and numerical reasoning
  Tolerance-based numerical scoring: |predicted - truth| / |truth| <= 0.01

This file provides:

  score_answer(ground_truth, predicted, tolerance)
    Scores a single prediction. Tries numeric comparison first (with
    tolerance), falls back to exact string match.

  DocumentProcessor
    Converts PDFs to structured graph representations:
    [NODE: table_0_header] type=header | Year | Revenue | Expenses
    [NODE: table_0_row_1] type=row | 1941 | $12,345 | ...
    [EDGE] table_0_header -> table_0_row_1

  CodeExecutor
    Sandboxed Python execution for numerical computation.

  OfficeQAEvaluator
    End-to-end evaluation: router encodes query, selects mode, dispatches
    to specialists, scores the final answer. Monitors operator conditioning
    throughout and rebuilds any operator whose condition number exceeds 1e4.

  AblationRunner
    Runs the full ablation matrix (Table 2 from the spec):
    baseline -> +SKA -> +router -> +shared_memory -> +multi_Koopman
    measuring each component's marginal contribution.

Dependencies:
  - core/structures.py for data types
  - router/adaptive_router.py is used if available
  - pipeline.py provides the full system for evaluation
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import numpy as np


# Scoring (§9.3)

def score_answer(
    ground_truth: str,
    predicted: str,
    tolerance: float = 0.01,
) -> int:
    """
    Score a predicted answer against ground truth.

    Uses tolerance-based numerical matching (§9.1):
        |predicted - ground_truth| / |ground_truth| ≤ tolerance

    For non-numeric answers, falls back to exact string match.

    Returns: 0 or 1
    """
    # Try numeric comparison first
    try:
        gt_val = float(ground_truth.replace(',', '').replace('$', '').replace('%', '').strip())
        pred_val = float(predicted.replace(',', '').replace('$', '').replace('%', '').strip())

        if abs(gt_val) < 1e-10:
            return int(abs(pred_val) < tolerance)

        relative_error = abs(pred_val - gt_val) / abs(gt_val)
        return int(relative_error <= tolerance)
    except (ValueError, AttributeError):
        pass

    # Fallback: normalized string match
    gt_norm = str(ground_truth).strip().lower()
    pred_norm = str(predicted).strip().lower()
    return int(gt_norm == pred_norm)


# Data Structures

@dataclass
class OfficeQAQuestion:
    """A single OfficeQA question."""
    question_id: str
    question: str
    answer: str
    source_documents: List[str] = field(default_factory=list)
    question_type: str = ""
    difficulty: str = ""


@dataclass
class EvalResult:
    """Result of evaluating a single question."""
    question_id: str
    question: str
    ground_truth: str
    predicted: str
    score: int
    latency_ms: float
    cost: float = 0.0
    retrieval_segments: int = 0
    mode_selected: str = ""
    actions_taken: int = 0
    operator_condition: float = 1.0
    error: str = ""


@dataclass
class AblationResult:
    """Aggregated results for one ablation configuration."""
    config_name: str
    results: List[EvalResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)

    @property
    def avg_cost(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.cost for r in self.results) / len(self.results)

    @property
    def avg_segments(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.retrieval_segments for r in self.results) / len(self.results)

    @property
    def avg_condition(self) -> float:
        conditions = [r.operator_condition for r in self.results if r.operator_condition > 0]
        return sum(conditions) / len(conditions) if conditions else 1.0

    @property
    def max_condition(self) -> float:
        conditions = [r.operator_condition for r in self.results if r.operator_condition > 0]
        return max(conditions) if conditions else 1.0

    def summary(self) -> str:
        return (
            f" {self.config_name}:\n"
            f" Accuracy: {self.accuracy:.4f} ({sum(r.score for r in self.results)}/{len(self.results)})\n"
            f" Avg latency: {self.avg_latency_ms:.1f}ms\n"
            f" Avg cost: ${self.avg_cost:.4f}\n"
            f" Avg segments: {self.avg_segments:.1f}\n"
            f" Operator cond: avg={self.avg_condition:.1f}, max={self.max_condition:.1f}"
        )


# Document Processing Pipeline (§9.2)

class DocumentProcessor:
    """
    Document processing pipeline for Treasury Bulletin PDFs (§9.2).

    Steps:
      1. PDF parsing: Convert scanned/digital PDFs to text + tables
      2. Graph construction: Tables -> node-edge graph format
      3. Indexing: Embed graph nodes for retrieval
    """

    def __init__(self, ocr_backend: str = "pdfplumber"):
        self.ocr_backend = ocr_backend

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a single PDF into structured graph representation.

        Returns:
            {'text': str, 'tables': List[Dict], 'graph': str}
        """
        text = ""
        tables = []

        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"

                    page_tables = page.extract_tables()
                    for t in page_tables:
                        tables.append(t)
        except ImportError:
            # Fallback: basic text extraction
            try:
                import PyPDF2
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += (page.extract_text() or "") + "\n"
            except ImportError:
                print(f" No PDF library available for {pdf_path}")

        graph = self._tables_to_graph(tables)

        return {
            'text': text,
            'tables': tables,
            'graph': graph,
            'source': pdf_path,
        }

    def _tables_to_graph(self, tables: List) -> str:
        """
        Convert extracted tables to graph serialization format (Listing 1).

        [NODE: table_N_header] type=header | col1 | col2 | ...
        [NODE: table_N_row_i] type=row | val1 | val2 | ...
        [EDGE] table_N_header --has_row--> table_N_row_i
        """
        lines = []
        for t_idx, table in enumerate(tables):
            if not table or len(table) == 0:
                continue

            # First row is typically headers
            headers = table[0] if table else []
            header_str = ' | '.join(str(h or '') for h in headers)
            lines.append(f"[NODE: table_{t_idx}_header] type=header | {header_str}")

            for r_idx, row in enumerate(table[1:]):
                row_str = ' | '.join(str(v or '') for v in row)
                lines.append(f"[NODE: table_{t_idx}_row_{r_idx}] type=row | {row_str}")
                lines.append(f"[EDGE] table_{t_idx}_header --has_row--> table_{t_idx}_row_{r_idx}")

                # Cell-level nodes for fine-grained retrieval
                for c_idx, (header, value) in enumerate(zip(headers, row)):
                    if value:
                        lines.append(
                            f"[NODE: table_{t_idx}_cell_{r_idx}_{c_idx}] "
                            f"type=cell | {header}: {value}"
                        )
                        lines.append(
                            f"[EDGE] table_{t_idx}_row_{r_idx} --has_value--> "
                            f"table_{t_idx}_cell_{r_idx}_{c_idx}"
                        )

        return '\n'.join(lines)


# Code Executor (sandbox)

class CodeExecutor:
    """
    Sandboxed Python execution for numerical computation (§3.1).

    Deterministic and exact. Handles:
      - Arithmetic and aggregation
      - Statistical computation
      - Regression
    """

    def __init__(self, timeout: int = 30, max_memory_mb: int = 512):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

    def execute(self, code: str) -> Tuple[str, bool]:
        """
        Execute Python code in sandbox.

        Args:
            code: Python code to execute

        Returns:
            (output, success)
        """
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                result = subprocess.run(
                    ['python', f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                if result.returncode == 0:
                    return result.stdout.strip(), True
                else:
                    return result.stderr.strip(), False
            except subprocess.TimeoutExpired:
                return "Execution timed out", False
            finally:
                os.unlink(f.name)


# Evaluation Pipeline

class OfficeQAEvaluator:
    """
    End-to-end OfficeQA evaluation pipeline (§9.3).

    For each question:
      1. Router encodes query, selects mode, identifies source documents
      2. Parser processes relevant PDFs -> graph representations
      3. Jamba+SKA retrieves relevant subgraphs and extracts values
      4. Code executor performs required computation
      5. DeepSeek V3 synthesizes final answer (with shared memory)
      6. score_answer(ground_truth, predicted, tolerance=0.01)
    """

    def __init__(
        self,
        router=None,
        retriever=None,
        code_executor=None,
        reasoner=None,
        parser=None,
        shared_memory=None,
        tolerance: float = 0.01,
    ):
        self.router = router
        self.retriever = retriever
        self.code_executor = code_executor or CodeExecutor()
        self.reasoner = reasoner
        self.parser = parser or DocumentProcessor()
        self.shared_memory = shared_memory
        self.tolerance = tolerance

    def evaluate_question(
        self,
        question: OfficeQAQuestion,
        verbose: bool = False,
    ) -> EvalResult:
        """Evaluate a single question through the full pipeline."""
        start_time = time.time()

        try:
            # Step 1: Route
            if self.router:
                specialists = {
                    'ska_retriever': lambda q: self._retrieve(q),
                    'code_executor': lambda q: self._compute(q),
                    'reasoner': lambda q: self._reason(q),
                    'parser': lambda q: self._parse(q),
                }
                actions = self.router.route(
                    question.question,
                    specialists=specialists,
                    verbose=verbose,
                )
                predicted = actions[-1].output if actions else ""
                mode = actions[0].action.target if actions else ""
                num_actions = len(actions)
            else:
                # Direct baseline: just use reasoner
                predicted = self._reason(question.question) if self.reasoner else ""
                mode = "direct"
                num_actions = 1

            # Get operator condition number if available
            op_cond = 1.0
            if self.shared_memory and self.shared_memory.operator:
                op_cond = self.shared_memory.operator.condition_number

        except Exception as e:
            predicted = ""
            mode = "error"
            num_actions = 0
            op_cond = 1.0
            error_msg = str(e)

            return EvalResult(
                question_id=question.question_id,
                question=question.question,
                ground_truth=question.answer,
                predicted=predicted,
                score=0,
                latency_ms=(time.time() - start_time) * 1000,
                mode_selected=mode,
                actions_taken=num_actions,
                operator_condition=op_cond,
                error=error_msg,
            )

        elapsed_ms = (time.time() - start_time) * 1000
        s = score_answer(question.answer, predicted, self.tolerance)

        return EvalResult(
            question_id=question.question_id,
            question=question.question,
            ground_truth=question.answer,
            predicted=predicted,
            score=s,
            latency_ms=elapsed_ms,
            mode_selected=mode,
            actions_taken=num_actions,
            operator_condition=op_cond,
        )

    def _retrieve(self, query: str) -> str:
        if self.retriever:
            result = self.retriever.retrieve(query, verbose=False)
            return result.get_context()
        return ""

    def _compute(self, query: str) -> str:
        output, success = self.code_executor.execute(query)
        return output

    def _reason(self, query: str) -> str:
        if self.reasoner:
            if callable(self.reasoner):
                return self.reasoner(query)
            elif hasattr(self.reasoner, 'generate'):
                return self.reasoner.generate(query, "")
        return ""

    def _parse(self, query: str) -> str:
        return f"[Parsed: {query[:100]}]"

    def evaluate_full(
        self,
        questions: List[OfficeQAQuestion],
        verbose: bool = False,
        condition_rebuild_threshold: float = 1e4,
    ) -> AblationResult:
        """
        Run full evaluation on all questions.

        Monitors operator conditioning throughout and rebuilds
        any operator whose condition number exceeds threshold.
        """
        config_name = self._get_config_name()
        ablation = AblationResult(config_name=config_name)

        print(f"\n=== Evaluating: {config_name} ===")
        print(f" Questions: {len(questions)}")

        for i, q in enumerate(questions):
            result = self.evaluate_question(q, verbose=verbose)
            ablation.results.append(result)

            # Monitor conditioning (§6.5)
            if result.operator_condition > condition_rebuild_threshold:
                print(f" WARNING: Condition {result.operator_condition:.1f} > {condition_rebuild_threshold}")
                if self.shared_memory and hasattr(self.shared_memory, 'reset'):
                    self.shared_memory.reset()
                    print(f" Operator rebuilt from scratch")

            if (i + 1) % 10 == 0:
                running_acc = sum(r.score for r in ablation.results) / len(ablation.results)
                print(f" Progress: {i+1}/{len(questions)}, running accuracy: {running_acc:.4f}")

        print(f"\n{ablation.summary()}")
        return ablation

    def _get_config_name(self) -> str:
        """Generate configuration name for ablation tracking."""
        parts = []
        if self.router:
            parts.append("router")
        if self.retriever:
            parts.append("ska_retriever")
        if self.shared_memory:
            parts.append("shared_mem")
        if self.reasoner:
            parts.append("reasoner")
        return "+".join(parts) if parts else "baseline"


# Ablation Runner (Table 2)

class AblationRunner:
    """
    Run the full ablation matrix (Table 2 in the spec).

    Configurations:
      1. Baseline (single LLM)
      2. + Parsed corpus
      3. + SKA retriever
      4. + Attention retriever (control)
      5. + Adaptive router
      6. + Shared memory
      7. + Multi-head Koopman
    """

    def __init__(
        self,
        questions: List[OfficeQAQuestion],
        tolerance: float = 0.01,
    ):
        self.questions = questions
        self.tolerance = tolerance
        self.results: Dict[str, AblationResult] = {}

    def run_ablation(
        self,
        config_name: str,
        evaluator: OfficeQAEvaluator,
        verbose: bool = False,
    ) -> AblationResult:
        """Run a single ablation configuration."""
        result = evaluator.evaluate_full(
            self.questions,
            verbose=verbose,
        )
        result.config_name = config_name
        self.results[config_name] = result
        return result

    def compare_results(self) -> str:
        """Generate comparison table of all ablation results."""
        lines = [
            "\n=== Ablation Comparison (Table 2) ===",
            f"{'Configuration':<35} {'Accuracy':>10} {'Latency':>12} {'Cost':>10} {'Segments':>10} {'Cond(max)':>10}",
            "-" * 87,
        ]

        for name, result in self.results.items():
            lines.append(
                f"{name:<35} {result.accuracy:>10.4f} {result.avg_latency_ms:>10.1f}ms "
                f"${result.avg_cost:>8.4f} {result.avg_segments:>10.1f} {result.max_condition:>10.1f}"
            )

        # Marginal contribution analysis
        if len(self.results) > 1:
            lines.append("\n--- Marginal Contributions ---")
            names = list(self.results.keys())
            for i in range(1, len(names)):
                prev = self.results[names[i - 1]]
                curr = self.results[names[i]]
                delta = curr.accuracy - prev.accuracy
                lines.append(f" {names[i-1]} -> {names[i]}: Δaccuracy = {delta:+.4f}")

        return '\n'.join(lines)

    def save_results(self, path: str):
        """Save all results to JSON."""
        data = {}
        for name, result in self.results.items():
            data[name] = {
                'config_name': name,
                'accuracy': result.accuracy,
                'avg_latency_ms': result.avg_latency_ms,
                'avg_cost': result.avg_cost,
                'num_questions': len(result.results),
                'per_question': [
                    {
                        'question_id': r.question_id,
                        'score': r.score,
                        'latency_ms': r.latency_ms,
                        'predicted': r.predicted,
                        'ground_truth': r.ground_truth,
                        'operator_condition': r.operator_condition,
                    }
                    for r in result.results
                ],
            }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {path}")
