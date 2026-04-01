"""
Command-line interface for SKA-Agent.

Provides the ska-agent command with subcommands:

  ska-agent build --phase N     Build a specific phase (1-6)
  ska-agent build --all         Build all phases sequentially
  ska-agent build --all --multi-koopman   Use K=4 parallel operators
  ska-agent eval --questions Q  Run OfficeQA evaluation
  ska-agent query "text"        Run a single query through the pipeline
  ska-agent status              Show system status and GPU info
  ska-agent ablation --questions Q   Run full ablation matrix

Entry point registered in setup.py as ska-agent console script.

Dependencies:
  - pipeline.py for SKAAgentPipeline
  - core/structures.py for SystemConfig
  - evaluation/officeqa.py for OfficeQAEvaluator, AblationRunner
"""

from __future__ import annotations

import argparse
import sys

from .core.structures import SystemConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ska-agent",
        description="SKA-Agent: Adaptive Multi-Model Orchestration",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Build command
    build_p = subparsers.add_parser("build", help="Build system phases")
    build_p.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6])
    build_p.add_argument("--all", action="store_true")
    build_p.add_argument("--skip-training", action="store_true")
    build_p.add_argument("--multi-koopman", action="store_true")

    # Eval command
    eval_p = subparsers.add_parser("eval", help="Run OfficeQA evaluation")
    eval_p.add_argument("--config", default="baseline")
    eval_p.add_argument("--questions", type=str, help="Path to questions JSON")
    eval_p.add_argument("--tolerance", type=float, default=0.01)
    eval_p.add_argument("--verbose", action="store_true")

    # Query command
    query_p = subparsers.add_parser("query", help="Run a single query")
    query_p.add_argument("text", type=str)
    query_p.add_argument("--verbose", action="store_true")

    # Status command
    subparsers.add_parser("status", help="Show system status")

    # Ablation command
    ablation_p = subparsers.add_parser("ablation", help="Run full ablation matrix")
    ablation_p.add_argument("--questions", type=str, required=True)
    ablation_p.add_argument("--output", type=str, default="ablation_results.json")
    ablation_p.add_argument("--verbose", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = SystemConfig()

    if args.command == "build":
        from .pipeline import SKAAgentPipeline

        agent = SKAAgentPipeline(config)

        if args.all:
            agent.build_all(skip_training=args.skip_training)
        elif args.phase:
            build_fn = getattr(agent, f"build_phase{args.phase}", None)
            if build_fn is None:
                print(f"Unknown phase: {args.phase}")
                return 1
            if args.phase == 3 and args.multi_koopman:
                build_fn(use_multi_koopman=True)
            else:
                build_fn()
        else:
            parser.error("Specify --phase N or --all")
        return 0

    if args.command == "eval":
        import json
        from .evaluation.officeqa import (
            OfficeQAQuestion, OfficeQAEvaluator,
        )

        if args.questions:
            with open(args.questions) as f:
                raw = json.load(f)
            questions = [
                OfficeQAQuestion(
                    question_id=q.get("id", str(i)),
                    question=q["question"],
                    answer=q["answer"],
                )
                for i, q in enumerate(raw)
            ]
        else:
            print("No questions file provided. Use --questions path/to/questions.json")
            return 1

        evaluator = OfficeQAEvaluator(tolerance=args.tolerance)
        result = evaluator.evaluate_full(questions, verbose=args.verbose)
        print(result.summary())
        return 0

    if args.command == "query":
        from .pipeline import SKAAgentPipeline

        agent = SKAAgentPipeline(config)
        agent.build_phase1()
        answer = agent.run(args.text, verbose=args.verbose)
        print(f"\nAnswer: {answer}")
        return 0

    if args.command == "status":
        print("SKA-Agent System Status")
        print(f" Jamba model: {config.jamba_model_name}")
        print(f" DeepSeek model: {config.deepseek_model_name}")
        print(f" SKA config: rank={config.ska.rank}, K={config.ska.power_K}")
        print(f" Multi-Koopman: K={config.multi_koopman.num_operators}, r={config.multi_koopman.rank_per_operator}")
        print(f" Router encoder: {config.encoder_model_name}")
        print(f" Surgery layers: {config.jamba_ska_layer_indices}")

        import torch
        if torch.cuda.is_available():
            print(f" GPU: {torch.cuda.get_device_name()}")
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f" GPU Memory: {mem:.1f} GB")
        else:
            print(" GPU: None (CPU mode)")
        return 0

    if args.command == "ablation":
        import json
        from .evaluation.officeqa import (
            OfficeQAQuestion, OfficeQAEvaluator, AblationRunner,
        )

        with open(args.questions) as f:
            raw = json.load(f)
        questions = [
            OfficeQAQuestion(
                question_id=q.get("id", str(i)),
                question=q["question"],
                answer=q["answer"],
            )
            for i, q in enumerate(raw)
        ]

        runner = AblationRunner(questions)

        # Run baseline
        evaluator = OfficeQAEvaluator(tolerance=0.01)
        runner.run_ablation("baseline", evaluator, verbose=args.verbose)

        print(runner.compare_results())
        runner.save_results(args.output)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
