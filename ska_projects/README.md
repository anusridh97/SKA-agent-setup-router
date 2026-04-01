# SKA-Agent: Student Project Collection

## Overview

Self-contained project specifications for extending SKA-Agent, an adaptive
multi-model orchestration framework with Structured Kernel Attention and
spectral shared memory. All projects are **8-week quarter projects** with
**low GPU requirements**.

## Critical Path

One project produces artifacts that 4 others depend on:

```
  ┌──────────────────────────────┐
  │ S5 Router Training (week 3)  │ ← ships trained checkpoints
  │    Person A: ModeSelector    │
  │    Person B: RewardPredictor │
  └──────────────┬───────────────┘
                 │  trained checkpoints
      ┌──────────┼──────────┬────────────┐
      ▼          ▼          ▼            ▼
   A1 Router  A4 Retrieval S4 PID    M4 Mode
   Dashboard  Feedback     Autotuning Calibration
```

S5 delivers v1 checkpoints at **week 3**. The 4 dependent projects can
start their Phase 1 work in weeks 1–3 (UI, infrastructure, data generation)
and plug in the trained router starting week 4.

All other projects (A2, A3, S1, S2, S3, M1, M2, M3) have **no dependencies**
and can start immediately.

## Project List

### Applied

| # | Project | Tier | GPU | What you build |
|---|---------|------|-----|----------------|
| A1 | [Router Dashboard](applied/router_dashboard/) | Starter | None | Routing UI + comparative analysis (needs S5) |
| A2 | [Memory Inspector](applied/memory_inspector/) | Starter | None | Memory debugger + anomaly detection |
| A3 | [OfficeQA Demo](applied/officeqa_demo/) | Starter | Minimal | PDF→answer app + retrieval comparison study |
| A4 | [Retrieval Feedback](applied/retrieval_feedback/) | Inter. | Minimal | Feedback loop → reward training (needs S5) |

### Systems

| # | Project | Tier | GPU | What you build |
|---|---------|------|-----|----------------|
| S1 | [Incremental Cholesky](systems/incremental_cholesky/) | Advanced | None | O(r²) operator updates via rank-1 Cholesky |
| S2 | [Streaming Segmentation](systems/streaming_segmentation/) | Inter. | Minimal | Bounded-memory DP segmentation |
| S3 | [TS Orchestrator](systems/ts_orchestrator/) | Advanced | None | TypeScript DAG scheduler + parallel dispatch |
| S4 | [PID Autotuning](systems/pid_autotuning/) | Inter. | Minimal | PID gain optimization (needs S5) |
| S5 | [Router Training](systems/router_training/) | Inter. | None | ModeSelector + RewardPredictor training pipeline |

### Statistical ML

| # | Project | Tier | GPU | What you build |
|---|---------|------|-----|----------------|
| M1 | [SVD Init Ablation](statml/svd_init_ablation/) | Inter. | ~1 hr | Warm-start strategy comparison |
| M2 | [Learned Lambda](statml/learned_lambda/) | Inter. | Minimal | Neural sparsity parameter prediction |
| M3 | [Reg. Analysis](statml/regularization_analysis/) | Advanced | ~2 hrs | Spectral + orthogonal reg interaction |
| M4 | [Mode Calibration](statml/mode_calibration/) | Inter. | Minimal | Calibration + calibrated routing (needs S5) |

## Difficulty Tiers

- **Starter** (3 projects), Good first contact. Working prototype in week 1.
  Weeks 1–4 build the core; weeks 5–8 extend into substantive analysis.
- **Intermediate** (7 projects), Requires understanding one subsystem.
- **Advanced** (3 projects), Deep spectral theory or distributed systems.

## Directory Structure

```
ska_projects/
├── README.md
├── applied/
│   ├── router_dashboard/       [STARTER, needs S5]
│   ├── memory_inspector/       [STARTER]
│   ├── officeqa_demo/          [STARTER]
│   └── retrieval_feedback/     [INTERMEDIATE, needs S5]
├── systems/
│   ├── incremental_cholesky/   [ADVANCED]
│   ├── streaming_segmentation/ [INTERMEDIATE]
│   ├── ts_orchestrator/        [ADVANCED]
│   ├── pid_autotuning/         [INTERMEDIATE, needs S5]
│   └── router_training/        [INTERMEDIATE, 2-person, critical path]
└── statml/
    ├── svd_init_ablation/      [INTERMEDIATE]
    ├── learned_lambda/         [INTERMEDIATE]
    ├── regularization_analysis/[ADVANCED]
    └── mode_calibration/       [INTERMEDIATE, needs S5]
```

Each project directory contains:
- `SPEC.md`, full specification with 8-week milestones
- `BACKGROUND.md`, self-contained learning material
- `STARTER.md`, concrete starting code and first experiments
- `EVALUATION.md`, grading rubric

## Suggested Paths

**"I want to understand the system quickly":**
A1 Router Dashboard → S4 PID Autotuning → S1 Incremental Cholesky

**"I'm a frontend/fullstack person":**
A1 Router Dashboard → A3 OfficeQA Demo → A2 Memory Inspector

**"I want to do ML research":**
M4 Mode Calibration → M2 Learned Lambda → M3 Reg. Analysis

**"I want to build infrastructure":**
A3 OfficeQA Demo → S2 Streaming Segmentation → S3 TS Orchestrator

## Prerequisites by Tier

**Starter:** Python, basic web dev (Streamlit), can read the codebase

**Intermediate**, additionally: linear algebra, basic ML, numpy/PyTorch

**Advanced**, additionally: spectral theory or distributed systems
