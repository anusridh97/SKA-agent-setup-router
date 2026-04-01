# Starter: Router Training Pipeline

## Step 0: Verify the Untrained Router

Run this first to see exactly why training is needed:

```python
import numpy as np
from ska_agent.router.adaptive_router import AdaptiveRouter
from ska_agent.core.structures import SystemConfig

router = AdaptiveRouter(config=SystemConfig())

# Mock specialists that always "succeed"
mock_specialists = {
    'ska_retriever': lambda q: f"[Retrieved: {q[:30]}]",
    'code_executor': lambda q: "42",
    'reasoner': lambda q: f"[Reasoned: {q[:30]}]",
    'parser': lambda q: f"[Parsed: {q[:30]}]",
}

# Try 5 queries
queries = [
    "What was the total federal debt in 2023?",
    "Compare healthcare spending between 2022 and 2023.",
    "What percentage of GDP was military spending?",
    "Which department had the highest growth after inflation?",
    "How much did the DOD spend on personnel?",
]

for q in queries:
    results = router.route(q, specialists=mock_specialists, verbose=True)
    print(f"  → {len(results)} actions taken\n")
```

You'll see: mode probabilities are ~uniform (25% each), reward predictions
are near-zero, the router stops after 0–1 actions every time. This is
what you're fixing.

## Person A: ModeSelector Data Generation

### Step 1: Template-Based Generation

```python
import numpy as np
import json

TEMPLATES = {
    "LOOKUP": [
        "What was the total {} in {}?",
        "How much did {} spend on {} in {}?",
        "What is the value of {} for {}?",
        "What was {}'s {} in fiscal year {}?",
        "How many {} were there in {}?",
        "What is the {} of {}?",
        "According to the {}, what was {}?",
        "What did the {} report as the total {}?",
    ],
    "MULTI_DOC": [
        "Compare {} between {} and {}.",
        "How did {} change from {} to {}?",
        "What is the difference in {} between {} and {}?",
        "Which had higher {}, {} or {}?",
        "Show me {} for both {} and {}.",
        "How does {}'s {} compare to {}'s?",
        "What were the {} figures for {} versus {}?",
        "Contrast the {} in {} with {}.",
    ],
    "COMPUTE": [
        "Calculate the {} growth rate from {} to {}.",
        "What percentage of {} was {}?",
        "Compute the ratio of {} to {}.",
        "What is the average {} across {}?",
        "By how much did {} exceed {} in percentage terms?",
        "What fraction of total {} came from {}?",
        "Estimate the per-capita {} given {} and population of {}.",
        "What would {} be if adjusted for {} inflation?",
    ],
    "MULTI_STEP": [
        "First find the total {}, then calculate what percentage went to {}, and compare with {}.",
        "Which {} had the highest {} after adjusting for {}?",
        "Rank the top {} by {} and show how each changed since {}.",
        "Determine whether {}'s {} exceeded {}'s {}, accounting for {}.",
        "If {} grew at the same rate as {}, what would {} be in {}?",
        "Analyze the trend in {} across {}, identify outliers, and explain them.",
        "Find all {} where {} exceeded {}, compute their aggregate {}, and compare to {}.",
        "What is the compound annual growth rate of {} from {} to {}, excluding {}?",
    ],
}

ENTITY_POOLS = {
    "financial": ["revenue", "expenses", "debt", "deficit", "surplus",
                  "spending", "outlays", "receipts", "appropriations"],
    "departments": ["Department of Defense", "HHS", "Treasury",
                    "Department of Education", "VA", "DHS", "DOJ"],
    "years": ["2019", "2020", "2021", "2022", "2023", "2024",
              "FY2022", "FY2023", "Q1 2023", "Q3 2024"],
    "metrics": ["GDP", "CPI", "unemployment rate", "interest rate",
                "inflation", "population", "workforce"],
}

def generate_queries(n_per_mode=250, seed=42):
    """Generate labeled (query, mode) pairs from templates."""
    np.random.seed(seed)
    dataset = []

    all_fillers = sum(ENTITY_POOLS.values(), [])

    for mode_name, templates in TEMPLATES.items():
        for _ in range(n_per_mode):
            tmpl = np.random.choice(templates)
            n_blanks = tmpl.count("{}")
            fills = np.random.choice(all_fillers, size=n_blanks, replace=False)
            query = tmpl.format(*fills)

            mode_idx = ["LOOKUP", "MULTI_DOC", "COMPUTE", "MULTI_STEP"].index(mode_name)
            dataset.append({"query": query, "mode": mode_name, "mode_idx": mode_idx})

    np.random.shuffle(dataset)
    return dataset

dataset = generate_queries(n_per_mode=250)
print(f"Generated {len(dataset)} queries")
for mode in ["LOOKUP", "MULTI_DOC", "COMPUTE", "MULTI_STEP"]:
    count = sum(1 for d in dataset if d["mode"] == mode)
    print(f"  {mode}: {count}")
print(f"\nExamples:")
for d in dataset[:5]:
    print(f"  [{d['mode']}] {d['query']}")

# Save
with open("mode_selector_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)
```

### Step 2: Embed and Train

```python
import torch
import numpy as np
from ska_agent.router.adaptive_router import QueryEncoder, ModeSelector
from ska_agent.core.structures import RouterTrainingConfig

# Encode all queries
encoder = QueryEncoder("all-MiniLM-L6-v2")
queries = [d["query"] for d in dataset]
embeddings = encoder.encode_batch(queries)

# Prepare training data
training_data = [
    {"query_embedding": embeddings[i], "mode_idx": dataset[i]["mode_idx"]}
    for i in range(len(dataset))
]

# Split
split = int(0.8 * len(training_data))
train_data = training_data[:split]
val_data = training_data[split:]
print(f"Train: {len(train_data)}, Val: {len(val_data)}")

# Train
mode_selector = ModeSelector(input_dim=384, hidden_dim=256, num_modes=4)
from ska_agent.training.trainers import RouterTrainer
trainer = RouterTrainer(
    reward_predictor=None,  # not training this one
    mode_selector=mode_selector,
    config=RouterTrainingConfig(mode_epochs=200, mode_lr=1e-3),
    device="cpu",
)
trainer.train_mode_selector(train_data)

# Evaluate
correct = 0
for sample in val_data:
    e_q = sample["query_embedding"]
    mode, probs = mode_selector.predict(e_q)
    pred_idx = np.argmax(probs)
    if pred_idx == sample["mode_idx"]:
        correct += 1
print(f"\nValidation accuracy: {correct}/{len(val_data)} = {correct/len(val_data):.1%}")

# Save
import os
os.makedirs("checkpoints", exist_ok=True)
torch.save(mode_selector.state_dict(), "checkpoints/mode_selector_v1.pt")
print("Saved checkpoint")
```

## Person B: RewardPredictor Data Generation

### Step 1: Quality Measurement via Mock Pipeline

```python
import numpy as np
from ska_agent.core.pricing import PricingEngine
from ska_agent.core.structures import Segment
from ska_agent.evaluation.officeqa import score_answer

def measure_specialist_quality(query, answer_gt, segments, embedder):
    """
    Run different specialist configurations and measure quality.
    Returns dict of {specialist: quality_score}.
    """
    results = {}

    # Retriever only
    engine = PricingEngine(
        segments=segments,
        embed_fn=lambda q: embedder.embed_single(q),
        lambda_sparsity=0.05,
        max_segments=5,
    )
    retrieval = engine.retrieve(query, verbose=False)
    context = retrieval.get_context()

    # Quality proxy: how much query embedding is explained by retrieved segments
    q_emb = embedder.embed_single(query)
    if len(retrieval.segments) > 0:
        seg_vecs = np.array([s.vector for s in retrieval.segments])
        # Project query onto span of retrieved segments
        Q, _ = np.linalg.qr(seg_vecs.T)
        proj = Q @ (Q.T @ q_emb)
        explained = np.dot(proj, proj) / (np.dot(q_emb, q_emb) + 1e-10)
        results["ska_retriever"] = float(explained)
    else:
        results["ska_retriever"] = 0.0

    # Reasoner (proxy: query complexity → higher reward for complex queries)
    # Simple heuristic: longer queries with more clauses benefit from reasoning
    word_count = len(query.split())
    has_comparison = any(w in query.lower() for w in ["compare", "difference", "versus", "change"])
    has_computation = any(w in query.lower() for w in ["calculate", "percentage", "ratio", "average"])
    results["reasoner"] = 0.3 + 0.2 * has_comparison + 0.1 * (word_count > 15)

    # Code executor (proxy: benefits computation queries)
    results["code_executor"] = 0.5 * has_computation + 0.1

    # Parser (proxy: modest benefit for all queries)
    results["parser"] = 0.15

    return results

# Generate training data
def generate_reward_training_data(queries, segments, embedder, n_queries=200):
    """Generate (query_embedding, model_idx, base_model_idx, delta_r) tuples."""
    encoder_data = []

    model_names = ["ska_retriever", "code_executor", "reasoner", "parser"]

    for query in queries[:n_queries]:
        q_emb = embedder.embed_single(query)
        qualities = measure_specialist_quality(query, "", segments, embedder)

        # baseline = parser (weakest)
        base_quality = qualities["parser"]
        base_idx = model_names.index("parser")

        for model_name, quality in qualities.items():
            model_idx = model_names.index(model_name)
            delta_r = quality - base_quality

            encoder_data.append({
                "query_embedding": q_emb,
                "model_idx": model_idx,
                "base_model_idx": base_idx,
                "delta_r": delta_r,
            })

    return encoder_data
```

### Step 2: Train

```python
from ska_agent.router.adaptive_router import RewardPredictor
from ska_agent.training.trainers import RouterTrainer
from ska_agent.core.structures import RouterTrainingConfig

reward_predictor = RewardPredictor(
    query_dim=384, model_embed_dim=64, hidden_dim=512, num_models=4
)

trainer = RouterTrainer(
    reward_predictor=reward_predictor,
    mode_selector=None,
    config=RouterTrainingConfig(reward_epochs=500, reward_lr=1e-3),
    device="cpu",
)
trainer.train_reward_predictor(training_data)

# Verify: does it predict positive rewards for retrieval?
import torch
test_emb = torch.randn(1, 384)
with torch.no_grad():
    # Retriever vs parser
    r_idx = torch.tensor([0])  # ska_retriever
    b_idx = torch.tensor([3])  # parser
    pred = reward_predictor(test_emb, r_idx, b_idx)
    print(f"Retriever vs parser: Δr̂ = {pred.item():.4f}")

    # Code executor vs parser
    c_idx = torch.tensor([1])
    pred = reward_predictor(test_emb, c_idx, b_idx)
    print(f"Code executor vs parser: Δr̂ = {pred.item():.4f}")

# Save
torch.save(reward_predictor.state_dict(), "checkpoints/reward_predictor_v1.pt")
```

## Both: Integration Test

```python
def test_trained_router():
    """Verify the router makes non-trivial decisions with trained checkpoints."""
    import torch
    from ska_agent.router.adaptive_router import AdaptiveRouter, ModeSelector, RewardPredictor
    from ska_agent.core.structures import SystemConfig

    config = SystemConfig()
    router = AdaptiveRouter(config=config)

    # Load checkpoints
    router.mode_selector.load_state_dict(
        torch.load("checkpoints/mode_selector_v1.pt", weights_only=True))
    router.reward_predictor.load_state_dict(
        torch.load("checkpoints/reward_predictor_v1.pt", weights_only=True))

    mock_specialists = {
        'ska_retriever': lambda q: f"[Retrieved: {q[:30]}]",
        'code_executor': lambda q: "42",
        'reasoner': lambda q: f"[Reasoned: {q[:30]}]",
        'parser': lambda q: f"[Parsed: {q[:30]}]",
    }

    queries = [
        "What was the total federal debt in 2023?",
        "Compare healthcare spending between 2022 and 2023.",
        "What percentage of GDP was military spending?",
        "Which department had the highest growth after inflation?",
    ]

    total_actions = 0
    for q in queries:
        results = router.route(q, specialists=mock_specialists, verbose=True)
        total_actions += len(results)
        print(f"  → {len(results)} actions\n")

    # With trained weights, the router should take at least some actions
    assert total_actions > 0, "Router still takes no actions, training didn't work"
    print(f"PASSED: {total_actions} total actions across {len(queries)} queries")

test_trained_router()
```
