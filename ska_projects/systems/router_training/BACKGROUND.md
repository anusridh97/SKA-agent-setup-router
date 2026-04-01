# Background: Training the Router Components

## 1. The Router's Decision Pipeline

The router makes two decisions per query:

**Decision 1, Mode Selection (ModeSelector):**
"What type of workflow does this query need?"
Input: query embedding e_Q ∈ R^384 (from frozen MiniLM encoder)
Output: probability distribution over 4 modes

**Decision 2, Action Scoring (RewardPredictor):**
"Is it worth calling specialist X on this query?"
Input: query embedding + specialist identity
Output: predicted quality improvement Δr̂ ∈ R

The PID controller adds a cost penalty. The final score is:
    S(a) = Δr̂(a) - λᵀ · Δĉ(a)

If S > 0 for some action, the router executes it. If all S ≤ 0, it stops.

## 2. The ModeSelector (Person A)

### Architecture

```
e_Q ∈ R^384 → Linear(384, 256) → ReLU → Dropout(0.1) → Linear(256, 4) → Softmax
```

97,540 trainable parameters. Trains in seconds on CPU.

### What Makes a Good Training Dataset

The 4 modes correspond to different query structures:

**LOOKUP**, needs exactly one piece of information from one place.
Linguistic markers: "what is", "how much was", "what was the value of",
simple noun phrases, single entity references.

**MULTI_DOC**, needs information from multiple sources to compare.
Linguistic markers: "compare", "difference between", "change from X to Y",
"across", two time periods or entities mentioned.

**COMPUTE**, needs numerical calculation beyond simple extraction.
Linguistic markers: "calculate", "percentage", "ratio", "growth rate",
"average", "sum of", mathematical operations implied.

**MULTI_STEP**, needs decomposition into sub-tasks.
Linguistic markers: "which X had the highest Y after adjusting for Z",
conditional reasoning, multiple operations chained, "first find ... then".

### Data Generation Strategy

1. **Templates**: Write 20–30 templates per mode with fill-in-the-blank slots.
   Fill with realistic entities (departments, years, financial terms).

2. **Paraphrasing**: For each template query, write 3–5 paraphrases that
   change the wording but keep the same mode. "What was the total?" →
   "Can you tell me the aggregate?" → "I need the sum."

3. **Adversarial examples**: Queries that look like one mode but are another.
   "What percentage..." looks COMPUTE but might be LOOKUP if the percentage
   is stated directly in the document. "Compare..." looks MULTI_DOC but
   might be LOOKUP if the comparison is in a single table.

4. **Human annotation**: Recruit 3–5 classmates. Give them 100 queries and
   the 4 mode definitions. Have them label independently. Use majority vote.
   Measure inter-annotator agreement (Cohen's κ).

### Training

Use the existing `RouterTrainer.train_mode_selector()` or write your own
training loop. Cross-entropy loss on mode labels.

```python
from ska_agent.training.trainers import RouterTrainer

trainer = RouterTrainer(
    reward_predictor=reward_predictor,  # Person B's model
    mode_selector=mode_selector,
    config=RouterTrainingConfig(mode_epochs=200, mode_lr=1e-3),
    device="cpu",
)

# training_data format: [{'query_embedding': np.ndarray, 'mode_idx': int}, ...]
trainer.train_mode_selector(training_data)
```

## 3. The RewardPredictor (Person B)

### Architecture

```
[e_Q; e_specialist - e_baseline] ∈ R^448
    → Linear(448, 512) → ReLU → Dropout(0.1)
    → Linear(512, 256) → ReLU → Dropout(0.1)
    → Linear(256, 1)
    → Δr̂ ∈ R
```

e_specialist and e_baseline are learned 64-dim embeddings for each of
the 4 specialists (ska_retriever, code_executor, reasoner, parser).
The input is e_Q concatenated with the embedding *difference* (specialist
minus baseline), so the model learns relative quality improvement.

362,497 trainable parameters. Still trains in seconds on CPU.

### What Makes Good Training Data

The predictor needs to learn: "for this query type, calling the retriever
is worth +0.3 quality points, but calling the code executor is only +0.05."

Data format per sample:
```python
{
    'query_embedding': np.ndarray,   # 384-dim from QueryEncoder
    'model_idx': int,                # specialist index (0-3)
    'base_model_idx': int,           # baseline specialist index
    'delta_r': float,                # actual quality improvement
}
```

### Data Generation Strategy

1. **Run the pipeline at different configurations.** For a set of queries
   with known answers:
   - Run with only retriever → measure answer quality q_retriever
   - Run with only reasoner → measure answer quality q_reasoner
   - Run with retriever + code → measure q_retriever_code
   - delta_r(code | retriever) = q_retriever_code - q_retriever

2. **Use mock specialists with controlled quality.** If you don't have
   ground truth answers, create synthetic scenarios:
   - "Retriever finds relevant context" → delta_r = +0.5
   - "Retriever finds irrelevant context" → delta_r = -0.2
   - "Code executor computes correctly" → delta_r = +0.8
   - "Code executor errors" → delta_r = -0.5

   This gives you a biased but usable training signal. The important
   thing is that the *ranking* is right (retriever > nothing for retrieval
   queries), not the exact magnitudes.

3. **Use the OfficeQA evaluation harness.** The `score_answer()` function
   gives binary 0/1 accuracy. For a set of questions, run with each
   specialist and record the score. delta_r = score_with - score_without.

### Training

```python
trainer.train_reward_predictor(training_data)
```

The loss is MSE: L = (Δr̂ - Δr_actual)².

### What "Working" Looks Like

After training, the predictor should:
- Predict positive Δr̂ (> 0) for retriever on retrieval-heavy queries
- Predict positive Δr̂ for code_executor on computation queries
- Predict near-zero or negative Δr̂ for inappropriate specialists
  (e.g. code_executor on a simple LOOKUP query)

This means the router will actually execute actions (because S(a) > 0)
instead of stopping immediately.

## 4. Checkpoint Format and Loading

### Saving

```python
import torch

# Save ModeSelector
torch.save(mode_selector.state_dict(), "checkpoints/mode_selector_v1.pt")

# Save RewardPredictor
torch.save(reward_predictor.state_dict(), "checkpoints/reward_predictor_v1.pt")
```

### Loading (for downstream projects)

```python
# ska_agent/router/pretrained.py

import os
import torch
from .adaptive_router import AdaptiveRouter, ModeSelector, RewardPredictor
from ..core.structures import SystemConfig

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

def load_trained_router(config: SystemConfig = None) -> AdaptiveRouter:
    """
    Load the router with trained ModeSelector and RewardPredictor.

    Usage by downstream projects:
        from ska_agent.router.pretrained import load_trained_router
        router = load_trained_router()
        results = router.route("What was the total debt?")
    """
    config = config or SystemConfig()

    router = AdaptiveRouter(config=config)

    # Load trained weights
    ms_path = os.path.join(CHECKPOINT_DIR, "mode_selector_v1.pt")
    rp_path = os.path.join(CHECKPOINT_DIR, "reward_predictor_v1.pt")

    if os.path.exists(ms_path):
        router.mode_selector.load_state_dict(torch.load(ms_path, weights_only=True))
        print(f"Loaded ModeSelector from {ms_path}")
    else:
        print(f"WARNING: No trained ModeSelector found at {ms_path}")

    if os.path.exists(rp_path):
        router.reward_predictor.load_state_dict(torch.load(rp_path, weights_only=True))
        print(f"Loaded RewardPredictor from {rp_path}")
    else:
        print(f"WARNING: No trained RewardPredictor found at {rp_path}")

    return router
```

## 5. References

1. The existing `RouterTrainer` in `ska_agent/training/trainers.py`, read the `train_mode_selector()` and `train_reward_predictor()` methods.
2. Sentence-BERT documentation, for understanding the query encoder
   that produces the input embeddings.
