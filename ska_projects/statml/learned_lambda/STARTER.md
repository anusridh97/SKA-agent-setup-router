# Starter: Learned Lambda

## Step 1: Generate Training Data

```python
import numpy as np
from ska_agent.core.pricing import PricingEngine
from ska_agent.core.structures import Segment
from ska_agent.evaluation.officeqa import score_answer

def generate_training_data(segments, queries, answers, embed_fn):
    """
    For each query, sweep lambda and find the best one.
    Returns: [(query_embedding, best_lambda), ...]
    """
    lambdas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    data = []

    for query, answer in zip(queries, answers):
        q_emb = embed_fn(query)
        best_lambda = lambdas[0]
        best_score = -1

        for lam in lambdas:
            engine = PricingEngine(
                segments=segments, embed_fn=embed_fn,
                lambda_sparsity=lam, max_segments=5,
            )
            result = engine.retrieve(query, verbose=False)
            context = result.get_context()
            # In practice, you'd generate an answer from context
            # For now, use context length as proxy
            score = len(result.segments)  # placeholder
            if score > best_score:
                best_score = score
                best_lambda = lam

        data.append((q_emb, best_lambda))

    return data
```

## Step 2: Train the MLP

```python
import torch
import torch.nn as nn

class LambdaPredictor(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus(),  # ensures λ > 0
        )

    def forward(self, e_q):
        return self.net(e_q)

# Training loop
predictor = LambdaPredictor()
optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)

for epoch in range(100):
    for e_q, target_lambda in training_data:
        e_q_t = torch.tensor(e_q, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor([[np.log(target_lambda)]], dtype=torch.float32)
        pred = torch.log(predictor(e_q_t))
        loss = nn.functional.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
