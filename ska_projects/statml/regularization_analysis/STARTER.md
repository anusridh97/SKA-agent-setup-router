# Starter: Regularization Analysis

The SPEC.md contains the main starter experiment. Here's the grid sweep:

```python
import itertools
import json
import torch
from ska_agent.core.structures import SKAConfig
from ska_agent.core.ska_module import SKAModule
from ska_agent.training.trainers import SpectralRegularization, OrthogonalRegularization

def run_grid_sweep(save_path="reg_results.json"):
    config = SKAConfig(d_model=256, n_heads=4, head_dim=64, rank=16)
    lambda_specs = [0, 0.001, 0.01, 0.1, 1.0]
    lambda_orthos = [0, 0.001, 0.01, 0.1, 1.0]
    results = {}

    for ls, lo in itertools.product(lambda_specs, lambda_orthos):
        key = f"spec={ls}_ortho={lo}"
        print(f"\nRunning {key}")

        ska = SKAModule(config)
        spec_reg = SpectralRegularization(lambda_spec=ls)
        ortho_reg = OrthogonalRegularization(lambda_ortho=lo)
        optimizer = torch.optim.Adam(ska.parameters(), lr=1e-3)
        modules = torch.nn.ModuleDict({"ska": ska})

        metrics = []
        for step in range(2000):
            x = torch.randn(4, 128, 256)
            prefix_len = 64

            out = ska(x, prefix_len=prefix_len)
            lm_loss = out.pow(2).mean()  # proxy LM loss

            l_spec = spec_reg(modules, x, prefix_len) if ls > 0 else torch.tensor(0.0)
            l_ortho = ortho_reg(modules) if lo > 0 else torch.tensor(0.0)
            loss = lm_loss + l_spec + l_ortho

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                stats = ska.get_operator_stats(x, prefix_len)
                stats['step'] = step
                stats['loss'] = loss.item()
                stats['lm_loss'] = lm_loss.item()
                metrics.append(stats)

        results[key] = metrics

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {save_path}")

run_grid_sweep()
```
