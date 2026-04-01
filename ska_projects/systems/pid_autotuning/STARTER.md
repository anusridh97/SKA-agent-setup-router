# Starter: PID Auto-Tuning

## Step 0: Understand the Current PID

```python
import numpy as np
from ska_agent.router.pid_controller import PIDController
from ska_agent.core.structures import CostVector, PIDConfig

# Default config
config = PIDConfig()
pid = PIDController(config)

print(f"Kp={config.Kp}, Ki={config.Ki}, Kd={config.Kd}")
print(f"budget_rate={config.budget_rate}")
print(f"lambda_max={config.lambda_max}")

# Simulate a sequence of costs and watch lambda evolve
np.random.seed(42)
lambdas = []
for t in range(100):
    cost = CostVector(
        input_tokens=np.random.uniform(100, 2000),
        output_tokens=np.random.uniform(50, 1000),
        latency_ms=np.random.uniform(10, 500),
        dollar_cost=np.random.uniform(0.001, 0.05),
        meta_overhead=np.random.uniform(0, 0.5),
    )
    lam = pid.update(cost)
    lambdas.append(lam.copy())

lambdas = np.array(lambdas)
print(f"\nFinal lambda: {lambdas[-1]}")
print(f"Lambda range: [{lambdas.min(axis=0)}, {lambdas.max(axis=0)}]")
```

## Step 1: Grid Search Tuner

```python
class PIDTuner:
    def __init__(self, queries, ground_truth, evaluate_fn):
        """
        queries: list of query strings
        ground_truth: list of expected answers
        evaluate_fn: (PIDConfig, queries, gt) -> (accuracy, total_cost_vector)
        """
        self.queries = queries
        self.ground_truth = ground_truth
        self.evaluate_fn = evaluate_fn

    def grid_search(self, budget_constraint=None):
        """Sweep Kp, Ki, Kd and find best config."""
        best_config = None
        best_score = -np.inf
        results = []

        for Kp in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
            for Ki in [0.001, 0.005, 0.01, 0.05]:
                for Kd in [0.01, 0.05, 0.1, 0.2]:
                    config = PIDConfig(Kp=Kp, Ki=Ki, Kd=Kd)
                    acc, cost = self.evaluate_fn(
                        config, self.queries, self.ground_truth)

                    if budget_constraint and cost > budget_constraint:
                        continue

                    results.append((Kp, Ki, Kd, acc, cost))
                    if acc > best_score:
                        best_score = acc
                        best_config = config

        return best_config, results
```

## Step 2: Bayesian Optimization

```bash
pip install scikit-optimize
```

See BACKGROUND.md Section 4 for the scikit-optimize integration pattern.

## Step 3: Budget Scheduling

```python
class BudgetSchedule:
    """Time-varying budget_rate."""

    @staticmethod
    def constant(b, T):
        return [b] * T

    @staticmethod
    def front_loaded(b_high, b_low, switch_frac, T):
        switch = int(T * switch_frac)
        return [b_high] * switch + [b_low] * (T - switch)

    @staticmethod
    def cosine(b_max, b_min, T):
        return [b_min + (b_max - b_min) * (1 + np.cos(np.pi * t / T)) / 2
                for t in range(T)]
```
