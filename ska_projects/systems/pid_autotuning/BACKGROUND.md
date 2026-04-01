# Background: PID Control and Auto-Tuning

## 1. PID Controllers

A PID controller computes a control signal u(t) from an error signal e(t):

    u(t) = Kp · e(t) + Ki · ∫e(τ)dτ + Kd · de/dt

- **Proportional (Kp):** Reacts to current error. Large Kp → aggressive response.
- **Integral (Ki):** Reacts to accumulated error. Eliminates steady-state offset.
- **Derivative (Kd):** Reacts to rate of change. Provides damping.

In our discrete setting (one update per router action):

    λ_{t+1} = clip(λ_t + Kp·e_t + Ki·Σe_τ + Kd·(e_t - e_{t-1}), 0, λ_max)

where e_t = avg_cost - budget_rate is the cost burn-rate error.

## 2. The 5-Dimensional Cost Vector

Each action produces a cost vector ĉ = (c_in, c_out, c_lat, c_$, c_meta):

| Dimension | Meaning | Units | Typical range |
|-----------|---------|-------|---------------|
| c_in | Input tokens consumed | tokens | 100–5000 |
| c_out | Output tokens generated | tokens | 50–2000 |
| c_lat | Wall-clock latency | ms | 10–2000 |
| c_$ | Dollar cost | USD | 0.001–0.10 |
| c_meta | Shared memory overhead | abstract | 0–1 |

The PID controller maintains a separate λ for each dimension. The action
scorer computes: S(a) = Δr̂(a) - λᵀ · Δĉ(a). High λ penalizes cost
more, making the router more conservative.

## 3. Ziegler-Nichols Method (Adapted)

The classic Ziegler-Nichols method:
1. Set Ki = 0, Kd = 0
2. Increase Kp until the system oscillates with constant amplitude
3. Record the critical gain Ku and oscillation period Tu
4. Set: Kp = 0.6·Ku, Ki = 2·Kp/Tu, Kd = Kp·Tu/8

**Adaptation for our setting:** Instead of continuous oscillation, we
look for cost oscillation in the λ trajectory over a sequence of queries.
Run the router on N queries with Ki=0, Kd=0. Gradually increase Kp.
When λ starts oscillating (alternating between high and low values
across queries), record the critical Kp and "period" (in query steps).

This gives a reasonable starting point. Fine-tune with grid search or BO.

## 4. Bayesian Optimization

For expensive-to-evaluate objectives (like running the full router on
100 queries), Bayesian optimization is more sample-efficient than grid
search.

```python
from skopt import gp_minimize

def objective(params):
    Kp, Ki, Kd = params
    config = PIDConfig(Kp=Kp, Ki=Ki, Kd=Kd)
    # Run router on held-out queries
    accuracy, total_cost = evaluate_router(config, queries)
    # Minimize negative accuracy (maximize accuracy)
    # subject to cost constraint
    if total_cost > budget:
        return 1.0  # penalty
    return -accuracy

result = gp_minimize(
    objective,
    dimensions=[(0.01, 1.0), (0.001, 0.1), (0.01, 0.5)],
    n_calls=50,
    random_state=42,
)
```

## 5. Budget Scheduling

Instead of a fixed budget_rate, use time-varying budgets:

- **Front-loaded:** High budget for first N queries (explore), then tighten
- **Adaptive:** Start tight, loosen if accuracy is below threshold
- **Cosine:** budget_rate = b_max · (1 + cos(π · t / T)) / 2

Each policy produces a different quality-cost Pareto frontier.

## 6. References

1. Ziegler & Nichols. "Optimum Settings for Automatic Controllers." 1942.
2. Åström & Hägglund. "PID Controllers: Theory, Design, and Tuning." 1995.
3. Snoek, Larochelle, Adams. "Practical Bayesian Optimization of Machine
   Learning Hyperparameters." NeurIPS, 2012.
