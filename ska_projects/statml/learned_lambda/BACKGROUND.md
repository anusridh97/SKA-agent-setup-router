# Background: Amortized Optimization and Lagrange Multipliers

## 1. The Lagrangian View

The retrieval problem is: min_V φ_recon(V) subject to |V| ≤ k.
The Lagrangian relaxation introduces multiplier λ:

    L(V, λ) = φ_recon(V) + λ(|V| - k)

The optimal λ* is the shadow price: the marginal value of relaxing
the constraint by one unit. It depends on the query, a hard query
benefits more from additional segments, so λ* is lower.

## 2. Amortized Optimization

Instead of solving for λ* per query, learn a function f(e_Q) → λ*
that predicts the optimal multiplier from query features. This is
"amortized" because the cost of learning f is spread over many queries.

## 3. Training Signal

For each query q_i in the training set:
1. Run retrieval at λ ∈ {0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}
2. For each λ, compute score_answer(ground_truth, predicted)
3. Label: λ* = argmax_λ accuracy(λ) (or argmax accuracy/|V| for efficiency)

This is a regression problem: e_Q → λ*.

## 4. Architecture

A 2-layer MLP is sufficient:

    e_Q ∈ R^384 → Linear(384, 128) → ReLU → Linear(128, 1) → Softplus → λ

Softplus ensures λ > 0. Train with MSE loss on log(λ) for stability.

## 5. References

1. Shu et al. "Amortized Bayesian Optimization over Discrete Spaces." NeurIPS 2019.
2. Boyd & Vandenberghe. "Convex Optimization." §5.1 (Lagrange duality).
