"""
PID Cost Controller for the adaptive router.

This is a pure feedback controller with no learned parameters. It
adjusts the 5-dimensional price vector lambda that weights the cost
dimensions (input tokens, output tokens, latency, dollar cost,
meta-overhead) in the router's action scoring formula:

    S(a) = delta_r_hat - lambda^T * delta_c_hat

The PID update rule:

    lambda_{t+1} = clip(
        lambda_t + Kp * e_t + Ki * sum(e_tau) + Kd * (e_t - e_{t-1}),
        0,
        lambda_max
    )

where e_t = average_recent_cost - budget_rate is the burn rate error.

Default gains: Kp=0.3, Ki=0.01, Kd=0.05
Default lambda_max: (5, 5, 2, 10, 3)

When spending exceeds budget, lambda increases, making actions more
expensive in the scoring formula, causing the router to prefer cheaper
specialists. When spending is below budget, lambda decreases, allowing
the router to invoke more expensive models.

Separated from adaptive_router.py to avoid torch dependency for basic
usage. This module requires only numpy.

Dependencies:
  - core/structures.py for CostVector, PIDConfig
"""

from __future__ import annotations

from collections import deque
import numpy as np

from ..core.structures import CostVector, PIDConfig


class PIDController:
    """
    PID cost controller (§5.1.4, Eq. 18, 25).

    Default gains (Eq. 25):
        K_p = 0.3, K_i = 0.01, K_d = 0.05
        λ_max = (5, 5, 2, 10, 3)
    """

    def __init__(self, config: PIDConfig = None):
        self.config = config or PIDConfig()
        self.lambda_vec = np.ones(5) * 0.1
        self.cost_history: deque = deque(maxlen=self.config.window_size)
        self.integral_error = np.zeros(5)
        self.prev_error = np.zeros(5)

    def update(self, cost: CostVector) -> np.ndarray:
        """Update price vector based on observed cost."""
        cost_arr = cost.to_array()
        self.cost_history.append(cost_arr)

        if len(self.cost_history) > 0:
            window = np.array(list(self.cost_history))
            avg_cost = window.mean(axis=0)
            error = avg_cost - self.config.budget_rate
        else:
            error = np.zeros(5)

        self.integral_error += error
        derivative = error - self.prev_error

        delta = (
            self.config.Kp * error
            + self.config.Ki * self.integral_error
            + self.config.Kd * derivative
        )

        self.lambda_vec = np.clip(
            self.lambda_vec + delta,
            0.0,
            self.config.lambda_max,
        )

        self.prev_error = error.copy()
        return self.lambda_vec

    def reset(self):
        """Reset controller state."""
        self.lambda_vec = np.ones(5) * 0.1
        self.cost_history.clear()
        self.integral_error = np.zeros(5)
        self.prev_error = np.zeros(5)
