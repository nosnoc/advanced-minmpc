"""Lotka-Volterra fishing model and RK4 multiple-shooting transcription.

Model/transcription reference: CasADi, lotka_volterra_minlp.py.
https://github.com/casadi/casadi/blob/main/docs/examples/python/lotka_volterra_minlp.py
Benchmark: mintOC, Lotka Volterra fishing problem.
https://mintoc.de/index.php?title=Lotka_Volterra_fishing_problem
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np


@dataclass(slots=True)
class LotkaVolterraProblem:
    nlp: dict[str, Any]
    discrete: list[bool]
    x0: ca.DM
    lbx: ca.DM
    ubx: ca.DM
    lbg: ca.DM
    ubg: ca.DM
    integrator: ca.Function
    state_indices: list[list[int]]
    binary_indices: list[int]
    binary_intervals: list[int]
    N: int
    T: float

    def trajectories(self, decision: Any) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(decision, dtype=float).reshape(-1)
        states = values[np.asarray(self.state_indices)].T
        binary = values[np.asarray(self.binary_indices)]
        return states, binary


def build_lotka_volterra_problem(
    initial_state: tuple[float, float] = (0.5, 0.7),
    N: int = 65,
    T: float = 16.0,
) -> LotkaVolterraProblem:
    """Build the reference multiple-shooting MINLP once for all examples."""

    c0, c1 = 0.4, 0.2
    x = ca.SX.sym("x", 2)
    v = ca.SX.sym("v")
    xdot = ca.vertcat(
        x[0] - x[0] * x[1] - c0 * x[0] * v,
        -x[1] + x[0] * x[1] - c1 * x[1] * v,
    )
    stage_cost = (x[0] - 1) ** 2 + (x[1] - 1) ** 2 + 1e-4 * v**2

    rk_steps = 4
    dt = T / N / rk_steps
    dynamics = ca.Function("lotka_rhs", [x, v], [xdot, stage_cost])
    X0 = ca.MX.sym("integrator_x0", 2)
    V = ca.MX.sym("integrator_v")
    X, Q = X0, 0
    for _ in range(rk_steps):
        k1, q1 = dynamics(X, V)
        k2, q2 = dynamics(X + dt / 2 * k1, V)
        k3, q3 = dynamics(X + dt / 2 * k2, V)
        k4, q4 = dynamics(X + dt * k3, V)
        X = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Q = Q + dt / 6 * (q1 + 2 * q2 + 2 * q3 + q4)
    integrator = ca.Function(
        "lotka_integrator", [X0, V], [X, Q], ["x0", "v"], ["xf", "qf"]
    )

    initial = ca.DM(initial_state)
    zero_controls = [ca.DM(0)] * N
    state_guess = [initial]
    state = initial
    for control in zero_controls:
        state = integrator(x0=state, v=control)["xf"]
        state_guess.append(state)

    variables: list[ca.MX] = []
    guesses: list[Any] = []
    lower_variables: list[float] = []
    upper_variables: list[float] = []
    discrete: list[bool] = []
    constraints: list[ca.MX] = []
    objective = 0
    state_indices: list[list[int]] = []
    binary_indices: list[int] = []
    binary_intervals: list[int] = []
    offset = 0

    state = ca.MX.sym("X_0", 2)
    variables.append(state)
    guesses.append(state_guess[0])
    lower_variables.extend(initial_state)
    upper_variables.extend(initial_state)
    discrete.extend([False, False])
    state_indices.append([offset, offset + 1])
    offset += 2

    for k in range(N):
        control = ca.MX.sym(f"V_{k}")
        variables.append(control)
        guesses.append(zero_controls[k])
        lower_variables.append(0.0)
        upper_variables.append(1.0)
        discrete.append(True)
        binary_indices.append(offset)
        binary_intervals.append(k)
        offset += 1

        integrated = integrator(x0=state, v=control)
        objective += integrated["qf"]
        next_state = ca.MX.sym(f"X_{k + 1}", 2)
        variables.append(next_state)
        guesses.append(state_guess[k + 1])
        lower_variables.extend([0.0, 0.0])
        upper_variables.extend([2.0, 2.0])
        discrete.extend([False, False])
        state_indices.append([offset, offset + 1])
        offset += 2
        constraints.append(integrated["xf"] - next_state)
        state = next_state

    decision = ca.vertcat(*variables)
    constraint = ca.vertcat(*constraints)
    return LotkaVolterraProblem(
        nlp={"f": objective, "x": decision, "g": constraint},
        discrete=discrete,
        x0=ca.vertcat(*guesses),
        lbx=ca.DM(lower_variables),
        ubx=ca.DM(upper_variables),
        lbg=ca.DM.zeros(constraint.numel(), 1),
        ubg=ca.DM.zeros(constraint.numel(), 1),
        integrator=integrator,
        state_indices=state_indices,
        binary_indices=binary_indices,
        binary_intervals=binary_intervals,
        N=N,
        T=T,
    )

