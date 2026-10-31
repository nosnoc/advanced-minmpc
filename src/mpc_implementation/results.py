from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np


@dataclass(slots=True)
class QuadraticModel:
    reference: ca.DM
    objective_value: float
    gradient: ca.DM
    hessian: np.ndarray | None
    constraint_value: ca.DM
    constraint_jacobian: ca.DM


@dataclass(slots=True)
class ThreePhaseResult:
    solution: dict[str, ca.DM]
    rnlp_solution: dict[str, ca.DM]
    miqp_solution: dict[str, ca.DM] | None
    milp_solution: dict[str, ca.DM] | None
    pycombina_binary: np.ndarray | None
    timings: dict[str, float]
    stats: dict[str, Any]


@dataclass(slots=True)
class PreparationResult:
    predicted_state: ca.DM
    initial_guess: ca.DM
    lbx: ca.DM
    ubx: ca.DM
    lbg: ca.DM
    ubg: ca.DM
    rnlp_solution: dict[str, ca.DM]
    rounded_rnlp_binary: np.ndarray | None
    pycombina_binary: np.ndarray | None
    quadratic_model: QuadraticModel
    predicted_miqp_solution: dict[str, ca.DM] | None
    predicted_milp_solution: dict[str, ca.DM] | None
    timings: dict[str, float]
    stats: dict[str, Any]


@dataclass(slots=True)
class FeedbackResult:
    solution: dict[str, ca.DM]
    miqp_solution: dict[str, ca.DM] | None
    milp_solution: dict[str, ca.DM] | None
    fixed_qp_solution: dict[str, ca.DM] | None
    pycombina_binary: np.ndarray | None
    timings: dict[str, float]
    stats: dict[str, Any]
