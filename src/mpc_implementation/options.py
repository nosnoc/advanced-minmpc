from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal, Sequence


@dataclass(slots=True)
class ThreePhaseOptions:
    """Configuration for three-phase and advanced-step solves."""

    # Keep every option documented with one line here and in OPTIONS.md.
    # Selects the CasADi conic backend used to solve the MIQP or MILP.
    mip_solver: Literal["gurobi", "daqp", "highs"] = "daqp"
    # Enables the MIQP-based integer reconstruction phase.
    enable_miqp: bool = True
    # Enables the MILP-based integer reconstruction phase.
    enable_milp: bool = False
    # Enables PyCombina reconstruction before or instead of the MIQP or MILP.
    enable_pycombina: bool = False
    # Declares each interval's binary controls as one mutually exclusive group.
    binary_controls_one_hot: bool = False
    # Sets the minimum eigenvalue used to regularize the MIQP Hessian.
    hessian_epsilon: float = 1e-6
    # Convexifies verified multiple-shooting stage Hessian blocks independently.
    exploit_structure_in_hessian_cvx: bool = False
    # Sets the Euclidean-distance tolerance for accepting rounded RNLP binaries.
    integrality_tol: float = 1e-7
    # Clamps relaxed PyCombina weights near zero or one without removing modes.
    pycombina_binary_threshold: float = 1e-6
    # Supplies options directly to the CasADi IPOPT solver used for RNLP and PNLP.
    ipopt_options: dict[str, Any] = field(
        default_factory=lambda: {
            "ipopt.print_level": 0,
            "print_time": False,
        }
    )
    # Supplies CasADi Gurobi options with strict relative and absolute MIP gaps by default.
    gurobi_options: dict[str, Any] = field(
        default_factory=lambda: {
            "gurobi": {"MIPGap": 0.0, "MIPGapAbs": 1e-9}
        }
    )
    # Supplies options directly to the CasADi DAQP conic interface.
    daqp_options: dict[str, Any] = field(default_factory=dict)
    # Supplies options directly to the CasADi HiGHS conic interface.
    highs_options: dict[str, Any] = field(default_factory=dict)
    # Configures PyCombina, including its time grid and switching constraints.
    pycombina_options: dict[str, Any] = field(default_factory=dict)

    # Gives the number of state variables at each prediction node.
    n_x: int | None = None
    # Gives the number of continuous controls in each control interval.
    n_u: int | None = None
    # Gives the number of binary controls in each control interval.
    n_v: int | None = None
    # Gives the number of control intervals in the prediction horizon.
    N: int | None = None
    # Keeps the first M_o intervals integer and fixes the remainder to PyCombina.
    M_o: int | None = None
    # Keeps the first M_f intervals integer and leaves the tail continuous in MIP and PNLP.
    M_f: int | None = None

    # Maps each discrete decision-vector entry to its control-interval index.
    binary_intervals: Sequence[int] | None = None
    # Maps every prediction-node state to its decision-vector indices.
    state_indices: Sequence[Sequence[int]] | None = None
    # Maps every continuous control to its decision-vector indices by interval.
    continuous_control_indices: Sequence[Sequence[int]] | None = None
    # Identifies the decision-vector entries fixed to the measured initial state.
    initial_state_indices: Sequence[int] | None = None

    # Enables shifting of the previous trajectory before advanced-step preparation.
    shift_initial_guess_enabled: bool = True
    # Solves a predicted MIQP during the advanced-step preparation phase.
    solve_miqp_in_preparation: bool = False
    # Solves a predicted MILP during the advanced-step preparation phase.
    solve_milp_in_preparation: bool = False
    # Refines an advanced-step MILP solution with a fixed-integer QP.
    solve_qp_after_milp: bool = True
    # Polishes the reconstructed solution with a fixed-binary NLP during feedback.
    solve_pnlp_in_feedback: bool = False
    # Selects MIQP slacks: none, eq, ineq_general, ineq_box, ineq_all, or all.
    miqp_slack_mode: Literal["none", "eq", "ineq_general", "ineq_box", "ineq_all", "all"] = "none"
    # Selects an L1 or squared-L2 penalty for MIQP slacks.
    miqp_slack_penalty_norm: Literal["l1", "l2"] = "l1"
    # Sets the positive weight of the MIQP slack penalty.
    miqp_slack_penalty: float = 1e6

    def __post_init__(self) -> None:
        gurobi_options = dict(self.gurobi_options)
        gurobi_settings = {"MIPGap": 0.0, "MIPGapAbs": 1e-9}
        gurobi_settings.update(gurobi_options.get("gurobi", {}))
        gurobi_options["gurobi"] = gurobi_settings
        self.gurobi_options = gurobi_options

        if self.mip_solver not in ("gurobi", "daqp", "highs"):
            raise ValueError("mip_solver must be 'gurobi', 'daqp', or 'highs'.")
        if self.enable_miqp and self.enable_milp:
            raise ValueError("MIQP and MILP reconstruction are mutually exclusive.")
        if self.miqp_slack_mode not in ("none", "eq", "ineq_general", "ineq_box", "ineq_all", "all"):
            raise ValueError("Invalid miqp_slack_mode.")
        if self.miqp_slack_penalty_norm not in ("l1", "l2"):
            raise ValueError("miqp_slack_penalty_norm must be 'l1' or 'l2'.")
        if not isfinite(self.miqp_slack_penalty) or self.miqp_slack_penalty <= 0:
            raise ValueError("miqp_slack_penalty must be finite and positive.")
        if self.miqp_slack_mode != "none" and not self.enable_miqp:
            raise ValueError("MIQP slacks require enable_miqp=True.")
        if not self.enable_miqp and not self.enable_milp and not self.enable_pycombina:
            raise ValueError("At least one of MIQP, MILP, and PyCombina must be enabled.")
        if self.binary_controls_one_hot:
            missing = [
                name
                for name in ("n_v", "N", "binary_intervals")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    "One-hot binary controls require " + ", ".join(missing) + "."
                )
        if self.enable_miqp and self.mip_solver == "highs":
            raise ValueError("HiGHS supports MILP reconstruction but not MIQP reconstruction.")
        if self.hessian_epsilon <= 0:
            raise ValueError("hessian_epsilon must be positive.")
        if not isfinite(self.integrality_tol) or self.integrality_tol < 0:
            raise ValueError("integrality_tol must be finite and nonnegative.")
        for name, threshold in (
            ("pycombina_binary_threshold", self.pycombina_binary_threshold),
            ('pycombina_options["binary_threshold"]',
             self.pycombina_options.get("binary_threshold", self.pycombina_binary_threshold)),
        ):
            if not isfinite(threshold) or not 0 <= threshold < 0.5:
                raise ValueError(f"{name} must be finite and satisfy 0 <= threshold < 0.5.")
        if self.M_o is not None and self.M_f is not None:
            raise ValueError("M_o and M_f are mutually exclusive.")

        dimensions = {
            "n_x": self.n_x,
            "n_u": self.n_u,
            "n_v": self.n_v,
            "N": self.N,
        }
        for name, value in dimensions.items():
            if value is not None and value < (0 if name == "n_u" else 1):
                raise ValueError(f"{name} has an invalid value: {value}.")

        partial_horizon = self.M_o is not None or self.M_f is not None
        if partial_horizon:
            missing = [name for name, value in dimensions.items() if value is None]
            if missing:
                raise ValueError(
                    "Partial horizons require " + ", ".join(missing) + "."
                )
            if self.binary_intervals is None:
                raise ValueError(
                    "Partial horizons require binary_intervals, with one interval "
                    "for every binary decision variable."
                )
            if not (self.enable_miqp or self.enable_milp):
                raise ValueError("M_o and M_f require MIQP or MILP to be enabled.")

        if self.M_o is not None:
            if not self.enable_pycombina:
                raise ValueError("M_o requires PyCombina to be enabled.")
            if not 0 <= self.M_o <= self.N:  # type: ignore[operator]
                raise ValueError("M_o must satisfy 0 <= M_o <= N.")
        if self.M_f is not None and not 1 <= self.M_f <= self.N:  # type: ignore[operator]
            raise ValueError("M_f must satisfy 1 <= M_f <= N.")

        if self.enable_pycombina:
            missing = [
                name
                for name in ("n_v", "N", "binary_intervals")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    "PyCombina requires " + ", ".join(missing) + "."
                )
