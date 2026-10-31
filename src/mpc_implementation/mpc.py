from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Callable, Sequence

import casadi as ca
import numpy as np

from .options import ThreePhaseOptions
from .results import FeedbackResult, PreparationResult
from .solver import ThreePhaseSolver


logger = logging.getLogger(__name__)


def shift_mpc_trajectory(
    previous_solution: Any,
    state_indices: Sequence[Sequence[int]],
    continuous_control_indices: Sequence[Sequence[int]] | None = None,
    binary_control_indices: Sequence[Sequence[int]] | Sequence[int] | None = None,
    current_state: Any | None = None,
) -> ca.DM:
    previous = np.asarray(ca.DM(previous_solution), dtype=float).reshape(-1)
    shifted = previous.copy()

    def index_grid(
        indices: Sequence[Sequence[int]] | Sequence[int] | None,
    ) -> np.ndarray | None:
        if indices is None:
            return None
        grid = np.asarray(indices, dtype=int)
        if grid.size == 0:
            return None
        if grid.ndim == 1:
            grid = grid.reshape(-1, 1)
        if grid.ndim != 2:
            raise ValueError("Trajectory index maps must be one- or two-dimensional.")
        if np.any(grid < 0) or np.any(grid >= previous.size):
            raise ValueError("Trajectory index maps contain an invalid decision index.")
        return grid

    state_grid = index_grid(state_indices)
    if state_grid is None:
        raise ValueError("state_indices must not be empty.")
    shifted[state_grid[:-1].reshape(-1)] = previous[state_grid[1:].reshape(-1)]
    shifted[state_grid[-1]] = previous[state_grid[-1]]

    for indices in (continuous_control_indices, binary_control_indices):
        control_grid = index_grid(indices)
        if control_grid is None:
            continue
        shifted[control_grid[:-1].reshape(-1)] = previous[
            control_grid[1:].reshape(-1)
        ]
        shifted[control_grid[-1]] = previous[control_grid[-1]]

    if current_state is not None:
        state = np.asarray(ca.DM(current_state), dtype=float).reshape(-1)
        if state.size != state_grid.shape[1]:
            raise ValueError("current_state has an incompatible dimension.")
        shifted[state_grid[0]] = state
    return ca.DM(shifted)


def prepare_mpc_initial_guess(
    nominal_guess: Any,
    previous_solution: Any | None,
    state_indices: Sequence[Sequence[int]],
    continuous_control_indices: Sequence[Sequence[int]] | None,
    binary_control_indices: Sequence[Sequence[int]] | Sequence[int] | None,
    current_state: Any,
    warm_start_enabled: bool,
) -> ca.DM:
    if not warm_start_enabled or previous_solution is None:
        return ca.DM(nominal_guess)
    return shift_mpc_trajectory(
        previous_solution,
        state_indices,
        continuous_control_indices,
        binary_control_indices,
        current_state,
    )


class AdvancedStepMPC:
    """Preparation/feedback MPC built from a reusable ThreePhaseSolver."""

    def __init__(
        self,
        name: str,
        nlp: dict[str, Any],
        discrete: Sequence[bool],
        options: ThreePhaseOptions,
        integrator: Callable[[ca.DM, ca.DM, ca.DM], Any] | None = None,
    ) -> None:
        self.options = options
        self.solver = ThreePhaseSolver(name, nlp, discrete, options)
        self.integrator = integrator
        self._preparation: PreparationResult | None = None
        self._last_solution: ca.DM | None = None
        self._validate_trajectory_mapping()

    def _validate_trajectory_mapping(self) -> None:
        required = {
            "state_indices": self.options.state_indices,
            "initial_state_indices": self.options.initial_state_indices,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError("AdvancedStepMPC requires " + ", ".join(missing) + ".")
        if self.options.N is None or self.options.n_x is None:
            raise ValueError("AdvancedStepMPC requires N and n_x.")
        state_indices = np.asarray(self.options.state_indices, dtype=int)
        if state_indices.shape != (self.options.N + 1, self.options.n_x):
            raise ValueError("state_indices must have shape (N+1, n_x).")
        initial_indices = np.asarray(self.options.initial_state_indices, dtype=int)
        if initial_indices.size != self.options.n_x:
            raise ValueError("initial_state_indices must contain n_x entries.")
        if self.options.n_u:
            controls = np.asarray(self.options.continuous_control_indices, dtype=int)
            if controls.shape != (self.options.N, self.options.n_u):
                raise ValueError(
                    "continuous_control_indices must have shape (N, n_u)."
                )

    def _binary_index_grid(self) -> np.ndarray:
        intervals = np.asarray(self.options.binary_intervals, dtype=int)
        grid = np.empty((self.options.N, self.options.n_v), dtype=int)
        for interval in range(self.options.N):
            indices = self.solver.binary_indices[intervals == interval]
            if indices.size != self.options.n_v:
                raise ValueError("Each interval must map exactly n_v binary variables.")
            grid[interval] = indices
        return grid

    def shift_initial_guess(self, previous_solution: Any) -> ca.DM:
        """Shift mapped state and control trajectories forward by one interval."""
        return shift_mpc_trajectory(
            previous_solution,
            self.options.state_indices,
            self.options.continuous_control_indices,
            self._binary_index_grid() if self.options.n_v else None,
        )

    def get_predicted_state(self, previous_solution: Any | None = None) -> ca.DM:
        """Return x_1, or one integrator step when an integrator was supplied."""

        source = self._last_solution if previous_solution is None else ca.DM(previous_solution)
        if source is None:
            raise ValueError("A previous solution is required to predict the next state.")
        state_grid = np.asarray(self.options.state_indices, dtype=int)
        if self.integrator is None:
            return ca.DM(source[state_grid[1]])

        x_now = ca.DM(source[state_grid[0]])
        if self.options.n_u:
            continuous_grid = np.asarray(
                self.options.continuous_control_indices, dtype=int
            )
            u_now = ca.DM(source[continuous_grid[0]])
        else:
            u_now = ca.DM.zeros(0, 1)
        v_now = ca.DM(source[self._binary_index_grid()[0]])
        predicted = self.integrator(x_now, u_now, v_now)
        if isinstance(predicted, dict):
            predicted = predicted.get("xf", predicted.get("x"))
        return ca.DM(predicted)

    def _set_initial_state(
        self, lbx: ca.DM, ubx: ca.DM, state: ca.DM
    ) -> tuple[ca.DM, ca.DM]:
        updated_lbx, updated_ubx = ca.DM(lbx), ca.DM(ubx)
        indices = np.asarray(self.options.initial_state_indices, dtype=int)
        state = ca.DM(state)
        if state.numel() != indices.size:
            raise ValueError("The supplied state does not have n_x entries.")
        updated_lbx[indices] = state
        updated_ubx[indices] = state
        return updated_lbx, updated_ubx

    @staticmethod
    def _mip_objective(model: Any, solution: dict[str, ca.DM]) -> float:
        constant = model.objective_value - float(
            ca.dot(model.gradient, model.reference)
        )
        if model.hessian is not None:
            hessian = ca.DM(model.hessian)
            constant += 0.5 * float(
                ca.mtimes([model.reference.T, hessian, model.reference])
            )
        return float(solution["cost"]) + constant

    def _log_solve_summary(
        self,
        preparation: PreparationResult,
        mip: dict[str, ca.DM] | None,
        mip_stats: dict[str, Any] | None,
        fixed_qp: dict[str, ca.DM] | None,
        fixed_qp_stats: dict[str, Any] | None,
        pnlp: dict[str, ca.DM] | None,
        pnlp_stats: dict[str, Any] | None,
        timings: dict[str, float],
    ) -> None:
        rows: list[list[str]] = []
        rnlp_stats = preparation.stats["rnlp"]
        rnlp_inf_pr, rnlp_inf_du = self.solver._solver_infeasibilities(rnlp_stats)
        rows.append(
            [
                "Preparation RNLP",
                f"{float(preparation.rnlp_solution['f']):.4f}",
                f"{self.solver._solver_cpu_time(rnlp_stats, preparation.timings['rnlp_solution']):.4f}",
                self.solver._solver_message(rnlp_stats),
                rnlp_inf_pr,
                rnlp_inf_du,
                "-",
            ]
        )

        predicted_pycombina_stats = preparation.stats.get("pycombina")
        if predicted_pycombina_stats is not None:
            rows.append(
                [
                    "Preparation PyCombina",
                    "-",
                    f"{float(predicted_pycombina_stats.get('cpu_time', 0.0)):.4f}",
                    self.solver._solver_message(predicted_pycombina_stats),
                    "-",
                    "-",
                    "-",
                ]
            )

        predicted_mip = (
            preparation.predicted_miqp_solution
            if self.options.enable_miqp
            else preparation.predicted_milp_solution
        )
        method = "MIQP" if self.options.enable_miqp else "MILP"
        predicted_mip_stats = preparation.stats.get(method.lower())
        if predicted_mip is not None:
            hessian_cpu = "-"
            if self.options.enable_miqp:
                hessian_cpu = (
                    f"{preparation.timings['hessian_evaluation_regularization']:.4f}"
                )
            rows.append(
                [
                    f"Preparation {method}",
                    f"{self._mip_objective(preparation.quadratic_model, predicted_mip):.4f}",
                    f"{self.solver._solver_cpu_time(predicted_mip_stats, preparation.timings[f'predicted_{method.lower()}_solution']):.4f}",
                    self.solver._solver_message(predicted_mip_stats or {}),
                    "-",
                    "-",
                    hessian_cpu,
                ]
            )

        if mip is not None:
            rows.append(
                [
                    f"Feedback {method}",
                    f"{self._mip_objective(preparation.quadratic_model, mip):.4f}",
                    f"{self.solver._solver_cpu_time(mip_stats, timings[f'{method.lower()}_solution']):.4f}",
                    self.solver._solver_message(mip_stats or {}),
                    "-",
                    "-",
                    "-",
                ]
            )

        if fixed_qp is not None:
            rows.append(
                [
                    "Feedback fixed QP",
                    f"{self._mip_objective(preparation.quadratic_model, fixed_qp):.4f}",
                    f"{self.solver._solver_cpu_time(fixed_qp_stats, timings['fixed_qp_solution']):.4f}",
                    self.solver._solver_message(fixed_qp_stats or {}),
                    "-",
                    "-",
                    f"{preparation.timings['hessian_evaluation_regularization']:.4f}",
                ]
            )

        if pnlp is not None:
            pnlp_inf_pr, pnlp_inf_du = self.solver._solver_infeasibilities(
                pnlp_stats or {}
            )
            rows.append(
                [
                    "Feedback PNLP",
                    f"{float(pnlp['f']):.4f}",
                    f"{self.solver._solver_cpu_time(pnlp_stats, timings['pnlp_solution']):.4f}",
                    self.solver._solver_message(pnlp_stats or {}),
                    pnlp_inf_pr,
                    pnlp_inf_du,
                    "-",
                ]
            )

        headings = [
            "Phase",
            "Objective",
            "CPU [s]",
            "Solver status",
            "inf_pr",
            "inf_du",
            "Hess. reg. CPU [s]",
        ]
        if self.options.enable_miqp and self.options.miqp_slack_mode != "none":
            headings.append("Slack inf")
            for row in rows:
                stats = (predicted_mip_stats if row[0] == "Preparation MIQP" else
                         mip_stats if row[0] == "Feedback MIQP" else None)
                value = None if stats is None else stats.get("slack_inf_norm")
                row.append("-" if value is None else f"{value:.3e}")
        logger.info(
            "Advanced-step MPC solve summary\n%s",
            self.solver._format_table(rows, headings, left_aligned=(0, 3)),
        )

    def preparation_phase(
        self,
        x0: Any,
        lbx: Any,
        ubx: Any,
        lbg: Any,
        ubg: Any,
        predicted_state: Any | None = None,
        previous_solution: Any | None = None,
        p: Any | None = None,
    ) -> PreparationResult:
        """Solve the predicted RNLP and prepare its local reconstruction model."""

        total_start = perf_counter()
        supplied_previous = (
            self._last_solution if previous_solution is None else ca.DM(previous_solution)
        )
        initial_guess = ca.DM(x0)
        if self.options.shift_initial_guess_enabled and supplied_previous is not None:
            initial_guess = self.shift_initial_guess(supplied_previous)

        if predicted_state is None:
            if supplied_previous is None:
                state_indices = np.asarray(self.options.initial_state_indices, dtype=int)
                predicted = ca.DM(initial_guess[state_indices])
            else:
                predicted = self.get_predicted_state(supplied_previous)
        else:
            predicted = ca.DM(predicted_state)

        prepared_lbx, prepared_ubx = self._set_initial_state(
            ca.DM(lbx), ca.DM(ubx), predicted
        )
        parameter = None if p is None else ca.DM(p)
        rnlp, rnlp_time = self.solver._solve_rnlp(
            initial_guess,
            prepared_lbx,
            prepared_ubx,
            ca.DM(lbg),
            ca.DM(ubg),
            parameter,
            log_call=False,
        )
        rnlp_stats = dict(self.solver._nlp_solver.stats())
        integer_found, rounded_rnlp_binary = (
            self.solver.rnlp_binary_integrality(rnlp)
        )
        model, linearization_time, hessian_time = self.solver.prepare_quadratic_model(
            rnlp,
            parameter,
            require_hessian=(
                integer_found
                or self.options.enable_milp
                and self.options.solve_qp_after_milp
                or self.options.M_o == 0
            ),
        )

        predicted_mip = None
        predicted_mip_stats = None
        predicted_pycombina_binary = None
        predicted_pycombina_stats = None
        predicted_pycombina_time = 0.0
        predicted_mip_build = 0.0
        predicted_mip_time = 0.0
        if self.options.enable_pycombina and not integer_found:
            predicted_pycombina_binary, predicted_pycombina_time = (
                self.solver.solve_pycombina(rnlp["x"], log_call=False)
            )
            predicted_pycombina_stats = dict(
                self.solver._last_pycombina_stats or {}
            )
        solve_mip_in_preparation = (
            self.options.enable_miqp and self.options.solve_miqp_in_preparation
        ) or (
            self.options.enable_milp and self.options.solve_milp_in_preparation
        )
        if solve_mip_in_preparation and not integer_found and self.options.M_o != 0:
            warm_start = None
            if predicted_pycombina_binary is not None:
                warm_start = ca.DM(rnlp["x"])
                warm_start[self.solver.binary_indices] = predicted_pycombina_binary
            predicted_mip, predicted_mip_build, predicted_mip_time = (
                self.solver.solve_mip(
                    model,
                    prepared_lbx,
                    prepared_ubx,
                    ca.DM(lbg),
                    ca.DM(ubg),
                    predicted_pycombina_binary,
                    warm_start,
                    log_call=False,
                )
            )
            predicted_mip_stats = self.solver._mip_stats(predicted_mip)

        timings = {
            "total_preparation": perf_counter() - total_start,
            "rnlp_solution": rnlp_time,
            "derivative_linearization": linearization_time,
            "hessian_evaluation_regularization": hessian_time,
            "predicted_pycombina_solution": predicted_pycombina_time,
            "predicted_miqp_construction": (
                predicted_mip_build if self.options.enable_miqp else 0.0
            ),
            "predicted_miqp_solution": (
                predicted_mip_time if self.options.enable_miqp else 0.0
            ),
            "predicted_milp_construction": (
                predicted_mip_build if self.options.enable_milp else 0.0
            ),
            "predicted_milp_solution": (
                predicted_mip_time if self.options.enable_milp else 0.0
            ),
        }
        result = PreparationResult(
            predicted_state=predicted,
            initial_guess=initial_guess,
            lbx=prepared_lbx,
            ubx=prepared_ubx,
            lbg=ca.DM(lbg),
            ubg=ca.DM(ubg),
            rnlp_solution=rnlp,
            rounded_rnlp_binary=(
                rounded_rnlp_binary if integer_found else None
            ),
            pycombina_binary=predicted_pycombina_binary,
            quadratic_model=model,
            predicted_miqp_solution=(
                predicted_mip if self.options.enable_miqp else None
            ),
            predicted_milp_solution=(
                predicted_mip if self.options.enable_milp else None
            ),
            timings=timings,
            stats={
                "rnlp": rnlp_stats,
                "pycombina": predicted_pycombina_stats,
                "miqp": (
                    self.solver.skipped_stats("integral RNLP")
                    if integer_found
                    and self.options.enable_miqp
                    and self.options.solve_miqp_in_preparation
                    else self.solver.skipped_stats("M_o=0 uses fixed QP")
                    if self.options.M_o == 0
                    and self.options.enable_miqp
                    and self.options.solve_miqp_in_preparation
                    else predicted_mip_stats if self.options.enable_miqp else None
                ),
                "milp": (
                    self.solver.skipped_stats("integral RNLP")
                    if integer_found
                    and self.options.enable_milp
                    and self.options.solve_milp_in_preparation
                    else self.solver.skipped_stats("M_o=0 uses fixed QP")
                    if self.options.M_o == 0
                    and self.options.enable_milp
                    and self.options.solve_milp_in_preparation
                    else predicted_mip_stats if self.options.enable_milp else None
                ),
                "integer_found_in_rnlp": integer_found,
            },
        )
        self._preparation = result
        return result

    def feedback_phase(
        self,
        x_0: Any,
        preparation: PreparationResult | None = None,
        p: Any | None = None,
    ) -> FeedbackResult:
        """Update the prepared model with the measured state and reconstruct controls."""

        total_start = perf_counter()
        prepared = preparation or self._preparation
        if prepared is None:
            raise ValueError("preparation_phase must be called before feedback_phase.")

        function_start = perf_counter()
        lbx, ubx = self._set_initial_state(prepared.lbx, prepared.ubx, ca.DM(x_0))
        function_time = perf_counter() - function_start

        integer_found = bool(prepared.stats.get("integer_found_in_rnlp", False))
        py_binary = prepared.pycombina_binary
        py_guess = None
        mip = None
        mip_stats = None
        mip_build = 0.0
        mip_time = 0.0
        fixed_qp = None
        fixed_qp_stats = None
        fixed_qp_build = 0.0
        fixed_qp_time = 0.0
        if integer_found:
            fixed_qp, fixed_qp_build, fixed_qp_time = self.solver.solve_fixed_qp(
                prepared.quadratic_model,
                prepared.rounded_rnlp_binary,
                lbx,
                ubx,
                prepared.lbg,
                prepared.ubg,
            )
            fixed_qp_stats = dict(self.solver._fixed_qp_solver.stats())
            solution = fixed_qp
            reconstructed = prepared.rounded_rnlp_binary
        else:
            if self.options.enable_pycombina:
                if py_binary is None:
                    raise RuntimeError("Preparation did not produce a PyCombina solution.")
                py_guess = ca.DM(prepared.rnlp_solution["x"])
                py_guess[self.solver.binary_indices] = py_binary

            if self.options.M_o == 0:
                fixed_qp, fixed_qp_build, fixed_qp_time = (
                    self.solver.solve_fixed_qp(
                        prepared.quadratic_model,
                        py_binary,
                        lbx,
                        ubx,
                        prepared.lbg,
                        prepared.ubg,
                        initial_guess=py_guess,
                    )
                )
                fixed_qp_stats = dict(self.solver._fixed_qp_solver.stats())
                solution = fixed_qp
                reconstructed = py_binary
            elif self.options.enable_miqp or self.options.enable_milp:
                predicted_mip = (
                    prepared.predicted_miqp_solution
                    if self.options.enable_miqp
                    else prepared.predicted_milp_solution
                )
                warm_start = (
                    predicted_mip["x"] if predicted_mip is not None else py_guess
                )
                mip, mip_build, mip_time = self.solver.solve_mip(
                    prepared.quadratic_model,
                    lbx,
                    ubx,
                    prepared.lbg,
                    prepared.ubg,
                    py_binary,
                    warm_start,
                    log_call=False,
                )
                mip_stats = self.solver._mip_stats(mip)
                solution = mip
                reconstructed = np.asarray(mip["x"], dtype=float).reshape(-1)[
                    self.solver.binary_indices
                ]
                if self.options.enable_milp and self.options.solve_qp_after_milp:
                    fixed_qp, fixed_qp_build, fixed_qp_time = (
                        self.solver.solve_fixed_qp(
                            prepared.quadratic_model,
                            reconstructed,
                            lbx,
                            ubx,
                            prepared.lbg,
                            prepared.ubg,
                            initial_guess=mip["x"],
                            fix_mip_integers_only=True,
                        )
                    )
                    fixed_qp_stats = dict(self.solver._fixed_qp_solver.stats())
                    solution = fixed_qp
            else:
                solution = {"x": py_guess}
                reconstructed = py_binary

        pnlp_time = 0.0
        pnlp = None
        pnlp_stats = None
        if self.options.solve_pnlp_in_feedback and not integer_found:
            parameter = None if p is None else ca.DM(p)
            pnlp, pnlp_time = self.solver.solve_pnlp(
                solution["x"],
                reconstructed,
                lbx,
                ubx,
                prepared.lbg,
                prepared.ubg,
                parameter,
                log_call=False,
            )
            pnlp_stats = dict(self.solver._nlp_solver.stats())
            solution = pnlp

        self._last_solution = ca.DM(solution["x"])
        timings = {
            "total_feedback": perf_counter() - total_start,
            "function_evaluation": function_time,
            "miqp_update_construction": (
                mip_build if self.options.enable_miqp else 0.0
            ),
            "milp_update_construction": (
                mip_build if self.options.enable_milp else 0.0
            ),
            "miqp_solution": mip_time if self.options.enable_miqp else 0.0,
            "milp_solution": mip_time if self.options.enable_milp else 0.0,
            "fixed_qp_construction": fixed_qp_build,
            "fixed_qp_solution": fixed_qp_time,
            "pnlp_solution": pnlp_time,
        }
        self._log_solve_summary(
            prepared,
            mip,
            mip_stats,
            fixed_qp,
            fixed_qp_stats,
            pnlp,
            pnlp_stats,
            timings,
        )
        return FeedbackResult(
            solution=solution,
            miqp_solution=mip if self.options.enable_miqp else None,
            milp_solution=mip if self.options.enable_milp else None,
            fixed_qp_solution=fixed_qp,
            pycombina_binary=py_binary,
            timings=timings,
            stats={
                "pycombina": None,
                "miqp": (
                    self.solver.skipped_stats("integral RNLP")
                    if integer_found and self.options.enable_miqp
                    else self.solver.skipped_stats("M_o=0 uses fixed QP")
                    if self.options.M_o == 0 and self.options.enable_miqp
                    else mip_stats if self.options.enable_miqp else None
                ),
                "milp": (
                    self.solver.skipped_stats("integral RNLP")
                    if integer_found and self.options.enable_milp
                    else self.solver.skipped_stats("M_o=0 uses fixed QP")
                    if self.options.M_o == 0 and self.options.enable_milp
                    else mip_stats if self.options.enable_milp else None
                ),
                "fixed_qp": fixed_qp_stats,
                "pnlp": (
                    self.solver.skipped_stats("integral RNLP")
                    if integer_found and self.options.solve_pnlp_in_feedback
                    else pnlp_stats
                ),
                "integer_found_in_rnlp": integer_found,
            },
        )
