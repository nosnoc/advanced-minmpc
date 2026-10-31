from __future__ import annotations

import logging
from time import perf_counter, process_time
from typing import Any, Sequence

import casadi as ca
import numpy as np

from .options import ThreePhaseOptions
from .results import QuadraticModel, ThreePhaseResult


logger = logging.getLogger(__name__)


def regularize_hessian(
    hessian: np.ndarray,
    epsilon: float,
    blocks: Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    """Return a symmetric Hessian whose eigenvalues are at least epsilon."""

    symmetric = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian).T)
    if blocks is not None:
        normalized = [np.asarray(block, dtype=int).reshape(-1) for block in blocks]
        covered = np.concatenate(normalized) if normalized else np.empty(0, dtype=int)
        valid = (
            symmetric.ndim == 2
            and symmetric.shape[0] == symmetric.shape[1]
            and covered.size == symmetric.shape[0]
            and np.array_equal(np.sort(covered), np.arange(symmetric.shape[0]))
        )
        if not valid:
            logger.warning(
                "Structured Hessian blocks do not cover every variable exactly once; "
                "using dense convexification."
            )
        else:
            off_block = np.ones(symmetric.shape, dtype=bool)
            for block in normalized:
                off_block[np.ix_(block, block)] = False
            scale = max(1.0, float(np.max(np.abs(symmetric), initial=0.0)))
            tolerance = 100.0 * np.finfo(float).eps * symmetric.shape[0] * scale
            maximum_off_block = float(
                np.max(np.abs(symmetric[off_block]), initial=0.0)
            )
            if maximum_off_block <= tolerance:
                regularized = np.zeros_like(symmetric)
                for block in normalized:
                    values, vectors = np.linalg.eigh(symmetric[np.ix_(block, block)])
                    values = np.maximum(values, epsilon)
                    block_hessian = (vectors * values) @ vectors.T
                    regularized[np.ix_(block, block)] = 0.5 * (
                        block_hessian + block_hessian.T
                    )
                return 0.5 * (regularized + regularized.T)
            logger.warning(
                "Structured Hessian has nonzero off-block entries "
                "(maximum %.3e, tolerance %.3e); using dense convexification.",
                maximum_off_block,
                tolerance,
            )
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.maximum(eigenvalues, epsilon)
    regularized = (eigenvectors * eigenvalues) @ eigenvectors.T
    return 0.5 * (regularized + regularized.T)


def check_integrality(
    values: Any, tolerance: float
) -> tuple[bool, np.ndarray]:
    rounded = np.rint(np.asarray(values, dtype=float).reshape(-1))
    residual = np.asarray(values, dtype=float).reshape(-1) - rounded
    return bool(np.linalg.norm(residual) <= tolerance), rounded


class ThreePhaseSolver:
    """Relax an MINLP, reconstruct its binaries, and polish the result."""

    def __init__(
        self,
        name: str,
        nlp: dict[str, Any],
        discrete: Sequence[bool],
        options: ThreePhaseOptions | None = None,
    ) -> None:
        self.name = name
        self.nlp = dict(nlp)
        self.options = options or ThreePhaseOptions()
        self.x = self.nlp["x"]
        self.f = self.nlp["f"]
        zero_constraints = (
            ca.MX.zeros(0, 1) if isinstance(self.x, ca.MX) else ca.SX.zeros(0, 1)
        )
        self.g = self.nlp.get("g", zero_constraints)
        self.p = self.nlp.get("p")
        self.n_w = int(self.x.numel())
        self.n_g = int(self.g.numel())
        self.discrete = np.asarray(discrete, dtype=bool).reshape(-1)
        if self.discrete.size != self.n_w:
            raise ValueError(
                f"discrete has length {self.discrete.size}; expected {self.n_w}."
            )
        self.binary_indices = np.flatnonzero(self.discrete)
        self._validate_binary_mapping()

        nlp_options = dict(self.options.ipopt_options)
        self._nlp_solver = ca.nlpsol(f"{name}_nlp", "ipopt", self.nlp, nlp_options)
        self._build_derivative_functions()
        mip_enabled = self.options.enable_miqp or self.options.enable_milp
        self._mip_solver = self._build_mip_solver() if mip_enabled else None
        self._slack_mip_solvers: dict[tuple[int, int], Any] = {}
        self._fixed_qp_solver = None
        self._last_pycombina_stats: dict[str, Any] | None = None

    def _validate_binary_mapping(self) -> None:
        intervals = self.options.binary_intervals
        if intervals is None:
            return
        values = np.asarray(intervals, dtype=int).reshape(-1)
        if values.size != self.binary_indices.size:
            raise ValueError(
                "binary_intervals must contain one entry per True value in discrete."
            )
        if self.options.N is not None and (
            np.any(values < 0) or np.any(values >= self.options.N)
        ):
            raise ValueError("Every binary interval must lie in [0, N).")
        if self.options.n_v is not None and self.options.N is not None:
            counts = np.bincount(values, minlength=self.options.N)
            if np.any(counts != self.options.n_v):
                raise ValueError("Each control interval must map exactly n_v binaries.")

    def _multiple_shooting_hessian_blocks(self) -> list[np.ndarray]:
        required = {
            "N": self.options.N,
            "n_x": self.options.n_x,
            "n_u": self.options.n_u,
            "n_v": self.options.n_v,
            "state_indices": self.options.state_indices,
            "binary_intervals": self.options.binary_intervals,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError("missing " + ", ".join(missing))

        state = np.asarray(self.options.state_indices, dtype=int)
        if state.shape != (self.options.N + 1, self.options.n_x):
            raise ValueError("state_indices has an incompatible shape")

        if self.options.n_u:
            if self.options.continuous_control_indices is None:
                raise ValueError("missing continuous_control_indices")
            continuous = np.asarray(
                self.options.continuous_control_indices, dtype=int
            )
            if continuous.shape != (self.options.N, self.options.n_u):
                raise ValueError("continuous_control_indices has an incompatible shape")
        else:
            continuous = np.empty((self.options.N, 0), dtype=int)

        intervals = np.asarray(self.options.binary_intervals, dtype=int).reshape(-1)
        binary = []
        for interval in range(self.options.N):
            indices = self.binary_indices[intervals == interval]
            if indices.size != self.options.n_v:
                raise ValueError("binary_intervals has an incompatible mapping")
            binary.append(indices)

        blocks = [
            np.concatenate((state[k], continuous[k], binary[k]))
            for k in range(self.options.N)
        ]
        blocks.append(state[-1].copy())
        covered = np.concatenate(blocks)
        if (
            np.any(covered < 0)
            or np.any(covered >= self.n_w)
            or covered.size != self.n_w
            or np.unique(covered).size != self.n_w
        ):
            raise ValueError(
                "stage blocks do not cover every decision variable exactly once"
            )
        return blocks

    def _function(self, name: str, expression: Any, extra_inputs: list[Any] | None = None) -> ca.Function:
        inputs = [self.x]
        if extra_inputs:
            inputs.extend(extra_inputs)
        if self.p is not None:
            inputs.append(self.p)
        return ca.Function(f"{self.name}_{name}", inputs, [expression])

    def _build_derivative_functions(self) -> None:
        self._objective_gradient = self._function(
            "objective_gradient", ca.gradient(self.f, self.x)
        )
        self._constraint_linearization = self._function(
            "constraint_linearization",
            ca.vertcat(self.g, ca.reshape(ca.jacobian(self.g, self.x), -1, 1)),
        )
        self._lagrangian_hessian = None
        if self.options.enable_miqp or self.options.enable_milp:
            symbol_type = ca.MX if isinstance(self.x, ca.MX) else ca.SX
            lam_g = symbol_type.sym(f"{self.name}_lam_g", self.n_g)
            lagrangian = self.f + ca.dot(lam_g, self.g)
            self._lagrangian_hessian = self._function(
                "lagrangian_hessian", ca.hessian(lagrangian, self.x)[0], [lam_g]
            )

    def _mip_discrete_mask(self) -> list[bool]:
        mask = self.discrete.copy()
        intervals = self.options.binary_intervals
        if self.options.M_f is not None:
            mapped = np.asarray(intervals, dtype=int)
            mask[self.binary_indices[mapped >= self.options.M_f]] = False
        if (
            self.options.binary_controls_one_hot
            and self.options.mip_solver == "daqp"
            and self.options.miqp_slack_mode not in ("eq", "all")
        ):
            # DAQP obtains the final mode exactly from the one-hot equality.
            mapped = np.asarray(intervals, dtype=int)
            for interval in range(self.options.N):
                indices = self.binary_indices[mapped == interval]
                mask[indices[-1]] = False
        return mask.tolist()

    def _build_mip_solver(
        self, n_w: int | None = None, n_g: int | None = None,
        *, daqp_discrete: Sequence[bool] | None = None,
    ) -> Any:
        n_w = self.n_w if n_w is None else n_w
        n_g = self.n_g if n_g is None else n_g
        plugin = self.options.mip_solver
        if not ca.has_conic(plugin):
            raise RuntimeError(f"Requested MIP solver '{plugin}' is unavailable in CasADi.")
        plugin_options = {
            "gurobi": self.options.gurobi_options,
            "daqp": self.options.daqp_options,
            "highs": self.options.highs_options,
        }[plugin]
        solver_options = dict(plugin_options)
        solver_options["discrete"] = self._mip_discrete_mask() + [False] * (n_w - self.n_w)
        if plugin == "daqp" and daqp_discrete is not None:
            solver_options["discrete"] = list(daqp_discrete)
        if n_w > self.n_w and plugin == "daqp" and self.options.miqp_slack_penalty_norm == "l1":
            settings = dict(solver_options.get("daqp", {}))
            settings.setdefault("eps_prox", self.options.hessian_epsilon)
            solver_options["daqp"] = settings
        hessian_sparsity = (
            ca.Sparsity.dense(n_w, n_w)
            if self.options.enable_miqp
            else ca.Sparsity(n_w, n_w)
        )
        structure = {
            "h": hessian_sparsity,
            "a": ca.Sparsity.dense(n_g, n_w),
        }
        method = "miqp" if self.options.enable_miqp else "milp"
        suffix = f"_slack_{n_w}_{n_g}" if n_w != self.n_w or n_g != self.n_g else ""
        return ca.conic(f"{self.name}_{method}{suffix}", plugin, structure, solver_options)

    def _prepare_daqp(
        self, arguments: dict[str, ca.DM], *, fixed_qp: bool = False,
    ) -> tuple[dict[str, ca.DM], np.ndarray]:
        """Retain fixed bounds, remove decided rows, and cache DAQP integrality masks."""
        n_w, n_g = arguments["g"].numel(), arguments["a"].size1()
        discrete = np.asarray(
            [False] * n_w if fixed_qp else self._mip_discrete_mask() + [False] * (n_w - self.n_w),
            dtype=bool,
        )
        attribute = "_daqp_qp_solvers" if fixed_qp else "_daqp_mip_solvers"
        cache = getattr(self, attribute, None)
        if cache is None:
            cache = {}
            existing = self._fixed_qp_solver if fixed_qp else self._mip_solver
            if existing is not None:
                cache[(n_w, n_g, tuple(discrete))] = existing
            setattr(self, attribute, cache)

        lower, upper = (np.asarray(arguments[key]).reshape(-1) for key in ("lbx", "ubx"))
        fixed = np.isfinite(lower) & (lower == upper)
        if np.any(discrete & fixed & ~np.isin(lower, (0., 1.))):
            raise RuntimeError("A fixed DAQP binary must equal zero or one.")
        # CasADi requires [0,1] bounds for every DAQP-declared binary. A fixed
        # coordinate is already integral, so it needs no branch-and-bound flag.
        discrete[fixed] = False
        matrix = np.asarray(arguments["a"])
        row_lower, row_upper = (np.asarray(arguments[key]).reshape(-1) for key in ("lba", "uba"))
        # Match DAQP's primal tolerance when checking already-decided rows.
        tolerance = float(self.options.daqp_options.get("daqp", {}).get("primal_tol", 1e-6))
        # Dwell/one-hot rows can also force a binary without fixing its box
        # bounds. Identify these implications without changing any call bounds.
        implied_lower, implied_upper = lower.copy(), upper.copy()
        while True:
            implied_fixed = np.isfinite(implied_lower) & (implied_lower == implied_upper)
            free = np.flatnonzero(~implied_fixed)
            singleton_rows = np.flatnonzero(np.count_nonzero(matrix[:, free], axis=1) == 1)
            changed = False
            for row in singleton_rows:
                column = int(free[np.flatnonzero(matrix[row, free])[0]])
                if implied_lower[column] == implied_upper[column]:
                    continue
                coefficient = matrix[row, column]
                constant = matrix[row, implied_fixed] @ implied_lower[implied_fixed]
                limits = ((row_lower[row] - constant) / coefficient,
                          (row_upper[row] - constant) / coefficient)
                lo = max(implied_lower[column], min(limits))
                hi = min(implied_upper[column], max(limits))
                if np.isfinite(lo) and lo == hi:
                    implied_lower[column] = implied_upper[column] = lo
                    changed = True
            if not changed:
                break
        implied_fixed = np.isfinite(implied_lower) & (implied_lower == implied_upper)
        discrete[implied_fixed & np.isin(implied_lower, (0., 1.))] = False

        decided = ~np.any(matrix[:, ~fixed] != 0., axis=1)
        values = matrix[:, fixed] @ lower[fixed]
        invalid = decided & (
            ~np.isfinite(values) | (row_lower > row_upper)
            | (values < row_lower - tolerance) | (values > row_upper + tolerance)
        )
        if np.any(invalid):
            row = int(np.flatnonzero(invalid)[0])
            raise RuntimeError(
                f"DAQP preprocessing found infeasible constraint row {row}: "
                f"fixed value {values[row]} is outside [{row_lower[row]}, {row_upper[row]}]."
            )
        # In particular, a one-hot equality becomes redundant when all modes
        # have been fixed. Keeping it can make DAQP's initial system dependent.
        rows = np.flatnonzero(~decided)
        if rows.size != n_g:
            arguments = {**arguments, "a": arguments["a"][rows, :],
                         "lba": arguments["lba"][rows], "uba": arguments["uba"][rows]}
        key = (n_w, rows.size, tuple(discrete))
        if key not in cache:
            cache[key] = (
                self._build_fixed_qp_solver(n_g=rows.size) if fixed_qp
                else self._build_mip_solver(n_w, rows.size, daqp_discrete=discrete)
            )
        if fixed_qp:
            self._fixed_qp_solver = cache[key]
        else:
            self._mip_solver = cache[key]
        return arguments, rows

    def _restore_daqp_solution(
        self, solution: dict[str, ca.DM], rows: np.ndarray,
        arguments: dict[str, ca.DM], *, fixed_qp: bool = False,
    ) -> dict[str, ca.DM]:
        n_g = arguments["a"].size1()
        if rows.size != n_g:
            solution = dict(solution)
            multipliers = ca.DM.zeros(n_g, 1)
            multipliers[rows] = solution["lam_a"]
            solution["lam_a"] = multipliers
        solver = self._fixed_qp_solver if fixed_qp else self._mip_solver
        if solver.stats().get("success", False):
            indices = self.binary_indices
            if self.options.M_f is not None:
                intervals = np.asarray(self.options.binary_intervals, dtype=int)
                indices = indices[intervals < self.options.M_f]
            values = np.asarray(solution["x"]).reshape(-1)
            rounded = np.rint(values[indices])
            tolerance = float(self.options.daqp_options.get("daqp", {}).get("primal_tol", 1e-6))
            if np.any(np.abs(values[indices] - rounded) > tolerance):
                raise RuntimeError("DAQP returned a nonintegral binary beyond its primal tolerance.")
            if not np.all(np.isfinite(values)) or np.any(~np.isin(rounded, (0., 1.))):
                raise RuntimeError("DAQP returned invalid binary values.")
            if np.any(values[indices] != rounded):
                # DAQP's native primal tolerance permits numerical mode noise.
                # Normalize only integral coordinates, leaving an M_f tail relaxed.
                values[indices] = rounded
                solution = dict(solution)
                solution["x"] = ca.DM(values)
                solution["cost"] = (0.5 * ca.mtimes([solution["x"].T, arguments["h"], solution["x"]])
                                    + ca.dot(arguments["g"], solution["x"]))
        return solution

    def _arguments(self, w: ca.DM, p: ca.DM | None = None) -> list[ca.DM]:
        args = [w]
        if self.p is not None:
            if p is None:
                raise ValueError("This NLP has parameters; solve(..., p=...) is required.")
            args.append(ca.DM(p))
        return args

    def rnlp_binary_integrality(
        self, solution: dict[str, ca.DM]
    ) -> tuple[bool, np.ndarray]:
        binary = np.asarray(solution["x"], dtype=float).reshape(-1)[
            self.binary_indices
        ]
        return check_integrality(binary, self.options.integrality_tol)

    def _solve_rnlp(
        self,
        x0: ca.DM,
        lbx: ca.DM,
        ubx: ca.DM,
        lbg: ca.DM,
        ubg: ca.DM,
        p: ca.DM | None,
        log_call: bool = True,
    ) -> tuple[dict[str, ca.DM], float]:
        start = perf_counter()
        kwargs: dict[str, Any] = dict(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        if self.p is not None:
            kwargs["p"] = p
        solution = self._nlp_solver(**kwargs)
        elapsed = perf_counter() - start
        stats = self._nlp_solver.stats()
        if log_call:
            logger.info(
                "RNLP success=%s time=%.6fs status=%s",
                stats.get("success"), elapsed, stats.get("return_status", "unknown"),
            )
        return solution, elapsed

    def prepare_quadratic_model(
        self,
        rnlp_solution: dict[str, ca.DM],
        p: ca.DM | None = None,
        require_hessian: bool = False,
    ) -> tuple[QuadraticModel, float, float]:
        """Evaluate the local model and, for MIQP, its regularized Hessian."""

        reference = ca.DM(rnlp_solution["x"])
        objective_value = float(rnlp_solution["f"])
        start_linearization = perf_counter()
        args = self._arguments(reference, p)
        gradient = self._objective_gradient(*args)
        packed = self._constraint_linearization(*args)
        constraint_value = packed[: self.n_g]
        jacobian = ca.reshape(packed[self.n_g :], self.n_g, self.n_w)
        linearization_time = perf_counter() - start_linearization

        hessian = None
        hessian_time = 0.0
        if self.options.enable_miqp or require_hessian:
            start_hessian = perf_counter()
            hessian_args = [reference, ca.DM(rnlp_solution["lam_g"])]
            if self.p is not None:
                hessian_args.append(ca.DM(p))
            raw_hessian = np.asarray(
                self._lagrangian_hessian(*hessian_args), dtype=float
            )
            blocks = None
            if self.options.exploit_structure_in_hessian_cvx:
                try:
                    blocks = self._multiple_shooting_hessian_blocks()
                except ValueError as error:
                    logger.warning(
                        "Cannot exploit multiple-shooting Hessian structure (%s); "
                        "using dense convexification.",
                        error,
                    )
            hessian = regularize_hessian(
                raw_hessian, self.options.hessian_epsilon, blocks
            )
            hessian_time = perf_counter() - start_hessian
        return (
            QuadraticModel(
                reference,
                objective_value,
                gradient,
                hessian,
                constraint_value,
                jacobian,
            ),
            linearization_time,
            hessian_time,
        )

    def _pycombina_time_grid(self) -> np.ndarray:
        raw = self.options.pycombina_options.get("time_grid")
        if raw is None:
            return np.arange(self.options.N + 1, dtype=float)
        grid = np.asarray(raw, dtype=float).reshape(-1)
        if grid.size != self.options.N + 1 or np.any(np.diff(grid) <= 0):
            raise ValueError("PyCombina time_grid must contain N+1 increasing values.")
        return grid

    @staticmethod
    def _component_option(value: Any, component: int, n_v: int) -> Any:
        if value is None or np.isscalar(value):
            return value
        values = np.asarray(value).reshape(-1)
        if values.size != n_v:
            raise ValueError("A PyCombina dwell-time option must be scalar or length n_v.")
        return values[component]

    @staticmethod
    def _group_option(value: Any, n_v: int) -> list[Any] | None:
        if value is None:
            return None
        if np.isscalar(value):
            return [value] * n_v
        values = np.asarray(value).reshape(-1)
        if values.size != n_v:
            raise ValueError("A PyCombina group option must be scalar or length n_v.")
        return values.tolist()

    def _solve_pycombina_one_hot(
        self,
        relaxed: np.ndarray,
        intervals: np.ndarray,
        time_grid: np.ndarray,
        py_options: dict[str, Any],
        BinApprox: Any,
        CombinaBnB: Any,
    ) -> tuple[np.ndarray, list[str], list[float]]:
        positions = np.asarray(
            [np.flatnonzero(intervals == interval) for interval in range(self.options.N)]
        )
        relaxed_grid = relaxed[positions]
        row_sums = np.sum(relaxed_grid, axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise RuntimeError("PyCombina received an invalid one-hot relaxation.")
        relaxed_grid = relaxed_grid / row_sums
        approximation = BinApprox(
            time_grid,
            relaxed_grid,
            binary_threshold=py_options.get(
                "binary_threshold", self.options.pycombina_binary_threshold
            ),
            reduce_problem_size_before_solve=py_options.get(
                "reduce_problem_size_before_solve", False
            ),
        )
        for option_name, setter_name in {
            "min_up_times": "set_min_up_times",
            "min_down_times": "set_min_down_times",
            "max_up_times": "set_max_up_times",
            "total_max_up_times": "set_total_max_up_times",
            "n_max_switches": "set_n_max_switches",
        }.items():
            value = self._group_option(
                py_options.get(option_name), self.options.n_v
            )
            if value is not None:
                getattr(approximation, setter_name)(value)
        previous = self._group_option(
            py_options.get("b_bin_pre"), self.options.n_v
        )
        if previous is not None:
            approximation.set_b_bin_pre(previous)
        transitions = py_options.get("valid_control_transitions")
        if transitions is not None:
            transitions = np.asarray(transitions)
            if transitions.shape != (self.options.n_v, self.options.n_v) or not np.all(
                np.isin(transitions, (0, 1))
            ):
                raise ValueError("valid_control_transitions must be an (n_v, n_v) binary matrix.")
            for mode, upcoming in enumerate(transitions):
                approximation.set_valid_control_transitions(mode, upcoming)
        valid_controls = py_options.get("valid_controls")
        if valid_controls is not None:
            valid = np.asarray(valid_controls)
            if valid.shape != (self.options.N, self.options.n_v) or not np.all(
                np.isin(valid, (0, 1))
            ) or np.any(np.sum(valid, axis=1) == 0):
                raise ValueError("valid_controls must be an (N, n_v) binary admissibility mask.")
            for k, allowed in enumerate(valid):
                if not np.all(allowed):
                    approximation.set_valid_controls_for_interval(
                        (time_grid[k], time_grid[k + 1]), allowed
                    )
        solver = CombinaBnB(approximation)
        solve_options = {"verbosity": 0}
        solve_options.update(py_options.get("solve_options", {}))
        solver.solve(**solve_options)
        raw_binary = np.asarray(approximation.b_bin, dtype=float)
        if raw_binary.shape == (self.options.N, self.options.n_v):
            raw_binary = raw_binary.T
        if raw_binary.shape != (self.options.n_v, self.options.N):
            raise RuntimeError("PyCombina returned an invalid one-hot trajectory shape.")
        binary_grid = raw_binary.T
        if not np.all(np.isin(binary_grid, (0, 1))) or not np.all(
            np.sum(binary_grid, axis=1) == 1
        ):
            raise RuntimeError("PyCombina did not return a one-hot trajectory.")
        binary = np.empty_like(relaxed)
        binary[positions.reshape(-1)] = binary_grid.reshape(-1)
        objectives: list[float] = []
        try:
            objectives.append(float(approximation.eta))
        except (AttributeError, TypeError, ValueError):
            pass
        return binary, [str(getattr(solver, "status", "solution returned"))], objectives

    def solve_pycombina(
        self, reference: ca.DM, log_call: bool = True
    ) -> tuple[np.ndarray, float]:
        """Run CIA for independent binaries or one mutually exclusive group."""

        try:
            from pycombina import BinApprox, CombinaBnB
        except ImportError as exc:
            raise RuntimeError("PyCombina was requested but is not installed.") from exc

        start = perf_counter()
        cpu_start = process_time()
        intervals = np.asarray(self.options.binary_intervals, dtype=int)
        relaxed = np.asarray(reference, dtype=float).reshape(-1)[self.binary_indices]
        relaxed = np.clip(relaxed, 0.0, 1.0)
        binary = np.empty_like(relaxed)
        time_grid = self._pycombina_time_grid()
        py_options = dict(self.options.pycombina_options)
        if callable(py_options.get("valid_controls")):
            py_options["valid_controls"] = py_options["valid_controls"](reference)
        horizon_duration = float(time_grid[-1] - time_grid[0])
        setters = {
            "min_up_times": ("set_min_up_times", 0),
            "min_down_times": ("set_min_down_times", 0),
            "max_up_times": ("set_max_up_times", horizon_duration),
            "total_max_up_times": ("set_total_max_up_times", horizon_duration),
            "n_max_switches": ("set_n_max_switches", self.options.N),
        }
        statuses: list[str] = []
        objectives: list[float] = []

        if self.options.binary_controls_one_hot:
            binary, statuses, objectives = self._solve_pycombina_one_hot(
                relaxed,
                intervals,
                time_grid,
                py_options,
                BinApprox,
                CombinaBnB,
            )
        else:
            positions_by_interval = [
                np.flatnonzero(intervals == interval)
                for interval in range(self.options.N)
            ]
            for component in range(self.options.n_v):
                positions = np.asarray(
                    [positions_by_interval[k][component] for k in range(self.options.N)]
                )
                component_relaxed = relaxed[positions]
                approximation = BinApprox(
                    time_grid,
                    np.column_stack((component_relaxed, 1.0 - component_relaxed)),
                    binary_threshold=py_options.get(
                        "binary_threshold", self.options.pycombina_binary_threshold
                    ),
                    reduce_problem_size_before_solve=py_options.get(
                        "reduce_problem_size_before_solve", False
                    ),
                )
                for option_name, (setter_name, complement_default) in setters.items():
                    value = self._component_option(
                        py_options.get(option_name), component, self.options.n_v
                    )
                    if value is not None:
                        values = [value, complement_default]
                        getattr(approximation, setter_name)(values)
                previous = self._component_option(
                    py_options.get("b_bin_pre"), component, self.options.n_v
                )
                if previous is not None:
                    approximation.set_b_bin_pre([previous, 1 - previous])
                solver = CombinaBnB(approximation)
                solve_options = {"verbosity": 0}
                solve_options.update(py_options.get("solve_options", {}))
                solver.solve(**solve_options)
                component_binary = np.asarray(approximation.b_bin)[0, :]
                if component_binary.size != self.options.N or not np.all(
                    np.isin(component_binary, (0, 1))
                ):
                    raise RuntimeError(
                        "PyCombina did not return a valid binary trajectory."
                    )
                binary[positions] = component_binary
                statuses.append(
                    str(getattr(solver, "status", "solution returned"))
                )
                try:
                    objectives.append(float(approximation.eta))
                except (AttributeError, TypeError, ValueError):
                    pass

        elapsed = perf_counter() - start
        self._last_pycombina_stats = {
            "success": True,
            "return_status": ", ".join(statuses),
            "objective": max(objectives) if objectives else None,
            "cpu_time": process_time() - cpu_start,
        }
        if log_call:
            logger.info(
                "PyCombina success=True time=%.6fs status=%s",
                elapsed,
                self._last_pycombina_stats["return_status"],
            )
        return binary, elapsed

    def _canonical_mip(
        self,
        model: QuadraticModel,
        lbx: ca.DM,
        ubx: ca.DM,
        lbg: ca.DM,
        ubg: ca.DM,
        pycombina_binary: np.ndarray | None,
    ) -> dict[str, ca.DM]:
        if self.options.enable_miqp:
            hessian = ca.DM(model.hessian)
            linear_cost = model.gradient - hessian @ model.reference
        else:
            hessian = ca.DM(ca.Sparsity(self.n_w, self.n_w))
            linear_cost = model.gradient
        shift = model.constraint_jacobian @ model.reference - model.constraint_value
        mip_lbx = ca.DM(lbx)
        mip_ubx = ca.DM(ubx)

        if self.options.M_o is not None:
            intervals = np.asarray(self.options.binary_intervals, dtype=int)
            tail = self.binary_indices[intervals >= self.options.M_o]
            if pycombina_binary is None:
                raise ValueError("M_o requires a PyCombina solution.")
            tail_values = pycombina_binary[intervals >= self.options.M_o]
            mip_lbx[tail] = tail_values
            mip_ubx[tail] = tail_values

        return {
            "h": hessian,
            "g": linear_cost,
            "a": model.constraint_jacobian,
            "lba": ca.DM(lbg) + shift,
            "uba": ca.DM(ubg) + shift,
            "lbx": mip_lbx,
            "ubx": mip_ubx,
        }

    def _slacken_miqp(self, arguments: dict[str, ca.DM]) -> tuple[dict[str, ca.DM], np.ndarray]:
        mode = self.options.miqp_slack_mode
        lower = np.asarray(arguments["lba"]).reshape(-1)
        upper = np.asarray(arguments["uba"]).reshape(-1)
        lbx = np.asarray(arguments["lbx"]).reshape(-1).copy()
        ubx = np.asarray(arguments["ubx"]).reshape(-1).copy()
        equality = np.isfinite(lower) & (lower == upper)
        selected = np.zeros(self.n_g, dtype=bool)
        if mode in ("eq", "all"):
            selected |= equality
        if mode in ("ineq_general", "ineq_all", "all"):
            selected |= ~equality
        box = np.empty(0, dtype=int)
        if mode in ("ineq_box", "ineq_all", "all"):
            eligible = ~self.discrete & (lbx < ubx) & (np.isfinite(lbx) | np.isfinite(ubx))
            if self.options.initial_state_indices is not None:
                eligible[np.asarray(self.options.initial_state_indices, dtype=int)] = False
            box = np.flatnonzero(eligible)
        lower = np.r_[lower, lbx[box]]
        upper = np.r_[upper, ubx[box]]
        rows = np.r_[np.flatnonzero(selected), np.arange(self.n_g, lower.size)]
        lower_rows = rows[np.isfinite(lower[rows])]
        upper_rows = rows[np.isfinite(upper[rows])]
        slack_rows = np.r_[lower_rows, upper_rows]
        count = slack_rows.size
        matrix = np.zeros((lower.size, self.n_w + count))
        matrix[:self.n_g, :self.n_w] = np.asarray(arguments["a"])
        matrix[self.n_g + np.arange(box.size), box] = 1.0
        matrix[slack_rows, self.n_w + np.arange(count)] = np.r_[
            np.ones(lower_rows.size), -np.ones(upper_rows.size)
        ]
        hessian = np.zeros((self.n_w + count, self.n_w + count))
        hessian[:self.n_w, :self.n_w] = np.asarray(arguments["h"])
        slack_gradient = np.zeros(count)
        if self.options.miqp_slack_penalty_norm == "l1":
            slack_gradient[:] = self.options.miqp_slack_penalty
        else:
            hessian[self.n_w:, self.n_w:] = self.options.miqp_slack_penalty * np.eye(count)
        lbx[box], ubx[box] = -np.inf, np.inf
        augmented = {
            "h": ca.DM(hessian),
            "g": ca.vertcat(arguments["g"], ca.DM(slack_gradient)),
            "a": ca.DM(matrix), "lba": ca.DM(lower), "uba": ca.DM(upper),
            "lbx": ca.DM(np.r_[lbx, np.zeros(count)]),
            "ubx": ca.DM(np.r_[ubx, np.full(count, np.inf)]),
        }
        if "x0" in arguments:
            guess = np.asarray(arguments["x0"]).reshape(-1)
            value = matrix[:, :self.n_w] @ guess
            slack_guess = np.r_[np.maximum(lower[lower_rows] - value[lower_rows], 0.0),
                                np.maximum(value[upper_rows] - upper[upper_rows], 0.0)]
            augmented["x0"] = ca.DM(np.r_[guess, slack_guess])
        return augmented, box

    def _mip_stats(self, solution: dict[str, ca.DM]) -> dict[str, Any]:
        stats = dict(self._mip_solver.stats())
        for key in ("slack_inf_norm", "slack_penalty", "penalized_cost"):
            if key in solution:
                stats[key] = float(solution[key])
        return stats

    def solve_mip(
        self,
        model: QuadraticModel,
        lbx: ca.DM,
        ubx: ca.DM,
        lbg: ca.DM,
        ubg: ca.DM,
        pycombina_binary: np.ndarray | None = None,
        warm_start: ca.DM | None = None,
        log_call: bool = True,
    ) -> tuple[dict[str, ca.DM], float, float]:
        """Construct and solve the configured MIQP or MILP around model.reference."""

        start_build = perf_counter()
        arguments = self._canonical_mip(
            model, lbx, ubx, lbg, ubg, pycombina_binary
        )
        construction_time = perf_counter() - start_build
        if warm_start is not None:
            arguments["x0"] = ca.DM(warm_start)
        slack_enabled = self.options.enable_miqp and self.options.miqp_slack_mode != "none"
        if slack_enabled:
            original = arguments
            arguments, box = self._slacken_miqp(arguments)
            shape = (arguments["g"].numel(), arguments["a"].size1())
            if shape not in self._slack_mip_solvers:
                self._slack_mip_solvers[shape] = self._build_mip_solver(*shape)
            self._mip_solver = self._slack_mip_solvers[shape]
            construction_time = perf_counter() - start_build

        if self.options.mip_solver == "daqp":
            daqp_arguments = arguments
            arguments, daqp_rows = self._prepare_daqp(arguments)
            construction_time = perf_counter() - start_build

        start_solve = perf_counter()
        solution = self._mip_solver(**arguments)
        solve_time = perf_counter() - start_solve
        if self.options.mip_solver == "daqp":
            solution = self._restore_daqp_solution(solution, daqp_rows, daqp_arguments)
        if slack_enabled:
            solution = dict(solution)
            slacks = np.asarray(solution["x"])[self.n_w:].reshape(-1)
            solution["x"] = solution["x"][:self.n_w]
            solution["slacks"] = ca.DM(slacks)
            solution["slack_inf_norm"] = ca.DM(np.max(np.abs(slacks), initial=0.0))
            penalty = (np.sum(slacks) if self.options.miqp_slack_penalty_norm == "l1"
                       else 0.5 * np.dot(slacks, slacks))
            solution["slack_penalty"] = ca.DM(self.options.miqp_slack_penalty * penalty)
            solution["penalized_cost"] = solution["cost"]
            w = solution["x"]
            solution["cost"] = 0.5 * ca.mtimes([w.T, original["h"], w]) + ca.dot(original["g"], w)
            solution["lam_x"] = solution["lam_x"][:self.n_w]
            if box.size:
                solution["lam_x"][box] += solution["lam_a"][self.n_g:]
            solution["lam_a"] = solution["lam_a"][:self.n_g]
        stats = self._mip_stats(solution)
        if log_call:
            method = "MIQP" if self.options.enable_miqp else "MILP"
            logger.info(
                "%s(%s) success=%s time=%.6fs status=%s",
                method,
                self.options.mip_solver,
                stats.get("success"),
                solve_time,
                stats.get("return_status", "unknown"),
            )
        return solution, construction_time, solve_time

    def _build_fixed_qp_solver(self, n_g: int | None = None) -> Any:
        plugin = self.options.mip_solver
        plugin_options = {
            "gurobi": self.options.gurobi_options,
            "daqp": self.options.daqp_options,
            "highs": self.options.highs_options,
        }[plugin]
        structure = {
            "h": ca.Sparsity.dense(self.n_w, self.n_w),
            "a": ca.Sparsity.dense(self.n_g if n_g is None else n_g, self.n_w),
        }
        return ca.conic(
            f"{self.name}_fixed_qp", plugin, structure, dict(plugin_options)
        )

    def solve_fixed_qp(
        self,
        model: QuadraticModel,
        binary: np.ndarray,
        lbx: ca.DM,
        ubx: ca.DM,
        lbg: ca.DM,
        ubg: ca.DM,
        initial_guess: ca.DM | None = None,
        fix_mip_integers_only: bool = False,
    ) -> tuple[dict[str, ca.DM], float, float]:
        """Solve the local QP with reconstructed integer coordinates fixed."""

        if model.hessian is None:
            raise ValueError("Fixed-binary QP requires a regularized Hessian.")
        values = np.asarray(binary, dtype=float).reshape(-1)
        if values.size != self.binary_indices.size:
            raise ValueError("Fixed-binary QP requires one value per binary variable.")
        fixed_positions = np.ones(values.size, dtype=bool)
        if fix_mip_integers_only and self.options.M_f is not None:
            intervals = np.asarray(self.options.binary_intervals, dtype=int)
            fixed_positions = intervals < self.options.M_f
        integral, rounded = check_integrality(
            values[fixed_positions], self.options.integrality_tol
        )
        if not integral:
            raise ValueError("Fixed-binary QP requires integral fixed values.")
        start_build = perf_counter()
        hessian = ca.DM(model.hessian)
        shift = model.constraint_jacobian @ model.reference - model.constraint_value
        fixed_lbx, fixed_ubx = ca.DM(lbx), ca.DM(ubx)
        fixed_indices = self.binary_indices[fixed_positions]
        fixed_lbx[fixed_indices] = rounded
        fixed_ubx[fixed_indices] = rounded
        warm_start = (
            ca.DM(model.reference) if initial_guess is None else ca.DM(initial_guess)
        )
        warm_start[fixed_indices] = rounded
        arguments = {
            "h": hessian,
            "g": model.gradient - hessian @ model.reference,
            "a": model.constraint_jacobian,
            "lba": ca.DM(lbg) + shift,
            "uba": ca.DM(ubg) + shift,
            "lbx": fixed_lbx,
            "ubx": fixed_ubx,
            "x0": warm_start,
        }
        if self.options.mip_solver == "daqp":
            daqp_arguments = arguments
            arguments, daqp_rows = self._prepare_daqp(arguments, fixed_qp=True)
        elif self._fixed_qp_solver is None:
            self._fixed_qp_solver = self._build_fixed_qp_solver()
        construction_time = perf_counter() - start_build
        start_solve = perf_counter()
        solution = self._fixed_qp_solver(**arguments)
        solve_time = perf_counter() - start_solve
        if self.options.mip_solver == "daqp":
            solution = self._restore_daqp_solution(solution, daqp_rows, daqp_arguments, fixed_qp=True)
        return solution, construction_time, solve_time

    def solve_pnlp(
        self,
        initial_guess: ca.DM,
        binary: np.ndarray,
        lbx: ca.DM,
        ubx: ca.DM,
        lbg: ca.DM,
        ubg: ca.DM,
        p: ca.DM | None = None,
        log_call: bool = True,
    ) -> tuple[dict[str, ca.DM], float]:
        """Solve the NLP after fixing the coordinates kept integer by reconstruction."""

        values = np.asarray(binary, dtype=float).reshape(-1)
        if values.size != self.binary_indices.size:
            raise ValueError("Expected one reconstructed value per binary variable.")
        fixed_positions = np.ones(values.size, dtype=bool)
        if self.options.M_f is not None:
            intervals = np.asarray(self.options.binary_intervals, dtype=int)
            fixed_positions = intervals < self.options.M_f
        fixed_values = values[fixed_positions]
        integral, rounded = check_integrality(
            fixed_values, self.options.integrality_tol
        )
        if not integral:
            raise RuntimeError(
                "PNLP requires integral values for every coordinate kept integer "
                "by the MIP."
            )
        fixed_lbx = ca.DM(lbx)
        fixed_ubx = ca.DM(ubx)
        fixed_indices = self.binary_indices[fixed_positions]
        fixed_lbx[fixed_indices] = rounded
        fixed_ubx[fixed_indices] = rounded
        start = perf_counter()
        kwargs: dict[str, Any] = dict(
            x0=initial_guess,
            lbx=fixed_lbx,
            ubx=fixed_ubx,
            lbg=lbg,
            ubg=ubg,
        )
        if self.p is not None:
            kwargs["p"] = p
        solution = self._nlp_solver(**kwargs)
        elapsed = perf_counter() - start
        stats = self._nlp_solver.stats()
        if log_call:
            logger.info(
                "PNLP success=%s time=%.6fs status=%s",
                stats.get("success"), elapsed, stats.get("return_status", "unknown"),
            )
        return solution, elapsed

    @staticmethod
    def _solver_message(stats: dict[str, Any]) -> str:
        outcome = "success" if stats.get("success", False) else "failure"
        status = str(stats.get("return_status", "unknown"))
        return f"{outcome} ({status})"

    @staticmethod
    def _solver_infeasibilities(stats: dict[str, Any]) -> tuple[str, str]:
        iterations = stats.get("iterations", {})
        residuals: list[str] = []
        for key in ("inf_pr", "inf_du"):
            values = iterations.get(key) if isinstance(iterations, dict) else None
            if values is not None and len(values):
                residuals.append(f"{float(values[-1]):.2e}")
            else:
                residuals.append("-")
        return residuals[0], residuals[1]

    @staticmethod
    def _solver_cpu_time(stats: dict[str, Any] | None, fallback: float) -> float:
        if stats:
            for key in ("t_proc_total", "t_proc_solver"):
                value = stats.get(key)
                if value is not None and float(value) > 0.0:
                    return float(value)
        return fallback

    @staticmethod
    def skipped_stats(reason: str) -> dict[str, Any]:
        return {
            "success": True,
            "return_status": f"skipped ({reason})",
            "skipped": True,
        }

    @staticmethod
    def _format_table(
        rows: list[list[str]],
        headings: Sequence[str] | None = None,
        left_aligned: Sequence[int] = (0, 4),
    ) -> str:
        if headings is None:
            headings = [
                "Phase",
                "Objective",
                "CPU [s]",
                "Hess. reg. CPU [s]",
                "Solver message",
                "inf_pr",
                "inf_du",
                "Gap [%]",
            ]
        widths = [
            max(len(headings[i]), *(len(row[i]) for row in rows))
            for i in range(len(headings))
        ]
        separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

        def line(values: list[str]) -> str:
            return "| " + " | ".join(
                (value.ljust(width) if index in left_aligned else value.rjust(width))
                for index, (value, width) in enumerate(zip(values, widths))
            ) + " |"

        return "\n".join(
            [separator, line(headings), separator]
            + [line(row) for row in rows]
            + [separator]
        )

    def _log_three_phase_summary(
        self,
        rnlp: dict[str, ca.DM],
        rnlp_stats: dict[str, Any],
        rnlp_time: float,
        model: QuadraticModel,
        hessian_time: float,
        mip: dict[str, ca.DM] | None,
        mip_stats: dict[str, Any] | None,
        mip_time: float,
        pycombina_time: float,
        pnlp: dict[str, ca.DM],
        pnlp_stats: dict[str, Any],
        pnlp_time: float,
    ) -> None:
        rnlp_objective = float(rnlp["f"])
        pnlp_objective = float(pnlp["f"])
        absolute_gap = pnlp_objective - rnlp_objective
        gap_tolerance = 1e-7 * max(1.0, abs(rnlp_objective), abs(pnlp_objective))
        if absolute_gap >= -gap_tolerance:
            absolute_gap = max(0.0, absolute_gap)
            relative_gap = absolute_gap / max(abs(pnlp_objective), 1e-12)
            gap_message = f"{100.0 * relative_gap:.3f}"
        else:
            gap_message = "invalid"

        if mip is not None:
            method = "MIQP" if self.options.enable_miqp else "MILP"
            constant = model.objective_value - float(
                ca.dot(model.gradient, model.reference)
            )
            if self.options.enable_miqp:
                hessian = ca.DM(model.hessian)
                constant += 0.5 * float(
                    ca.mtimes([model.reference.T, hessian, model.reference])
                )
            integer_objective = float(mip["cost"]) + constant
            mip_message = self._solver_message(mip_stats or {})
            integer_message = (
                mip_message if self.options.enable_miqp else f"MILP {mip_message}"
            )
            integer_cpu = self._solver_cpu_time(mip_stats, mip_time)
            if self.options.enable_pycombina and self._last_pycombina_stats:
                py_message = self._solver_message(self._last_pycombina_stats)
                integer_message = (
                    f"PyCombina {py_message}; {method} {mip_message}"
                )
                integer_cpu += float(self._last_pycombina_stats["cpu_time"])
        else:
            py_stats = self._last_pycombina_stats or {}
            py_objective = py_stats.get("objective")
            integer_objective = None if py_objective is None else float(py_objective)
            integer_message = f"PyCombina {self._solver_message(py_stats)}"
            integer_cpu = float(py_stats.get("cpu_time", pycombina_time))

        rnlp_inf_pr, rnlp_inf_du = self._solver_infeasibilities(rnlp_stats)
        pnlp_inf_pr, pnlp_inf_du = self._solver_infeasibilities(pnlp_stats)
        rows = [
            [
                "RNLP",
                f"{rnlp_objective:.4f}",
                f"{self._solver_cpu_time(rnlp_stats, rnlp_time):.4f}",
                "-",
                self._solver_message(rnlp_stats),
                rnlp_inf_pr,
                rnlp_inf_du,
                "-",
            ],
            [
                "Integer",
                "-" if integer_objective is None else f"{integer_objective:.4f}",
                f"{integer_cpu:.4f}",
                f"{hessian_time:.4f}" if self.options.enable_miqp else "-",
                integer_message,
                "-",
                "-",
                "-",
            ],
            [
                "PNLP",
                f"{pnlp_objective:.4f}",
                f"{self._solver_cpu_time(pnlp_stats, pnlp_time):.4f}",
                "-",
                self._solver_message(pnlp_stats),
                pnlp_inf_pr,
                pnlp_inf_du,
                gap_message,
            ],
        ]
        logger.info("Three-phase solve summary\n%s", self._format_table(rows))

    def _log_integral_rnlp_summary(
        self,
        rnlp: dict[str, ca.DM],
        rnlp_stats: dict[str, Any],
        rnlp_time: float,
    ) -> None:
        inf_pr, inf_du = self._solver_infeasibilities(rnlp_stats)
        skipped = self._solver_message(self.skipped_stats("integral RNLP"))
        rows = [
            [
                "RNLP",
                f"{float(rnlp['f']):.4f}",
                f"{self._solver_cpu_time(rnlp_stats, rnlp_time):.4f}",
                "-",
                self._solver_message(rnlp_stats),
                inf_pr,
                inf_du,
                "-",
            ],
            ["Integer", "-", "-", "-", skipped, "-", "-", "-"],
            ["PNLP", "-", "-", "-", skipped, "-", "-", "-"],
        ]
        logger.info("Three-phase solve summary\n%s", self._format_table(rows))

    def solve(
        self,
        x0: Any,
        lbx: Any,
        ubx: Any,
        lbg: Any,
        ubg: Any,
        p: Any | None = None,
    ) -> ThreePhaseResult:
        """Run all enabled phases for the supplied numerical data."""

        total_start = perf_counter()
        x0, lbx, ubx = ca.DM(x0), ca.DM(lbx), ca.DM(ubx)
        lbg, ubg = ca.DM(lbg), ca.DM(ubg)
        parameter = None if p is None else ca.DM(p)

        rnlp, rnlp_time = self._solve_rnlp(
            x0, lbx, ubx, lbg, ubg, parameter, log_call=False
        )
        rnlp_stats = dict(self._nlp_solver.stats())
        integer_found, rounded_binary = self.rnlp_binary_integrality(rnlp)
        if integer_found:
            completed = dict(rnlp)
            completed_x = ca.DM(rnlp["x"])
            completed_x[self.binary_indices] = rounded_binary
            completed["x"] = completed_x
            skipped = self.skipped_stats("integral RNLP")
            timings = {
                "rnlp_solver": rnlp_time,
                "miqp_construction": 0.0,
                "milp_construction": 0.0,
                "hessian_evaluation_regularization": 0.0,
                "pycombina_solution": 0.0,
                "miqp_solution": 0.0,
                "milp_solution": 0.0,
                "pnlp_solver": 0.0,
                "total": perf_counter() - total_start,
            }
            stats = {
                "rnlp": rnlp_stats,
                "miqp": skipped if self.options.enable_miqp else None,
                "milp": skipped if self.options.enable_milp else None,
                "pycombina": (
                    skipped if self.options.enable_pycombina else None
                ),
                "pnlp": skipped,
                "integer_found_in_rnlp": True,
            }
            self._log_integral_rnlp_summary(completed, rnlp_stats, rnlp_time)
            return ThreePhaseResult(
                solution=completed,
                rnlp_solution=completed,
                miqp_solution=None,
                milp_solution=None,
                pycombina_binary=None,
                timings=timings,
                stats=stats,
            )
        model, linearization_time, hessian_time = self.prepare_quadratic_model(
            rnlp, parameter
        )

        pycombina_binary = None
        pycombina_time = 0.0
        pycombina_guess = None
        if self.options.enable_pycombina:
            pycombina_binary, pycombina_time = self.solve_pycombina(
                rnlp["x"], log_call=False
            )
            pycombina_guess = ca.DM(rnlp["x"])
            pycombina_guess[self.binary_indices] = pycombina_binary

        mip = None
        mip_construction_time = 0.0
        mip_time = 0.0
        if self.options.enable_miqp or self.options.enable_milp:
            mip, mip_construction_time, mip_time = self.solve_mip(
                model,
                lbx,
                ubx,
                lbg,
                ubg,
                pycombina_binary,
                pycombina_guess,
                log_call=False,
            )
            polish_guess = mip["x"]
            reconstructed = np.asarray(mip["x"], dtype=float).reshape(-1)[
                self.binary_indices
            ]
            mip_stats = self._mip_stats(mip)
        else:
            polish_guess = pycombina_guess
            reconstructed = pycombina_binary
            mip_stats = None

        pnlp, pnlp_time = self.solve_pnlp(
            polish_guess,
            reconstructed,
            lbx,
            ubx,
            lbg,
            ubg,
            parameter,
            log_call=False,
        )
        pnlp_stats = dict(self._nlp_solver.stats())
        self._log_three_phase_summary(
            rnlp,
            rnlp_stats,
            rnlp_time,
            model,
            hessian_time,
            mip,
            mip_stats,
            mip_time,
            pycombina_time,
            pnlp,
            pnlp_stats,
            pnlp_time,
        )
        timings = {
            "rnlp_solver": rnlp_time,
            "miqp_construction": (
                mip_construction_time + linearization_time
                if self.options.enable_miqp
                else 0.0
            ),
            "milp_construction": (
                mip_construction_time + linearization_time
                if self.options.enable_milp
                else 0.0
            ),
            "hessian_evaluation_regularization": hessian_time,
            "pycombina_solution": pycombina_time,
            "miqp_solution": mip_time if self.options.enable_miqp else 0.0,
            "milp_solution": mip_time if self.options.enable_milp else 0.0,
            "pnlp_solver": pnlp_time,
            "total": perf_counter() - total_start,
        }
        stats = {
            "rnlp": rnlp_stats,
            "miqp": mip_stats if self.options.enable_miqp else None,
            "milp": mip_stats if self.options.enable_milp else None,
            "pycombina": self._last_pycombina_stats,
            "pnlp": pnlp_stats,
            "integer_found_in_rnlp": False,
        }
        return ThreePhaseResult(
            solution=pnlp,
            rnlp_solution=rnlp,
            miqp_solution=mip if self.options.enable_miqp else None,
            milp_solution=mip if self.options.enable_milp else None,
            pycombina_binary=pycombina_binary,
            timings=timings,
            stats=stats,
        )
