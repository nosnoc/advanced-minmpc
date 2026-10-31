from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence
import tempfile

import casadi as ca


@dataclass(slots=True)
class CaminoResult:
    solution: dict[str, Any]
    timings: dict[str, float]
    stats: dict[str, Any]


def _column(value: Any) -> ca.DM:
    result = ca.DM(value)
    return ca.reshape(result, result.numel(), 1)


def _integer_indices(discrete: Sequence[bool | int], number_variables: int) -> list[int]:
    values = list(discrete)
    if len(values) == number_variables and all(
        value in (False, True, 0, 1) for value in values
    ):
        return [index for index, value in enumerate(values) if bool(value)]
    indices = [int(value) for value in values]
    if any(index < 0 or index >= number_variables for index in indices):
        raise ValueError("Every discrete-variable index must lie inside the decision vector.")
    return indices


def _install_casadi_compatibility() -> None:
    def zeros(matrix_type: type, *shape: Any) -> Any:
        dimensions = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        return matrix_type.zeros(*dimensions)

    if not hasattr(ca, "GenDM_zeros"):
        ca.GenDM_zeros = lambda *shape: zeros(ca.DM, *shape)
    if not hasattr(ca, "GenMX_zeros"):
        ca.GenMX_zeros = lambda *shape: zeros(ca.MX, *shape)


def _load_camino() -> tuple[Any, ...]:
    _install_casadi_compatibility()
    try:
        from camino.settings import GlobalSettings, Settings
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "CAMINO is unavailable. Install the optional dependency with "
            "'python -m pip install -r requirements-camino.txt'."
        ) from error

    temporary_directory = tempfile.gettempdir()
    GlobalSettings._OUT_DIR = temporary_directory
    GlobalSettings._DATA_FOLDER = temporary_directory
    GlobalSettings._CACHE_FOLDER = temporary_directory
    GlobalSettings._IMG_DIR = temporary_directory

    from camino.data import MinlpData
    from camino.problem import MinlpProblem
    from camino.solver import MinlpSolver
    from camino.stats import Stats

    return MinlpData, MinlpProblem, MinlpSolver, Settings, Stats


class CaminoSolver:
    def __init__(
        self,
        name: str,
        nlp: dict[str, Any],
        discrete: Sequence[bool | int],
        *,
        mip_solver: str = "gurobi",
        max_time: float = 120.0,
        max_iterations: int | None = None,
        hessian_not_psd: bool = False,
    ) -> None:
        if max_time <= 0:
            raise ValueError("max_time must be positive.")
        if max_iterations is not None and max_iterations < 1:
            raise ValueError("max_iterations must be positive or None.")
        for key in ("f", "g", "x"):
            if key not in nlp:
                raise ValueError(f"The NLP dictionary must contain '{key}'.")

        MinlpData, MinlpProblem, MinlpSolver, Settings, Stats = _load_camino()
        decision = nlp["x"]
        parameter = nlp.get("p")
        if parameter is None:
            symbol_type = ca.SX if isinstance(decision, ca.SX) else ca.MX
            parameter = symbol_type.sym(f"{name}_p", 0)

        self.name = name
        self._MinlpData = MinlpData
        self._MinlpSolver = MinlpSolver
        self._Settings = Settings
        self._Stats = Stats
        self._number_variables = int(decision.numel())
        self._number_constraints = int(nlp["g"].numel())
        self._number_parameters = int(parameter.numel())
        self._mip_solver = mip_solver
        self._max_time = float(max_time)
        self._max_iterations = max_iterations
        self._problem = MinlpProblem(
            f=nlp["f"],
            g=nlp["g"],
            x=decision,
            p=parameter,
            idx_x_integer=_integer_indices(discrete, self._number_variables),
            hessian_not_psd=hessian_not_psd,
            idx_g_dwelltime=[],
            idx_g_without_dwelltime=list(range(self._number_constraints)),
        )

    def _limit_iterations(self, solver: Any) -> None:
        algorithm = solver._subsolvers[0]
        native_termination = algorithm.termination_condition

        def termination(
            stats: Any,
            settings: Any,
            lb: Any,
            ub: Any,
            x_best: Any,
            x_current: Any,
        ) -> bool:
            if native_termination(stats, settings, lb, ub, x_best, x_current):
                return True
            if stats["iter_nr"] >= self._max_iterations:
                self._iteration_limit_reached = True
                return True
            return False

        algorithm.termination_condition = termination

    def solve(
        self,
        *,
        x0: Any,
        lbx: Any,
        ubx: Any,
        lbg: Any,
        ubg: Any,
        p: Any | None = None,
    ) -> CaminoResult:
        values = {
            "x0": _column(x0),
            "lbx": _column(lbx),
            "ubx": _column(ubx),
            "lbg": _column(lbg),
            "ubg": _column(ubg),
            "p": ca.DM.zeros(self._number_parameters, 1) if p is None else _column(p),
        }
        expected = {
            "x0": self._number_variables,
            "lbx": self._number_variables,
            "ubx": self._number_variables,
            "lbg": self._number_constraints,
            "ubg": self._number_constraints,
            "p": self._number_parameters,
        }
        for key, value in values.items():
            if value.numel() != expected[key]:
                raise ValueError(
                    f"{key} has {value.numel()} entries; expected {expected[key]}."
                )

        data = self._MinlpData(
            p=values["p"],
            x0=values["x0"],
            _lbx=values["lbx"],
            _ubx=values["ubx"],
            _lbg=values["lbg"],
            _ubg=values["ubg"],
        )
        settings = self._Settings(TIME_LIMIT=self._max_time)
        settings.MIP_SOLVER = self._mip_solver
        if not settings.WITH_DEBUG:
            settings.IPOPT_SETTINGS["ipopt.sb"] = "yes"
        if self._mip_solver == "gurobi":
            settings.MIP_SETTINGS_ALL["gurobi"]["gurobi.TimeLimit"] = self._max_time
            if not settings.WITH_DEBUG:
                settings.MIP_SETTINGS_ALL["gurobi"].pop("gurobi.output_flag", None)
                settings.MIP_SETTINGS_ALL["gurobi"].update({"gurobi.OutputFlag": 0, "gurobi.LogToConsole": 0})
        stats = self._Stats("s-b-miqp", self.name, out_dir=tempfile.gettempdir())
        solver = self._MinlpSolver("s-b-miqp", self._problem, data, stats, settings)
        self._iteration_limit_reached = False
        if self._max_iterations is not None:
            self._limit_iterations(solver)

        started = perf_counter()
        result = solver.solve(data)
        total_time = perf_counter() - started
        solution = dict(result._sol)
        statistics = dict(stats.data)
        statistics.update(
            {
                "success": bool(result.solved),
                "iteration_limit_reached": self._iteration_limit_reached,
                "return_status": (
                    "iteration_limit_feasible"
                    if self._iteration_limit_reached and result.solved
                    else "iteration_limit_without_incumbent"
                    if self._iteration_limit_reached
                    else "success"
                    if result.solved
                    else "failure"
                ),
            }
        )
        return CaminoResult(
            solution=solution,
            timings={
                "total": total_time,
                "solver": float(statistics.get("solver_wall_time", 0.0)),
                "python": float(statistics.get("python_wall_time", 0.0)),
            },
            stats=statistics,
        )
