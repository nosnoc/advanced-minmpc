"""Bevel-tip needle: position and yaw/pitch/roll, insertion and axial rotation.

Y. Chen and M. Lazar, An Efficient MPC Algorithm for Switched Systems with
Minimum Dwell Time Constraints (2022), Section 5.2, equations (37)–(39).
https://arxiv.org/pdf/2002.09658v3
"""
from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np


@dataclass(frozen=True)
class NeedleConfig:
    dt: float = 0.1
    ell: int = 5
    kappa: float = 0.22
    x_initial: tuple = (0., 0., 0., 0., 0., 0.)
    x_target: tuple = (-2., 3.5, 10., 0., 0., 0.)
    centers: tuple = ((0., 0., 5.), (1., 3., 7.), (-2., 0., 10.))
    radius: float = 2.
    initial_mode: int = 0
    initial_remaining_dwell: int = 0
    # Numerical choices: the paper leaves integration order unspecified; use RK4 substeps.
    N_int_ocp: int = 2
    N_int_sim: int = 4

    def __post_init__(self):
        for value in (self.ell, self.N_int_ocp, self.N_int_sim):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError("ell and RK4 substep counts must be positive integers.")
        if self.dt <= 0 or self.initial_mode not in (0, 1):
            raise ValueError("dt must be positive and initial_mode must be 0 or 1.")
        if (isinstance(self.initial_remaining_dwell, bool)
                or not isinstance(self.initial_remaining_dwell, (int, np.integer))
                or not 0 <= self.initial_remaining_dwell < self.ell):
            raise ValueError("initial_remaining_dwell must lie in [0, ell).")


@dataclass
class ModeHistory:
    mode: int
    remaining: int = 0

    @property
    def previous(self):
        return np.eye(2)[self.mode]

    def advance(self, v, ell):
        mode = int(np.argmax(v))
        if mode != self.mode:
            if self.remaining:
                raise RuntimeError("Applied mode switches before its dwell obligation expires.")
            self.mode, self.remaining = mode, ell
        self.remaining = max(0, self.remaining - 1)


class NeedleModel:
    def __init__(self, config):
        self.config = c = config
        x, u, v = ca.MX.sym("x", 6), ca.MX.sym("u", 2), ca.MX.sym("v", 2)
        # Preserve x4=yaw, x5=pitch, x6=roll and the signs of equation (37).
        insertion = ca.vertcat(ca.sin(x[4]), -ca.cos(x[4])*ca.sin(x[3]),
            ca.cos(x[3])*ca.cos(x[4]), c.kappa*ca.cos(x[5])/ca.cos(x[4]),
            c.kappa*ca.sin(x[5]), -c.kappa*ca.cos(x[5])*ca.tan(x[4]))
        rhs = v[0]*u[0]*insertion + v[1]*u[1]*ca.DM([0, 0, 0, 0, 0, 1])
        self.rhs = ca.Function("needle_rhs", [x, u, v], [rhs])

        def rk4(substeps):
            z, h = x, c.dt/substeps
            for _ in range(substeps):
                k1 = self.rhs(z, u, v)
                k2 = self.rhs(z+h*k1/2, u, v)
                k3 = self.rhs(z+h*k2/2, u, v)
                k4 = self.rhs(z+h*k3, u, v)
                z = z+h*(k1+2*k2+2*k3+k4)/6
            return ca.Function("needle_rk4", [x, u, v], [z])
        self.prediction_integrator = rk4(c.N_int_ocp)
        self.integrator = rk4(c.N_int_sim)

    def step(self, state, continuous, mode):
        return np.asarray(self.integrator(state, continuous, mode)).reshape(6)

    def clearance(self, states):
        p = np.atleast_2d(states)[:, :3]
        return np.linalg.norm(p[:, None, :]-np.asarray(self.config.centers), axis=2)-self.config.radius


@dataclass
class NeedleProblem:
    model: NeedleModel
    N: int
    nlp: dict[str, Any]
    discrete: list
    x0: ca.DM
    lbx: ca.DM
    ubx: ca.DM
    lbg: ca.DM
    ubg: ca.DM
    state_indices: list
    continuous_control_indices: list
    binary_indices: list
    binary_intervals: list
    mode_constraint_indices: list
    evaluate: ca.Function

    def history(self):
        c = self.model.config
        return ModeHistory(c.initial_mode, c.initial_remaining_dwell)

    def bounds(self, state, history):
        lower, upper = ca.DM(self.lbx), ca.DM(self.ubx)
        lower[self.state_indices[0]] = upper[self.state_indices[0]] = state
        for indices in self.binary_indices[:history.remaining]:
            lower[indices] = upper[indices] = history.previous
        return lower, upper

    def reconstruction_options(self, history):
        c = self.model.config
        valid = np.ones((self.N, 2), dtype=int)
        valid[:history.remaining] = history.previous
        return {"time_grid": np.arange(self.N+1)*c.dt, "min_up_times": [c.ell*c.dt]*2,
                "b_bin_pre": history.previous, "valid_controls": valid}

    def trajectories(self, decision):
        w = np.asarray(decision).reshape(-1)
        return tuple(w[np.asarray(indices)] for indices in (
            self.state_indices, self.continuous_control_indices, self.binary_indices))

    def check(self, decision, state, history, *, M_f=None, nonlinear=True, tol=1e-5):
        w = np.asarray(decision).reshape(-1)
        if not np.all(np.isfinite(w)):
            raise RuntimeError("Solver returned nonfinite variables.")
        _, _, v = self.trajectories(w)
        integral = v if M_f is None else v[:M_f]
        if np.max(np.abs(integral-np.rint(integral))) > tol:
            raise RuntimeError("Solver returned fractional modes in the integer horizon.")
        lower, upper = self.bounds(state, history)
        _, g = self.evaluate(w, history.previous)
        g = np.asarray(g).reshape(-1)
        residual = np.maximum(np.asarray(self.lbg).reshape(-1)-g, g-np.asarray(self.ubg).reshape(-1))
        selected = residual if nonlinear else residual[self.mode_constraint_indices]
        violation = max(0., np.max(np.asarray(lower).reshape(-1)-w),
                        np.max(w-np.asarray(upper).reshape(-1)), np.max(selected))
        if violation > tol:
            raise RuntimeError(f"Hard constraint violation {violation:.3g}.")
        return float(max(0., np.max(residual)))


def build_problem(config=NeedleConfig(), N=40):
    if not isinstance(N, int) or N < 1:
        raise ValueError("N must be a positive integer.")
    c, model = config, NeedleModel(config)
    previous = ca.MX.sym("previous_mode", 2)
    variables, lower, upper, guesses, discrete = [], [], [], [], []
    constraints, lbg, ubg, mode_rows = [], [], [], []
    states, controls, binaries, intervals, modes = [], [], [], [], []

    def variable(name, guess, lb, ub, integer=False):
        indices = list(range(len(guesses), len(guesses)+len(guess)))
        symbol = ca.MX.sym(name, len(guess))
        variables.append(symbol)
        guesses.extend(guess)
        lower.extend(np.broadcast_to(lb, (len(guess),)))
        upper.extend(np.broadcast_to(ub, (len(guess),)))
        discrete.extend([integer]*len(guess))
        return symbol, indices

    def constrain(expression, lb, ub, mode=False):
        if mode:
            mode_rows.extend(range(len(lbg), len(lbg)+expression.numel()))
        constraints.append(expression)
        lbg.extend(np.broadcast_to(lb, (expression.numel(),)))
        ubg.extend(np.broadcast_to(ub, (expression.numel(),)))

    x, indices = variable("X0", c.x_initial, c.x_initial, c.x_initial)
    states.append(indices)
    objective, guess_state = 0, np.array(c.x_initial)
    for k in range(N):
        # A rotating-then-inserting seed selects a useful local branch of this nonconvex OCP.
        rotating = k < min(max(c.ell, int(1.5/c.dt)), N//2)
        nominal_u, nominal_v = ([0., -np.pi/2], [0., 1.]) if rotating else ([4., 0.], [1., 0.])
        u, indices = variable(f"U{k}", nominal_u, [0, -np.pi/2], [5, np.pi/2])
        controls.append(indices)
        v, indices = variable(f"V{k}", nominal_v, 0, 1, True)
        binaries.append(indices)
        intervals.extend([k, k])
        modes.append(v)
        constrain(ca.sum1(v), 1, 1, mode=True)
        objective += 0.01*ca.sumsqr(u)  # Discrete sum: no dt factor.
        for center in c.centers:
            constrain(ca.sumsqr(x[:3]-ca.DM(center)), c.radius**2, np.inf)
        integrated = model.prediction_integrator(x, u, v)
        guess_state = np.asarray(model.prediction_integrator(guess_state, nominal_u, nominal_v)).reshape(6)
        next_x, indices = variable(f"X{k+1}", guess_state, -np.inf, np.inf)
        states.append(indices)
        constrain(integrated-next_x, 0, 0)
        x = next_x
    for center in c.centers:
        constrain(ca.sumsqr(x[:3]-ca.DM(center)), c.radius**2, np.inf)
    objective += 10*ca.sumsqr(x-ca.DM(c.x_target))
    # Each activation implies ell consecutive active samples, across either tail boundary.
    # A dwell started near the horizon end may finish beyond it; MPC carries that obligation.
    for k, v in enumerate(modes):
        rise = v-(previous if k == 0 else modes[k-1])
        for j in range(k+1, min(N, k+c.ell)):
            constrain(rise-modes[j], -np.inf, 0, mode=True)
    nlp = {"x": ca.vertcat(*variables), "f": objective,
           "g": ca.vertcat(*constraints), "p": previous}
    evaluate = ca.Function("needle_evaluate", [nlp["x"], previous], [objective, nlp["g"]])
    return NeedleProblem(model, N, nlp, discrete, ca.DM(guesses), ca.DM(lower), ca.DM(upper),
                         ca.DM(lbg), ca.DM(ubg), states, controls, binaries, intervals,
                         mode_rows, evaluate)
