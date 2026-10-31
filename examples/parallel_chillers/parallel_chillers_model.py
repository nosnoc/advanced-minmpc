"""Nonlinear parallel-chiller model and multiple-shooting OCP, Eqs. (1)–(4).

J. Boldocky, C. Faulkner, E. Michael, M. Gulan, A. Tuor, and J. Drgona,
Data Center Chiller Plant Optimization via Mixed-Integer Nonlinear
Differentiable Predictive Control, 2026. https://arxiv.org/pdf/2603.28323v1
Parameters and bounds: paper Table 1; actuator conventions: pnnl/MI-DPC.
Demand data: chiller_staging_with_ramp_limits/results/{RBC,MIDPC}/data_N15_Ts_180_M_3.pt.
Controls optionally use active mass flow a=v*m instead of the paper's mass flow m.
Model implementation and data: https://github.com/pnnl/MI-DPC
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import warnings

import casadi as ca
import numpy as np

SEED = 210
FILTER = np.array([0.45, 0.20, 0.15, 0.10, 0.05, 0.05])
COP_COEFFICIENTS = (1.0, 19.33, -18.33)  # Eq. (3b), Table 1.
INITIAL_TEMPERATURE = 8.0
NOMINAL_FLOW = 10.0
NOMINAL_LOAD = 150.0  # Used only to roll out the initial guess.


@dataclass(frozen=True)
class ChillerConfig:
    M: int = 3
    dt: float = 180.0
    with_ramp_limits: bool = False
    permanent_on_unit: int | None = 0
    #tracking_weight: float = 0.001
    #switching_weight: float = 20.0
    tracking_weight: float = 0.01
    switching_weight: float = 200.0
    cp: float = 4.184
    eta_r: float = 0.75
    eta_s: float = 0.7
    C_r: float = 29288.0
    C_i: float = 14644.0
    rho: float = 10.0
    capacity: float = 500.0
    gamma: float = 9.62e-4
    temperature_min: float = 8.0
    return_max: float = 40.0
    flow_min: float = 5.0
    ramp_fraction: float = 0.08
    N_int_ocp: int = 1
    N_int_sim: int = 1
    # True substitutes a=v*m; False retains mass-flow controls in Eqs. (1), (3c), and (4e).
    use_active_flow: bool = True
    flow_max: float = 20.0
    temperature_max: float = 12.0
    # Lift cooling at shooting nodes and RK4 stages using defining equalities, Eq. (1a).
    lift_Q_expression: bool = False

    # Validate the chiller count, sampling time, RK4 substeps, and fixed-on unit.
    def __post_init__(self):
        if self.M < 1 or self.dt <= 0:
            raise ValueError("M and dt must be positive.")
        for value in (self.N_int_ocp, self.N_int_sim):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError("N_int_ocp and N_int_sim must be positive integers.")
        if self.permanent_on_unit is not None and not 0 <= self.permanent_on_unit < self.M:
            raise ValueError("permanent_on_unit must be None or a valid unit index.")

    # Count the return temperature and all supply temperatures.
    @property
    def n_thermal(self):
        return self.M + 1

    # Include previous-cooling memory states when ramp limits are enabled.
    @property
    def n_x(self):
        return self.n_thermal + (self.M if self.with_ramp_limits else 0)

    # Initialize all temperatures to 8 degrees C and cooling memory to zero.
    def initial_state(self):
        return np.r_[np.full(self.n_thermal, INITIAL_TEMPERATURE), np.zeros(self.n_x - self.n_thermal)]

    # Start with the permanently-on unit, or the first unit, switched on.
    def initial_switches(self):
        value = np.zeros(self.M)
        value[self.permanent_on_unit or 0] = 1.0
        return value


# Apply the causal FIR load filter with zero prehistory.
def filtered_demand(raw: np.ndarray) -> np.ndarray:
    return np.convolve(np.asarray(raw, dtype=float), FILTER, mode="full")[:len(raw)]  # Eq. (2).


# Repeat, scale by chiller count, and resample the recorded raw and filtered demand.
def demand_record(config: ChillerConfig, count: int) -> tuple[np.ndarray, np.ndarray]:
    if count < 1:
        raise ValueError("Demand record must contain at least one sample.")
    reference = np.loadtxt(Path(__file__).with_name("demand.csv"), delimiter=",", skiprows=1)
    source_dt = reference[1, 0] - reference[0, 0]
    time = np.arange(count) * config.dt
    source_count = int(np.ceil(time[-1] / source_dt)) + 1
    source_time = np.arange(source_count) * source_dt
    raw = np.tile(reference[:, 1], int(np.ceil(source_count / len(reference))))[:source_count]
    raw = raw * config.M / 3.0
    # Preserve the physical FIR delays on the source grid when MPC sampling changes.
    filtered = filtered_demand(raw)
    indices = np.searchsorted(source_time, time, side="right") - 1
    return raw[indices], filtered[indices]


class ChillerModel:
    # Build symbolic cooling, power, and RK4 functions for prediction and simulation.
    def __init__(self, config: ChillerConfig):
        self.config = c = config
        x = ca.MX.sym("x", c.n_x)
        u = ca.MX.sym("u", 2 * c.M)
        v = ca.MX.sym("v", c.M)
        load = ca.MX.sym("filtered_load")
        # Build per-unit cooling and power outputs with optional plant ramp clipping.
        def output_function(clip, cooling=None):
            q = self.cooling_expression(x, u, v, clip=clip) if cooling is None else cooling
            plr = q / c.capacity  # Eq. (3b): part-load ratio.
            a0, a1, a2 = COP_COEFFICIENTS
            cop = a0 + a1 * plr + a2 * plr**2  # Eq. (3b), coefficients from Table 1.
            # COP is >=1 on the hard feasible interval; extend outside it for NLP trials.
            chiller = q / ca.fmax(cop, 0.5) + c.rho * v  # Eq. (3a), with the numerical COP floor above.
            pump = c.gamma * self.active_flow(u, v)**3  # Eq. (3c): gamma*(v*m)^3 = gamma*a^3.
            inputs = [x, u, v] if cooling is None else [cooling, u, v]
            return ca.Function("chiller_outputs", inputs, [q, chiller, pump])
        self.outputs = output_function(True)
        self.prediction_outputs = output_function(False)
        q_hold = ca.MX.sym("q_hold", c.M)
        thermal_rhs = ca.vertcat(
            (load - ca.sum1(q_hold)) / c.C_r,  # Eq. (1b).
            -c.eta_s * c.cp * self.active_flow(u, v) * (x[1:c.n_thermal] - u[c.M:]) / c.C_i,  # Eq. (1c).
        )
        self.rhs = ca.Function("chiller_rhs", [x, u, v, load, q_hold], [thermal_rhs])

        # Build a one-interval RK4 map with the selected clipping and substep count.
        def integration_function(clip, substeps):
            next_state, _ = self.integration_expression(x, u, v, load, substeps, clip=clip)
            return ca.Function("chiller_rk4", [x, u, v, load], [next_state])
        self.integrator = integration_function(True, c.N_int_sim)
        self.prediction_integrator = integration_function(False, c.N_int_ocp)
        if c.lift_Q_expression:
            self.lifted_outputs = output_function(False, ca.MX.sym("cooling", c.M))
            cooling_stages = []

            # Collect the original cooling evaluations for consistent auxiliary initial guesses.
            def collect_cooling(expression):
                cooling_stages.append(expression)
                return expression

            self.integration_expression(x, u, v, load, c.N_int_ocp, clip=False, cooling_lift=collect_cooling)
            self.prediction_cooling = ca.Function("chiller_cooling_stages", [x, u, v, load],
                                                  [ca.horzcat(*cooling_stages)])

    # Build RK4 expressions, optionally replacing each cooling evaluation by an auxiliary.
    def integration_expression(self, x, u, v, load, substeps, *, clip=False, cooling_lift=None):
        c = self.config
        q = self.cooling_expression(x, u, v, clip=clip)
        if cooling_lift is not None:
            q = cooling_lift(q)
        first_stage = True  # The first RK4 cooling value is also the shooting-node value.

        # Evaluate thermal derivatives, holding cooling fixed only in ramp mode.
        def rhs(thermal):
            nonlocal first_stage
            state = ca.vertcat(thermal, x[c.n_thermal:])
            if c.with_ramp_limits or (cooling_lift is not None and first_stage):
                delivered = q
            else:
                delivered = self.cooling_expression(state, u, v)
                if cooling_lift is not None:
                    delivered = cooling_lift(delivered)
            first_stage = False
            return self.rhs(state, u, v, load, delivered)

        t = x[:c.n_thermal]
        h = c.dt / substeps
        for _ in range(substeps):
            k1 = rhs(t)
            k2 = rhs(t + h * k1 / 2)
            k3 = rhs(t + h * k2 / 2)
            k4 = rhs(t + h * k3)
            t = t + h * (k1 + 2*k2 + 2*k3 + k4) / 6  # Eq. (4b): RK4 of Eqs. (1b)-(1c).
        next_state = ca.vertcat(t, q) if c.with_ramp_limits else t
        return next_state, q

    # Return actual flow from either active-flow or original mass-flow controls.
    def active_flow(self, continuous, switches):
        flow = continuous[:self.config.M]
        # Substitute a=m*v: controls store a when enabled; otherwise compute m*v in Eqs. (1), (3c).
        return flow if self.config.use_active_flow else flow * switches

    # Compute cooling, optionally applying plant capacity and ramp saturation.
    def cooling_expression(self, x, u, v, clip=True):
        c = self.config
        requested = c.eta_r * c.cp * self.active_flow(u, v) * (x[0] - x[1:c.n_thermal])  # Eq. (1a).
        if not c.with_ramp_limits or not clip:
            return requested
        requested = ca.fmin(ca.fmax(requested, 0), c.capacity)  # Eqs. (1a), (4g): plant saturation only.
        previous = x[c.n_thermal:]
        change = c.ramp_fraction * c.capacity  # Sec. 4.3: source ramp increment per sampling interval.
        return ca.fmax(ca.fmin(requested, previous + change), previous - change)

    # Advance the simulated plant by one control interval.
    def step(self, state, continuous, switches, load):
        return np.asarray(self.integrator(state, continuous, switches, load)).reshape(-1)

    # Return simulated cooling, chiller power, and pump power for each unit.
    def evaluate(self, state, continuous, switches):
        return tuple(np.asarray(a).reshape(-1) for a in self.outputs(state, continuous, switches))

    # Share the stage objective between symbolic optimization and numerical recording.
    def cost_expression(self, outputs, switches, raw, previous):
        q, chiller, pump = outputs
        c = self.config
        return (ca.sum1(chiller + pump) + c.tracking_weight * (ca.sum1(q)-raw)**2
                + c.switching_weight * ca.sumsqr(switches-previous))  # Eq. (4a), using applied switch history.

    # Sum electrical power, demand-tracking penalty, and switch-change penalty.
    def stage_cost(self, state, continuous, switches, raw, previous, *, outputs=None):
        if outputs is None:
            outputs = self.evaluate(state, continuous, switches)
        return float(self.cost_expression(outputs, switches, raw, previous))


@dataclass
class ChillerProblem:
    model: ChillerModel
    N: int
    nlp: dict[str, Any]
    discrete: list[bool]
    x0: ca.DM
    lbx: ca.DM
    ubx: ca.DM
    lbg: ca.DM
    ubg: ca.DM
    state_indices: list[list[int]]
    continuous_control_indices: list[list[int]]
    binary_indices: list[list[int]]
    binary_intervals: list[int]
    cooling_indices: list[list[int]] = field(default_factory=list)

    # Include algebraic cooling in trajectory shifting without adding physical actuators.
    @property
    def warm_start_control_indices(self):
        if not self.cooling_indices:
            return self.continuous_control_indices
        return [u+q for u, q in zip(self.continuous_control_indices, self.cooling_indices)]

    # Pack raw demand, filtered load forecasts, and previously applied switches.
    def parameters(self, raw, filtered, previous):
        if len(raw) != self.N or len(filtered) != self.N:
            raise ValueError("Each demand forecast must have N samples.")
        return ca.DM(np.r_[raw, filtered, previous])

    # Fix the measured initial state and adjust first-interval bounds for negative cooling.
    def bounds(self, state, cooling_tolerance=0.0):
        if not np.isfinite(cooling_tolerance) or cooling_tolerance < 0:
            raise ValueError("cooling_tolerance must be finite and nonnegative.")
        lower, upper = ca.DM(self.lbx), ca.DM(self.ubx)
        lower[self.state_indices[0]] = state  # Eq. (4j): fix the measured initial state through bounds.
        upper[self.state_indices[0]] = state
        c = self.model.config
        temperatures = np.asarray(state)[:c.n_thermal]
        for i, difference in enumerate(temperatures[0]-temperatures[1:]):
            if difference < 0:
                max_flow = cooling_tolerance / (-c.eta_r * c.cp * difference)
                flow_index = self.continuous_control_indices[0][i]
                switch_index = self.binary_indices[0][i]
                if c.use_active_flow or max_flow >= c.flow_min:
                    upper[flow_index] = min(float(upper[flow_index]), max_flow)
                if max_flow < c.flow_min:
                    if float(lower[switch_index]) > 0:
                        raise RuntimeError("A permanently-on unit has negative initial cooling beyond tolerance.")
                    upper[switch_index] = 0.0
                if c.with_ramp_limits:
                    # Keep the first memory update consistent with the accepted cooling tolerance.
                    memory_index = self.state_indices[1][c.n_thermal+i]
                    flow_limit = float(upper[flow_index])
                    if not c.use_active_flow:
                        flow_limit *= float(upper[switch_index])
                    lower[memory_index] = c.eta_r*c.cp*flow_limit*difference
        return lower, upper


# Assemble the multiple-shooting MINLP, nominal guess, bounds, and index maps.
def build_problem(config: ChillerConfig, N: int = 15) -> ChillerProblem:
    if N < 1:
        raise ValueError("N must be positive.")
    c, model = config, ChillerModel(config)
    parameter = ca.MX.sym("parameters", 2*N + c.M)
    variables, lower, upper, guesses, discrete = [], [], [], [], []
    constraints, lbg, ubg = [], [], []
    states, controls, binaries, intervals, cooling = [], [], [], [], []
    objective, offset = 0, 0

    # Append a decision block with its guess, bounds, integrality, and indices.
    def add_variable(name, guess, lb, ub, integer=False):
        nonlocal offset
        variable = ca.MX.sym(name, len(guess))
        variables.append(variable)
        guesses.extend(guess)
        lower.extend(lb)
        upper.extend(ub)
        discrete.extend([integer]*len(guess))
        indices = list(range(offset, offset+len(guess)))
        offset += len(guess)
        return variable, indices

    # Append a general constraint and its lower and upper bounds.
    def constrain(expression, lb, ub):
        constraints.append(expression)
        lbg.extend(np.broadcast_to(lb, (expression.numel(),)).tolist())
        ubg.extend(np.broadcast_to(ub, (expression.numel(),)).tolist())

    xmin = [c.temperature_min]*c.n_thermal + [0.0]*(c.n_x-c.n_thermal)  # Eq. (4f); cooling memory is additional.
    xmax = [c.return_max] + [c.temperature_max]*c.M + [c.capacity]*(c.n_x-c.n_thermal)  # Eq. (4f), Table 1.
    x, indices = add_variable("X0", c.initial_state(), c.initial_state(), c.initial_state())  # Eq. (4j).
    states.append(indices)
    previous = parameter[2*N:]
    initial_v = c.initial_switches()
    nominal_flow = NOMINAL_FLOW*initial_v if c.use_active_flow else np.full(c.M, NOMINAL_FLOW)
    nominal_u = np.r_[nominal_flow, np.full(c.M, INITIAL_TEMPERATURE)]
    guess_state = c.initial_state()
    for k in range(N):
        u, indices = add_variable(f"U{k}", nominal_u,
                                 [0.0 if c.use_active_flow else c.flow_min]*c.M+[c.temperature_min]*c.M,
                                 [c.flow_max]*c.M+[c.temperature_max]*c.M)  # Eq. (4e); active-flow bounds also depend on v below.
        controls.append(indices)
        vmin = np.zeros(c.M)
        if c.permanent_on_unit is not None:
            vmin[c.permanent_on_unit] = 1.0  # Eq. (7g): optional always-on convention, configurable unit.
        v, indices = add_variable(f"V{k}", initial_v, vmin, np.ones(c.M), True)  # Eq. (4i).
        binaries.append(indices)
        intervals.extend([k]*c.M)
        if c.lift_Q_expression:
            cooling_guess = np.asarray(model.prediction_cooling(
                guess_state, nominal_u, initial_v, NOMINAL_LOAD)).T
            stage_cooling_indices = []

            # Define auxiliary cooling exactly; intermediate RK4 values receive no bounds, Eq. (1a).
            def lift_cooling(expression):
                stage = len(stage_cooling_indices) // c.M
                value, indices = add_variable(f"Q{k}_{stage}", cooling_guess[stage],
                                              [-np.inf]*c.M, [np.inf]*c.M)
                stage_cooling_indices.extend(indices)
                constrain((value-expression)/c.capacity, 0, 0)  # Eq. (1a), scaled defining equality.
                return value

            next_expression, q = model.integration_expression(
                x, u, v, parameter[N+k], c.N_int_ocp, clip=False, cooling_lift=lift_cooling)
            cooling.append(stage_cooling_indices)
            outputs = model.lifted_outputs(q, u, v)
        else:
            outputs = model.prediction_outputs(x, u, v)
            q = outputs[0]
        objective += model.cost_expression(outputs, v, parameter[k], previous)  # Eqs. (4a), (4c).
        # Eq. (4a), using v_k-v_{k-1} and applied history at k=0.
        previous = v
        constrain(ca.sum1(v), 1, np.inf)  # Eq. (4d).
        if c.use_active_flow:  # Substitute a=m*v in Eq. (4e): flow_min*v <= a <= flow_max*v.
            constrain(u[:c.M] - c.flow_min*v, 0, np.inf)  # Eq. (4e), lifted lower flow bound.
            constrain(u[:c.M] - c.flow_max*v, -np.inf, 0)  # Eq. (4e), lifted upper flow bound.
        # At fixed x0, nonnegative cooling is enforced by switch bounds in bounds().
        constrain(q/c.capacity, 0 if k else -np.inf, 1)  # Eqs. (1a), (4g): nonnegative cooling and capacity.
        if c.with_ramp_limits:
            constrain((q-x[c.n_thermal:])/c.capacity, -c.ramp_fraction, c.ramp_fraction)  # Sec. 4.3: source ramp bound.
        if not c.lift_Q_expression:
            next_expression = model.prediction_integrator(x, u, v, parameter[N+k])  # Eq. (4b).
        guess_state = np.asarray(model.prediction_integrator(guess_state, nominal_u, initial_v, NOMINAL_LOAD)).reshape(-1)
        next_x, indices = add_variable(f"X{k+1}", guess_state, xmin, xmax)
        states.append(indices)
        scale = ca.DM([1.0]*c.n_thermal+[c.capacity]*(c.n_x-c.n_thermal))
        constrain((next_expression-next_x)/scale, 0, 0)  # Eq. (4b): scaled multiple-shooting continuity.
        x = next_x
    return ChillerProblem(model, N, {"x": ca.vertcat(*variables), "f": objective,
                          "g": ca.vertcat(*constraints), "p": parameter},
                          discrete, ca.DM(guesses), ca.DM(lower), ca.DM(upper),
                          ca.DM(lbg), ca.DM(ubg), states, controls, binaries, intervals, cooling)


# Extract first-interval controls, round switches, warn on violations, and zero off-unit active flows.
def applied_control(problem, decision):
    w = np.asarray(decision).reshape(-1)
    u = w[problem.continuous_control_indices[0]].copy()
    values = w[problem.binary_indices[0]]
    if np.max(np.abs(values-np.rint(values))) > 1e-6:
        warnings.warn("Applied switches are fractional.", RuntimeWarning, stacklevel=2)
    v = np.rint(values)
    c = problem.model.config
    if np.any(v < 0) or np.any(v > 1) or sum(v) < 1:
        warnings.warn("Applied switches violate the staging constraints.", RuntimeWarning, stacklevel=2)
    if c.permanent_on_unit is not None and v[c.permanent_on_unit] != 1:
        warnings.warn("The permanently-on chiller was switched off.", RuntimeWarning, stacklevel=2)
    flow_lower = c.flow_min*v if c.use_active_flow else c.flow_min
    flow_upper = c.flow_max*v if c.use_active_flow else c.flow_max
    if np.any(u[:c.M] < flow_lower-1e-5) or np.any(u[:c.M] > flow_upper+1e-5):
        warnings.warn("Applied active flow violates its switch-dependent bounds." if c.use_active_flow
                      else "Applied mass flow violates a hard bound.", RuntimeWarning, stacklevel=2)
    if c.use_active_flow:
        u[:c.M] = np.where(v == 0, 0.0, u[:c.M])
    if np.any(u[c.M:] < c.temperature_min-1e-5) or np.any(u[c.M:] > c.temperature_max+1e-5):
        warnings.warn("Applied evaporator command violates a hard bound.", RuntimeWarning, stacklevel=2)
    return u, v


# Measure temperature, cooling-capacity, and cooling-ramp violations for a simulated step.
def physical_violations(model, state, u, v, next_state, *, cooling=None):
    c = model.config
    thermal = next_state[:c.n_thermal]
    lo = np.full(c.n_thermal, c.temperature_min)
    hi = np.r_[c.return_max, np.full(c.M, c.temperature_max)]
    q = model.evaluate(state, u, v)[0] if cooling is None else cooling
    ramp = 0.0
    if c.with_ramp_limits:
        ramp = max(0.0, float(np.max(np.abs(q-state[c.n_thermal:])-c.ramp_fraction*c.capacity)))
    return {"temperature_C": max(0.0, float(np.max(lo-thermal)), float(np.max(thermal-hi))),
            "capacity_kW": max(0.0, float(-np.min(q)), float(np.max(q-c.capacity))),
            "ramp_kW": ramp}
