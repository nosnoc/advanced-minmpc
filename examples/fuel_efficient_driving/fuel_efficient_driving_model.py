"""Longitudinal position/speed model with engine torque, braking and six gears.

S. Mallick et al., Learning-Based MPC for Fuel Efficient Control of Autonomous
Vehicles with Discrete Gear Selection (2025), equations (1)–(12), Section VI.
https://arxiv.org/pdf/2503.11359v3
Parameters/reference: https://github.com/SamuelMallick/mpcrl-vehicle-gears/tree/paper-2025
"""
from dataclasses import dataclass

import casadi as ca
import numpy as np

SOURCE_REVISION = "a2bdd29130602b44f4515d30583c39e653fc68bd"
GEAR_NUMBERS = np.arange(1, 7)
# Authors' adjacency matrix: row = upcoming gear, column = preceding gear.
ADJACENCY = (np.abs(GEAR_NUMBERS[:, None]-GEAR_NUMBERS[None, :]) <= 1).astype(int)


@dataclass(frozen=True)
class DrivingConfig:
    dt: float = 1.
    seed: int = 10
    m: float = 1500.
    C: float = 0.4071
    mu: float = 0.015
    g: float = 9.81
    r: float = 0.3554
    z_f: float = 3.39
    z: tuple = (4.484, 2.872, 1.842, 1.414, 1., 0.742)
    c0: float = 0.04918
    c1: float = 0.001897
    c2: float = 4.5232e-5
    alpha: float = 0.
    beta: float = 0.01
    Q: tuple = (1., 0.1)
    torque_min: float = 15.
    torque_max: float = 300.
    brake_max: float = 9000.
    rpm_min: float = 900.
    rpm_max: float = 3000.
    torque_rate: float = 100.
    acceleration_max: float = 3.
    # Requested numerical change: classical RK4 replaces the source's Euler state update.
    N_int_ocp: int = 2
    N_int_sim: int = 4

    def __post_init__(self):
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        for value in (self.N_int_ocp, self.N_int_sim):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError("RK4 substep counts must be positive integers.")

    @property
    def ratios(self):
        return self.z_f*np.asarray(self.z)/self.r

    @property
    def G(self):
        return self.m*self.g*(self.mu*np.cos(self.alpha)+np.sin(self.alpha))

    @property
    def speed_bounds(self):
        return (np.pi*self.rpm_min/(30*self.ratios[0]),
                np.pi*self.rpm_max/(30*self.ratios[-1]))


def driving_record(config, episode_steps=100, N=15, initial_state=None):
    """Reproduce env.reset/type_2, including RNG draw order and the preview length."""
    c, rng = config, np.random.default_rng(config.seed)
    lo, hi = c.speed_bounds
    state = (np.array([0., rng.uniform(lo+5, hi-5)]) if initial_state is None
             else np.asarray(initial_state, dtype=float).copy())
    length = episode_steps+N+1
    changes = np.r_[0, np.sort(rng.integers(0, length-1, size=4)), length-1]
    speed = rng.uniform(15, 25)
    accelerations = np.r_[0., rng.uniform(-0.6, 0.6, size=3), 0.]
    reference = np.zeros((length, 2))
    reference[0] = [0., speed]
    for start, stop, acceleration in zip(changes[:-1], changes[1:], accelerations):
        for k in range(start, stop):
            speed = np.clip(speed+acceleration*c.dt, 5., 28.)
            reference[k+1] = [reference[k, 0]+c.dt*speed, speed]
    return state, reference, changes, accelerations


class DrivingModel:
    def __init__(self, config):
        self.config = c = config
        x, u, v = ca.MX.sym("x", 2), ca.MX.sym("u", 2), ca.MX.sym("v", 6)
        n = ca.dot(ca.DM(c.ratios), v)
        rhs = ca.vertcat(x[1], (n*u[0]-u[1]-c.C*x[1]**2-c.G)/c.m)
        self.rhs = ca.Function("driving_rhs", [x, u, v], [rhs])
        rpm = (30/np.pi)*n*x[1]
        self.outputs = ca.Function("driving_outputs", [x, u, v],
                                   [rpm, c.c0+c.c1*rpm+c.c2*rpm*u[0]])

        def rk4(substeps):
            state, h = x, c.dt/substeps
            for _ in range(substeps):
                k1 = self.rhs(state, u, v)
                k2 = self.rhs(state+h*k1/2, u, v)
                k3 = self.rhs(state+h*k2/2, u, v)
                k4 = self.rhs(state+h*k3, u, v)
                state = state+h*(k1+2*k2+2*k3+k4)/6
            return ca.Function("driving_rk4", [x, u, v], [state])
        self.prediction_integrator = rk4(c.N_int_ocp)
        self.integrator = rk4(c.N_int_sim)

    def step(self, state, continuous, gear):
        return np.asarray(self.integrator(state, continuous, gear)).reshape(2)

    def tracking_cost(self, state, reference):
        return float(self.config.beta*np.dot(self.config.Q, (np.asarray(state)-reference)**2))

    def initial_controls(self, state, gear=None, torque=None):
        """Default to the highest rpm-feasible gear and a feasible steady-speed input."""
        c = self.config
        speeds = (30/np.pi)*c.ratios*state[1]
        feasible = np.flatnonzero((speeds >= c.rpm_min) & (speeds <= c.rpm_max))
        if not len(feasible):
            raise ValueError("Initial speed admits no gear.")
        index = int(feasible[-1]) if gear is None else int(gear)-1
        if index not in feasible:
            raise ValueError("Initial gear must be rpm-feasible (numbered 1 through 6).")
        resistance = c.C*state[1]**2+c.G
        steady_torque = np.clip(resistance/c.ratios[index], c.torque_min, c.torque_max)
        torque = float(steady_torque if torque is None else torque)
        if not c.torque_min <= torque <= c.torque_max:
            raise ValueError("Initial torque exceeds its bounds.")
        brake = max(0., torque*c.ratios[index]-resistance)
        return np.array([torque, brake]), np.eye(6)[index]

    def physical_violations(self, state, u, v, next_state, previous_torque, previous_gear):
        c = self.config
        lo, hi = c.speed_bounds
        rpm = np.array([float(self.outputs(s, u, v)[0]) for s in (state, next_state)])
        # Report physical units, including the endpoint rpm under the gear actually applied.
        return {"torque_Nm": max(0., c.torque_min-u[0], u[0]-c.torque_max),
                "brake_N": max(0., -u[1], u[1]-c.brake_max),
                "rpm": max(0., c.rpm_min-rpm.min(), rpm.max()-c.rpm_max),
                "speed_m_s": max(0., lo-next_state[1], next_state[1]-hi),
                "speed_increment_m_s": max(0., abs(next_state[1]-state[1])-c.acceleration_max*c.dt),
                "torque_increment_Nm": max(0., abs(u[0]-previous_torque)-c.torque_rate*c.dt),
                "gear": max(0., float(np.max(v-ADJACENCY@previous_gear)))}


@dataclass
class DrivingProblem:
    model: DrivingModel
    N: int
    nlp: dict
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
    linear_rows: list
    initial_state: np.ndarray
    initial_input: np.ndarray
    initial_gear: np.ndarray
    evaluate: ca.Function

    def parameters(self, reference, previous_torque, previous_gear):
        reference = np.asarray(reference)
        if reference.shape != (self.N+1, 2):
            raise ValueError("The reference preview must have shape (N+1, 2).")
        return ca.DM(np.r_[reference.reshape(-1), previous_torque, previous_gear])

    def bounds(self, state):
        lower, upper = ca.DM(self.lbx), ca.DM(self.ubx)
        lower[self.state_indices[0]] = upper[self.state_indices[0]] = state
        return lower, upper

    def reconstruction_options(self, previous_gear):
        return {"time_grid": np.arange(self.N+1)*self.model.config.dt,
                "b_bin_pre": previous_gear, "valid_control_transitions": ADJACENCY.T,
                "valid_controls": self.admissible_gears}

    def admissible_gears(self, relaxed_decision):
        c = self.model.config
        x, _, _ = self.trajectories(relaxed_decision)
        rpm = (30/np.pi)*x[:, 1, None]*c.ratios
        feasible = (rpm >= c.rpm_min-1e-4) & (rpm <= c.rpm_max+1e-4)
        return (feasible[:-1] & feasible[1:]).astype(int)

    def trajectories(self, decision):
        w = np.asarray(decision).reshape(-1)
        return tuple(w[np.asarray(indices)] for indices in (
            self.state_indices, self.continuous_control_indices, self.binary_indices))

    def check(self, decision, parameter, state, M_f=None, nonlinear=True, tol=1e-5):
        w = np.asarray(decision).reshape(-1)
        if not np.all(np.isfinite(w)):
            raise RuntimeError("Solver returned nonfinite variables.")
        _, _, gears = self.trajectories(w)
        integer = gears if M_f is None else gears[:M_f]
        if np.max(np.abs(integer-np.rint(integer))) > tol:
            raise RuntimeError("Fractional gear in the integer horizon.")
        lower, upper = self.bounds(state)
        _, g = self.evaluate(w, parameter)
        g = np.asarray(g).reshape(-1)
        residual = np.maximum(np.asarray(self.lbg).reshape(-1)-g, g-np.asarray(self.ubg).reshape(-1))
        selected = residual if nonlinear else residual[self.linear_rows]
        violation = max(0., np.max(np.asarray(lower).reshape(-1)-w),
                        np.max(w-np.asarray(upper).reshape(-1)), np.max(selected))
        if violation > tol:
            raise RuntimeError(f"Hard constraint violation {violation:.3g}.")
        return float(max(0., np.max(residual)))


def build_problem(config=DrivingConfig(), N=15, initial_state=None, initial_gear=None, initial_torque=None):
    if not isinstance(N, int) or N < 1:
        raise ValueError("N must be a positive integer.")
    c, model = config, DrivingModel(config)
    state = driving_record(c, N=N)[0] if initial_state is None else np.asarray(initial_state, dtype=float)
    nominal_u, nominal_v = model.initial_controls(state, initial_gear, initial_torque)
    p = ca.MX.sym("parameters", 2*(N+1)+7)
    reference = ca.reshape(p[:2*(N+1)], 2, N+1)
    prior_torque, prior_gear = p[2*(N+1)], p[2*(N+1)+1:]
    variables, guesses, lower, upper, discrete = [], [], [], [], []
    constraints, lbg, ubg, linear_rows = [], [], [], []
    states, controls, binaries, intervals = [], [], [], []

    def variable(name, guess, lb, ub, integer=False):
        indices = list(range(len(guesses), len(guesses)+len(guess)))
        symbol = ca.MX.sym(name, len(guess))
        variables.append(symbol)
        guesses.extend(guess)
        lower.extend(np.broadcast_to(lb, (len(guess),)))
        upper.extend(np.broadcast_to(ub, (len(guess),)))
        discrete.extend([integer]*len(guess))
        return symbol, indices

    def constrain(expression, lb, ub, linear=False):
        if linear:
            linear_rows.extend(range(len(lbg), len(lbg)+expression.numel()))
        constraints.append(expression)
        lbg.extend(np.broadcast_to(lb, (expression.numel(),)))
        ubg.extend(np.broadcast_to(ub, (expression.numel(),)))

    x, indices = variable("X0", state, state, state)
    states.append(indices)
    objective, guess_state = 0, state.copy()
    speed_min, speed_max = c.speed_bounds
    gear_speed_min = ca.DM(np.pi*c.rpm_min/(30*c.ratios))
    gear_speed_max = ca.DM(np.pi*c.rpm_max/(30*c.ratios))
    for k in range(N):
        u, indices = variable(f"U{k}", nominal_u, [c.torque_min, 0], [c.torque_max, c.brake_max])
        controls.append(indices)
        v, indices = variable(f"V{k}", nominal_v, 0, 1, True)
        binaries.append(indices)
        intervals.extend([k]*6)
        constrain(ca.sum1(v), 1, 1, linear=True)
        constrain(v-ca.DM(ADJACENCY)@prior_gear, -np.inf, 0, linear=True)
        constrain((u[0]-prior_torque)/c.torque_rate, -c.dt, c.dt, linear=True)
        rpm, fuel_rate = model.outputs(x, u, v)
        error = x-reference[:, k]
        objective += c.beta*ca.dot(ca.DM(c.Q), error**2)+c.dt*fuel_rate
        constrain(rpm/c.rpm_max, c.rpm_min/c.rpm_max, 1)
        # Binary-equivalent speed envelopes strengthen the relaxation and keep rpm
        # endpoint bounds in the feedback MIQP even when its gear changes substantially.
        constrain((x[1]-ca.dot(gear_speed_min, v))/speed_max, 0, np.inf, linear=True)
        constrain((x[1]-ca.dot(gear_speed_max, v))/speed_max, -np.inf, 0, linear=True)
        integrated = model.prediction_integrator(x, u, v)
        guess_state = np.asarray(model.prediction_integrator(guess_state, nominal_u, nominal_v)).reshape(2)
        next_x, indices = variable(f"X{k+1}", guess_state, [-np.inf, speed_min], [np.inf, speed_max])
        states.append(indices)
        constrain(integrated-next_x, 0, 0)
        constrain((next_x[1]-x[1])/c.acceleration_max, -c.dt, c.dt, linear=True)
        # This includes the terminal endpoint, under the final interval's gear.
        constrain(model.outputs(next_x, u, v)[0]/c.rpm_max, c.rpm_min/c.rpm_max, 1)
        constrain((next_x[1]-ca.dot(gear_speed_min, v))/speed_max, 0, np.inf, linear=True)
        constrain((next_x[1]-ca.dot(gear_speed_max, v))/speed_max, -np.inf, 0, linear=True)
        x, prior_torque, prior_gear = next_x, u[0], v
    error = x-reference[:, N]
    objective += c.beta*ca.dot(ca.DM(c.Q), error**2)
    nlp = {"x": ca.vertcat(*variables), "f": objective, "g": ca.vertcat(*constraints), "p": p}
    evaluate = ca.Function("driving_evaluate", [nlp["x"], p], [objective, nlp["g"]])
    return DrivingProblem(model, N, nlp, discrete, ca.DM(guesses), ca.DM(lower), ca.DM(upper),
        ca.DM(lbg), ca.DM(ubg), states, controls, binaries, intervals, linear_rows,
        state.copy(), nominal_u, nominal_v, evaluate)
