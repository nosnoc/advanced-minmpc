# Solver and MPC options

[Back to the README](README.md) · [Code structure](docs/CODE_STRUCTURE.md)

`ThreePhaseOptions` configures both `ThreePhaseSolver` and `AdvancedStepMPC`. The fields below are defined in [`options.py`](src/mpc_implementation/options.py).

| Option | Default | Meaning |
|---|---:|---|
| `mip_solver` | `"daqp"` | Selects the CasADi conic backend used to solve the MIQP or MILP: `"gurobi"`, `"daqp"`, or `"highs"`. |
| `enable_miqp` | `True` | Enables the MIQP-based integer reconstruction phase. |
| `enable_milp` | `False` | Enables the MILP-based integer reconstruction phase. |
| `miqp_slack_mode` | `"none"` | Softens MIQP general equalities (`"eq"`), general inequalities (`"ineq_general"`), continuous box bounds (`"ineq_box"`), both inequality types (`"ineq_all"`), or all three (`"all"`); binaries, fixed variables, and mapped initial states stay hard. |
| `miqp_slack_penalty_norm` | `"l1"` | Selects `weight * sum(slacks)` (`"l1"`) or `0.5 * weight * sum(slacks**2)` (`"l2"`) for MIQP slacks. |
| `miqp_slack_penalty` | `1e6` | Sets the finite positive weight of the MIQP slack penalty. |
| `enable_pycombina` | `False` | Enables PyCombina reconstruction before or instead of the MIQP or MILP. |
| `binary_controls_one_hot` | `False` | Declares each interval's binary controls as one mutually exclusive group. |
| `hessian_epsilon` | `1e-6` | Sets the minimum eigenvalue used to regularize the MIQP Hessian. |
| `exploit_structure_in_hessian_cvx` | `False` | Convexifies verified multiple-shooting stage Hessian blocks independently. |
| `integrality_tol` | `1e-7` | Sets the Euclidean-distance tolerance for accepting rounded RNLP binaries. |
| `pycombina_binary_threshold` | `1e-6` | Clamps relaxed PyCombina weights near zero or one without removing modes; finite values in `[0, 0.5)` are accepted. |
| `ipopt_options` | quiet IPOPT output | Supplies options directly to the CasADi IPOPT solver used for RNLP and PNLP. |
| `gurobi_options` | `MIPGap=0`, `MIPGapAbs=1e-9` | Supplies CasADi Gurobi options with strict relative and absolute MIP gaps by default. |
| `daqp_options` | `{}` | Supplies options directly to the CasADi DAQP conic interface. |
| `highs_options` | `{}` | Supplies options directly to the CasADi HiGHS conic interface. |
| `pycombina_options` | `{}` | Configures PyCombina, including its time grid and switching constraints. |
| `n_x` | `None` | Gives the number of state variables at each prediction node. |
| `n_u` | `None` | Gives the number of continuous controls in each control interval. |
| `n_v` | `None` | Gives the number of binary controls in each control interval. |
| `N` | `None` | Gives the number of control intervals in the prediction horizon. |
| `M_o` | `None` | Keeps the first `M_o` intervals integer and fixes the remainder to the PyCombina solution. |
| `M_f` | `None` | Keeps the first `M_f` intervals integer and leaves the remaining `[0,1]` variables continuous in both MIP and PNLP. |
| `binary_intervals` | `None` | Maps each discrete decision-vector entry to its control-interval index. |
| `state_indices` | `None` | Maps every prediction-node state to its decision-vector indices. |
| `continuous_control_indices` | `None` | Maps every continuous control to its decision-vector indices by interval. |
| `initial_state_indices` | `None` | Identifies the decision-vector entries fixed to the measured initial state. |
| `shift_initial_guess_enabled` | `True` | Enables shifting of the previous trajectory before advanced-step preparation. |
| `solve_miqp_in_preparation` | `False` | Solves a predicted MIQP during the advanced-step preparation phase. |
| `solve_milp_in_preparation` | `False` | Solves a predicted MILP during the advanced-step preparation phase. |
| `solve_qp_after_milp` | `True` | Refines an advanced-step MILP solution with a fixed-integer QP. |
| `solve_pnlp_in_feedback` | `False` | Polishes the reconstructed solution with a fixed-binary NLP during feedback. |

## PyCombina settings

For one-hot reconstruction, `pycombina_options` also accepts:

- `valid_controls`: an `(N, n_v)` binary mask of allowed modes, with at least one allowed mode per interval. It can also be a callable of the complete relaxed decision vector. Together with `b_bin_pre` and `min_up_times`, it can enforce a remaining initial dwell time.
- `valid_control_transitions`: an `(n_v, n_v)` binary matrix whose row `i` lists the modes allowed after mode `i`. With `b_bin_pre`, it also restricts the initial transition.

`pycombina_binary_threshold` clamps input weights near zero or one while retaining all candidate modes. `integrality_tol` instead determines whether the relaxed solution is already integral enough to skip reconstruction. The legacy `pycombina_options["binary_threshold"]` overrides `pycombina_binary_threshold` when supplied.
