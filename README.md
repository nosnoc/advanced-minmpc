[![Tests](https://github.com/nosnoc/advanced-minmpc/actions/workflows/tests.yml/badge.svg)](https://github.com/nosnoc/advanced-minmpc/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CasADi](https://img.shields.io/badge/CasADi-3.6.7%2B-009999)](https://web.casadi.org/)
[![Paper](https://img.shields.io/badge/arXiv-pending-b31b1b)](#citation)

# Advanced-step nonlinear mixed-integer MPC

`advanced-minmpc` implements the advanced-step nonlinear mixed-integer model predictive control algorithms introduced in:

> Armin Nurkanović. **Real-Time Nonlinear Mixed-Integer Model Predictive Control With Horizon-Tail Relaxations.** [arXiv preprint](https://arxiv.org/abs/XXXX.XXXXX) (placeholder; identifier to be updated).

The method combines nonlinear optimization with discrete control reconstruction. Horizon-tail relaxations significantly reduce the combinatorial search while preserving an integer-feasible first control input. In the paper's numerical experiments, the resulting closed-loop performance typically differs only slightly or negligibly from optimization with integrality enforced over the full horizon.

## Algorithm

`ThreePhaseSolver` implements the three-phase decomposition:

1. **Continuous relaxation:** solve the NLP with binary controls relaxed to `[0, 1]` using [IPOPT](https://coin-or.github.io/Ipopt/).
2. **Integer reconstruction:** recover discrete controls through a mixed-integer quadratic program (MIQP), a mixed-integer linear program (MILP), or [PyCombina](https://github.com/adbuerger/pycombina). PyCombina can also provide a starting trajectory for the MIP.
3. **Nonlinear polishing:** solve an NLP with the reconstructed binary decisions fixed. With a feasibility-relaxed tail, only the integer prefix is fixed and the tail remains continuous.

`AdvancedStepMPC` implements AS-MINMPC in two phases:

1. **Preparation:** before the next measurement arrives, solve the relaxed NLP at a predicted state and prepare derivatives and the quadratic model.
2. **Feedback:** update the initial state with the measurement and solve the reconstruction MIP. An MIQP selects discrete controls and performs one sequential quadratic programming (SQP) step; a MILP can be followed by a fixed-integer QP. Nonlinear polishing is optional and disabled by default.

### Horizon-tail relaxations

MPC applies only the first control input. The algorithm therefore retains the full prediction horizon for the objective and constraints, while reducing the number of free binary decisions in the reconstruction problem:

- **Optimality relaxation (`M_o`):** optimize binary controls over the first `M_o` intervals and fix the tail to a binary trajectory supplied by PyCombina. The entire control sequence remains integer feasible.
- **Feasibility relaxation (`M_f`):** enforce integrality over the first `M_f` intervals and allow controls in the tail to vary continuously in `[0, 1]`. For `M_f >= 1`, the applied first input remains integer feasible, although the predicted tail may be fractional.

[![Full MIQP, feasibility relaxation with M_f = 8, and optimality relaxation with M_o = 3](src/horizon_relaxation_single_columns.png)](src/horizon_relaxation_single_columns.pdf)

*Horizon-tail approximations from the paper's parallel-chiller example. Shading marks the relaxed or fixed tail.*

For `n_v` binary controls per interval, either approximation reduces the number of free binary decisions from `N * n_v` to `M * n_v`, where `M` is the selected prefix length. This can substantially reduce combinatorial complexity and feedback computation time while retaining predictions over all `N` intervals.

## Installation

Use Python 3.10 or newer. The paper's experiments use CasADi 3.8.0. Clone the repository and run the following from its root in Windows PowerShell:

```powershell
git clone https://github.com/nosnoc/advanced-minmpc.git
cd advanced-minmpc
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
```

If you already have a local copy, start with the virtual-environment commands in that directory.

On Linux or macOS, use the corresponding shell commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

### Dependencies

[`requirements.txt`](requirements.txt) installs:

- [CasADi](https://web.casadi.org/) for symbolic models, automatic differentiation, and solver interfaces; [IPOPT](https://coin-or.github.io/Ipopt/) solves the NLPs through CasADi.
- [NumPy](https://numpy.org/) and [Matplotlib](https://matplotlib.org/) for numerical operations and plots.
- The bundled [PyCombina](https://github.com/adbuerger/pycombina) source in [`pycombina_win/`](pycombina_win), including Windows build adaptations and reconstruction fixes. Building it requires a C++ toolchain. On Windows, use MinGW-w64 or Visual Studio Build Tools with C++ support; see the [Windows build notes](pycombina_win/README_WINDOWS.md).

## Getting started

Solve a MINLP with the three-phase method using the [Lotka–Volterra example](examples/lotka_volterra/lotka_volterra_minlp.py):

```powershell
python examples/lotka_volterra/lotka_volterra_minlp.py
```

Run an MPC loop with AS-MINMPC using the [Lotka–Volterra MPC example](examples/lotka_volterra/lotka_volterra_mpc.py):

```powershell
python examples/lotka_volterra/lotka_volterra_mpc.py
```

Both examples use DAQP for MIQP reconstruction. The optional MILP comparisons use HiGHS through CasADi.

The [example library](examples/) also includes [parallel chillers](examples/parallel_chillers/), [fuel-efficient driving](examples/fuel_efficient_driving/), a [steerable needle](examples/steerable_needle/), [double-tank](examples/double_tank/), and [three-tank multimode](examples/three_tank_multimode/) systems, with optimal control, MPC, and benchmark scripts.

## Options

Both solvers use `ThreePhaseOptions`. The complete reference, including defaults, solver-specific settings, trajectory index maps, and switching constraints, is in **[OPTIONS.md](OPTIONS.md)**.

| Setting | Purpose |
|---|---|
| `mip_solver` | Choose `"daqp"`, `"gurobi"`, or `"highs"` (MILP only). |
| `enable_miqp`, `enable_milp`, `enable_pycombina` | Select reconstruction; MIQP and MILP are mutually exclusive, and either can be combined with PyCombina. |
| `M_o` | Fix the tail to PyCombina's binary trajectory; requires `enable_pycombina=True` and `0 <= M_o <= N`. |
| `M_f` | Relax tail integrality; requires `1 <= M_f <= N`. |
| `solve_pnlp_in_feedback` | Enable optional nonlinear polishing in advanced-step feedback (default: `False`). |

`M_o` and `M_f` are mutually exclusive. Leave both as `None` for full-horizon integer reconstruction, or set either to `N` to recover that horizon. Partial horizons require the problem dimensions and `binary_intervals` mapping; see the examples and [full options reference](OPTIONS.md).

## Solver backends and comparisons

MIQP and MILP reconstruction use CasADi conic plugins:

| Backend | Reconstruction | Setup |
|---|---|---|
| [DAQP](https://darnstrom.github.io/daqp/) | MIQP or MILP | Default backend; requires a CasADi build with the DAQP plugin and mixed-integer support. |
| [HiGHS](https://highs.dev/) | MILP | Requires the CasADi HiGHS plugin. |
| [Gurobi](https://docs.gurobi.com/projects/optimizer/en/current/) | MIQP or MILP | Optional; requires a compatible Gurobi installation, license, and CasADi Gurobi plugin. |

Check plugin availability in the active environment:

```powershell
python -c "import casadi as ca; print({s: ca.has_conic(s) for s in ('daqp', 'highs', 'gurobi')})"
```

[CAMINO](https://github.com/minlp-toolbox/CAMINO) is optional and used for comparisons. Install the pinned version separately:

```powershell
python -m pip install -r requirements-camino.txt
```

<details>
<summary>Configure Gurobi on Windows (optional)</summary>

Install Gurobi and configure a valid license following the [Gurobi documentation](https://docs.gurobi.com/projects/optimizer/en/current/). Install its Python interface if needed for comparisons or PyCombina's Gurobi-backed MILP solver:

```powershell
python -m pip install gurobipy
```

For a Gurobi 13.0 installation at `D:\gurobi1303`, check the DLL and set the following variables for the current PowerShell session. Adjust the path and version to your installation and CasADi build:

```powershell
Test-Path "D:\gurobi1303\win64\bin\gurobi130.dll"
$env:GUROBI_VERSION = "130"
$env:PATH = "D:\gurobi1303\win64\bin;$env:PATH"
```

To make these settings persistent, open **Edit environment variables for your account**, add `GUROBI_VERSION` with value `130`, and add the Gurobi `bin` directory to your user `Path`. Restart the terminal afterwards.

Verify from the active virtual environment:

```powershell
python -c "import os; print(os.environ.get('GUROBI_VERSION')); import casadi as ca; print(ca.has_conic('gurobi'))"
```

For this configuration, the output should contain `130` and `True`.

</details>

## Development

See the [code structure and solver interfaces](docs/CODE_STRUCTURE.md) for implementation details. Run the tests from the repository root with the environment configured as above:

```powershell
python -m unittest discover -s tests -v
```

## Citation

If you use this package in research, please cite the accompanying paper. The arXiv identifier below is a placeholder and will be updated after upload.

```bibtex
@article{Nurkanovic2026AdvancedMINMPC,
  title         = {Real-Time Nonlinear Mixed-Integer Model Predictive Control With Horizon-Tail Relaxations},
  author        = {Nurkanovi{\'c}, Armin},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv}
}
```

## Contact

Questions, remarks, bug reports, and feature requests are best submitted via a [new issue](https://github.com/nosnoc/advanced-minmpc/issues/new) in this repository.

- [Armin Nurkanović](https://www.syscop.de/people/armin-nurkanovic) — [armin.nurkanovic@imtek.uni-freiburg.de](mailto:armin.nurkanovic@imtek.uni-freiburg.de)

## Acknowledgments

OpenAI's Codex and ChatGPT assisted in the implementation of this package.
