# Code structure

[Back to the README](../README.md) · [Options reference](../OPTIONS.md)

```text
src/mpc_implementation/
├── solver.py         Three-phase RNLP, integer reconstruction, and PNLP
├── mpc.py            Advanced-step preparation/feedback and trajectory shifting
├── options.py        Shared solver configuration
├── results.py        Result and quadratic-model data classes
└── camino_solver.py  Optional CAMINO adapter

examples/             Optimal control, MPC, and benchmark scripts
tests/                Unit and solver-path tests
```

Each example constructs a CasADi NLP, bounds, a discrete-variable mask, and trajectory index maps. It then calls `ThreePhaseSolver`, `AdvancedStepMPC`, or the optional `CaminoSolver` adapter.

- `ThreePhaseSolver` solves the relaxed NLP (RNLP), reconstructs the integer controls, and solves the polishing NLP (PNLP).
- `AdvancedStepMPC` separates preparation at a predicted state from feedback at the measured state and manages trajectory shifting.
- `ThreePhaseOptions` configures both solvers; see [OPTIONS.md](../OPTIONS.md).
- Result classes expose solutions, timings, and solver statistics.
- `CaminoSolver` provides an optional comparison with CAMINO.

```mermaid
classDiagram
    direction LR
    class ThreePhaseOptions
    class ThreePhaseSolver {
        +solve() ThreePhaseResult
        +prepare_quadratic_model() QuadraticModel
        +solve_mip()
        +solve_fixed_qp()
        +solve_pnlp()
    }
    class AdvancedStepMPC {
        +preparation_phase() PreparationResult
        +feedback_phase() FeedbackResult
        +shift_initial_guess()
    }
    class CaminoSolver {
        +solve() CaminoResult
    }
    class QuadraticModel
    class ThreePhaseResult
    class PreparationResult
    class FeedbackResult
    class CaminoResult

    ThreePhaseSolver --> ThreePhaseOptions
    ThreePhaseSolver --> ThreePhaseResult
    ThreePhaseSolver --> QuadraticModel
    AdvancedStepMPC *-- ThreePhaseSolver
    AdvancedStepMPC --> PreparationResult
    AdvancedStepMPC --> FeedbackResult
    PreparationResult *-- QuadraticModel
    CaminoSolver --> CaminoResult
```
