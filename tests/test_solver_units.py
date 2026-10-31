import unittest
from unittest.mock import Mock, patch
import sys
from types import SimpleNamespace

import casadi as ca
import numpy as np

from mpc_implementation import ThreePhaseOptions, ThreePhaseSolver, regularize_hessian
from mpc_implementation.results import QuadraticModel
from mpc_implementation.solver import check_integrality


class FakeNlpSolver:
    def __init__(self):
        self.arguments = None

    def __call__(self, **kwargs):
        self.arguments = kwargs
        return {"x": kwargs["x0"], "f": ca.DM(0)}

    def stats(self):
        return {"success": True, "return_status": "mock"}


class FakeBinApprox:
    instances = []

    def __init__(self, time_grid, relaxed, **kwargs):
        self.time_grid = np.asarray(time_grid)
        values = np.asarray(relaxed)
        self.relaxed = values.T if values.shape[0] == self.time_grid.size - 1 else values
        self.b_bin = None
        self.options = kwargs
        self.set_values = {}
        self.instances.append(self)

    def __getattr__(self, name):
        if name.startswith("set_"):
            return lambda value: self.set_values.__setitem__(name, value)
        raise AttributeError(name)


class FakeCombinaBnB:
    def __init__(self, approximation):
        self.approximation = approximation

    def solve(self, **kwargs):
        self.approximation.b_bin = np.rint(self.approximation.relaxed)


class SolverUnitTests(unittest.TestCase):
    def test_integrality_check_includes_tolerance_boundary(self):
        tolerance = 1e-7
        integral, rounded = check_integrality([tolerance], tolerance)
        self.assertTrue(integral)
        np.testing.assert_array_equal(rounded, [0])
        outside, _ = check_integrality(
            [np.nextafter(tolerance, np.inf)], tolerance
        )
        self.assertFalse(outside)

    def test_three_phase_integral_rnlp_at_boundary_skips_remaining_phases(self):
        solver = object.__new__(ThreePhaseSolver)
        solver.options = ThreePhaseOptions(integrality_tol=1e-7)
        solver.binary_indices = np.array([0])
        solver._nlp_solver = FakeNlpSolver()
        solver._solve_rnlp = Mock(
            return_value=({"x": ca.DM([1e-7]), "f": ca.DM(2.0)}, 0.01)
        )
        solver.prepare_quadratic_model = Mock()
        solver.solve_mip = Mock()
        solver.solve_pnlp = Mock()
        solver._log_integral_rnlp_summary = Mock()
        result = solver.solve(
            ca.DM([0.0]),
            ca.DM([0.0]),
            ca.DM([1.0]),
            ca.DM.zeros(0, 1),
            ca.DM.zeros(0, 1),
        )
        self.assertTrue(result.stats["integer_found_in_rnlp"])
        self.assertEqual(float(result.solution["x"]), 0.0)
        self.assertTrue(result.stats["miqp"]["skipped"])
        self.assertTrue(result.stats["pnlp"]["skipped"])
        self.assertEqual(result.timings["miqp_solution"], 0.0)
        self.assertEqual(result.timings["pnlp_solver"], 0.0)
        solver.prepare_quadratic_model.assert_not_called()
        solver.solve_mip.assert_not_called()
        solver.solve_pnlp.assert_not_called()

    def test_three_phase_nonintegral_rnlp_preserves_all_phases(self):
        solver = object.__new__(ThreePhaseSolver)
        solver.options = ThreePhaseOptions(integrality_tol=1e-7)
        solver.binary_indices = np.array([0])
        solver._last_pycombina_stats = None
        solver._nlp_solver = FakeNlpSolver()
        solver._mip_solver = FakeNlpSolver()
        solver._solve_rnlp = Mock(
            return_value=({"x": ca.DM([0.5]), "f": ca.DM(1.0)}, 0.01)
        )
        model = QuadraticModel(
            ca.DM([0.5]),
            1.0,
            ca.DM([0.0]),
            np.eye(1),
            ca.DM.zeros(0, 1),
            ca.DM.zeros(0, 1),
        )
        solver.prepare_quadratic_model = Mock(return_value=(model, 0.02, 0.03))
        solver.solve_mip = Mock(
            return_value=({"x": ca.DM([1.0]), "cost": ca.DM(1.0)}, 0.04, 0.05)
        )
        solver.solve_pnlp = Mock(
            return_value=({"x": ca.DM([1.0]), "f": ca.DM(1.0)}, 0.06)
        )
        solver._log_three_phase_summary = Mock()
        result = solver.solve(
            ca.DM([0.0]),
            ca.DM([0.0]),
            ca.DM([1.0]),
            ca.DM.zeros(0, 1),
            ca.DM.zeros(0, 1),
        )
        self.assertFalse(result.stats["integer_found_in_rnlp"])
        solver.prepare_quadratic_model.assert_called_once()
        solver.solve_mip.assert_called_once()
        solver.solve_pnlp.assert_called_once()

    def test_hessian_is_symmetric_positive_definite(self):
        raw = np.array([[1.0, 4.0], [-2.0, -3.0]])
        regularized = regularize_hessian(raw, 0.25)
        np.testing.assert_allclose(regularized, regularized.T, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(regularized).min(), 0.25 - 1e-12)

    def test_binary_mapping_length_is_checked(self):
        solver = object.__new__(ThreePhaseSolver)
        solver.options = ThreePhaseOptions(
            enable_pycombina=True,
            n_v=1,
            N=2,
            binary_intervals=[0],
        )
        solver.binary_indices = np.array([1, 3])
        with self.assertRaisesRegex(ValueError, "one entry"):
            solver._validate_binary_mapping()

    def test_pnlp_fixes_every_binary_bound(self):
        solver = object.__new__(ThreePhaseSolver)
        solver.binary_indices = np.array([1, 3])
        solver.options = ThreePhaseOptions()
        solver.p = None
        solver._nlp_solver = FakeNlpSolver()
        solver.solve_pnlp(
            ca.DM([0, 1, 2, 0]),
            np.array([1.0, 0.0]),
            ca.DM([-2, 0, -2, 0]),
            ca.DM([2, 1, 2, 1]),
            ca.DM.zeros(0, 1),
            ca.DM.zeros(0, 1),
        )
        np.testing.assert_array_equal(
            np.asarray(solver._nlp_solver.arguments["lbx"]).reshape(-1)[[1, 3]],
            [1, 0],
        )
        np.testing.assert_array_equal(
            np.asarray(solver._nlp_solver.arguments["ubx"]).reshape(-1)[[1, 3]],
            [1, 0],
        )

    def test_pnlp_with_m_f_fixes_only_integer_head(self):
        solver = object.__new__(ThreePhaseSolver)
        solver.binary_indices = np.array([0, 2, 3])
        solver.options = ThreePhaseOptions(
            n_x=1,
            n_u=0,
            n_v=1,
            N=3,
            M_f=1,
            binary_intervals=[0, 1, 2],
        )
        solver.p = None
        solver._nlp_solver = FakeNlpSolver()
        solver.solve_pnlp(
            ca.DM([1.0, 0.0, 0.4, 0.7]),
            np.array([1.0, 0.4, 0.7]),
            ca.DM([0, -2, 0, 0]),
            ca.DM([1, 2, 1, 1]),
            ca.DM.zeros(0, 1),
            ca.DM.zeros(0, 1),
        )
        lower = np.asarray(solver._nlp_solver.arguments["lbx"]).reshape(-1)
        upper = np.asarray(solver._nlp_solver.arguments["ubx"]).reshape(-1)
        np.testing.assert_array_equal(lower[[0, 2, 3]], [1, 0, 0])
        np.testing.assert_array_equal(upper[[0, 2, 3]], [1, 1, 1])

    def test_fixed_qp_with_m_f_fixes_only_milp_integer_head(self):
        solver = object.__new__(ThreePhaseSolver)
        solver.n_g = 0
        solver.n_w = 4
        solver.binary_indices = np.array([0, 2, 3])
        solver.options = ThreePhaseOptions(
            enable_miqp=False,
            enable_milp=True,
            n_x=1,
            n_u=0,
            n_v=1,
            N=3,
            M_f=1,
            binary_intervals=[0, 1, 2],
        )
        solver._fixed_qp_solver = FakeNlpSolver()
        model = QuadraticModel(
            ca.DM.zeros(4, 1),
            0.0,
            ca.DM.zeros(4, 1),
            np.eye(4),
            ca.DM.zeros(0, 1),
            ca.DM.zeros(0, 4),
        )
        solver.solve_fixed_qp(
            model,
            np.array([1.0, 0.4, 0.7]),
            ca.DM([0, -2, 0, 0]),
            ca.DM([1, 2, 1, 1]),
            ca.DM.zeros(0, 1),
            ca.DM.zeros(0, 1),
            initial_guess=ca.DM([1.0, 0.0, 0.4, 0.7]),
            fix_mip_integers_only=True,
        )
        lower = np.asarray(solver._fixed_qp_solver.arguments["lbx"]).reshape(-1)
        upper = np.asarray(solver._fixed_qp_solver.arguments["ubx"]).reshape(-1)
        np.testing.assert_array_equal(lower[[0, 2, 3]], [1, 0, 0])
        np.testing.assert_array_equal(upper[[0, 2, 3]], [1, 1, 1])

    def test_m_o_fixes_only_tail_to_pycombina(self):
        solver = object.__new__(ThreePhaseSolver)
        solver.n_w = 4
        solver.options = ThreePhaseOptions(
            enable_pycombina=True,
            n_x=1,
            n_u=0,
            n_v=1,
            N=3,
            M_o=1,
            binary_intervals=[0, 1, 2],
        )
        solver.binary_indices = np.array([0, 1, 2])
        model = QuadraticModel(
            ca.DM.zeros(4, 1),
            0.0,
            ca.DM.zeros(4, 1),
            np.eye(4),
            ca.DM.zeros(0, 1),
            ca.DM.zeros(0, 4),
        )
        arguments = solver._canonical_mip(
            model,
            ca.DM([0, 0, 0, -5]),
            ca.DM([1, 1, 1, 5]),
            ca.DM.zeros(0, 1),
            ca.DM.zeros(0, 1),
            np.array([1, 0, 1]),
        )
        lower = np.asarray(arguments["lbx"]).reshape(-1)
        upper = np.asarray(arguments["ubx"]).reshape(-1)
        self.assertEqual((lower[0], upper[0]), (0, 1))
        np.testing.assert_array_equal(lower[1:3], [0, 1])
        np.testing.assert_array_equal(upper[1:3], [0, 1])

    def test_m_f_relaxes_only_tail(self):
        solver = object.__new__(ThreePhaseSolver)
        solver.discrete = np.array([True, False, True, True])
        solver.binary_indices = np.array([0, 2, 3])
        solver.options = ThreePhaseOptions(
            n_x=1,
            n_u=0,
            n_v=1,
            N=3,
            M_f=1,
            binary_intervals=[0, 1, 2],
        )
        self.assertEqual(solver._mip_discrete_mask(), [True, False, False, False])

    def test_daqp_one_hot_uses_last_mode_as_continuous_complement(self):
        solver = object.__new__(ThreePhaseSolver)
        solver.discrete = np.ones(6, dtype=bool)
        solver.binary_indices = np.arange(6)
        solver.options = ThreePhaseOptions(
            binary_controls_one_hot=True,
            n_v=3,
            N=2,
            binary_intervals=[0, 0, 0, 1, 1, 1],
        )
        self.assertEqual(
            solver._mip_discrete_mask(),
            [True, True, False, True, True, False],
        )

    def test_milp_uses_gradient_and_zero_hessian(self):
        solver = object.__new__(ThreePhaseSolver)
        solver.n_w = 2
        solver.options = ThreePhaseOptions(enable_miqp=False, enable_milp=True)
        solver.binary_indices = np.array([0])
        model = QuadraticModel(
            ca.DM([0.25, 0.5]),
            2.0,
            ca.DM([3.0, 4.0]),
            None,
            ca.DM([1.0]),
            ca.DM([[2.0, 1.0]]),
        )
        arguments = solver._canonical_mip(
            model,
            ca.DM([0.0, -1.0]),
            ca.DM([1.0, 2.0]),
            ca.DM([0.0]),
            ca.DM([0.0]),
            None,
        )
        np.testing.assert_array_equal(np.asarray(arguments["h"]), np.zeros((2, 2)))
        np.testing.assert_array_equal(
            np.asarray(arguments["g"]).reshape(-1), [3.0, 4.0]
        )

    def test_milp_model_skips_hessian_evaluation(self):
        solver = object.__new__(ThreePhaseSolver)
        solver.options = ThreePhaseOptions(enable_miqp=False, enable_milp=True)
        solver.p = None
        solver.n_g = 1
        solver.n_w = 2
        solver._objective_gradient = lambda *_args: ca.DM([2.0, 3.0])
        solver._constraint_linearization = lambda *_args: ca.DM([4.0, 5.0, 6.0])
        solver._lagrangian_hessian = lambda *_args: self.fail(
            "MILP must not evaluate the Hessian"
        )
        model, _, hessian_time = solver.prepare_quadratic_model(
            {"x": ca.DM([0.0, 0.0]), "f": ca.DM(1.0)}
        )
        self.assertIsNone(model.hessian)
        self.assertEqual(hessian_time, 0.0)

    def test_pycombina_result_orientation_and_interval_mapping(self):
        solver = object.__new__(ThreePhaseSolver)
        solver.binary_indices = np.array([1, 2, 4, 5])
        solver.options = ThreePhaseOptions(
            enable_miqp=False,
            enable_pycombina=True,
            n_v=2,
            N=2,
            binary_intervals=[0, 0, 1, 1],
            pycombina_options={"n_max_switches": [1, 2]},
        )
        fake_module = SimpleNamespace(BinApprox=FakeBinApprox, CombinaBnB=FakeCombinaBnB)
        FakeBinApprox.instances.clear()
        reference = ca.DM([9, 0.2, 0.8, 9, 0.7, 0.1])
        with patch.dict(sys.modules, {"pycombina": fake_module}):
            binary, _ = solver.solve_pycombina(reference)
        np.testing.assert_array_equal(binary, [0, 1, 1, 0])
        self.assertEqual(
            FakeBinApprox.instances[0].set_values["set_n_max_switches"], [1, 2]
        )

    def test_pycombina_binary_threshold_forwarding(self):
        fake_module = SimpleNamespace(BinApprox=FakeBinApprox, CombinaBnB=FakeCombinaBnB)
        for one_hot in (False, True):
            for threshold in (1e-6, 1e-8):
                for override in (None, 1e-5):
                    with self.subTest(one_hot=one_hot, threshold=threshold, override=override):
                        solver = object.__new__(ThreePhaseSolver)
                        solver.binary_indices = np.arange(6)
                        solver.options = ThreePhaseOptions(
                            enable_miqp=False, enable_pycombina=True,
                            binary_controls_one_hot=one_hot, n_v=3, N=2,
                            binary_intervals=[0, 0, 0, 1, 1, 1],
                            pycombina_binary_threshold=threshold,
                            pycombina_options={} if override is None else {"binary_threshold": override})
                        FakeBinApprox.instances.clear()
                        with patch.dict(sys.modules, {"pycombina": fake_module}):
                            solver.solve_pycombina(ca.DM([0.6, 0.3, 0.1, 0.2, 0.7, 0.1]))
                        self.assertTrue(FakeBinApprox.instances)
                        for approximation in FakeBinApprox.instances:
                            self.assertEqual(approximation.options["binary_threshold"],
                                             threshold if override is None else override)

    def test_pycombina_one_hot_reconstructs_modes_jointly(self):
        solver = object.__new__(ThreePhaseSolver)
        solver.binary_indices = np.arange(6)
        solver.options = ThreePhaseOptions(
            enable_miqp=False,
            enable_pycombina=True,
            binary_controls_one_hot=True,
            n_v=3,
            N=2,
            binary_intervals=[0, 0, 0, 1, 1, 1],
        )
        fake_module = SimpleNamespace(BinApprox=FakeBinApprox, CombinaBnB=FakeCombinaBnB)
        FakeBinApprox.instances.clear()
        with patch.dict(sys.modules, {"pycombina": fake_module}):
            binary, _ = solver.solve_pycombina(
                ca.DM([0.6, 0.3, 0.1, 0.2, 0.7, 0.1])
            )
        np.testing.assert_array_equal(binary, [1, 0, 0, 0, 1, 0])
        self.assertEqual(len(FakeBinApprox.instances), 1)

    def test_pycombina_one_hot_passes_carried_dwell_admissibility(self):
        solver = object.__new__(ThreePhaseSolver)
        solver.binary_indices = np.arange(6)
        solver.options = ThreePhaseOptions(
            enable_miqp=False, enable_pycombina=True, binary_controls_one_hot=True,
            n_v=2, N=3, binary_intervals=[0, 0, 1, 1, 2, 2],
            pycombina_options={"time_grid": [0., 0.1, 0.2, 0.3],
                              "b_bin_pre": [0, 1],
                              "valid_controls": [[0, 1], [0, 1], [1, 1]]},
        )
        approximation = FakeBinApprox([0, 1, 2, 3], [[0, 1], [0, 1], [1, 0]])
        approximation.set_valid_controls_for_interval = Mock()
        fake_module = SimpleNamespace(BinApprox=Mock(return_value=approximation),
                                      CombinaBnB=FakeCombinaBnB)
        with patch.dict(sys.modules, {"pycombina": fake_module}):
            solver.solve_pycombina(ca.DM([0, 1, 0, 1, 1, 0]))
        calls = approximation.set_valid_controls_for_interval.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args[0], (0., 0.1))
        self.assertEqual(calls[1].args[0], (0.1, 0.2))
        np.testing.assert_array_equal(calls[0].args[1], [0, 1])
        self.assertEqual(approximation.set_values["set_b_bin_pre"], [0, 1])
        solver.options.pycombina_options["valid_controls"] = [[0, 0]]*3
        with patch.dict(sys.modules, {"pycombina": fake_module}):
            with self.assertRaisesRegex(ValueError, "admissibility mask"):
                solver.solve_pycombina(ca.DM([0, 1, 0, 1, 1, 0]))

        mask = Mock(return_value=[[0, 1], [0, 1], [1, 1]])
        solver.options.pycombina_options["valid_controls"] = mask
        reference = ca.DM([0, 1, 0, 1, 1, 0])
        with patch.dict(sys.modules, {"pycombina": fake_module}):
            solver.solve_pycombina(reference)
        mask.assert_called_once()
        np.testing.assert_array_equal(mask.call_args.args[0], reference)
        self.assertIs(solver.options.pycombina_options["valid_controls"], mask)

    def test_pycombina_one_hot_forwards_directional_transitions(self):
        solver = object.__new__(ThreePhaseSolver)
        solver.binary_indices = np.arange(6)
        # Asymmetric transitions detect accidental transposition of the API input.
        transitions = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
        solver.options = ThreePhaseOptions(
            enable_miqp=False, enable_pycombina=True, binary_controls_one_hot=True,
            n_v=3, N=2, binary_intervals=[0, 0, 0, 1, 1, 1],
            pycombina_options={"b_bin_pre": [1, 0, 0], "valid_control_transitions": transitions})
        approximation = FakeBinApprox([0, 1, 2], [[1, 0, 0], [0, 1, 0]])
        approximation.set_valid_control_transitions = Mock()
        fake_module = SimpleNamespace(BinApprox=Mock(return_value=approximation), CombinaBnB=FakeCombinaBnB)
        with patch.dict(sys.modules, {"pycombina": fake_module}):
            solver.solve_pycombina(ca.DM([1, 0, 0, 0, 1, 0]))
        calls = approximation.set_valid_control_transitions.call_args_list
        self.assertEqual(len(calls), 3)
        for i, call in enumerate(calls):
            self.assertEqual(call.args[0], i)
            np.testing.assert_array_equal(call.args[1], transitions[i])
        solver.options.pycombina_options["valid_control_transitions"] = [[1, 2, 0]]*3
        with patch.dict(sys.modules, {"pycombina": fake_module}):
            with self.assertRaisesRegex(ValueError, "binary matrix"):
                solver.solve_pycombina(ca.DM([1, 0, 0, 0, 1, 0]))

if __name__ == "__main__":
    unittest.main()
