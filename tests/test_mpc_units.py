import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import casadi as ca
import numpy as np

from mpc_implementation import (
    AdvancedStepMPC,
    ThreePhaseOptions,
    ThreePhaseSolver,
    prepare_mpc_initial_guess,
    shift_mpc_trajectory,
)
from mpc_implementation.results import QuadraticModel


class FakeSolver:
    def stats(self):
        return {"success": True, "return_status": "mock"}


class MpcUnitTests(unittest.TestCase):
    def make_controller(
        self, binary, method="miqp", solve_qp_after_milp=True, M_o=None
    ):
        options = ThreePhaseOptions(
            enable_miqp=method == "miqp",
            enable_milp=method == "milp",
            enable_pycombina=M_o is not None,
            n_x=1,
            n_u=0,
            n_v=1,
            N=1,
            M_o=M_o,
            binary_intervals=[0],
            state_indices=[[1], [2]],
            continuous_control_indices=[],
            initial_state_indices=[1],
            solve_miqp_in_preparation=method == "miqp",
            solve_milp_in_preparation=method == "milp",
            solve_qp_after_milp=solve_qp_after_milp,
            integrality_tol=1e-7,
        )
        solver = object.__new__(ThreePhaseSolver)
        solver.options = options
        solver.binary_indices = np.array([0])
        solver._nlp_solver = FakeSolver()
        solver._mip_solver = FakeSolver()
        solver._fixed_qp_solver = FakeSolver()
        solver._last_pycombina_stats = {
            "success": True,
            "return_status": "mock",
            "cpu_time": 0.08,
        }
        solver._solve_rnlp = Mock(
            return_value=(
                {
                    "x": ca.DM([binary, 0.0, 0.0]),
                    "f": ca.DM(0.0),
                    "lam_g": ca.DM.zeros(0, 1),
                },
                0.01,
            )
        )
        model = QuadraticModel(
            ca.DM([binary, 0.0, 0.0]),
            0.0,
            ca.DM.zeros(3, 1),
            np.eye(3),
            ca.DM.zeros(0, 1),
            ca.DM.zeros(0, 3),
        )
        solver.prepare_quadratic_model = Mock(return_value=(model, 0.02, 0.03))
        solver.solve_mip = Mock(
            return_value=(
                {"x": ca.DM([1.0, 0.0, 0.0]), "cost": ca.DM(0.0)},
                0.04,
                0.05,
            )
        )
        solver.solve_fixed_qp = Mock(
            return_value=(
                {"x": ca.DM([0.0, 0.0, 0.0]), "cost": ca.DM(0.0)},
                0.06,
                0.07,
            )
        )
        solver.solve_pycombina = Mock(return_value=(np.array([1.0]), 0.08))
        solver.solve_pnlp = Mock()

        controller = object.__new__(AdvancedStepMPC)
        controller.options = options
        controller.solver = solver
        controller.integrator = None
        controller._preparation = None
        controller._last_solution = None
        controller._log_solve_summary = Mock()
        return controller

    def prepare(self, controller):
        return controller.preparation_phase(
            ca.DM.zeros(3, 1),
            ca.DM([-1.0, 0.0, -1.0]),
            ca.DM([1.0, 0.0, 1.0]),
            ca.DM.zeros(0, 1),
            ca.DM.zeros(0, 1),
            predicted_state=ca.DM([0.0]),
        )

    def test_integral_preparation_skips_mip_at_tolerance_boundary(self):
        for method in ("miqp", "milp"):
            with self.subTest(method=method):
                controller = self.make_controller(1e-7, method)
                preparation = self.prepare(controller)
                self.assertTrue(preparation.stats["integer_found_in_rnlp"])
                np.testing.assert_array_equal(preparation.rounded_rnlp_binary, [0])
                self.assertIsNotNone(preparation.quadratic_model.hessian)
                controller.solver.prepare_quadratic_model.assert_called_once_with(
                    preparation.rnlp_solution,
                    None,
                    require_hessian=True,
                )
                controller.solver.solve_mip.assert_not_called()

    def test_nonintegral_preparation_preserves_mip(self):
        controller = self.make_controller(0.5)
        preparation = self.prepare(controller)
        self.assertFalse(preparation.stats["integer_found_in_rnlp"])
        self.assertIsNone(preparation.rounded_rnlp_binary)
        controller.solver.solve_mip.assert_called_once()

    def test_preparation_passes_nominal_then_once_shifted_solution_to_rnlp(self):
        controller = self.make_controller(0.5)
        nominal = ca.DM([0.4, 10.0, 11.0])
        bounds = (
            ca.DM([-1.0, 0.0, -1.0]),
            ca.DM([1.0, 0.0, 1.0]),
            ca.DM.zeros(0, 1),
            ca.DM.zeros(0, 1),
        )
        controller.preparation_phase(
            nominal,
            *bounds,
            predicted_state=ca.DM([0.0]),
        )
        np.testing.assert_array_equal(
            np.asarray(controller.solver._solve_rnlp.call_args.args[0]).reshape(-1),
            np.asarray(nominal).reshape(-1),
        )

        controller.solver._solve_rnlp.reset_mock()
        previous_finalized = ca.DM([1.0, 20.0, 21.0])
        controller.preparation_phase(
            nominal,
            *bounds,
            predicted_state=ca.DM([0.0]),
            previous_solution=previous_finalized,
        )
        np.testing.assert_array_equal(
            np.asarray(controller.solver._solve_rnlp.call_args.args[0]).reshape(-1),
            [1.0, 21.0, 21.0],
        )

    def test_integral_feedback_uses_only_fixed_qp(self):
        for method in ("miqp", "milp"):
            with self.subTest(method=method):
                controller = self.make_controller(1e-8, method)
                controller.options.miqp_slack_mode = "all"
                preparation = self.prepare(controller)
                feedback = controller.feedback_phase(ca.DM([0.0]), preparation)
                self.assertTrue(feedback.stats["integer_found_in_rnlp"])
                self.assertIs(feedback.solution, feedback.fixed_qp_solution)
                self.assertTrue(feedback.stats[method]["skipped"])
                self.assertEqual(feedback.timings["fixed_qp_solution"], 0.07)
                self.assertEqual(feedback.timings[f"{method}_solution"], 0.0)
                controller.solver.solve_fixed_qp.assert_called_once()
                controller.solver.solve_mip.assert_not_called()
                controller.solver.solve_pycombina.assert_not_called()
                controller.solver.solve_pnlp.assert_not_called()

    def test_milp_feedback_refines_with_fixed_integer_qp(self):
        controller = self.make_controller(0.5, "milp")
        preparation = self.prepare(controller)
        controller.solver.solve_fixed_qp.reset_mock()

        feedback = controller.feedback_phase(ca.DM([0.0]), preparation)

        self.assertIs(feedback.solution, feedback.fixed_qp_solution)
        self.assertIsNotNone(feedback.milp_solution)
        self.assertEqual(feedback.timings["fixed_qp_construction"], 0.06)
        self.assertEqual(feedback.timings["fixed_qp_solution"], 0.07)
        call = controller.solver.solve_fixed_qp.call_args
        np.testing.assert_array_equal(call.args[1], [1.0])
        np.testing.assert_array_equal(
            np.asarray(call.kwargs["initial_guess"]).reshape(-1),
            [1.0, 0.0, 0.0],
        )
        self.assertTrue(call.kwargs["fix_mip_integers_only"])
        controller.solver.prepare_quadratic_model.assert_called_once_with(
            preparation.rnlp_solution,
            None,
            require_hessian=True,
        )

    def test_milp_feedback_can_skip_qp_refinement(self):
        controller = self.make_controller(
            0.5,
            "milp",
            solve_qp_after_milp=False,
        )
        preparation = self.prepare(controller)
        feedback = controller.feedback_phase(ca.DM([0.0]), preparation)

        self.assertIs(feedback.solution, feedback.milp_solution)
        self.assertIsNone(feedback.fixed_qp_solution)
        self.assertEqual(feedback.timings["fixed_qp_solution"], 0.0)
        controller.solver.solve_fixed_qp.assert_not_called()
        controller.solver.prepare_quadratic_model.assert_called_once_with(
            preparation.rnlp_solution,
            None,
            require_hessian=False,
        )

    def test_miqp_feedback_does_not_use_qp_refinement(self):
        controller = self.make_controller(0.5, "miqp")
        preparation = self.prepare(controller)
        feedback = controller.feedback_phase(ca.DM([0.0]), preparation)

        self.assertIs(feedback.solution, feedback.miqp_solution)
        self.assertIsNone(feedback.fixed_qp_solution)
        controller.solver.solve_fixed_qp.assert_not_called()
        feedback_call = controller.solver.solve_mip.call_args_list[-1]
        np.testing.assert_array_equal(
            np.asarray(feedback_call.args[6]).reshape(-1),
            np.asarray(preparation.predicted_miqp_solution["x"]).reshape(-1),
        )

    def test_m_o_zero_uses_preparation_pycombina_and_feedback_qp(self):
        controller = self.make_controller(0.5, "miqp", M_o=0)

        preparation = self.prepare(controller)

        np.testing.assert_array_equal(preparation.pycombina_binary, [1.0])
        controller.solver.solve_pycombina.assert_called_once()
        controller.solver.solve_mip.assert_not_called()

        feedback = controller.feedback_phase(ca.DM([0.0]), preparation)

        self.assertIs(feedback.solution, feedback.fixed_qp_solution)
        self.assertIsNone(feedback.miqp_solution)
        self.assertEqual(feedback.timings["miqp_solution"], 0.0)
        self.assertEqual(feedback.timings["fixed_qp_solution"], 0.07)
        controller.solver.solve_pycombina.assert_called_once()
        controller.solver.solve_mip.assert_not_called()
        fixed_qp_call = controller.solver.solve_fixed_qp.call_args
        np.testing.assert_array_equal(fixed_qp_call.args[1], [1.0])

    def test_shift_repeats_final_state_and_controls(self):
        controller = object.__new__(AdvancedStepMPC)
        controller.options = ThreePhaseOptions(
            n_x=1,
            n_u=1,
            n_v=1,
            N=2,
            binary_intervals=[0, 1],
            state_indices=[[0], [3], [6]],
            continuous_control_indices=[[1], [4]],
            initial_state_indices=[0],
        )
        controller.solver = SimpleNamespace(binary_indices=np.array([2, 5]))
        shifted = np.asarray(
            controller.shift_initial_guess(ca.DM([10, 20, 0, 11, 21, 1, 12]))
        ).reshape(-1)
        np.testing.assert_array_equal(shifted, [11, 21, 1, 12, 21, 1, 12])

    def test_shared_shift_handles_multidimensional_blocks_and_current_state(self):
        shifted = np.asarray(
            shift_mpc_trajectory(
                ca.DM(np.arange(100, 114)),
                [[9, 0], [4, 12], [2, 7]],
                [[11, 1], [5, 13]],
                [[8, 3], [10, 6]],
                current_state=[-1, -2],
            )
        ).reshape(-1)
        np.testing.assert_array_equal(
            shifted,
            [-2, 113, 102, 106, 102, 105, 106, 107, 110, -1, 110, 105, 107, 113],
        )

    def test_bonmin_initial_guess_respects_warm_start_switch(self):
        nominal = ca.DM([1, 2, 0, 3])
        previous = ca.DM([10, 20, 0, 30])
        arguments = (
            nominal,
            previous,
            [[0], [3]],
            [[1]],
            [2],
            [99],
        )
        cold = np.asarray(
            prepare_mpc_initial_guess(*arguments, warm_start_enabled=False)
        ).reshape(-1)
        warm = np.asarray(
            prepare_mpc_initial_guess(*arguments, warm_start_enabled=True)
        ).reshape(-1)
        first = np.asarray(
            prepare_mpc_initial_guess(
                nominal,
                None,
                [[0], [3]],
                [[1]],
                [2],
                [99],
                warm_start_enabled=True,
            )
        ).reshape(-1)
        np.testing.assert_array_equal(cold, np.asarray(nominal).reshape(-1))
        np.testing.assert_array_equal(first, np.asarray(nominal).reshape(-1))
        np.testing.assert_array_equal(warm, [99, 20, 0, 30])

if __name__ == "__main__":
    unittest.main()
