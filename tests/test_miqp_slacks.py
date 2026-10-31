import unittest

import casadi as ca
import numpy as np

from mpc_implementation import ThreePhaseOptions, ThreePhaseSolver


class MiqpSlackTests(unittest.TestCase):
    def make_arguments(self, mode, norm="l1"):
        solver = object.__new__(ThreePhaseSolver)
        solver.n_w, solver.n_g = 6, 5
        solver.discrete = np.array([False, True, True, False, False, False])
        solver.options = ThreePhaseOptions(
            miqp_slack_mode=mode, miqp_slack_penalty_norm=norm,
            miqp_slack_penalty=10.0, initial_state_indices=[0],
        )
        arguments = {
            "h": ca.DM.eye(6), "g": ca.DM.zeros(6, 1),
            "a": ca.DM(np.tile([0, 0, 0, 1, 0, 0], (5, 1))),
            "lba": ca.DM([2, 2, -np.inf, -1, -np.inf]),
            "uba": ca.DM([2, np.inf, 0, 0.5, np.inf]),
            "lbx": ca.DM([0, 0, 0, 0, -np.inf, 2]),
            "ubx": ca.DM([10, 1, 1, 1, np.inf, 2]),
            "x0": ca.DM([0, 1, 0, 3, 0, 2]),
        }
        return solver, arguments

    def test_modes_protect_fixed_binary_and_initial_state_bounds(self):
        for mode, count, boxes in (("none", 0, []), ("eq", 2, []),
                                   ("ineq_general", 4, []), ("ineq_box", 2, [3]),
                                   ("ineq_all", 6, [3]), ("all", 8, [3])):
            with self.subTest(mode=mode):
                solver, original = self.make_arguments(mode)
                before = {key: ca.DM(value) for key, value in original.items()}
                augmented, box = solver._slacken_miqp(original)
                self.assertEqual(augmented["g"].numel(), solver.n_w + count)
                np.testing.assert_array_equal(box, boxes)
                for key in original:
                    np.testing.assert_array_equal(np.asarray(original[key]), np.asarray(before[key]))
                for key in ("lbx", "ubx"):
                    np.testing.assert_array_equal(
                        np.asarray(augmented[key]).ravel()[[0, 1, 2, 5]],
                        np.asarray(original[key]).ravel()[[0, 1, 2, 5]],
                    )
                if boxes:
                    self.assertTrue(np.isneginf(float(augmented["lbx"][3])))
                    self.assertTrue(np.isposinf(float(augmented["ubx"][3])))
                guess = augmented["x0"]
                self.assertEqual(guess.numel(), solver.n_w + count)
                if count:
                    correction = np.asarray(augmented["a"])[:, solver.n_w:]
                    selected = np.any(correction != 0, axis=1)
                    values = np.asarray(augmented["a"] @ guess).ravel()[selected]
                    self.assertTrue(np.all(values >= np.asarray(augmented["lba"]).ravel()[selected]))
                    self.assertTrue(np.all(values <= np.asarray(augmented["uba"]).ravel()[selected]))

    def test_penalties_extend_only_the_slack_block(self):
        for norm in ("l1", "l2"):
            solver, original = self.make_arguments("all", norm)
            augmented, _ = solver._slacken_miqp(original)
            hessian = np.asarray(augmented["h"])
            np.testing.assert_array_equal(hessian[:6, :6], np.eye(6))
            np.testing.assert_array_equal(hessian[:6, 6:], 0)
            np.testing.assert_array_equal(hessian[6:, 6:], np.eye(8) * (0 if norm == "l1" else 10))
            np.testing.assert_array_equal(np.asarray(augmented["g"])[6:], 10 if norm == "l1" else 0)

    def test_daqp_complement_is_disabled_only_when_equalities_are_softened(self):
        solver = object.__new__(ThreePhaseSolver)
        solver.discrete = np.array([True, True])
        solver.binary_indices = np.array([0, 1])
        for mode in ("none", "eq", "ineq_general", "ineq_box", "ineq_all", "all"):
            solver.options = ThreePhaseOptions(
                binary_controls_one_hot=True, n_v=2, N=1,
                binary_intervals=[0, 0], miqp_slack_mode=mode,
            )
            self.assertEqual(solver._mip_discrete_mask(), [True, mode in ("eq", "all")])

    def test_partial_horizon_bounds_are_not_softened(self):
        solver, original = self.make_arguments("all")
        solver.binary_indices = np.array([1, 2])
        for mode in ("mf", "mo"):
            with self.subTest(mode=mode):
                solver.options = ThreePhaseOptions(
                    enable_pycombina=mode == "mo", n_x=1, n_u=0, n_v=1, N=2,
                    binary_intervals=[0, 1], M_f=1 if mode == "mf" else None,
                    M_o=1 if mode == "mo" else None, miqp_slack_mode="all",
                )
                arguments = {key: ca.DM(value) for key, value in original.items()}
                if mode == "mo":
                    arguments["lbx"][2] = arguments["ubx"][2] = 1
                augmented, _ = solver._slacken_miqp(arguments)
                self.assertEqual(float(augmented["lbx"][2]), 1 if mode == "mo" else 0)
                self.assertEqual(float(augmented["ubx"][2]), 1)
                self.assertEqual(solver._mip_discrete_mask()[2], mode == "mo")

    def build_solver(self, plugin, mode, norm="l1", equality=False):
        if not ca.has_conic(plugin):
            self.skipTest(f"CasADi {plugin} unavailable")
        w = ca.SX.sym("w", 2)
        options = ThreePhaseOptions(
            mip_solver=plugin, miqp_slack_mode=mode,
            miqp_slack_penalty_norm=norm, miqp_slack_penalty=10.0,
            gurobi_options={"gurobi": {"OutputFlag": 0, "LogToConsole": 0}},
        )
        solver = ThreePhaseSolver("slacks_" + plugin, {"x": w, "f": 0.5 * ca.sumsqr(w),
                                                     "g": w[1]}, [True, False], options)
        model, _, _ = solver.prepare_quadratic_model(
            {"x": ca.DM([0.5, 0.5]), "f": ca.DM(0.25), "lam_g": ca.DM([0.0])})
        bounds = (ca.DM([0.0, 0.0]), ca.DM([1.0, 1.0]), ca.DM([2.0]),
                  ca.DM([2.0 if equality else np.inf]))
        return solver, model, bounds

    def test_slack_solutions_and_objective_values(self):
        for plugin in ("gurobi", "daqp"):
            # Mode selection and penalty matrices are covered above; exercise
            # representative equality, box-bound, and combined solves here.
            for mode, norm in (("eq", "l1"), ("ineq_box", "l2"), ("all", "l1")):
                with self.subTest(plugin=plugin, mode=mode, norm=norm):
                    solver, model, bounds = self.build_solver(plugin, mode, norm, mode == "eq")
                    solution, _, _ = solver.solve_mip(
                        model, *bounds, warm_start=model.reference, log_call=False)
                    self.assertTrue(solver._mip_solver.stats()["success"])
                    self.assertEqual(solution["x"].numel(), 2)
                    self.assertEqual(solution["lam_x"].numel(), 2)
                    self.assertEqual(solution["lam_a"].numel(), 1)
                    self.assertAlmostEqual(float(solution["x"][0]), 0, places=5)
                    self.assertGreater(float(solution["slack_inf_norm"]), 0.1)
                    self.assertAlmostEqual(float(solution["cost"]),
                                           0.5 * float(ca.sumsqr(solution["x"])), places=4)
                    self.assertAlmostEqual(float(solution["penalized_cost"]),
                                           float(solution["cost"] + solution["slack_penalty"]), places=4)
                    if mode == "ineq_box":
                        self.assertGreaterEqual(float(solution["x"][1]), 2 - 1e-5)

    def test_default_path_returns_unaugmented_solution(self):
        for plugin in ("gurobi", "daqp"):
            with self.subTest(plugin=plugin):
                solver, model, bounds = self.build_solver(plugin, "none")
                bounds = (bounds[0], ca.DM([1.0, 3.0]), bounds[2], bounds[3])
                result, _, _ = solver.solve_mip(model, *bounds, warm_start=model.reference, log_call=False)
                np.testing.assert_allclose(np.asarray(result["x"]).ravel(), [0, 2], atol=1e-6)
                self.assertNotIn("slacks", result)
                self.assertNotIn("slack_inf_norm", solver._mip_stats(result))


if __name__ == "__main__":
    unittest.main()
