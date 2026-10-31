import unittest

import casadi as ca
import numpy as np

from mpc_implementation import ThreePhaseOptions, ThreePhaseSolver, regularize_hessian


class HessianStructureTests(unittest.TestCase):
    def test_structured_matches_dense_for_noncontiguous_blocks(self):
        hessian = np.array(
            [
                [-2.0, 0.0, 1.0, 0.0],
                [0.0, 3.0, 0.0, -0.5],
                [1.0, 0.0, 2.0, 0.0],
                [0.0, -0.5, 0.0, -1.0],
            ]
        )
        dense = regularize_hessian(hessian, 0.1)
        structured = regularize_hessian(hessian, 0.1, [[0, 2], [1, 3]])

        np.testing.assert_allclose(structured, dense, atol=1e-12)
        np.testing.assert_allclose(structured, structured.T, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(structured).min(), 0.1 - 1e-12)

    def test_non_block_diagonal_hessian_falls_back_to_dense(self):
        hessian = np.array([[-1.0, 0.25], [0.25, 2.0]])
        dense = regularize_hessian(hessian, 0.1)

        with self.assertLogs("mpc_implementation.solver", level="WARNING") as logs:
            structured = regularize_hessian(hessian, 0.1, [[0], [1]])

        np.testing.assert_allclose(structured, dense, atol=1e-12)
        self.assertIn("nonzero off-block entries", " ".join(logs.output))

    def test_stage_blocks_follow_index_mappings(self):
        solver = object.__new__(ThreePhaseSolver)
        solver.n_w = 7
        solver.binary_indices = np.array([5, 2])
        solver.options = ThreePhaseOptions(
            n_x=1,
            n_u=1,
            n_v=1,
            N=2,
            state_indices=[[3], [0], [6]],
            continuous_control_indices=[[4], [1]],
            binary_intervals=[1, 0],
        )

        blocks = solver._multiple_shooting_hessian_blocks()

        np.testing.assert_array_equal(blocks[0], [3, 4, 2])
        np.testing.assert_array_equal(blocks[1], [0, 1, 5])
        np.testing.assert_array_equal(blocks[2], [6])

    def _build_solver(self, plugin, structured, suffix):
        w = ca.SX.sym("w", 7)
        x_0, u_0, v_0, x_1, u_1, v_1, x_2 = ca.vertsplit(w)
        stage_0 = (x_0 + u_0 + v_0 - 0.7) ** 2
        stage_1 = (x_1 + u_1 + v_1 - 1.1) ** 2
        objective = stage_0 + stage_1 + 0.1 * ca.dot(w, w) + (x_2 - 1.0) ** 2
        constraints = ca.vertcat(
            x_1 - x_0 - u_0 - v_0,
            x_2 - x_1 - u_1 - v_1,
        )
        options = ThreePhaseOptions(
            mip_solver=plugin,
            exploit_structure_in_hessian_cvx=structured,
            hessian_epsilon=1e-8,
            n_x=1,
            n_u=1,
            n_v=1,
            N=2,
            state_indices=[[0], [3], [6]],
            continuous_control_indices=[[1], [4]],
            initial_state_indices=[0],
            binary_intervals=[0, 1],
            gurobi_options={"gurobi": {"OutputFlag": 0, "LogToConsole": 0}},
        )
        solver = ThreePhaseSolver(
            f"hessian_structure_{plugin}_{suffix}",
            {"x": w, "f": objective, "g": constraints},
            [False, False, True, False, False, True, False],
            options,
        )
        reference = ca.DM([0.0, 0.2, 0.4, 0.6, -0.1, 0.4, 0.9])
        rnlp = {
            "x": reference,
            "f": ca.Function("objective", [w], [objective])(reference),
            "lam_g": ca.DM.zeros(2, 1),
        }
        bounds = (
            ca.DM([0.0, -1.0, 0.0, -3.0, -1.0, 0.0, -3.0]),
            ca.DM([0.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0]),
            ca.DM.zeros(2, 1),
            ca.DM.zeros(2, 1),
        )
        return solver, rnlp, bounds, ca.Function(
            f"objective_{plugin}_{suffix}", [w], [objective]
        )

    def _assert_solver_equivalence(self, plugin):
        if not ca.has_conic(plugin):
            self.skipTest(f"CasADi {plugin} conic plugin is unavailable")
        try:
            dense_solver, rnlp, bounds, objective = self._build_solver(
                plugin, False, "dense"
            )
            structured_solver, _, _, _ = self._build_solver(
                plugin, True, "structured"
            )
            dense_model, _, _ = dense_solver.prepare_quadratic_model(rnlp)
            structured_model, _, _ = structured_solver.prepare_quadratic_model(rnlp)
            dense_mip, _, _ = dense_solver.solve_mip(
                dense_model, *bounds, warm_start=rnlp["x"], log_call=False
            )
            structured_mip, _, _ = structured_solver.solve_mip(
                structured_model, *bounds, warm_start=rnlp["x"], log_call=False
            )
        except RuntimeError as error:
            if plugin == "gurobi" and "license" in str(error).lower():
                self.skipTest(f"Gurobi licence unavailable: {error}")
            raise

        np.testing.assert_allclose(
            structured_model.hessian, dense_model.hessian, atol=1e-10
        )
        dense_x = np.asarray(dense_mip["x"], dtype=float).reshape(-1)
        structured_x = np.asarray(structured_mip["x"], dtype=float).reshape(-1)
        np.testing.assert_allclose(structured_x, dense_x, atol=1e-7)
        np.testing.assert_array_equal(
            np.rint(structured_x[[2, 5]]), np.rint(dense_x[[2, 5]])
        )
        self.assertAlmostEqual(
            float(dense_mip["cost"]), float(structured_mip["cost"]), places=8
        )
        self.assertAlmostEqual(
            float(objective(dense_mip["x"])),
            float(objective(structured_mip["x"])),
            places=8,
        )

        binary = np.rint(dense_x[[2, 5]])
        dense_qp, _, _ = dense_solver.solve_fixed_qp(
            dense_model, binary, *bounds, initial_guess=dense_mip["x"]
        )
        structured_qp, _, _ = structured_solver.solve_fixed_qp(
            structured_model, binary, *bounds, initial_guess=structured_mip["x"]
        )
        np.testing.assert_allclose(
            np.asarray(structured_qp["x"], dtype=float),
            np.asarray(dense_qp["x"], dtype=float),
            atol=1e-7,
        )
        self.assertAlmostEqual(
            float(objective(dense_qp["x"])),
            float(objective(structured_qp["x"])),
            places=8,
        )

    def test_gurobi_small_mpc_matches_dense(self):
        self._assert_solver_equivalence("gurobi")

    def test_daqp_small_mpc_matches_dense(self):
        self._assert_solver_equivalence("daqp")


if __name__ == "__main__":
    unittest.main()
