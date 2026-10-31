"""Native regressions for constrained one-hot reconstruction."""
import unittest

import numpy as np

try:
    from pycombina import BinApprox, CombinaBnB
except ImportError:
    BinApprox = CombinaBnB = None


@unittest.skipIf(BinApprox is None, "Native PyCombina is unavailable")
class PyCombinaRegressionTests(unittest.TestCase):
    def test_zero_weight_modes_remain_available_with_adjacency_and_bounds(self):
        for threshold in (1e-3, 1e-6, 1e-8):
            for reduce in (False, True):
                with self.subTest(threshold=threshold, reduce=reduce):
                    relaxed = np.zeros((6, 7))
                    relaxed[2] = 1.
                    relaxed[1, 0], relaxed[2, 0] = 5e-7, 1.-5e-7
                    problem = BinApprox(np.arange(8.), relaxed,
                        binary_threshold=threshold, reduce_problem_size_before_solve=reduce)
                    problem.set_b_bin_pre([0, 0, 1, 0, 0, 0])
                    transitions = (np.abs(np.arange(6)[:, None]-np.arange(6)) <= 1).astype(int)
                    for mode, allowed in enumerate(transitions):
                        problem.set_valid_control_transitions(mode, allowed)
                    # Gear 1 has exactly zero relaxed weight but is required here;
                    # gear 2 must bridge the transition from the previous gear 3.
                    problem.set_valid_controls_for_interval((2., 4.), [1, 0, 0, 0, 0, 0])
                    problem.set_min_up_times([2., 0., 0., 0., 0., 0.])
                    solver = CombinaBnB(problem)
                    self.assertEqual(solver._binapprox_p.n_c, 6)
                    solver.solve(verbosity=0)
                    binary = np.asarray(problem.b_bin)
                    np.testing.assert_array_equal(binary.sum(axis=0), np.ones(7))
                    self.assertTrue(np.all(np.isin(binary, (0, 1))))
                    self.assertTrue(np.all(binary[0, 2:4] == 1))
                    gears = np.r_[2, binary.argmax(axis=0)]
                    self.assertTrue(np.all(transitions[gears[:-1], gears[1:]] == 1))
                    self.assertEqual(solver.status, "Optimal solution found")

    def test_feasible_schedule_at_full_horizon_error_is_not_pruned(self):
        for strategy in CombinaBnB.get_search_strategies():
            with self.subTest(strategy=strategy):
                problem = BinApprox(np.arange(6.), [[1.]*5, [0.]*5])
                problem.set_valid_controls_for_interval((0., 5.), [0, 1])
                solver = CombinaBnB(problem)
                solver.solve(verbosity=0, strategy=strategy)
                np.testing.assert_array_equal(problem.b_bin, [[0]*5, [1]*5])
                self.assertEqual(problem.eta, 5.)
                self.assertEqual(solver.status, "Optimal solution found")

    def test_infeasible_problem_reports_failure_instead_of_zero_schedule(self):
        problem = BinApprox(np.arange(6.), [[1.]*5, [0.]*5])
        problem.set_b_bin_pre([1, 0])
        problem.set_valid_control_transitions(0, [1, 0])
        problem.set_valid_control_transitions(1, [0, 1])
        problem.set_valid_controls_for_interval((0., 5.), [0, 1])
        solver = CombinaBnB(problem)
        with self.assertRaisesRegex(RuntimeError, "Problem infeasible.*no feasible one-hot"):
            solver.solve(verbosity=0)
        self.assertEqual(solver.status, "Problem infeasible")
        self.assertTrue(np.isinf(solver._bnb_solver.get_eta()))

    def test_iteration_limit_without_incumbent_reports_failure(self):
        problem = BinApprox(np.arange(6.), [[0.6]*5, [0.4]*5])
        solver = CombinaBnB(problem)
        with self.assertRaisesRegex(RuntimeError, "Maximum number of iterations.*no feasible one-hot"):
            solver.solve(verbosity=0, max_iter=0)
        self.assertEqual(solver.status, "Maximum number of iterations exceeded")


if __name__ == "__main__":
    unittest.main()
