import unittest

from mpc_implementation import ThreePhaseOptions


class OptionValidationTests(unittest.TestCase):
    def test_rejects_invalid_miqp_slack_options(self):
        for kwargs in ({"miqp_slack_mode": "both"}, {"miqp_slack_penalty_norm": "bad"},
                       {"miqp_slack_penalty": 0}, {"miqp_slack_penalty": -1},
                       {"miqp_slack_penalty": float("inf")}, {"miqp_slack_penalty": float("nan")},
                       {"enable_miqp": False, "enable_milp": True, "miqp_slack_mode": "all"}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "slack"):
                    ThreePhaseOptions(**kwargs)

    def test_gurobi_defaults_are_merged_with_user_options(self):
        options = ThreePhaseOptions(
            gurobi_options={"gurobi": {"OutputFlag": 0}}
        )

        self.assertEqual(options.gurobi_options["gurobi"]["MIPGap"], 0.0)
        self.assertEqual(options.gurobi_options["gurobi"]["MIPGapAbs"], 1e-9)
        self.assertEqual(options.gurobi_options["gurobi"]["OutputFlag"], 0)

    def test_rejects_negative_integrality_tolerance(self):
        with self.assertRaisesRegex(ValueError, "integrality_tol"):
            ThreePhaseOptions(integrality_tol=-1e-7)

    def test_rejects_invalid_pycombina_binary_thresholds(self):
        for threshold in (-1e-8, 0.5, 1., float("nan"), float("inf")):
            for kwargs in ({"pycombina_binary_threshold": threshold},
                           {"pycombina_options": {"binary_threshold": threshold}}):
                with self.subTest(kwargs=kwargs):
                    with self.assertRaisesRegex(ValueError, "threshold"):
                        ThreePhaseOptions(**kwargs)

    def test_rejects_disabled_reconstruction(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            ThreePhaseOptions(enable_miqp=False, enable_pycombina=False)

    def test_rejects_incompatible_partial_horizons(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            ThreePhaseOptions(M_o=2, M_f=2)

    def test_rejects_simultaneous_miqp_and_milp(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            ThreePhaseOptions(enable_miqp=True, enable_milp=True)

    def test_rejects_highs_for_miqp(self):
        with self.assertRaisesRegex(ValueError, "HiGHS"):
            ThreePhaseOptions(mip_solver="highs", enable_miqp=True)

    def test_partial_horizon_requires_mapping(self):
        with self.assertRaisesRegex(ValueError, "binary_intervals"):
            ThreePhaseOptions(n_x=2, n_u=0, n_v=1, N=4, M_f=2)

    def test_m_o_requires_pycombina(self):
        with self.assertRaisesRegex(ValueError, "PyCombina"):
            ThreePhaseOptions(
                n_x=2,
                n_u=0,
                n_v=1,
                N=4,
                M_o=2,
                binary_intervals=[0, 1, 2, 3],
            )


if __name__ == "__main__":
    unittest.main()
