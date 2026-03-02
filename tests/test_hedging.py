"""
test_hedging.py
===============
Unit tests for hedging optimizer and efficient frontier.
"""

import unittest
import numpy as np
from python.risk.hedging import compute_efficient_frontier


class TestComputeEfficientFrontier(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n_scenarios = 77  # 11 spot x 7 vol
        self.portfolio_pnl = np.random.randn(n_scenarios) * 100.0
        self.hedge_pnl = np.random.randn(n_scenarios, 3) * 50.0
        self.hedge_costs = np.array([0.10, 0.15, 0.12])

    def test_returns_list(self):
        frontier = compute_efficient_frontier(
            self.portfolio_pnl, self.hedge_pnl, self.hedge_costs, budget_steps=5)
        self.assertIsInstance(frontier, list)
        self.assertEqual(len(frontier), 5)

    def test_frontier_keys(self):
        frontier = compute_efficient_frontier(
            self.portfolio_pnl, self.hedge_pnl, self.hedge_costs, budget_steps=3)
        for point in frontier:
            self.assertIn("budget", point)
            self.assertIn("residual_variance", point)
            self.assertIn("optimal_quantities", point)
            self.assertIn("actual_cost", point)

    def test_variance_decreasing_trend(self):
        frontier = compute_efficient_frontier(
            self.portfolio_pnl, self.hedge_pnl, self.hedge_costs, budget_steps=10)
        variances = [f["residual_variance"] for f in frontier]
        # The best hedged variance should be less than or equal to unhedged
        self.assertLessEqual(min(variances), variances[0] + 1e-6)

    def test_zero_budget_is_unhedged(self):
        frontier = compute_efficient_frontier(
            self.portfolio_pnl, self.hedge_pnl, self.hedge_costs, budget_steps=5)
        # First point should be near zero cost
        self.assertAlmostEqual(frontier[0]["budget"], 0.0, places=4)
        # Its variance should be close to the raw portfolio variance
        raw_var = float(np.var(self.portfolio_pnl))
        self.assertAlmostEqual(frontier[0]["residual_variance"], raw_var, delta=raw_var * 0.01)

    def test_actual_cost_within_budget(self):
        frontier = compute_efficient_frontier(
            self.portfolio_pnl, self.hedge_pnl, self.hedge_costs, budget_steps=5)
        for point in frontier:
            self.assertLessEqual(point["actual_cost"], point["budget"] + 1e-6)


if __name__ == "__main__":
    unittest.main()
