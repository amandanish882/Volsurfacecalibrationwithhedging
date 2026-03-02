"""
test_etf_premium.py
===================
Unit tests for ETF premium/discount estimation and vol adjustment.
"""

import unittest
from python.models.etf_premium import estimate_premium, adjust_surface_for_premium


class TestEstimatePremium(unittest.TestCase):

    def test_positive_premium(self):
        result = estimate_premium(
            spot=585.0, nav=580.0,
            implied_forward=590.0, nav_implied_forward=585.0,
            T=0.25,
        )
        self.assertGreater(result["spot_premium_pct"], 0.0)

    def test_negative_premium(self):
        result = estimate_premium(
            spot=575.0, nav=580.0,
            implied_forward=580.0, nav_implied_forward=585.0,
            T=0.25,
        )
        self.assertLess(result["spot_premium_pct"], 0.0)

    def test_zero_premium(self):
        result = estimate_premium(
            spot=580.0, nav=580.0,
            implied_forward=585.0, nav_implied_forward=585.0,
            T=0.25,
        )
        self.assertAlmostEqual(result["spot_premium_pct"], 0.0, places=6)

    def test_description_present(self):
        result = estimate_premium(585.0, 580.0, 590.0, 585.0, 0.25)
        self.assertIn("ETF", result["description"])

    def test_all_keys_present(self):
        result = estimate_premium(585.0, 580.0, 590.0, 585.0, 0.25)
        for key in ["spot_premium_pct", "fwd_premium_pct",
                     "reversion_rate_annual", "forward_adjustment", "description"]:
            self.assertIn(key, result)


class TestAdjustSurfaceForPremium(unittest.TestCase):

    def test_positive_premium_reduces_iv(self):
        adj = adjust_surface_for_premium(0.20, premium_pct=1.0, T=0.1)
        self.assertLess(adj, 0.20)

    def test_negative_premium_increases_iv(self):
        adj = adjust_surface_for_premium(0.20, premium_pct=-1.0, T=0.1)
        self.assertGreater(adj, 0.20)

    def test_zero_premium_unchanged(self):
        adj = adjust_surface_for_premium(0.20, premium_pct=0.0, T=0.1)
        self.assertAlmostEqual(adj, 0.20, places=6)

    def test_iv_floor(self):
        adj = adjust_surface_for_premium(0.02, premium_pct=50.0, T=0.01)
        self.assertGreaterEqual(adj, 0.01)

    def test_long_dated_less_impact(self):
        adj_short = adjust_surface_for_premium(0.20, premium_pct=1.0, T=0.1)
        adj_long = adjust_surface_for_premium(0.20, premium_pct=1.0, T=2.0)
        # Short-dated should have more premium drag
        self.assertLess(adj_short, adj_long)


if __name__ == "__main__":
    unittest.main()
