"""
test_engine.py
==============
Unit tests for the C++ engine (qr_engine): SSVI and Greeks.
Requires compiled qr_engine.
"""

import unittest
import numpy as np


def _engine_available():
    try:
        import qr_engine
        return True
    except ImportError:
        return False


@unittest.skipUnless(_engine_available(), "qr_engine not compiled")
class TestSSVI(unittest.TestCase):

    def setUp(self):
        from qr_engine.ssvi import total_variance, surface_vec, derivatives
        self.total_variance = total_variance
        self.surface_vec = surface_vec
        self.derivatives = derivatives

    def test_atm_total_variance_positive(self):
        w = self.total_variance(0.0, 0.04, -0.25, 1.0)
        self.assertGreater(w, 0.0)

    def test_total_variance_symmetric_wings(self):
        # With rho=0, variance should be symmetric in k
        w_left = self.total_variance(-0.2, 0.04, 0.0, 1.0)
        w_right = self.total_variance(0.2, 0.04, 0.0, 1.0)
        self.assertAlmostEqual(w_left, w_right, places=10)

    def test_total_variance_skew_with_rho(self):
        # Negative rho -> left wing higher
        w_left = self.total_variance(-0.3, 0.04, -0.5, 1.0)
        w_right = self.total_variance(0.3, 0.04, -0.5, 1.0)
        self.assertGreater(w_left, w_right)

    def test_surface_vec_length(self):
        k_vec = np.linspace(-0.3, 0.3, 7).tolist()
        w_vec = self.surface_vec(k_vec, 0.04, -0.25, 1.0)
        self.assertEqual(len(w_vec), 7)

    def test_surface_vec_all_positive(self):
        k_vec = np.linspace(-0.5, 0.5, 20).tolist()
        w_vec = self.surface_vec(k_vec, 0.04, -0.25, 1.0)
        for w in w_vec:
            self.assertGreater(w, 0.0)

    def test_derivatives_consistency(self):
        d = self.derivatives(0.0, 0.04, -0.25, 1.0)
        w_direct = self.total_variance(0.0, 0.04, -0.25, 1.0)
        self.assertAlmostEqual(d.w, w_direct, places=12)

    def test_derivatives_numerical_dw_dk(self):
        k, theta, rho, eta = 0.1, 0.04, -0.25, 1.0
        dk = 1e-6
        d = self.derivatives(k, theta, rho, eta)
        w_up = self.total_variance(k + dk, theta, rho, eta)
        w_dn = self.total_variance(k - dk, theta, rho, eta)
        numerical_dw_dk = (w_up - w_dn) / (2 * dk)
        self.assertAlmostEqual(d.dw_dk, numerical_dw_dk, places=4)


@unittest.skipUnless(_engine_available(), "qr_engine not compiled")
class TestGreeks(unittest.TestCase):

    def setUp(self):
        from qr_engine.greeks import bs_price, bs_implied_vol, compute
        self.bs_price = bs_price
        self.bs_implied_vol = bs_implied_vol
        self.compute = compute

    def test_call_positive(self):
        px = self.bs_price(100.0, 100.0, 0.25, 0.05, 0.20, True)
        self.assertGreater(px, 0.0)

    def test_put_positive(self):
        px = self.bs_price(100.0, 100.0, 0.25, 0.05, 0.20, False)
        self.assertGreater(px, 0.0)

    def test_put_call_parity(self):
        F, K, T, r, sigma = 585.0, 590.0, 0.25, 0.045, 0.18
        call = self.bs_price(F, K, T, r, sigma, True)
        put = self.bs_price(F, K, T, r, sigma, False)
        parity = (F - K) * np.exp(-r * T)
        self.assertAlmostEqual(call - put, parity, places=6)

    def test_implied_vol_roundtrip(self):
        F, K, T, r, sigma = 585.0, 590.0, 0.25, 0.045, 0.18
        px = self.bs_price(F, K, T, r, sigma, True)
        iv_back = self.bs_implied_vol(px, F, K, T, r, True)
        self.assertAlmostEqual(iv_back, sigma, places=6)

    def test_delta_call_between_0_and_1(self):
        g = self.compute(100.0, 100.0, 0.25, 0.05, 0.20, True)
        self.assertGreater(g.delta, 0.0)
        self.assertLess(g.delta, 1.0)

    def test_gamma_positive(self):
        g = self.compute(100.0, 100.0, 0.25, 0.05, 0.20, True)
        self.assertGreater(g.gamma, 0.0)

    def test_vega_positive(self):
        g = self.compute(100.0, 100.0, 0.25, 0.05, 0.20, True)
        self.assertGreater(g.vega, 0.0)

    def test_theta_nonzero_for_long_option(self):
        # Theta convention: finite-diff (V(T-dT) - V(T)) / (-dT)
        # Sign depends on convention; just verify it's nonzero (time decay exists)
        g = self.compute(100.0, 100.0, 0.25, 0.05, 0.20, True)
        self.assertNotAlmostEqual(g.theta, 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
