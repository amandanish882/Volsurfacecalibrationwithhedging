
"""
test_dividends.py
=================
Unit tests for implied forward curve building via put-call parity.
No network access needed — uses synthetic test data.
"""

import unittest
import numpy as np
import pandas as pd

from python.types import YieldCurve, OptionQuote, OptionChain
from python.data.dividends import (
    build_forward_curve_index, ForwardCurve, _extract_implied_forward,
)


def _make_synthetic_chain(spot, curve, div_yield, expiries):
    """
    Build a synthetic option chain where put-call parity holds exactly
    with a known continuous dividend yield q.

    F(T) = S * exp((r - q) * T)
    C - P = exp(-rT) * (F - K) = exp(-rT) * (S*exp((r-q)*T) - K)
    """
    quotes = []
    for expiry in expiries:
        T = (expiry - pd.Timestamp.now()).days / 365.25
        if T < 0.01:
            continue
        r = curve.rate(T)
        F = spot * np.exp((r - div_yield) * T)

        # Create ATM and near-ATM strikes
        for K in [spot * 0.95, spot * 0.98, spot, spot * 1.02, spot * 1.05]:
            # Approximate BS prices for mid (enough for put-call parity test)
            # Use simple intrinsic + time value approximation
            sigma = 0.18
            d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            from scipy.stats import norm
            call_px = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
            put_px = np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

            # Add small spread
            spread = 0.50
            quotes.append(OptionQuote(
                strike=K, expiry=expiry, is_call=True,
                bid=max(call_px - spread / 2, 0.01),
                ask=call_px + spread / 2,
                volume=1000, open_interest=5000,
            ))
            quotes.append(OptionQuote(
                strike=K, expiry=expiry, is_call=False,
                bid=max(put_px - spread / 2, 0.01),
                ask=put_px + spread / 2,
                volume=800, open_interest=4000,
            ))

    return OptionChain("SPX", spot, quotes, str(pd.Timestamp.now().date()))


class TestExtractImpliedForward(unittest.TestCase):

    def setUp(self):
        tenors = np.array([0.25, 1.0, 2.0, 5.0])
        rates = np.array([0.04, 0.042, 0.045, 0.048])
        self.curve = YieldCurve(tenors, rates, "2026-02-22")
        self.spot = 5800.0
        self.div_yield = 0.014  # ~1.4%
        self.expiries = [
            pd.Timestamp.now() + pd.DateOffset(months=3),
            pd.Timestamp.now() + pd.DateOffset(months=6),
            pd.Timestamp.now() + pd.DateOffset(months=12),
        ]
        self.chain = _make_synthetic_chain(
            self.spot, self.curve, self.div_yield, self.expiries,
        )

    def test_implied_forward_recovers_true_forward(self):
        """Implied forward from put-call parity should match F = S*exp((r-q)*T)."""
        for expiry in self.expiries:
            result = _extract_implied_forward(self.chain, self.curve, expiry)
            self.assertIsNotNone(result)
            T, F_impl = result
            r = self.curve.rate(T)
            F_true = self.spot * np.exp((r - self.div_yield) * T)
            # Should be close (within bid-ask noise)
            self.assertAlmostEqual(F_impl, F_true, delta=F_true * 0.002)

    def test_returns_none_for_short_expiry(self):
        short_exp = pd.Timestamp.now() + pd.Timedelta(days=3)
        result = _extract_implied_forward(self.chain, self.curve, short_exp)
        self.assertIsNone(result)


class TestBuildForwardCurveIndex(unittest.TestCase):

    def setUp(self):
        tenors = np.array([0.25, 1.0, 2.0, 5.0])
        rates = np.array([0.04, 0.042, 0.045, 0.048])
        self.curve = YieldCurve(tenors, rates, "2026-02-22")
        self.spot = 5800.0
        self.div_yield = 0.014
        self.expiries = [
            pd.Timestamp.now() + pd.DateOffset(months=3),
            pd.Timestamp.now() + pd.DateOffset(months=6),
            pd.Timestamp.now() + pd.DateOffset(months=12),
        ]
        self.chain = _make_synthetic_chain(
            self.spot, self.curve, self.div_yield, self.expiries,
        )

    def test_returns_dict_and_curve(self):
        fwd_dict, fwd_curve = build_forward_curve_index(self.spot, self.chain, self.curve)
        self.assertIsInstance(fwd_dict, dict)
        self.assertIsInstance(fwd_curve, ForwardCurve)

    def test_forward_dict_keys_match_expiries(self):
        fwd_dict, _ = build_forward_curve_index(self.spot, self.chain, self.curve)
        for exp in self.expiries:
            self.assertIn(exp, fwd_dict)

    def test_implied_div_yield_is_realistic(self):
        """Bootstrapped q(T) should be close to the true dividend yield."""
        _, fwd_curve = build_forward_curve_index(self.spot, self.chain, self.curve)
        for T in [0.25, 0.5, 1.0]:
            q = fwd_curve.div_yield_at(T)
            self.assertAlmostEqual(q, self.div_yield, delta=0.005)

    def test_forward_below_naive_no_div_forward(self):
        """Implied forward should be below S*exp(rT) due to dividends."""
        fwd_dict, _ = build_forward_curve_index(self.spot, self.chain, self.curve)
        now = pd.Timestamp.now()
        for exp, F_impl in fwd_dict.items():
            T = (exp - now).days / 365.25
            r = self.curve.rate(T)
            F_no_div = self.spot * np.exp(r * T)
            self.assertLess(F_impl, F_no_div)

    def test_continuous_forward_at_arbitrary_tenor(self):
        """ForwardCurve.forward_at() should work at non-listed tenors."""
        _, fwd_curve = build_forward_curve_index(self.spot, self.chain, self.curve)
        # Interpolated tenor between listed expiries
        F_interp = fwd_curve.forward_at(0.4)
        self.assertGreater(F_interp, 0)
        self.assertGreater(F_interp, self.spot * 0.95)
        self.assertLess(F_interp, self.spot * 1.10)

    def test_forward_increases_with_tenor_when_r_gt_q(self):
        """When r > q, forwards should increase with tenor."""
        fwd_dict, _ = build_forward_curve_index(self.spot, self.chain, self.curve)
        fwd_vals = [fwd_dict[exp] for exp in sorted(fwd_dict.keys())]
        for i in range(len(fwd_vals) - 1):
            self.assertLess(fwd_vals[i], fwd_vals[i + 1])


if __name__ == "__main__":
    unittest.main()
