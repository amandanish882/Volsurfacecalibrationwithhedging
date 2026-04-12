"""
test_types.py
=============
Unit tests for python/types.py — YieldCurve, OptionQuote, OptionChain.
"""

import unittest
import numpy as np
import pandas as pd

from python.types import (
    YieldCurve, OptionQuote, OptionChain,
    DiscreteDividend, SSVIParams, ArbViolation, FittedSurface,
)


class TestYieldCurve(unittest.TestCase):

    def setUp(self):
        self.tenors = np.array([0.25, 1.0, 5.0, 10.0, 30.0])
        self.rates = np.array([0.04, 0.042, 0.045, 0.048, 0.05])
        self.curve = YieldCurve(self.tenors, self.rates, as_of_date="2026-02-22")

    def test_rate_interpolation(self):
        r = self.curve.rate(2.0)
        self.assertGreater(r, 0.04)
        self.assertLess(r, 0.045)

    def test_rate_at_node(self):
        r = self.curve.rate(1.0)
        self.assertAlmostEqual(r, 0.042, places=6)

    def test_discount_at_zero(self):
        df = self.curve.discount(0.0)
        self.assertAlmostEqual(df, 1.0, places=6)

    def test_discount_decreasing(self):
        df1 = self.curve.discount(1.0)
        df5 = self.curve.discount(5.0)
        df10 = self.curve.discount(10.0)
        self.assertGreater(df1, df5)
        self.assertGreater(df5, df10)

    def test_discount_positive(self):
        for t in [0.25, 1.0, 5.0, 10.0, 30.0]:
            self.assertGreater(self.curve.discount(t), 0.0)


class TestOptionQuote(unittest.TestCase):

    def test_mid(self):
        q = OptionQuote(strike=100, expiry=pd.Timestamp("2026-06-01"),
                        is_call=True, bid=2.0, ask=2.5, volume=100, open_interest=500)
        self.assertAlmostEqual(q.mid(), 2.25)

    def test_spread(self):
        q = OptionQuote(strike=100, expiry=pd.Timestamp("2026-06-01"),
                        is_call=True, bid=2.0, ask=2.5, volume=100, open_interest=500)
        self.assertAlmostEqual(q.spread(), 0.5)


class TestOptionChain(unittest.TestCase):

    def setUp(self):
        exp1 = pd.Timestamp("2026-03-20")
        exp2 = pd.Timestamp("2026-06-19")
        self.quotes = [
            OptionQuote(100, exp1, True, 5.0, 5.5, 200, 1000),
            OptionQuote(105, exp1, True, 2.0, 2.5, 150, 800),
            OptionQuote(100, exp1, False, 3.0, 3.5, 100, 600),
            OptionQuote(100, exp2, True, 8.0, 8.5, 300, 1500),
        ]
        self.chain = OptionChain("SPX", spot=5800.0, quotes=self.quotes,
                                 as_of="2026-02-22")

    def test_expiries_sorted(self):
        exps = self.chain.expiries()
        self.assertEqual(len(exps), 2)
        self.assertLess(exps[0], exps[1])

    def test_for_expiry_count(self):
        exp1_quotes = self.chain.for_expiry(pd.Timestamp("2026-03-20"))
        self.assertEqual(len(exp1_quotes), 3)

    def test_for_expiry_empty(self):
        empty = self.chain.for_expiry(pd.Timestamp("2099-01-01"))
        self.assertEqual(len(empty), 0)


class TestDiscreteDividend(unittest.TestCase):

    def test_default_source(self):
        d = DiscreteDividend(pd.Timestamp("2026-06-15"), 1.75)
        self.assertEqual(d.source, "historical")

    def test_custom_source(self):
        d = DiscreteDividend(pd.Timestamp("2026-06-15"), 1.75, source="implied")
        self.assertEqual(d.source, "implied")


class TestSSVIParams(unittest.TestCase):

    def test_attributes(self):
        p = SSVIParams(theta=0.04, rho=-0.3, eta=1.0, tenor=0.25)
        self.assertAlmostEqual(p.theta, 0.04)
        self.assertAlmostEqual(p.rho, -0.3)
        self.assertAlmostEqual(p.eta, 1.0)
        self.assertAlmostEqual(p.tenor, 0.25)


if __name__ == "__main__":
    unittest.main()
