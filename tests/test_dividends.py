"""
test_dividends.py
=================
Unit tests for dividend projection and forward curve building.
No network access needed — uses hardcoded test data.
"""

import unittest
import numpy as np
import pandas as pd

from python.types import YieldCurve, DiscreteDividend, OptionQuote, OptionChain
from python.data.dividends import (
    project_from_history, build_forward_curve,
    decompose_implied_dividends, blend_dividends,
)


class TestProjectFromHistory(unittest.TestCase):

    def setUp(self):
        self.hist = pd.DataFrame({
            "ex_date": pd.date_range("2025-03-15", periods=8, freq="QS"),
            "amount": [1.65, 1.65, 1.70, 1.70, 1.75, 1.75, 1.78, 1.78],
        })

    def test_returns_list(self):
        result = project_from_history(self.hist, horizon_years=1.0)
        self.assertIsInstance(result, list)

    def test_all_future(self):
        result = project_from_history(self.hist, horizon_years=2.0)
        now = pd.Timestamp.now()
        for d in result:
            self.assertGreater(d.ex_date, now)

    def test_amount_is_average_of_recent(self):
        result = project_from_history(self.hist, horizon_years=1.0)
        if result:
            expected_avg = self.hist.tail(4)["amount"].mean()
            self.assertAlmostEqual(result[0].amount, expected_avg, places=4)

    def test_source_label(self):
        result = project_from_history(self.hist, horizon_years=1.0)
        for d in result:
            self.assertEqual(d.source, "historical")

    def test_empty_history_handles_gracefully(self):
        empty = pd.DataFrame({"ex_date": pd.Series([], dtype="datetime64[ns]"),
                               "amount": pd.Series([], dtype="float64")})
        # Empty history should either return empty or raise — not crash unexpectedly
        try:
            result = project_from_history(empty, horizon_years=1.0)
            self.assertEqual(len(result), 0)
        except (IndexError, ValueError):
            pass  # acceptable: no data to project from


class TestBuildForwardCurve(unittest.TestCase):

    def setUp(self):
        tenors = np.array([0.25, 1.0, 2.0, 5.0])
        rates = np.array([0.04, 0.042, 0.045, 0.048])
        self.curve = YieldCurve(tenors, rates, "2026-02-22")
        self.spot = 600.0
        self.expiries = [
            pd.Timestamp.now() + pd.DateOffset(months=3),
            pd.Timestamp.now() + pd.DateOffset(months=6),
            pd.Timestamp.now() + pd.DateOffset(months=12),
        ]

    def test_no_dividends_forward_above_spot(self):
        forwards = build_forward_curve(self.spot, [], self.curve, self.expiries)
        for exp in self.expiries:
            self.assertGreater(forwards[exp], self.spot)

    def test_dividends_reduce_forward(self):
        no_div_fwd = build_forward_curve(self.spot, [], self.curve, self.expiries)
        divs = [
            DiscreteDividend(pd.Timestamp.now() + pd.DateOffset(months=1), 2.0),
            DiscreteDividend(pd.Timestamp.now() + pd.DateOffset(months=4), 2.0),
        ]
        with_div_fwd = build_forward_curve(self.spot, divs, self.curve, self.expiries)
        # Forwards with dividends should be lower
        for exp in self.expiries:
            self.assertLess(with_div_fwd[exp], no_div_fwd[exp])

    def test_forward_keys_match_expiries(self):
        forwards = build_forward_curve(self.spot, [], self.curve, self.expiries)
        self.assertEqual(set(forwards.keys()), set(self.expiries))


class TestDecomposeImpliedDividends(unittest.TestCase):

    def setUp(self):
        tenors = np.array([0.25, 1.0, 2.0, 5.0])
        rates = np.array([0.04, 0.042, 0.045, 0.048])
        self.curve = YieldCurve(tenors, rates, "2026-02-22")
        self.spot = 600.0
        now = pd.Timestamp.now()

        self.exp1 = now + pd.DateOffset(months=3)
        self.exp2 = now + pd.DateOffset(months=6)

        self.implied_pvs = {
            self.exp1: 2.0,
            self.exp2: 4.5,
        }

        self.hist = [
            DiscreteDividend(now + pd.DateOffset(months=1), 1.80, source="historical"),
            DiscreteDividend(now + pd.DateOffset(months=4), 1.80, source="historical"),
            DiscreteDividend(now + pd.DateOffset(months=7), 1.80, source="historical"),
        ]

    def test_returns_list_of_discrete_dividends(self):
        result = decompose_implied_dividends(
            self.implied_pvs, self.hist, self.spot, self.curve
        )
        self.assertIsInstance(result, list)
        for d in result:
            self.assertIsInstance(d, DiscreteDividend)

    def test_implied_source_within_expiry_range(self):
        result = decompose_implied_dividends(
            self.implied_pvs, self.hist, self.spot, self.curve
        )
        implied = [d for d in result if d.source == "implied"]
        self.assertGreater(len(implied), 0)

    def test_historical_tail_preserved(self):
        result = decompose_implied_dividends(
            self.implied_pvs, self.hist, self.spot, self.curve
        )
        tail = [d for d in result if d.ex_date > self.exp2]
        for d in tail:
            self.assertEqual(d.source, "historical")

    def test_total_pv_matches_last_expiry(self):
        result = decompose_implied_dividends(
            self.implied_pvs, self.hist, self.spot, self.curve
        )
        now = pd.Timestamp.now()
        total_pv = 0.0
        for d in result:
            if d.ex_date <= self.exp2 and d.source == "implied":
                T_d = (d.ex_date - now).days / 365.25
                r = self.curve.rate(max(T_d, 0.001))
                total_pv += d.amount * np.exp(-r * T_d)
        self.assertAlmostEqual(total_pv, 4.5, places=1)

    def test_empty_implied_returns_historical(self):
        result = decompose_implied_dividends({}, self.hist, self.spot, self.curve)
        self.assertEqual(len(result), len(self.hist))
        for d in result:
            self.assertEqual(d.source, "historical")

    def test_amounts_are_positive(self):
        result = decompose_implied_dividends(
            self.implied_pvs, self.hist, self.spot, self.curve
        )
        for d in result:
            self.assertGreater(d.amount, 0.0)


class TestBlendDividends(unittest.TestCase):

    def setUp(self):
        now = pd.Timestamp.now()
        self.hist = [
            DiscreteDividend(now + pd.DateOffset(months=1), 1.80, source="historical"),
            DiscreteDividend(now + pd.DateOffset(months=4), 1.80, source="historical"),
        ]
        self.implied = [
            DiscreteDividend(now + pd.DateOffset(months=1), 2.00, source="implied"),
            DiscreteDividend(now + pd.DateOffset(months=4), 2.10, source="implied"),
        ]

    def test_prefer_implied_returns_implied(self):
        result = blend_dividends(self.hist, self.implied, method="prefer_implied")
        self.assertEqual(len(result), len(self.implied))
        self.assertAlmostEqual(result[0].amount, 2.00, places=2)

    def test_average_method(self):
        result = blend_dividends(self.hist, self.implied, method="average")
        self.assertAlmostEqual(result[0].amount, (1.80 + 2.00) / 2.0, places=2)

    def test_empty_implied_returns_historical(self):
        result = blend_dividends(self.hist, [], method="prefer_implied")
        self.assertEqual(len(result), len(self.hist))


if __name__ == "__main__":
    unittest.main()
