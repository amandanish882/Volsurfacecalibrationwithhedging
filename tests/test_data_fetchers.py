"""
test_data_fetchers.py
=====================
Integration tests for live data fetchers (FRED, yfinance).
These tests hit external APIs and require network access.

Run:  python -m pytest tests/test_data_fetchers.py -v
Skip: python -m pytest tests/ -v -k "not integration"
"""

import unittest
import os
import pandas as pd

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _has_fred_key():
    return bool(os.environ.get("FRED_API_KEY"))


class TestFredYieldCurve(unittest.TestCase):
    """Integration: FRED API yield curve fetch."""

    @unittest.skipUnless(_has_fred_key(), "FRED_API_KEY not set")
    def test_fetch_yield_curve_integration(self):
        from python.data.fred_rates import fetch_yield_curve
        key = os.environ["FRED_API_KEY"]
        curve = fetch_yield_curve(key)

        # Should have interpolated tenors
        self.assertGreater(len(curve.tenors), 10)
        self.assertGreater(len(curve.zero_rates), 10)

        # Rates should be positive and reasonable (0-15%)
        for r in curve.zero_rates:
            self.assertGreater(r, 0.0)
            self.assertLess(r, 0.15)

        # as_of_date should be a valid date string
        self.assertIsInstance(curve.as_of_date, str)
        pd.Timestamp(curve.as_of_date)  # should not raise

        # Discount factor sanity
        self.assertAlmostEqual(curve.discount(0.0), 1.0, places=3)
        self.assertGreater(curve.discount(1.0), 0.0)
        self.assertLess(curve.discount(1.0), 1.0)


class TestYfinanceOptionsChain(unittest.TestCase):
    """Integration: yfinance option chain fetch."""

    def test_fetch_spy_chain_integration(self):
        from python.data.options_chain import fetch_yfinance
        chain = fetch_yfinance("SPY")

        # Should have quotes
        self.assertGreater(len(chain.quotes), 100)
        self.assertGreater(len(chain.expiries()), 3)

        # Spot should be positive and reasonable for SPY
        self.assertGreater(chain.spot, 100.0)
        self.assertLess(chain.spot, 2000.0)

        # All quotes should have positive strikes and valid bid/ask
        for q in chain.quotes[:50]:
            self.assertGreater(q.strike, 0)
            self.assertGreater(q.bid, 0)
            self.assertGreater(q.ask, q.bid)
            self.assertIsInstance(q.is_call, bool)
            self.assertGreaterEqual(q.volume, 0)
            self.assertGreaterEqual(q.open_interest, 0)


class TestYfinanceDividends(unittest.TestCase):
    """Integration: yfinance dividend fetch."""

    def test_fetch_spy_dividends_integration(self):
        from python.data.dividends import fetch_dividends
        df = fetch_dividends("SPY", period="3y")

        # Should have quarterly dividends over 3 years
        self.assertGreater(len(df), 8)

        # Columns
        self.assertIn("ex_date", df.columns)
        self.assertIn("amount", df.columns)

        # Amounts should be positive
        for amt in df["amount"]:
            self.assertGreater(amt, 0.0)

        # Dates should be tz-naive
        self.assertIsNone(pd.DatetimeIndex(df["ex_date"]).tz)

    def test_dividend_projection_from_live_integration(self):
        from python.data.dividends import fetch_dividends, project_from_history
        df = fetch_dividends("SPY", period="3y")
        projected = project_from_history(df, horizon_years=1.0)

        # Should project at least 3 quarterly dividends
        self.assertGreater(len(projected), 2)

        # All projected should be in the future
        now = pd.Timestamp.now()
        for d in projected:
            self.assertGreater(d.ex_date, now)
            self.assertGreater(d.amount, 0.0)
            self.assertEqual(d.source, "historical")


if __name__ == "__main__":
    unittest.main()
