"""
test_data_fetchers.py
=====================
Integration tests for live data fetchers (FRED, Databento).
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


def _has_databento_key():
    return bool(os.environ.get("DATABENTO_API_KEY"))


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


class TestDabentoOptionsChain(unittest.TestCase):
    """Integration: Databento SPX option chain fetch."""

    @unittest.skipUnless(_has_databento_key(), "DATABENTO_API_KEY not set")
    def test_fetch_spx_chain_integration(self):
        from python.data.databento_chain import fetch_databento
        chain = fetch_databento("SPX")

        # Should have quotes
        self.assertGreater(len(chain.quotes), 100)
        self.assertGreater(len(chain.expiries()), 3)

        # Spot should be positive and reasonable for SPX
        self.assertGreater(chain.spot, 1000.0)
        self.assertLess(chain.spot, 10000.0)

        # All quotes should have positive strikes and valid bid/ask
        for q in chain.quotes[:50]:
            self.assertGreater(q.strike, 0)
            self.assertGreater(q.bid, 0)
            self.assertGreater(q.ask, q.bid)
            self.assertIsInstance(q.is_call, bool)
            self.assertGreaterEqual(q.volume, 0)
            self.assertGreaterEqual(q.open_interest, 0)


if __name__ == "__main__":
    unittest.main()
