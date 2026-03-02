
"""
types.py
========
Shared data structures. Plain classes only.
"""

import numpy as np
import pandas as pd


class YieldCurve:
    """Bootstrapped zero-rate curve from FRED Treasury yields."""

    def __init__(self, tenors, zero_rates, as_of_date):
        self.tenors = tenors          # np.ndarray, years
        self.zero_rates = zero_rates  # np.ndarray, continuous compounding
        self.as_of_date = as_of_date  # str

    def discount(self, T):
        r = np.interp(T, self.tenors, self.zero_rates)
        return np.exp(-r * T)

    def rate(self, T):
        return float(np.interp(T, self.tenors, self.zero_rates))


class DiscreteDividend:
    """A single projected discrete dividend."""

    def __init__(self, ex_date, amount, source="historical"):
        self.ex_date = ex_date   # pd.Timestamp
        self.amount = amount     # float
        self.source = source     # "historical" | "implied"


class OptionQuote:
    """One row of a listed option chain."""

    def __init__(self, strike, expiry, is_call, bid, ask, volume, open_interest):
        self.strike = strike
        self.expiry = expiry
        self.is_call = is_call
        self.bid = bid
        self.ask = ask
        self.volume = volume
        self.open_interest = open_interest

    def mid(self):
        return (self.bid + self.ask) / 2.0

    def spread(self):
        return self.ask - self.bid


class OptionChain:
    """Full chain for one underlying, one snapshot."""

    def __init__(self, underlying, spot, quotes, as_of):
        self.underlying = underlying
        self.spot = spot
        self.quotes = quotes   # list of OptionQuote
        self.as_of = as_of

    def for_expiry(self, expiry):
        return [q for q in self.quotes if q.expiry == expiry]

    def expiries(self):
        return sorted(set(q.expiry for q in self.quotes))


class SSVIParams:
    """Fitted SSVI parameters for one tenor slice or global surface."""

    def __init__(self, theta, rho, eta, tenor=None):
        self.theta = theta
        self.rho = rho
        self.eta = eta
        self.tenor = tenor


class ArbViolation:
    """Record of a detected arbitrage violation."""

    def __init__(self, violation_type, strike, tenor, severity, description):
        self.violation_type = violation_type
        self.strike = strike
        self.tenor = tenor
        self.severity = severity
        self.description = description


class FittedSurface:
    """Calibrated SSVI surface with arb report."""

    def __init__(self, params_by_tenor, global_rho, global_eta,
                 violations_pre=None, violations_post=None):
        self.params_by_tenor = params_by_tenor
        self.global_rho = global_rho
        self.global_eta = global_eta
        self.violations_pre = violations_pre if violations_pre is not None else []
        self.violations_post = violations_post if violations_post is not None else []

    def implied_vol(self, k, T):
        """Read implied vol from calibrated surface at log-moneyness k, tenor T."""
        from qr_engine.ssvi import total_variance
        tenors = sorted(self.params_by_tenor.keys())
        thetas = [self.params_by_tenor[t].theta for t in tenors]
        theta_T = float(np.interp(T, tenors, thetas))
        w = total_variance(k, theta_T, self.global_rho, self.global_eta)
        return np.sqrt(max(w / T, 1e-12))
