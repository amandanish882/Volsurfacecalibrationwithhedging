
"""
dividends.py
============
Forward curve construction for equity index options (SPX).

Extracts implied forwards from put-call parity on the option chain,
bootstraps a continuous dividend yield term structure q(T), and
provides a continuous forward curve F(T) = S * exp((r(T) - q(T)) * T).

This is the standard sell-side approach: model-free, self-consistent
with the options market, requires no external dividend data.
"""

import numpy as np
import pandas as pd
from python.types import OptionChain, YieldCurve


class ForwardCurve:
    """
    Continuous forward curve bootstrapped from implied forwards.

    Stores the implied dividend yield term structure q(T) and provides
    forward prices at arbitrary tenors via interpolation.
    """

    def __init__(self, spot, tenors, forwards, div_yields, curve):
        self.spot = spot
        self.tenors = np.array(tenors)       # T at each listed expiry
        self.forwards = np.array(forwards)   # implied F at each expiry
        self.div_yields = np.array(div_yields)  # q(T) at each expiry
        self._curve = curve

    def forward_at(self, T):
        """Continuous forward at arbitrary tenor T."""
        T = max(T, 0.001)
        q = float(np.interp(T, self.tenors, self.div_yields))
        r = self._curve.rate(T)
        return self.spot * np.exp((r - q) * T)

    def div_yield_at(self, T):
        """Interpolated continuous dividend yield at tenor T."""
        return float(np.interp(max(T, 0.001), self.tenors, self.div_yields))


def _extract_implied_forward(chain, curve, expiry):
    """
    Extract implied forward at a single expiry from near-ATM put-call parity.

    F_implied = K + exp(rT) * (C_mid - P_mid)

    Averages across the 3 nearest-to-ATM strikes with both calls and puts.

    Returns (T, F_implied) or None if insufficient data.
    """
    now = pd.Timestamp.now()
    T = (expiry - now).days / 365.25
    if T < 7 / 365.25:
        return None

    r = curve.rate(T)
    S = chain.spot

    calls = {}
    puts = {}
    for q in chain.for_expiry(expiry):
        if q.is_call:
            calls[q.strike] = q
        else:
            puts[q.strike] = q

    common_strikes = sorted(set(calls.keys()) & set(puts.keys()))
    if len(common_strikes) < 2:
        return None

    # Pick 3 nearest-to-ATM strikes
    atm_strikes = sorted(common_strikes, key=lambda k: abs(k - S))[:3]

    implied_forwards = []
    for K in atm_strikes:
        c_mid = calls[K].mid()
        p_mid = puts[K].mid()
        F = K + np.exp(r * T) * (c_mid - p_mid)
        if F > 0:
            implied_forwards.append(F)

    if not implied_forwards:
        return None

    return T, float(np.mean(implied_forwards))


def build_forward_curve_index(spot, chain, curve):
    """
    Build a continuous forward curve from implied forwards.

    Steps:
      1. Extract implied forwards at each listed expiry via put-call parity
      2. Back out implied dividend yield: q(T) = r(T) - ln(F/S) / T
      3. Return ForwardCurve with interpolated q(T) for arbitrary tenors

    Also returns a dict[expiry -> forward] for direct use by the calibrator.

    Parameters
    ----------
    spot : float
        Current index level.
    chain : OptionChain
        Option chain with puts and calls at multiple expiries.
    curve : YieldCurve
        Risk-free zero-rate curve.

    Returns
    -------
    forwards_dict : dict[pd.Timestamp -> float]
        Implied forward at each listed expiry (for calibration).
    fwd_curve : ForwardCurve
        Continuous forward curve (for pricing at arbitrary tenors).
    """
    now = pd.Timestamp.now()
    tenors = []
    forwards = []
    div_yields = []
    expiry_map = {}

    for expiry in chain.expiries():
        result = _extract_implied_forward(chain, curve, expiry)
        if result is None:
            continue
        T, F_impl = result
        r = curve.rate(T)

        # Back out implied continuous dividend yield
        # q(T) = r(T) - ln(F/S) / T
        q = r - np.log(F_impl / spot) / T

        tenors.append(T)
        forwards.append(F_impl)
        div_yields.append(q)
        expiry_map[expiry] = F_impl

    if not tenors:
        raise RuntimeError(
            "Cannot extract implied forwards: no expiries with valid "
            "put-call pairs found in the option chain."
        )

    fwd_curve = ForwardCurve(
        spot=spot,
        tenors=tenors,
        forwards=forwards,
        div_yields=div_yields,
        curve=curve,
    )

    return expiry_map, fwd_curve
