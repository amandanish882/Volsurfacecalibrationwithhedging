
"""
risk_ladder.py
==============
Two-dimensional risk ladder: full revaluation across spot x vol shocks.
Requires the compiled C++ engine (qr_engine).
"""

import numpy as np
from python.types import FittedSurface, YieldCurve
from qr_engine.greeks import bs_price


def compute_risk_ladder(positions, surface, curve, spot,
                        spot_shocks_pct=None, vol_shocks_abs=None):
    """
    Full-revaluation risk ladder.

    Parameters
    ----------
    positions : list of dicts with keys:
        strike, expiry_years, is_call, quantity, forward
    surface : FittedSurface
    curve : YieldCurve
    spot : current spot
    spot_shocks_pct : array, default -10 to +10 in 1% steps
    vol_shocks_abs : array, default -0.05 to +0.05 in 0.5vol steps

    Returns dict with pnl_matrix, spot_levels, vol_shocks, base_value
    """
    if spot_shocks_pct is None:
        spot_shocks_pct = np.linspace(-10, 10, 21)
    if vol_shocks_abs is None:
        vol_shocks_abs = np.linspace(-0.05, 0.05, 21)

    def portfolio_value(S_shock_pct, vol_shock):
        S_new = spot * (1.0 + S_shock_pct / 100.0)
        total = 0.0
        for pos in positions:
            K = pos["strike"]
            T = pos["expiry_years"]
            F = S_new / spot * pos["forward"]
            k = np.log(K / F)
            sigma = surface.implied_vol(k, T) + vol_shock
            sigma = max(sigma, 0.01)
            r = curve.rate(T)
            px = bs_price(F, K, T, r, sigma, pos["is_call"])
            total += px * pos["quantity"]
        return total

    base_val = portfolio_value(0.0, 0.0)

    pnl = np.zeros((len(spot_shocks_pct), len(vol_shocks_abs)))
    for i, ds in enumerate(spot_shocks_pct):
        for j, dv in enumerate(vol_shocks_abs):
            pnl[i, j] = portfolio_value(ds, dv) - base_val

    return {
        "pnl_matrix": pnl,
        "spot_levels": spot * (1.0 + spot_shocks_pct / 100.0),
        "spot_shocks_pct": spot_shocks_pct,
        "vol_shocks": vol_shocks_abs,
        "base_value": base_val,
    }
