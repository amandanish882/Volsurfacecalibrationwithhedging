
"""
etf_premium.py
==============
ETF premium/discount estimation and forward adjustment.

The listed option is on the ETF share price, not the NAV. For international
or less-liquid ETFs, the spread between price and NAV is a second source
of uncertainty that affects the vol surface.
"""

import numpy as np


def estimate_premium(spot, nav, implied_forward, nav_implied_forward, T):
    """
    Estimate the ETF premium/discount and its impact.
    Returns dict with premium metrics and adjustment factors.
    """
    spot_premium_pct = (spot / nav - 1.0) * 100.0
    fwd_premium_pct = (implied_forward / nav_implied_forward - 1.0) * 100.0
    reversion_rate = (spot_premium_pct - fwd_premium_pct) / max(T, 0.01)

    return {
        "spot_premium_pct": spot_premium_pct,
        "fwd_premium_pct": fwd_premium_pct,
        "reversion_rate_annual": reversion_rate,
        "forward_adjustment": implied_forward / nav_implied_forward,
        "description": (
            "ETF trades at %+.2f%% to NAV (spot). "
            "Forward implies %+.2f%% premium at T=%.2fy. "
            "Estimated premium mean-reversion: %.1f%%/yr."
            % (spot_premium_pct, fwd_premium_pct, T, reversion_rate)
        ),
    }


def adjust_surface_for_premium(raw_iv, premium_pct, T,
                               reversion_half_life=0.25):
    """
    Adjust implied vol for premium/discount dynamics.

    Short-dated options on an ETF at premium have depressed realised vol
    (premium mean-reverts, dampening moves).
    """
    decay = np.exp(-np.log(2) * T / reversion_half_life)
    premium_vol_drag = 0.4 * abs(premium_pct) / 100.0 * decay
    if premium_pct > 0:
        adjusted_iv = raw_iv - premium_vol_drag
    else:
        adjusted_iv = raw_iv + premium_vol_drag
    return max(adjusted_iv, 0.01)
