
"""
dividends.py
============
Discrete dividend modeling: historical projection + implied extraction.
Supports fetching dividend history via yfinance.
"""

import numpy as np
import pandas as pd
from python.types import DiscreteDividend, OptionChain, YieldCurve


def fetch_dividends(ticker="SPY", period="3y"):
    """
    Fetch historical dividend data from Yahoo Finance via yfinance.

    Returns a DataFrame with columns ["ex_date", "amount"] suitable
    for project_from_history().

    Parameters
    ----------
    ticker : str, default "SPY"
    period : str, default "3y" — how far back to fetch
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    divs = tk.dividends
    if divs.empty:
        raise RuntimeError("No dividend data found for %s" % ticker)

    # Normalize to tz-naive for consistency with rest of pipeline
    if divs.index.tz is not None:
        divs.index = divs.index.tz_localize(None)

    # Filter to requested period
    years_back = int(period.replace("y", "")) if period.endswith("y") else 3
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=years_back)
    divs = divs[divs.index >= cutoff]

    df = pd.DataFrame({
        "ex_date": divs.index,
        "amount": divs.values,
    }).reset_index(drop=True)

    print("  [dividends] Fetched %d dividends for %s over %s" % (len(df), ticker, period))
    return df


def project_from_history(historical_divs, horizon_years=2.0):
    """
    Project future discrete dividends from historical ex-date schedule.

    Parameters
    ----------
    historical_divs : DataFrame with columns ["ex_date", "amount"]
    """
    historical_divs = historical_divs.copy()
    historical_divs["ex_date"] = pd.to_datetime(historical_divs["ex_date"])
    historical_divs = historical_divs.sort_values("ex_date")

    gaps = historical_divs["ex_date"].diff().dropna().dt.days
    median_gap = gaps.median()

    recent = historical_divs.tail(4)
    avg_amount = recent["amount"].mean()

    last_ex = historical_divs["ex_date"].iloc[-1]
    projections = []
    current = last_ex + pd.Timedelta(days=int(median_gap))
    cutoff = pd.Timestamp.now() + pd.Timedelta(days=int(horizon_years * 365))

    while current <= cutoff:
        if current > pd.Timestamp.now():
            projections.append(DiscreteDividend(
                ex_date=current, amount=avg_amount, source="historical"
            ))
        current += pd.Timedelta(days=int(median_gap))

    return projections


def extract_implied_dividends(chain, curve):
    """
    Extract market-implied discrete dividends from put-call parity per expiry.

    At each expiry T:
        F_implied = (C - P) * e^{rT} + K
        PV(Divs to T) = S - F_implied * e^{-rT}
    """
    S = chain.spot
    implied_divs = {}

    for expiry in chain.expiries():
        T = (expiry - pd.Timestamp.now()).days / 365.25
        if T < 7 / 365.25:
            continue

        r = curve.rate(T)
        calls = {}
        puts = {}
        for q in chain.for_expiry(expiry):
            if q.is_call:
                calls[q.strike] = q
            else:
                puts[q.strike] = q

        common_strikes = sorted(set(calls.keys()) & set(puts.keys()))
        if not common_strikes:
            continue

        atm_strikes = sorted(common_strikes, key=lambda k: abs(k - S))[:3]

        forwards = []
        for K in atm_strikes:
            c_mid = calls[K].mid()
            p_mid = puts[K].mid()
            F_implied = (c_mid - p_mid) * np.exp(r * T) + K
            forwards.append(F_implied)

        F_avg = np.mean(forwards)
        div_pv = S - F_avg * np.exp(-r * T)
        implied_divs[expiry] = max(div_pv, 0.0)

    return implied_divs


def decompose_implied_dividends(implied_pvs, historical_projected, spot, curve):
    """
    Decompose cumulative implied PV-of-dividends into individual
    DiscreteDividend objects, using the historical schedule as a
    timing template.

    Parameters
    ----------
    implied_pvs : dict[pd.Timestamp -> float]
        Output of extract_implied_dividends(): cumulative PV of all
        dividends between now and each expiry.
    historical_projected : list[DiscreteDividend]
        Output of project_from_history(): expected ex-dates and amounts.
    spot : float
        Current spot price.
    curve : YieldCurve
        For discounting.

    Returns
    -------
    list[DiscreteDividend]
        Same ex_dates as historical, but amounts rescaled to match
        market-implied PV.  source="implied" within expiry range,
        source="historical" for dates beyond the last expiry.
    """
    if not implied_pvs or not historical_projected:
        return list(historical_projected)

    now = pd.Timestamp.now()
    sorted_expiries = sorted(implied_pvs.keys())
    boundaries = [now] + sorted_expiries
    result = []

    for i in range(len(boundaries) - 1):
        window_start = boundaries[i]
        window_end = boundaries[i + 1]

        cum_pv_end = implied_pvs[window_end]
        cum_pv_start = implied_pvs.get(window_start, 0.0) if window_start != now else 0.0
        marginal_pv = max(cum_pv_end - cum_pv_start, 0.0)

        divs_in_window = [
            d for d in historical_projected
            if window_start < d.ex_date <= window_end
        ]

        if not divs_in_window:
            continue

        weights = []
        for d in divs_in_window:
            T_d = (d.ex_date - now).days / 365.25
            r = curve.rate(max(T_d, 0.001))
            df = np.exp(-r * T_d)
            weights.append(d.amount * df)

        total_weight = sum(weights)
        if total_weight < 1e-12:
            continue

        for d, w in zip(divs_in_window, weights):
            allocated_pv = marginal_pv * (w / total_weight)
            T_d = (d.ex_date - now).days / 365.25
            r = curve.rate(max(T_d, 0.001))
            implied_amount = allocated_pv * np.exp(r * T_d)
            result.append(DiscreteDividend(
                ex_date=d.ex_date,
                amount=implied_amount,
                source="implied",
            ))

    last_expiry = sorted_expiries[-1]
    for d in historical_projected:
        if d.ex_date > last_expiry:
            result.append(DiscreteDividend(
                ex_date=d.ex_date, amount=d.amount, source="historical"
            ))

    return sorted(result, key=lambda d: d.ex_date)


def blend_dividends(historical_projected, implied_projected,
                    method="prefer_implied"):
    """
    Combine historical and implied dividend projections.

    Parameters
    ----------
    historical_projected : list[DiscreteDividend]
    implied_projected : list[DiscreteDividend]
    method : str
        "prefer_implied" -- use implied where available, historical tail.
        "average"        -- average amounts at matching ex_dates.
    """
    if not implied_projected:
        return list(historical_projected)

    if method == "prefer_implied":
        return list(implied_projected)

    if method == "average":
        impl_by_date = {d.ex_date: d for d in implied_projected}
        hist_by_date = {d.ex_date: d for d in historical_projected}
        all_dates = sorted(set(list(impl_by_date.keys()) + list(hist_by_date.keys())))
        result = []
        for dt in all_dates:
            h = hist_by_date.get(dt)
            m = impl_by_date.get(dt)
            if h and m:
                avg = (h.amount + m.amount) / 2.0
                result.append(DiscreteDividend(dt, avg, source="implied"))
            elif m:
                result.append(m)
            else:
                result.append(h)
        return result

    raise ValueError("Unknown blend method: %s" % method)


def build_forward_curve(spot, dividends, curve, expiries):
    """
    Discrete-dividend-adjusted forward for each expiry:
        F(T) = (S - sum PV(D_i) for ex_i < T) * e^{rT}
    """
    now = pd.Timestamp.now()
    forwards = {}

    for expiry in expiries:
        T = (expiry - now).days / 365.25
        r = curve.rate(max(T, 0.001))

        pv_divs = sum(
            d.amount * np.exp(-r * (d.ex_date - now).days / 365.25)
            for d in dividends
            if now < d.ex_date <= expiry
        )

        forwards[expiry] = (spot - pv_divs) * np.exp(r * T)

    return forwards
