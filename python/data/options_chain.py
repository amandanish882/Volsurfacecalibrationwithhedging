
"""
options_chain.py
================
Ingest listed option chains from CBOE delayed quotes CSV or
fetch live chains via yfinance.
Filters out stale / zero-bid quotes.
"""

import pandas as pd
from python.types import OptionQuote, OptionChain


def fetch_yfinance(ticker="SPY"):
    """
    Fetch a full option chain from Yahoo Finance via yfinance.

    Returns an OptionChain with all available expiries.
    Drops zero-bid and invalid-spread rows.

    Parameters
    ----------
    ticker : str, default "SPY"
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    spot = tk.info.get("regularMarketPrice") or tk.info.get("previousClose")
    if spot is None:
        hist = tk.history(period="1d")
        if hist.empty:
            raise RuntimeError("Cannot fetch spot price for %s" % ticker)
        spot = float(hist["Close"].iloc[-1])

    expiry_strings = tk.options
    if not expiry_strings:
        raise RuntimeError("No option expiries found for %s" % ticker)

    def _safe_int(val):
        """Convert a possibly-NaN value to int, defaulting to 0."""
        try:
            if pd.isna(val):
                return 0
            return int(val)
        except (ValueError, TypeError):
            return 0

    def _safe_float(val, default=0.0):
        """Convert a possibly-NaN value to float."""
        try:
            if pd.isna(val):
                return default
            return float(val)
        except (ValueError, TypeError):
            return default

    quotes = []
    for exp_str in expiry_strings:
        chain = tk.option_chain(exp_str)
        expiry = pd.Timestamp(exp_str)

        for _, row in chain.calls.iterrows():
            bid = _safe_float(row.get("bid", 0))
            ask = _safe_float(row.get("ask", 0))
            if bid <= 0 or ask <= bid:
                continue
            quotes.append(OptionQuote(
                strike=float(row["strike"]),
                expiry=expiry,
                is_call=True,
                bid=bid,
                ask=ask,
                volume=_safe_int(row.get("volume", 0)),
                open_interest=_safe_int(row.get("openInterest", 0)),
            ))

        for _, row in chain.puts.iterrows():
            bid = _safe_float(row.get("bid", 0))
            ask = _safe_float(row.get("ask", 0))
            if bid <= 0 or ask <= bid:
                continue
            quotes.append(OptionQuote(
                strike=float(row["strike"]),
                expiry=expiry,
                is_call=False,
                bid=bid,
                ask=ask,
                volume=_safe_int(row.get("volume", 0)),
                open_interest=_safe_int(row.get("openInterest", 0)),
            ))

    print("  [options_chain] Fetched %d quotes across %d expiries for %s (spot=%.2f)"
          % (len(quotes), len(expiry_strings), ticker, spot))

    return OptionChain(
        underlying=ticker,
        spot=float(spot),
        quotes=quotes,
        as_of=str(pd.Timestamp.now().date()),
    )


def load_cboe_csv(filepath, underlying, spot):
    """
    Parse a CBOE-format CSV into an OptionChain.

    Expected columns:
        expiration, strike, option_type, bid, ask, volume, open_interest

    Drops zero-bid and invalid-spread rows.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()

    required = {"expiration", "strike", "option_type", "bid", "ask", "volume", "open_interest"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("CBOE CSV missing columns: %s" % missing)

    df["expiration"] = pd.to_datetime(df["expiration"])
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce").fillna(0)
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce").fillna(0)

    n_before = len(df)
    df = df[df["bid"] > 0].copy()
    df = df[df["ask"] > df["bid"]].copy()
    n_after = len(df)
    if n_before > n_after:
        print("  [options_chain] Dropped %d zero-bid/invalid rows." % (n_before - n_after))

    quotes = []
    for _, row in df.iterrows():
        quotes.append(OptionQuote(
            strike=float(row["strike"]),
            expiry=row["expiration"],
            is_call=(str(row["option_type"]).strip().upper() in ("C", "CALL")),
            bid=float(row["bid"]),
            ask=float(row["ask"]),
            volume=int(row.get("volume", 0)),
            open_interest=int(row.get("open_interest", 0)),
        ))

    return OptionChain(
        underlying=underlying,
        spot=spot,
        quotes=quotes,
        as_of=str(pd.Timestamp.now().date()),
    )
