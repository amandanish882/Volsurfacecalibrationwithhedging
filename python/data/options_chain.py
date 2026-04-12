
"""
options_chain.py
================
Ingest listed option chains from CBOE delayed quotes CSV.
Filters out stale / zero-bid quotes.
"""

import pandas as pd
from python.types import OptionQuote, OptionChain


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
