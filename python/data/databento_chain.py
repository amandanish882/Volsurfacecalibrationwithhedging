
"""
databento_chain.py
==================
Fetch SPX index option chains and spot level from Databento.
Supports both live snapshots and historical date lookups.
"""

import os
import numpy as np
import pandas as pd
from python.types import OptionQuote, OptionChain
from python.data._cache import load_or_fetch, _normalize_date


def _get_api_key():
    """Return Databento API key from env var or .env file."""
    key = os.environ.get("DATABENTO_API_KEY")
    if key:
        return key
    from pathlib import Path
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("DATABENTO_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _fetch_spot_from_fred(symbol="SPX"):
    """
    Fetch index spot level from FRED.

    SPX is a calculated index, not a traded security, so it is not
    available in Databento's equity datasets. FRED publishes the
    S&P 500 index as series 'SP500'.
    """
    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        from pathlib import Path
        env_file = Path(__file__).resolve().parents[2] / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("FRED_API_KEY="):
                    fred_key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not fred_key:
        raise RuntimeError("FRED_API_KEY not set — needed for %s spot price." % symbol)

    from fredapi import Fred
    fred = Fred(api_key=fred_key)
    series = fred.get_series("SP500").dropna()
    if series.empty:
        raise RuntimeError("FRED SP500 series returned no data.")
    return float(series.iloc[-1])


def fetch_databento(symbol="SPX", valuation_date=None):
    """
    Fetch SPX option chain from Databento (OPRA dataset).

    Spot price is fetched from FRED (SP500 series) since SPX is a
    calculated index not available in Databento's equity datasets.

    Parameters
    ----------
    symbol : str, default "SPX"
        Underlying symbol (index).
    valuation_date : date, datetime, ISO string, or None
        Date to snapshot. If None, walk back from ~2 business days
        ago to find the freshest finalized data; both the cache key
        and the fetched snapshot use that resolved date.

    Returns
    -------
    OptionChain
        With spot set to SPX index level, quotes filtered to
        drop zero-bid and invalid-spread rows.
    """
    if valuation_date is None:
        candidate = pd.Timestamp.now().normalize() - pd.Timedelta(days=2)
        while candidate.weekday() >= 5:
            candidate -= pd.Timedelta(days=1)
        resolved_date = candidate.date()
    else:
        resolved_date = _normalize_date(valuation_date)
        # If the requested date is today or in the future, walk back to the
        # most recent business day with finalized data (Databento finalizes
        # data with a ~1 day lag).
        import datetime as _dt
        if resolved_date >= _dt.date.today():
            candidate = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
            while candidate.weekday() >= 5:
                candidate -= pd.Timedelta(days=1)
            resolved_date = candidate.date()

    def _fetch():
        return _fetch_databento_impl(symbol, resolved_date)

    cache_key = "databento_%s" % symbol.lower()
    return load_or_fetch(cache_key, resolved_date, _fetch)


def _fetch_databento_impl(symbol, date):
    import databento as db

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "DATABENTO_API_KEY not set. Add it to .env or export it."
        )

    client = db.Historical(api_key)

    # ------------------------------------------------------------------
    # Date range — caller always passes a concrete date
    # ------------------------------------------------------------------
    snap_date = pd.Timestamp(date)

    start = snap_date.strftime("%Y-%m-%d")
    # Databento requires end > start; use next day
    end = (snap_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Fetch SPX spot from FRED (SP500 series)
    # ------------------------------------------------------------------
    spot = _fetch_spot_from_fred(symbol)

    # ------------------------------------------------------------------
    # Fetch option definitions to get strike/expiry metadata
    # ------------------------------------------------------------------
    defn_data = client.timeseries.get_range(
        dataset="OPRA.PILLAR",
        symbols=f"{symbol}.OPT",
        stype_in="parent",
        schema="definition",
        start=start,
        end=end,
    )
    defn_df = defn_data.to_df()

    if defn_df.empty:
        raise RuntimeError("No option definitions found for %s on %s" % (symbol, start))

    # Build instrument_id -> metadata mapping
    instrument_meta = {}
    for _, row in defn_df.iterrows():
        iid = row.get("instrument_id") or row.get("hd.instrument_id")
        if iid is None:
            continue
        strike_price = float(row.get("strike_price", 0))
        # Databento stores strike_price as fixed-point (price * 1e9)
        if strike_price > 1e6:
            strike_price = strike_price / 1e9
        expiry = pd.Timestamp(row.get("expiration", row.get("expiry", pd.NaT)))
        if pd.isna(expiry):
            continue
        expiry = expiry.tz_localize(None) if expiry.tzinfo else expiry
        is_call = str(row.get("instrument_class", "")).upper() == "C"
        instrument_meta[iid] = {
            "strike": strike_price,
            "expiry": expiry,
            "is_call": is_call,
        }

    # ------------------------------------------------------------------
    # Fetch consolidated BBO snapshot near market close (15:49 ET)
    # Uses cbbo-1m with a 1-minute window for real bid/ask at low cost
    # ------------------------------------------------------------------
    snap_start = snap_date.strftime("%Y-%m-%dT15:49:00")
    snap_end = snap_date.strftime("%Y-%m-%dT15:50:00")

    bbo_data = client.timeseries.get_range(
        dataset="OPRA.PILLAR",
        symbols=f"{symbol}.OPT",
        stype_in="parent",
        schema="cbbo-1m",
        start=snap_start,
        end=snap_end,
    )
    bbo_df = bbo_data.to_df()

    if bbo_df.empty:
        raise RuntimeError("No BBO data found for %s options on %s" % (symbol, start))

    # Keep the last snapshot per contract (closest to close)
    iid_col = "instrument_id" if "instrument_id" in bbo_df.columns else "hd.instrument_id"
    bbo_df = bbo_df.sort_index().groupby(iid_col).last()

    # ------------------------------------------------------------------
    # Fetch daily volume from ohlcv-1d (cheap, needed for vega*volume
    # weighting in surface calibration)
    # ------------------------------------------------------------------
    vol_data = client.timeseries.get_range(
        dataset="OPRA.PILLAR",
        symbols=f"{symbol}.OPT",
        stype_in="parent",
        schema="ohlcv-1d",
        start=start,
        end=end,
    )
    vol_df = vol_data.to_df()
    vol_iid_col = "instrument_id" if "instrument_id" in vol_df.columns else "hd.instrument_id"
    volume_map = dict(zip(vol_df[vol_iid_col], vol_df["volume"]))

    # ------------------------------------------------------------------
    # Build OptionQuote list
    # ------------------------------------------------------------------
    quotes = []
    for iid, row in bbo_df.iterrows():
        meta = instrument_meta.get(iid)
        if meta is None:
            continue

        bid = _extract_price(row, "bid")
        ask = _extract_price(row, "ask")

        if bid <= 0 or ask <= bid:
            continue

        vol = volume_map.get(iid, 0)
        volume = int(vol) if not pd.isna(vol) else 0

        quotes.append(OptionQuote(
            strike=meta["strike"],
            expiry=meta["expiry"],
            is_call=meta["is_call"],
            bid=bid,
            ask=ask,
            volume=volume,
            open_interest=0,
        ))

    print("  [databento] Fetched %d quotes across %d expiries for %s (spot=%.2f)"
          % (len(quotes), len(set(q.expiry for q in quotes)), symbol, spot))

    return OptionChain(
        underlying=symbol,
        spot=float(spot),
        quotes=quotes,
        as_of=start,
    )


def _extract_price(row, side):
    """
    Extract bid or ask price from a Databento mbp-1 row.
    Handles both nested and flattened column naming conventions.
    """
    # Try flattened naming first (bid_px_00 / ask_px_00)
    for col in [f"{side}_px_00", f"{side}_px", f"levels[0].{side}_px", side]:
        val = row.get(col, None)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            price = float(val)
            # Databento fixed-point: if value looks scaled, divide
            if price > 1e6:
                price = price / 1e9
            return price
    return 0.0
