
"""
fred_rates.py
=============
Fetch the full US Treasury yield curve from FRED and bootstrap
a continuous zero-rate term structure via cubic spline.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from fredapi import Fred

from python.types import YieldCurve
from python.data._cache import load_or_fetch

SERIES = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 3 / 12,
    "DGS6MO": 6 / 12,
    "DGS1":   1.0,
    "DGS2":   2.0,
    "DGS3":   3.0,
    "DGS5":   5.0,
    "DGS7":   7.0,
    "DGS10":  10.0,
    "DGS20":  20.0,
    "DGS30":  30.0,
}


def fetch_yield_curve(api_key, valuation_date=None):
    """
    Pull the latest observation for each Treasury tenor from FRED,
    convert par yields to continuous zero rates, return YieldCurve.
    Raises RuntimeError if any series is unavailable.
    """
    def _fetch():
        fred = Fred(api_key=api_key)
        tenors, par_yields = [], []
        as_of_date = None

        for series_id, T in SERIES.items():
            obs = fred.get_series(series_id).dropna()
            if obs.empty:
                raise RuntimeError("FRED series %s returned no data." % series_id)
            if as_of_date is None:
                as_of_date = obs.index[-1].date()
            tenors.append(T)
            par_yields.append(obs.iloc[-1] / 100.0)

        tenors_arr = np.array(tenors)
        par_yields_arr = np.array(par_yields)

        zero_rates = 2.0 * np.log(1.0 + par_yields_arr / 2.0)

        cs = CubicSpline(tenors_arr, zero_rates, bc_type="natural")
        fine_tenors = np.linspace(tenors_arr[0], tenors_arr[-1], 200)
        fine_rates = cs(fine_tenors)

        as_of = str(as_of_date)

        return YieldCurve(tenors=fine_tenors, zero_rates=fine_rates, as_of_date=as_of)

    return load_or_fetch("fred_yield_curve", valuation_date, _fetch)
