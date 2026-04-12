
"""
warm_start.py
=============
Gradient-boosted tree model that predicts SSVI parameter updates from
observable market features, providing a warm-start to the constrained
calibration optimizer.

Arb-free guarantee: UNCONDITIONALLY PRESERVED.
This model only supplies an initial guess to the L-BFGS-B optimizer,
which still runs to convergence with full arb penalty enforcement.
A bad prediction simply means more optimizer iterations — never an
arb-violating surface.

Training data is self-generated: run the classical calibration on N days
of historical CBOE snapshots. Each row is:
    (features at time t) -> (calibrated params at time t)
No external labelled dataset required.

All features derive from FRED + CBOE data already in the pipeline:
    - Prior calibration output (theta, rho, eta from t-1)
    - VIX level (FRED series VIXCLS)
    - Yield curve slope (DGS10 - DGS2)
    - Realised vol: 1d, 5d, 20d (from CBOE underlying close history)
    - Put-call volume ratio (from option chain)
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from python.types import SSVIParams


FEATURE_COLS = [
    "prev_theta", "prev_rho", "prev_eta",
    "vix",
    "yield_slope_10y_2y",
    "rvol_1d", "rvol_5d", "rvol_20d",
    "put_call_volume_ratio",
]

TARGET_COLS = ["d_theta", "d_rho", "d_eta"]


class WarmStartModel:
    """
    Predicts SSVI parameter deltas from market features.

    Usage:
        model = WarmStartModel()
        model.train(history_df)          # one-time on historical data
        model.save("warm_start.pkl")

        model = WarmStartModel.load("warm_start.pkl")
        predicted = model.predict(features_dict, prev_params)
        # predicted is an SSVIParams used as initial guess
    """

    def __init__(self, n_estimators=200, max_depth=4, learning_rate=0.05):
        base = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
        )
        self.model = MultiOutputRegressor(base)
        self.is_trained = False

    def train(self, history_df):
        """
        Train on historical calibration log.

        Parameters
        ----------
        history_df : DataFrame with columns FEATURE_COLS + TARGET_COLS.
            TARGET_COLS are the *changes* in SSVI params from t-1 to t:
                d_theta = theta_t - theta_{t-1}
                d_rho   = rho_t   - rho_{t-1}
                d_eta   = eta_t   - eta_{t-1}
            Predicting deltas rather than levels makes the model more
            stable and generalisable across vol regimes.
        """
        missing = set(FEATURE_COLS + TARGET_COLS) - set(history_df.columns)
        if missing:
            raise ValueError("Training data missing columns: %s" % missing)

        X = history_df[FEATURE_COLS].values
        y = history_df[TARGET_COLS].values

        # Drop rows with NaN (first row has no prev, etc.)
        mask = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(y), axis=1)
        X, y = X[mask], y[mask]

        if len(X) < 20:
            raise ValueError("Need at least 20 valid training rows, got %d." % len(X))

        self.model.fit(X, y)
        self.is_trained = True
        print("  [warm_start] Trained on %d samples." % len(X))

    def predict(self, features, prev_params):
        """
        Predict SSVI params for the current snapshot.

        Parameters
        ----------
        features : dict with keys matching FEATURE_COLS
            (prev_theta/rho/eta will be filled from prev_params)
        prev_params : SSVIParams from last calibration

        Returns
        -------
        SSVIParams — predicted initial guess for the optimizer.
        """
        if not self.is_trained:
            return prev_params  # untrained model is a no-op

        row = {
            "prev_theta": prev_params.theta,
            "prev_rho":   prev_params.rho,
            "prev_eta":   prev_params.eta,
        }
        row.update(features)

        X = np.array([[row[c] for c in FEATURE_COLS]])
        deltas = self.model.predict(X)[0]

        # Apply deltas to prev params, clamp to valid bounds
        theta = max(prev_params.theta + deltas[0], 1e-5)
        rho   = max(-0.99, min(0.99, prev_params.rho + deltas[1]))
        eta   = max(1e-4, prev_params.eta + deltas[2])

        return SSVIParams(theta=theta, rho=rho, eta=eta)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "is_trained": self.is_trained}, f)

    def load_from(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.is_trained = data["is_trained"]
        return self


def build_training_set(calibration_log, market_features_log):
    """
    Join calibration history with market features to produce training data.

    Parameters
    ----------
    calibration_log : DataFrame with columns [date, tenor, theta, rho, eta]
        One row per (date, tenor) from historical calibration runs.
    market_features_log : DataFrame with columns [date, vix, yield_slope_10y_2y,
        rvol_1d, rvol_5d, rvol_20d, put_call_volume_ratio]
        One row per date.

    Returns
    -------
    DataFrame ready for WarmStartModel.train().
    """
    cal = calibration_log.sort_values(["tenor", "date"]).copy()

    # Compute deltas within each tenor group
    for col in ["theta", "rho", "eta"]:
        cal["prev_" + col] = cal.groupby("tenor")[col].shift(1)
        cal["d_" + col] = cal[col] - cal["prev_" + col]

    cal = cal.dropna(subset=["prev_theta", "d_theta"])

    merged = cal.merge(market_features_log, on="date", how="inner")
    return merged


def _atm_iv_at_tenor(chain, curve, target_days=30):
    """Find the near-ATM call closest to target_days and return its IV."""
    from qr_engine.greeks import bs_implied_vol as _bs_iv
    best = None
    for q in chain.quotes:
        if not q.is_call:
            continue
        days = (q.expiry - pd.Timestamp.now()).days
        if days < 2:
            continue
        tenor_dist = abs(days - target_days)
        moneyness = abs(q.strike / chain.spot - 1.0)
        score = tenor_dist / 30.0 + moneyness  # prefer close tenor + ATM
        if best is None or score < best[0]:
            best = (score, q)
    if best is None:
        return None
    q = best[1]
    T = (q.expiry - pd.Timestamp.now()).days / 365.25
    try:
        iv = _bs_iv(q.mid(), chain.spot, q.strike, T, curve.rate(T), True)
        if 0.01 < iv < 3.0:
            return iv
    except Exception:
        pass
    return None


def extract_live_features(curve, chain, fred_api_key=None):
    """
    Compute the market-feature vector for the current snapshot
    from data already available in the pipeline.

    Parameters
    ----------
    curve : YieldCurve (already fetched from FRED)
    chain : OptionChain (already loaded from CBOE)
    fred_api_key : str, optional. If provided, fetches VIXCLS from FRED.
        If None, VIX is approximated from ATM implied vol.

    Returns
    -------
    dict with keys matching FEATURE_COLS (excluding prev_theta/rho/eta,
    which are filled by the model from prev_params).
    """
    # Yield curve slope
    r_10y = curve.rate(10.0)
    r_2y  = curve.rate(2.0)
    yield_slope = (r_10y - r_2y) * 100.0  # in percentage points

    # VIX from FRED if key available, else approximate from ATM vol
    vix = None
    if fred_api_key is not None:
        from fredapi import Fred
        fred = Fred(api_key=fred_api_key)
        vix_series = fred.get_series("VIXCLS").dropna()
        if not vix_series.empty:
            vix = float(vix_series.iloc[-1])

    if vix is None:
        # Approximate: find near-ATM, near-30d call, use its IV * 100
        best = None
        for q in chain.quotes:
            if not q.is_call:
                continue
            T = (q.expiry - pd.Timestamp.now()).days / 365.25
            if T < 0.05 or T > 0.15:
                continue
            moneyness = abs(q.strike / chain.spot - 1.0)
            if best is None or moneyness < best[0]:
                best = (moneyness, q)
        if best is not None:
            from qr_engine.greeks import bs_implied_vol
            q = best[1]
            T = (q.expiry - pd.Timestamp.now()).days / 365.25
            iv = bs_implied_vol(q.mid(), chain.spot, q.strike, T, curve.rate(T), True)
            vix = iv * 100.0
        else:
            vix = 20.0  # last resort neutral assumption

    # Put-call volume ratio
    call_vol = sum(q.volume for q in chain.quotes if q.is_call)
    put_vol  = sum(q.volume for q in chain.quotes if not q.is_call)
    pc_ratio = put_vol / max(call_vol, 1.0)

    # Realised vol proxies from ATM implied vols at different tenors.
    # In production, compute from historical daily closes. Here we use
    # near-ATM IV at short tenors as a reasonable proxy.
    rvol_1d = _atm_iv_at_tenor(chain, curve, target_days=5) or vix / 100.0
    rvol_5d = _atm_iv_at_tenor(chain, curve, target_days=10) or vix / 100.0
    rvol_20d = _atm_iv_at_tenor(chain, curve, target_days=30) or vix / 100.0

    return {
        "vix": vix,
        "yield_slope_10y_2y": yield_slope,
        "rvol_1d": rvol_1d,
        "rvol_5d": rvol_5d,
        "rvol_20d": rvol_20d,
        "put_call_volume_ratio": pc_ratio,
    }
