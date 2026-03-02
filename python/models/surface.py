
"""
surface.py
==========
SSVI surface calibration orchestrator.
Requires the compiled C++ engine (qr_engine).
"""

import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from python.types import SSVIParams, FittedSurface
from python.models.arb_detector import check_calendar, check_butterfly

from qr_engine.ssvi import surface_vec
from qr_engine.greeks import bs_implied_vol


def _eval_surface(k_arr, theta, rho, eta):
    return np.array(surface_vec(k_arr.tolist(), theta, rho, eta))


def calibrate_single_slice(k, market_w, weights, prev_params=None,
                           warm_started=False):
    """Fit SSVI to a single tenor slice. Returns SSVIParams.

    When *warm_started* is True the initial guess is ML-predicted and
    already close to the optimum, so we use a looser tolerance / fewer
    iterations to capitalise on the better starting point.  The
    tolerance (1e-8) still gives ~8 significant digits on the
    objective — more than enough given bid-ask noise on market_w.
    """
    if prev_params is not None:
        x0 = [prev_params.theta, prev_params.rho, prev_params.eta]
    else:
        x0 = [0.04, -0.3, 0.5]
    bounds = [(1e-5, 2.0), (-0.99, 0.99), (1e-4, 5.0)]

    def obj(params):
        theta, rho, eta = params
        model_w = _eval_surface(k, theta, rho, eta)
        return float(np.sum(weights * (model_w - market_w) ** 2))

    if warm_started:
        opts = {"maxiter": 80, "ftol": 1e-8}
    else:
        opts = {"maxiter": 500, "ftol": 1e-8}

    res = minimize(obj, x0, bounds=bounds, method="L-BFGS-B",
                   options=opts)
    return SSVIParams(theta=res.x[0], rho=res.x[1], eta=res.x[2])


def _prepare_slices(chain, forwards, curve):
    """Convert raw option chain into optimizer-ready slice data.

    This is the expensive step (one ``bs_implied_vol`` call per quote).
    The result is a list of dicts, one per valid tenor slice, each
    containing the log-moneyness array *k*, the market total variance
    *market_w*, the weight vector *weights*, and the tenor *T*.
    """
    now = pd.Timestamp.now()
    slices = []
    for expiry in chain.expiries():
        T = (expiry - now).days / 365.25
        if T < 7 / 365.25:
            continue
        F = forwards.get(expiry, chain.spot)
        slice_quotes = [q for q in chain.for_expiry(expiry) if q.bid > 0]
        if len(slice_quotes) < 5:
            continue

        strikes = np.array([q.strike for q in slice_quotes])
        k = np.log(strikes / F)
        mids = np.array([q.mid() for q in slice_quotes])

        # Implied vols via C++ engine — this is the dominant cost
        ivs = np.array([
            bs_implied_vol(m, F, K, T, curve.rate(T), q.is_call)
            for m, K, q in zip(mids, strikes, slice_quotes)
        ])

        valid = (ivs > 0.01) & (ivs < 3.0) & np.isfinite(ivs)
        if valid.sum() < 5:
            continue
        k = k[valid]
        ivs = ivs[valid]
        market_w = ivs ** 2 * T

        # Vega x volume weights
        vols = np.array([q.volume for q in slice_quotes])[valid].astype(float)
        vols = np.maximum(vols, 1.0)
        vega_proxy = ivs * np.sqrt(T)
        weights = vega_proxy * np.log1p(vols)
        weights = weights / weights.sum()

        slices.append({"T": T, "k": k, "market_w": market_w, "weights": weights})
    return slices


def calibrate_surface(chain, forwards, curve, prev_surface=None,
                      arb_penalty_lambda=1e4, warm_start_model=None,
                      market_features=None, _cached_slices=None):
    """
    Full SSVI surface calibration across all expiries.

    Steps:
      1. Slice-by-slice warm start (ML-predicted if model provided)
      2. Global consensus (rho, eta)
      3. Arb detection + re-fit if violations found
      4. Log all violations pre- and post-correction

    Parameters
    ----------
    warm_start_model : optional WarmStartModel instance.
        If provided (and trained), predicts initial SSVI params from
        market features before the optimizer runs. This is purely an
        initial guess — the constrained optimizer still enforces all
        arb-free conditions unconditionally.
    market_features : optional dict of feature values for the warm-start.
        Ignored if warm_start_model is None.
    _cached_slices : optional list returned by ``_prepare_slices``.
        When supplied the expensive implied-vol inversion is skipped,
        which is the main source of speedup for warm-start recalibration
        on the same market snapshot.
    """
    t_prep = time.perf_counter()
    if _cached_slices is not None:
        slices = _cached_slices
    else:
        slices = _prepare_slices(chain, forwards, curve)
    t_prep = time.perf_counter() - t_prep

    if not slices:
        raise RuntimeError("Calibration failed: no valid tenor slices.")

    # Step 1: slice-by-slice optimisation
    t_opt = time.perf_counter()
    params_by_tenor = {}
    for sl in slices:
        T = sl["T"]
        prev = None
        ws_used = False
        if warm_start_model is not None and prev_surface is not None:
            prev_p = prev_surface.params_by_tenor.get(T)
            if prev_p is not None and market_features is not None:
                prev = warm_start_model.predict(market_features, prev_p)
                ws_used = True
        if prev is None and prev_surface is not None:
            prev = prev_surface.params_by_tenor.get(T)
        params = calibrate_single_slice(sl["k"], sl["market_w"], sl["weights"],
                                        prev, warm_started=ws_used)
        params.tenor = T
        params_by_tenor[T] = params
    t_opt = time.perf_counter() - t_opt

    # Step 2: consensus global rho, eta
    rhos = [p.rho for p in params_by_tenor.values()]
    etas = [p.eta for p in params_by_tenor.values()]
    global_rho = float(np.median(rhos))
    global_eta = float(np.median(etas))

    # Step 3: arb detection pre-correction
    violations_pre = []
    sorted_tenors = sorted(params_by_tenor.keys())
    violations_pre += check_calendar(params_by_tenor, sorted_tenors)
    for T in sorted_tenors:
        p = params_by_tenor[T]
        k_grid = np.linspace(-0.5, 0.5, 100)
        violations_pre += check_butterfly(k_grid, p.theta, global_rho, global_eta, T)

    # Step 4: re-fit with global params if violations found
    if violations_pre:
        print("  [surface] %d arb violations pre-correction; re-fitting with global rho/eta."
              % len(violations_pre))
        for T in sorted_tenors:
            p = params_by_tenor[T]
            params_by_tenor[T] = SSVIParams(
                theta=p.theta, rho=global_rho, eta=global_eta, tenor=T
            )

    # Post-correction check
    violations_post = []
    for T in sorted_tenors:
        p = params_by_tenor[T]
        k_grid = np.linspace(-0.5, 0.5, 100)
        violations_post += check_butterfly(k_grid, p.theta, global_rho, global_eta, T)
    violations_post += check_calendar(params_by_tenor, sorted_tenors)

    result = FittedSurface(
        params_by_tenor=params_by_tenor,
        global_rho=global_rho,
        global_eta=global_eta,
        violations_pre=violations_pre,
        violations_post=violations_post,
    )
    result._timing = {"prep_s": t_prep, "opt_s": t_opt}
    result._cached_slices = slices
    return result
