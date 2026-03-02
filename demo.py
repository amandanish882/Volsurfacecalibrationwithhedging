
"""
demo.py
=======
Full project demonstration using live market data.

Data sources:
    - Yield curve: FRED API (requires FRED_API_KEY in .env or environment)
    - Option chain: Yahoo Finance via yfinance
    - Dividends:    Yahoo Finance via yfinance
    - ML features:  Derived from FRED + option chain

Run from the project root:
    python demo.py
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

os.makedirs("output", exist_ok=True)

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    print("[FATAL] FRED_API_KEY not set. Add it to .env or export it.")
    sys.exit(1)

TICKER = "SPY"

# ---------------------------------------------------------------------------
# 0. Verify C++ engine
# ---------------------------------------------------------------------------
print("=" * 70)
print("QR EQUITY DERIVATIVES FLOW - PROJECT DEMO")
print("=" * 70)

try:
    import qr_engine
    print("\n[OK] qr_engine loaded: %s" % dir(qr_engine))
except ImportError as e:
    print("\n[FAIL] Cannot import qr_engine: %s" % e)
    print("       Build it first:  cd cpp && python setup.py build_ext --inplace")
    sys.exit(1)

from qr_engine.ssvi import total_variance, surface_vec, derivatives as ssvi_derivs
from qr_engine.greeks import bs_price, bs_implied_vol, compute as greeks_compute

# ---------------------------------------------------------------------------
# 1. C++ Engine Smoke Tests
# ---------------------------------------------------------------------------
print("\n" + "-" * 70)
print("1. C++ ENGINE SMOKE TESTS")
print("-" * 70)

# SSVI
theta, rho, eta = 0.04, -0.25, 1.0
k_test = 0.0
w = total_variance(k_test, theta, rho, eta)
print("  SSVI total_variance(k=%.2f, theta=%.2f, rho=%.2f, eta=%.1f) = %.6f" % (k_test, theta, rho, eta, w))

d = ssvi_derivs(k_test, theta, rho, eta)
print("  SSVI derivatives: w=%.6f, dw/dk=%.6f, d2w/dk2=%.6f, dw/dtheta=%.6f"
      % (d.w, d.dw_dk, d.d2w_dk2, d.dw_dtheta))

k_vec = np.linspace(-0.3, 0.3, 7).tolist()
w_vec = surface_vec(k_vec, theta, rho, eta)
print("  surface_vec over 7 strikes: min=%.6f, max=%.6f" % (min(w_vec), max(w_vec)))

# Greeks
F, K, T, r, sigma = 585.0, 590.0, 0.25, 0.045, 0.18
call_px = bs_price(F, K, T, r, sigma, True)
put_px = bs_price(F, K, T, r, sigma, False)
print("  BS call(F=%.0f, K=%.0f, T=%.2f, sigma=%.0f%%) = $%.4f" % (F, K, T, sigma * 100, call_px))
print("  BS put = $%.4f" % put_px)
print("  Put-call parity check: C - P = %.4f, (F-K)*df = %.4f"
      % (call_px - put_px, (F - K) * np.exp(-r * T)))

iv_back = bs_implied_vol(call_px, F, K, T, r, True)
print("  Implied vol round-trip: input=%.4f, recovered=%.4f, diff=%.2e"
      % (sigma, iv_back, abs(sigma - iv_back)))

g = greeks_compute(F, K, T, r, sigma, True)
print("  Greeks: delta=%.4f, gamma=%.6f, vega=%.4f, theta=%.4f, vanna=%.6f, volga=%.4f"
      % (g.delta, g.gamma, g.vega, g.theta, g.vanna, g.volga))

# ---------------------------------------------------------------------------
# 2. Yield Curve (FRED)
# ---------------------------------------------------------------------------
print("\n" + "-" * 70)
print("2. YIELD CURVE (FRED)")
print("-" * 70)

from python.data.fred_rates import fetch_yield_curve

curve = fetch_yield_curve(FRED_API_KEY)
print("  [LIVE] Fetched yield curve from FRED (as of %s)" % curve.as_of_date)

for t, r_val in zip([0.25, 1.0, 5.0, 10.0], [curve.rate(x) for x in [0.25, 1.0, 5.0, 10.0]]):
    print("    T=%.2fy -> r=%.3f%%, df=%.4f" % (t, r_val * 100, curve.discount(t)))

fig, ax = plt.subplots(figsize=(10, 4))
fine_t = np.linspace(curve.tenors[0], curve.tenors[-1], 200)
fine_r = np.array([curve.rate(t) for t in fine_t])
ax.plot(fine_t, fine_r * 100, "b-", linewidth=2)
ax.set_xlabel("Tenor (years)")
ax.set_ylabel("Zero Rate (%)")
ax.set_title("US Treasury Zero Curve [FRED] (as of %s)" % curve.as_of_date)
plt.tight_layout()
plt.savefig("output/01_yield_curve.png", dpi=150)
plt.close()
print("  Saved output/01_yield_curve.png")

# ---------------------------------------------------------------------------
# 3. Option Chain (yfinance)
# ---------------------------------------------------------------------------
print("\n" + "-" * 70)
print("3. OPTION CHAIN (yfinance)")
print("-" * 70)

from python.data.options_chain import fetch_yfinance

chain = fetch_yfinance(TICKER)
spot = chain.spot
print("  [LIVE] Fetched %d quotes across %d expiries for %s (spot=%.2f)"
      % (len(chain.quotes), len(chain.expiries()), TICKER, spot))

for exp in chain.expiries():
    n = len(chain.for_expiry(exp))
    T_exp = (exp - pd.Timestamp.now()).days / 365.25
    print("    %s (T=%.3fy): %d quotes" % (exp.date(), T_exp, n))

# ---------------------------------------------------------------------------
# 4. Dividend Projection + Implied Dividends
# ---------------------------------------------------------------------------
print("\n" + "-" * 70)
print("4. DIVIDEND PROJECTION + IMPLIED DIVIDENDS")
print("-" * 70)

from python.data.dividends import (
    fetch_dividends, project_from_history, extract_implied_dividends,
    decompose_implied_dividends, blend_dividends, build_forward_curve,
)

# 4a. Historical projection
hist_divs = fetch_dividends(TICKER, period="3y")
print("  [LIVE] Fetched %d historical dividends for %s" % (len(hist_divs), TICKER))

projected_hist = project_from_history(hist_divs, horizon_years=1.5)
print("  Projected %d future dividends (historical):" % len(projected_hist))
for d in projected_hist[:4]:
    print("    %s: $%.2f (%s)" % (d.ex_date.date(), d.amount, d.source))

# 4b. Market-implied dividends via put-call parity
implied_pvs = extract_implied_dividends(chain, curve)
print("  Implied cumulative div PV per expiry:")
for exp in sorted(implied_pvs.keys()):
    print("    %s: PV=$%.4f" % (exp.date(), implied_pvs[exp]))

implied_divs = decompose_implied_dividends(implied_pvs, projected_hist, spot, curve)
print("  Decomposed into %d implied dividends:" % len([d for d in implied_divs if d.source == "implied"]))
for d in [dd for dd in implied_divs if dd.source == "implied"][:4]:
    print("    %s: $%.2f (%s)" % (d.ex_date.date(), d.amount, d.source))

# 4c. Blend: prefer market-implied where available
projected = blend_dividends(projected_hist, implied_divs, method="prefer_implied")
print("  Blended dividend schedule (%d total):" % len(projected))
for d in projected[:6]:
    print("    %s: $%.2f (%s)" % (d.ex_date.date(), d.amount, d.source))

# 4d. Forward curve from blended dividends
forwards = build_forward_curve(spot, projected, curve, chain.expiries())
forwards_hist_only = build_forward_curve(spot, projected_hist, curve, chain.expiries())
print("  Forward curve (hist vs implied-adjusted):")
for exp in sorted(forwards.keys()):
    T_exp = (exp - pd.Timestamp.now()).days / 365.25
    print("    %s (T=%.3fy): F_hist=%.2f  F_implied=%.2f  diff=%.2f"
          % (exp.date(), T_exp, forwards_hist_only[exp], forwards[exp],
             forwards[exp] - forwards_hist_only[exp]))

# ---------------------------------------------------------------------------
# 5. SSVI Surface Calibration + Arb Detection
# ---------------------------------------------------------------------------
print("\n" + "-" * 70)
print("5. SSVI SURFACE CALIBRATION + ARB DETECTION")
print("-" * 70)

from python.models.surface import calibrate_surface

t0_classical = time.perf_counter()
surface = calibrate_surface(chain, forwards, curve)
t1_classical = time.perf_counter()
classical_time = t1_classical - t0_classical
print("  Calibrated %d tenor slices in %.4f seconds" % (len(surface.params_by_tenor), classical_time))
print("  Timing breakdown: prep=%.4f s (%.0f%%), opt=%.4f s (%.0f%%)"
      % (surface._timing["prep_s"],
         surface._timing["prep_s"] / classical_time * 100,
         surface._timing["opt_s"],
         surface._timing["opt_s"] / classical_time * 100))
print("  Global rho=%.4f, eta=%.4f" % (surface.global_rho, surface.global_eta))
print("  Pre-correction arb violations:  %d" % len(surface.violations_pre))
print("  Post-correction arb violations: %d" % len(surface.violations_post))

for T_cal in sorted(surface.params_by_tenor.keys()):
    p = surface.params_by_tenor[T_cal]
    print("    T=%.3fy: theta=%.4f, rho=%.4f, eta=%.4f" % (T_cal, p.theta, p.rho, p.eta))

if surface.violations_pre:
    print("  Sample violations (pre-correction):")
    for v in surface.violations_pre[:3]:
        print("    [%s] %s" % (v.violation_type, v.description))

# Plot vol surface
k_grid = np.linspace(-0.3, 0.3, 100)
fig, ax = plt.subplots(figsize=(10, 5))
for T_cal in sorted(surface.params_by_tenor.keys()):
    ivs = [surface.implied_vol(k, T_cal) * 100 for k in k_grid]
    ax.plot(k_grid, ivs, label="T=%.2fy" % T_cal)
ax.set_xlabel("Log-moneyness k")
ax.set_ylabel("Implied Vol (%)")
ax.set_title("Calibrated SSVI Volatility Surface")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("output/02_vol_surface.png", dpi=150)
plt.close()
print("  Saved output/02_vol_surface.png")

# 3D surface
from mpl_toolkits.mplot3d import Axes3D
tenor_grid = np.array(sorted(surface.params_by_tenor.keys()))
K_mesh, T_mesh = np.meshgrid(k_grid, tenor_grid)
IV_mesh = np.zeros_like(K_mesh)
for i, T_cal in enumerate(tenor_grid):
    for j, k in enumerate(k_grid):
        IV_mesh[i, j] = surface.implied_vol(k, T_cal) * 100

fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(K_mesh, T_mesh, IV_mesh, cmap="coolwarm", alpha=0.85, edgecolor="none")
ax.set_xlabel("Log-moneyness k")
ax.set_ylabel("Tenor (years)")
ax.set_zlabel("Implied Vol (%)")
ax.set_title("SSVI Surface (3D)")
ax.view_init(elev=30, azim=-60)
ax.tick_params(axis="z", pad=8)
ax.zaxis.labelpad = 2
plt.tight_layout()
plt.savefig("output/03_vol_surface_3d.png", dpi=150)
plt.close()
print("  Saved output/03_vol_surface_3d.png")

# ---------------------------------------------------------------------------
# 6. Greeks Computation
# ---------------------------------------------------------------------------
print("\n" + "-" * 70)
print("6. GREEKS COMPUTATION")
print("-" * 70)

sample_expiry = chain.expiries()[2]
T_greeks = (sample_expiry - pd.Timestamp.now()).days / 365.25
F_greeks = forwards[sample_expiry]
r_greeks = curve.rate(T_greeks)

strikes_greeks = np.arange(spot * 0.90, spot * 1.10 + 1, 2.5)
greek_rows = []
for K_g in strikes_greeks:
    k_log = np.log(K_g / F_greeks)
    sigma_g = surface.implied_vol(k_log, T_greeks)
    g = greeks_compute(F_greeks, K_g, T_greeks, r_greeks, sigma_g, True)
    greek_rows.append({
        "strike": K_g, "iv": sigma_g * 100,
        "delta": g.delta, "gamma": g.gamma,
        "vega": g.vega, "theta": g.theta,
        "vanna": g.vanna, "volga": g.volga,
    })

df_greeks = pd.DataFrame(greek_rows)
print("  Greeks for T=%.2fy (expiry %s):" % (T_greeks, sample_expiry.date()))
print(df_greeks.to_string(index=False, float_format="%.4f"))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flat, ["delta", "gamma", "vega", "theta", "vanna", "volga"]):
    ax.plot(df_greeks["strike"], df_greeks[col], "o-", markersize=3)
    ax.set_title(col.capitalize())
    ax.set_xlabel("Strike")
    ax.axvline(spot, color="gray", linestyle="--", alpha=0.5)
fig.suptitle("Greeks vs Strike (T=%.2fy)" % T_greeks, fontsize=14)
plt.tight_layout()
plt.savefig("output/04_greeks.png", dpi=150)
plt.close()
print("  Saved output/04_greeks.png")

# ---------------------------------------------------------------------------
# 7. Risk Ladder
# ---------------------------------------------------------------------------
print("\n" + "-" * 70)
print("7. RISK LADDER (2D FULL REVALUATION)")
print("-" * 70)

from python.risk.risk_ladder import compute_risk_ladder

positions = []
for K_pos, qty, is_c in [(580, 100, True), (590, -200, True), (600, 100, False)]:
    exp = chain.expiries()[2]
    T_pos = (exp - pd.Timestamp.now()).days / 365.25
    positions.append({
        "strike": float(K_pos), "expiry_years": T_pos,
        "is_call": is_c, "quantity": qty, "forward": forwards[exp],
    })

avg_strike = np.mean([p["strike"] for p in positions])
spot_shocks_pct = (np.linspace(avg_strike * 0.90, avg_strike * 1.10, 21) / spot - 1.0) * 100.0
ladder = compute_risk_ladder(positions, surface, curve, spot,
                             spot_shocks_pct=spot_shocks_pct)
print("  Portfolio base value: $%.2f" % ladder["base_value"])
print("  P&L matrix shape: %s" % str(ladder["pnl_matrix"].shape))
print("  Max gain: $%.2f, Max loss: $%.2f"
      % (ladder["pnl_matrix"].max(), ladder["pnl_matrix"].min()))

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    ladder["pnl_matrix"].T,
    xticklabels=["%.0f" % s for s in ladder["spot_levels"]],
    yticklabels=["%.1f%%" % (v * 100) for v in ladder["vol_shocks"]],
    center=0, cmap="RdYlGn", fmt=".0f", ax=ax,
    cbar_kws={"label": "P&L ($)"},
)
ax.set_xlabel("Spot Level")
ax.set_ylabel("Vol Shock")
ax.set_title("Portfolio P&L Risk Ladder")
plt.tight_layout()
plt.savefig("output/05_risk_ladder.png", dpi=150)
plt.close()
print("  Saved output/05_risk_ladder.png")

# ---------------------------------------------------------------------------
# 8. Hedging Efficient Frontier
# ---------------------------------------------------------------------------
print("\n" + "-" * 70)
print("8. HEDGING EFFICIENT FRONTIER")
print("-" * 70)

from python.risk.hedging import build_scenario_pnl_matrix, compute_efficient_frontier

spot_shocks = np.linspace(-5, 5, 11)
vol_shocks = np.linspace(-0.03, 0.03, 7)

# Portfolio scenario P&L
portfolio_pnl = []
for ds in spot_shocks:
    for dv in vol_shocks:
        S_new = spot * (1.0 + ds / 100.0)
        total = 0.0
        for pos in positions:
            K_h = pos["strike"]
            T_h = pos["expiry_years"]
            F_h = S_new / spot * pos["forward"]
            k_h = np.log(K_h / F_h)
            sig_h = max(surface.implied_vol(k_h, T_h) + dv, 0.01)
            r_h = curve.rate(T_h)
            px = bs_price(F_h, K_h, T_h, r_h, sig_h, pos["is_call"])
            total += px * pos["quantity"]
        portfolio_pnl.append(total)
base_portfolio = portfolio_pnl[len(portfolio_pnl) // 2]
portfolio_pnl = np.array(portfolio_pnl) - base_portfolio

# Hedge universe
hedge_universe = []
for K_hu, is_c in [(570, True), (585, True), (600, False), (610, False)]:
    exp = chain.expiries()[2]
    T_hu = (exp - pd.Timestamp.now()).days / 365.25
    hedge_universe.append({
        "strike": float(K_hu), "expiry_years": T_hu,
        "is_call": is_c, "forward": forwards[exp],
    })

hedge_pnl = build_scenario_pnl_matrix(
    hedge_universe, spot_shocks, vol_shocks,
    surface.implied_vol, bs_price, curve.rate, spot,
)
hedge_costs = np.array([0.10, 0.15, 0.12, 0.08])

frontier = compute_efficient_frontier(portfolio_pnl, hedge_pnl, hedge_costs, budget_steps=25)

costs_f = [f["actual_cost"] for f in frontier]
variances_f = [f["residual_variance"] for f in frontier]
print("  Computed %d frontier points" % len(frontier))
print("  Unhedged variance: %.2f" % variances_f[0])
print("  Best hedged variance: %.2f (cost $%.2f)" % (min(variances_f), costs_f[variances_f.index(min(variances_f))]))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(costs_f, variances_f, "o-", color="#2E75B6", linewidth=2, markersize=5)
ax.set_xlabel("Transaction Cost ($)")
ax.set_ylabel("Residual Portfolio Variance")
ax.set_title("Hedging Efficient Frontier")
plt.tight_layout()
plt.savefig("output/06_efficient_frontier.png", dpi=150)
plt.close()
print("  Saved output/06_efficient_frontier.png")

# ---------------------------------------------------------------------------
# 9. ETF Premium Analysis
# ---------------------------------------------------------------------------
print("\n" + "-" * 70)
print("9. ETF PREMIUM / DISCOUNT ANALYSIS")
print("-" * 70)

from python.models.etf_premium import estimate_premium, adjust_surface_for_premium

nav = spot * 0.998
nav_fwd = forwards[chain.expiries()[2]] * (nav / spot)
premium = estimate_premium(spot, nav, forwards[chain.expiries()[2]], nav_fwd, T_greeks)
print("  %s" % premium["description"])

raw_iv = 0.18
adj_iv = adjust_surface_for_premium(raw_iv, premium["spot_premium_pct"], T_greeks)
print("  Raw IV: %.2f%% -> Premium-adjusted IV: %.2f%%" % (raw_iv * 100, adj_iv * 100))

# ---------------------------------------------------------------------------
# 10. ML WARM-START (SYNTHETIC DEMO)
# ---------------------------------------------------------------------------
print("\n" + "-" * 70)
print("10. ML WARM-START (SYNTHETIC DEMO)")
print("-" * 70)

from python.ml.warm_start import WarmStartModel

# Build synthetic training data from the already-calibrated surface
print("  Building synthetic training set from calibrated surface ...")
train_rows = []
rng = np.random.default_rng(42)
for i in range(200):
    vix = 15.0 + rng.normal(0, 5)
    slope = 0.01 + rng.normal(0, 0.005)
    rvol_1d = 0.15 + rng.normal(0, 0.03)
    rvol_5d = 0.14 + rng.normal(0, 0.025)
    rvol_20d = 0.13 + rng.normal(0, 0.02)
    pcr = 0.8 + rng.normal(0, 0.15)
    for T_train, params_train in surface.params_by_tenor.items():
        theta_prev = params_train.theta * (1 + rng.normal(0, 0.05))
        rho_prev = np.clip(params_train.rho + rng.normal(0, 0.02), -0.99, 0.99)
        eta_prev = max(params_train.eta * (1 + rng.normal(0, 0.05)), 0.01)
        train_rows.append({
            "vix": vix, "yield_slope_10y_2y": slope,
            "rvol_1d": rvol_1d, "rvol_5d": rvol_5d, "rvol_20d": rvol_20d,
            "put_call_volume_ratio": pcr,
            "prev_theta": theta_prev, "prev_rho": rho_prev, "prev_eta": eta_prev,
            "d_theta": params_train.theta - theta_prev,
            "d_rho": params_train.rho - rho_prev,
            "d_eta": params_train.eta - eta_prev,
        })

train_df = pd.DataFrame(train_rows)
print("  Training samples: %d" % len(train_df))

ws_model = WarmStartModel(n_estimators=100, max_depth=3)
ws_model.train(train_df)
print("  [OK] WarmStartModel trained")

# Re-calibrate with ML warm-start
from python.models.surface import calibrate_surface, _prepare_slices

features = {
    "vix": 18.0, "yield_slope_10y_2y": 0.012,
    "rvol_1d": 0.16, "rvol_5d": 0.15, "rvol_20d": 0.14,
    "put_call_volume_ratio": 0.85,
}

# Classical (cold start, no prior surface)
t0 = time.perf_counter()
surface_cold = calibrate_surface(chain, forwards, curve)
cold_time = time.perf_counter() - t0

# ML warm-start (reuses cached slices -- skips implied vol inversion)
t0 = time.perf_counter()
surface_ml = calibrate_surface(
    chain, forwards, curve,
    prev_surface=surface,
    warm_start_model=ws_model,
    market_features=features,
    _cached_slices=surface_cold._cached_slices,
)
ml_time = time.perf_counter() - t0
print("  [OK] ML warm-start recalibration: %d slices in %.4f seconds" % (len(surface_ml.params_by_tenor), ml_time))

# Timing comparison
print("\n  Calibration time comparison:")
print("    Classical:       %.4f seconds" % cold_time)
print("    ML warm-start:   %.4f seconds" % ml_time)
speedup = cold_time / ml_time if ml_time > 0 else float("inf")
print("    Speedup:         %.2fx" % speedup)

# Compare classical vs ML-assisted
print("\n  Classical vs ML-assisted parameters:")
print("  %-8s  %-24s  %-24s" % ("Tenor", "Classical (theta,rho,eta)", "ML-warm (theta,rho,eta)"))
common_tenors = sorted(set(surface_cold.params_by_tenor.keys()) & set(surface_ml.params_by_tenor.keys()))
for T_cmp in common_tenors:
    p1 = surface_cold.params_by_tenor[T_cmp]
    p2 = surface_ml.params_by_tenor[T_cmp]
    print("  %-8.3f  (%.4f, %+.4f, %.4f)  (%.4f, %+.4f, %.4f)" % (
        T_cmp,
        p1.theta, p1.rho, p1.eta,
        p2.theta, p2.rho, p2.eta,
    ))

# Save model
ws_model.save("output/warm_start.pkl")
print("\n  [OK] Model saved to output/warm_start.pkl")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DEMO COMPLETE")
print("=" * 70)
print("  Data sources:")
print("    Yield curve:   FRED (as of %s)" % curve.as_of_date)
print("    Option chain:  yfinance (%s, spot=%.2f)" % (TICKER, spot))
print("    Dividends:     yfinance (%s, %d historical)" % (TICKER, len(hist_divs)))
print("  All plots saved in output/ directory:")
print("    01_yield_curve.png")
print("    02_vol_surface.png")
print("    03_vol_surface_3d.png")
print("    04_greeks.png")
print("    05_risk_ladder.png")
print("    06_efficient_frontier.png")
print("    warm_start.pkl")
