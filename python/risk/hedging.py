
"""
hedging.py
==========
Multi-instrument hedging optimizer with efficient frontier.
Uses scenario-based P&L variance as the objective, with live bid-ask costs.
"""

import numpy as np
from scipy.optimize import minimize


def build_scenario_pnl_matrix(hedge_universe, spot_shocks, vol_shocks,
                              surface_eval_fn, bs_price_fn, curve_rate_fn, spot):
    """
    Build P&L-per-unit matrix for each hedge instrument across scenarios.
    Returns shape (n_scenarios, n_instruments).
    """
    scenarios = [(ds, dv) for ds in spot_shocks for dv in vol_shocks]
    n_scenarios = len(scenarios)
    n_instruments = len(hedge_universe)
    pnl_matrix = np.zeros((n_scenarios, n_instruments))

    for j, inst in enumerate(hedge_universe):
        K = inst["strike"]
        T = inst["expiry_years"]
        is_call = inst["is_call"]
        F = inst["forward"]
        r = curve_rate_fn(T)
        k0 = np.log(K / F)
        sigma0 = surface_eval_fn(k0, T)
        base_px = bs_price_fn(F, K, T, r, sigma0, is_call)

        for i, (ds, dv) in enumerate(scenarios):
            F_new = F * (1.0 + ds / 100.0)
            k_new = np.log(K / F_new)
            sigma_new = max(surface_eval_fn(k_new, T) + dv, 0.01)
            px_new = bs_price_fn(F_new, K, T, r, sigma_new, is_call)
            pnl_matrix[i, j] = px_new - base_px

    return pnl_matrix


def compute_efficient_frontier(portfolio_pnl_scenarios, hedge_pnl_matrix,
                               hedge_costs, budget_steps=20):
    """
    Sweep cost budgets and solve for optimal hedge at each level.

    Parameters
    ----------
    portfolio_pnl_scenarios : (n_scenarios,) existing portfolio P&L
    hedge_pnl_matrix : (n_scenarios, n_instruments) P&L per unit
    hedge_costs : (n_instruments,) half-spread cost per unit
    budget_steps : number of budget levels

    Returns list of dicts: budget, residual_variance, optimal_quantities, actual_cost
    """
    max_budget = float(np.sum(hedge_costs) * 50)
    budgets = np.linspace(0, max_budget, budget_steps)
    n_inst = hedge_pnl_matrix.shape[1]
    frontier = []

    for budget in budgets:
        def objective(h):
            hedged_pnl = portfolio_pnl_scenarios + hedge_pnl_matrix.dot(h)
            return float(np.var(hedged_pnl))

        def cost_constraint(h):
            return budget - np.sum(np.abs(h) * hedge_costs)

        res = minimize(
            objective,
            x0=np.zeros(n_inst),
            method="SLSQP",
            constraints=[{"type": "ineq", "fun": cost_constraint}],
            options={"maxiter": 300, "ftol": 1e-10},
        )

        frontier.append({
            "budget": float(budget),
            "residual_variance": float(res.fun),
            "optimal_quantities": np.round(res.x, 4).tolist(),
            "actual_cost": float(np.sum(np.abs(res.x) * hedge_costs)),
        })

    return frontier
