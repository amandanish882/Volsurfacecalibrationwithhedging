
#include "greeks.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace greeks {

static constexpr double PI = 3.14159265358979323846;

// ------------------------------------------------------------------ //
//  Utilities
// ------------------------------------------------------------------ //

static double norm_cdf(double x) noexcept {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

// ------------------------------------------------------------------ //
//  Black-Scholes pricing
// ------------------------------------------------------------------ //

double bs_price(double F, double K, double T, double r, double sigma, bool is_call) {
    if (K <= 0.0) {
        throw std::invalid_argument(
            "greeks::bs_price: strike must be > 0, got " + std::to_string(K));
    }

    // Near-expiry or near-zero vol: return discounted intrinsic
    if (T <= 1e-10 || sigma <= 1e-10) {
        const double intrinsic = is_call ? std::max(F - K, 0.0) : std::max(K - F, 0.0);
        return intrinsic * std::exp(-r * T);
    }

    const double sv = sigma * std::sqrt(T);
    const double d1 = (std::log(F / K) + 0.5 * sigma * sigma * T) / sv;
    const double d2 = d1 - sv;
    const double df = std::exp(-r * T);

    if (is_call)
        return df * (F * norm_cdf(d1) - K * norm_cdf(d2));
    else
        return df * (K * norm_cdf(-d2) - F * norm_cdf(-d1));
}

// ------------------------------------------------------------------ //
//  Implied volatility (Newton-Raphson)
// ------------------------------------------------------------------ //

double bs_implied_vol(double price, double F, double K, double T, double r,
                      bool is_call, double tol, int max_iter) {
    if (price <= 0.0) return 0.0;

    double sigma = 0.20;
    for (int i = 0; i < max_iter; ++i) {
        const double p  = bs_price(F, K, T, r, sigma, is_call);
        const double sv = sigma * std::sqrt(T);
        const double d1 = (std::log(F / K) + 0.5 * sigma * sigma * T) / sv;

        // Vega = F * df * sqrt(T) * n(d1)
        const double vega_val = F * std::exp(-r * T) * std::sqrt(T)
                              * std::exp(-0.5 * d1 * d1) / std::sqrt(2.0 * PI);
        if (vega_val < 1e-14) break;

        const double step = (p - price) / vega_val;
        sigma -= step;
        sigma = std::max(sigma, 1e-6);
        if (std::abs(step) < tol) break;
    }
    return sigma;
}

// ------------------------------------------------------------------ //
//  Finite-difference Greeks
// ------------------------------------------------------------------ //

GreekResult compute(double F, double K, double T, double r, double sigma,
                    bool is_call, double dS_frac, double dSigma, double dT) {
    GreekResult g{};
    const double dS = F * dS_frac;
    const double V0 = bs_price(F, K, T, r, sigma, is_call);

    // Delta, Gamma  (spot bump)
    const double Vup = bs_price(F + dS, K, T, r, sigma, is_call);
    const double Vdn = bs_price(F - dS, K, T, r, sigma, is_call);
    g.delta = (Vup - Vdn) / (2.0 * dS);
    g.gamma = (Vup - 2.0 * V0 + Vdn) / (dS * dS);

    // Vega  (vol bump)
    const double Vvu = bs_price(F, K, T, r, sigma + dSigma, is_call);
    const double Vvd = bs_price(F, K, T, r, sigma - dSigma, is_call);
    g.vega = (Vvu - Vvd) / (2.0 * dSigma);

    // Theta  (1-day time decay)
    const double T1 = std::max(T - dT, 1e-10);
    const double Vt = bs_price(F, K, T1, r, sigma, is_call);
    g.theta = (Vt - V0) / (-dT);

    // Vanna  (cross-partial: spot x vol)
    const double Vup_vu = bs_price(F + dS, K, T, r, sigma + dSigma, is_call);
    const double Vup_vd = bs_price(F + dS, K, T, r, sigma - dSigma, is_call);
    const double Vdn_vu = bs_price(F - dS, K, T, r, sigma + dSigma, is_call);
    const double Vdn_vd = bs_price(F - dS, K, T, r, sigma - dSigma, is_call);
    g.vanna = (Vup_vu - Vup_vd - Vdn_vu + Vdn_vd) / (4.0 * dS * dSigma);

    // Volga  (second-order vol)
    g.volga = (Vvu - 2.0 * V0 + Vvd) / (dSigma * dSigma);

    return g;
}

} // namespace greeks
