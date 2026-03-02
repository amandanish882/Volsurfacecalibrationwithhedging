
#pragma once

#include <cmath>
#include <vector>
#include <stdexcept>
#include <string>

namespace greeks {

/**
 * Black-Scholes pricing and finite-difference Greeks engine.
 *
 * All functions take the forward price F rather than spot S.
 * The discount factor e^{-rT} is applied internally.
 *
 * Conventions
 * -----------
 *   F      : forward price of the underlying
 *   K      : strike price
 *   T      : time to expiry in years
 *   r      : risk-free rate (continuous compounding)
 *   sigma  : Black implied volatility
 *   is_call: true = call, false = put
 */

/**
 * Black-Scholes European option price.
 *
 * Returns the discounted expected payoff under log-normal dynamics.
 * Handles T -> 0 and sigma -> 0 gracefully (returns intrinsic value).
 */
double bs_price(double F, double K, double T, double r, double sigma, bool is_call);

/**
 * Newton-Raphson implied volatility solver.
 *
 * Inverts bs_price to recover the Black implied vol from a market price.
 * Starts from sigma = 20% and iterates until |step| < tol.
 */
double bs_implied_vol(double price, double F, double K, double T, double r,
                      bool is_call, double tol = 1e-8, int max_iter = 100);

/**
 * GreekResult: full set of finite-difference Greeks for one option.
 *
 * Computed via central differences on bs_price.
 */
struct GreekResult {
    double delta = 0.0;   // dV/dS
    double gamma = 0.0;   // d2V/dS2
    double vega  = 0.0;   // dV/dsigma
    double theta = 0.0;   // dV/dT  (1-day decay)
    double vanna = 0.0;   // d2V/dS dsigma
    double volga = 0.0;   // d2V/dsigma2
};

/**
 * Compute all Greeks via central finite differences.
 *
 * Bump sizes are configurable; defaults are standard desk conventions.
 */
GreekResult compute(double F, double K, double T, double r, double sigma,
                    bool is_call,
                    double dS_frac = 0.005,
                    double dSigma  = 0.005,
                    double dT      = 1.0/365.0);

} // namespace greeks
