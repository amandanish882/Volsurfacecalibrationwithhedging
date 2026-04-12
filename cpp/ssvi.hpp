
#pragma once

#include <cmath>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <string>

namespace ssvi {

/**
 * SSVI (Surface Stochastic Volatility Inspired) parameterisation.
 *
 * Total variance:
 *   w(k, theta, rho, eta) = (theta/2)(1 + rho*phi*k + sqrt((phi*k+rho)^2 + 1-rho^2))
 *
 * Power-law mixing function:
 *   phi = eta / theta^gamma,  gamma = 0.5  (Gatheral-Jacquier default)
 *
 * Parameters
 * ----------
 *   k      : log-moneyness  ln(K / F)
 *   theta  : ATM total variance (sigma_ATM^2 * T), must be > 0
 *   rho    : skew parameter, in (-1, 1)
 *   eta    : curvature parameter, must be > 0
 */

/** Compute SSVI total variance at log-moneyness k. */
double total_variance(double k, double theta, double rho, double eta);

/**
 * SSVIDerivatives: analytical first and second partial derivatives.
 *
 * Used for Gatheral-Jacquier butterfly arbitrage checks and
 * chain-rule Greeks through the surface.
 */
struct SSVIDerivatives {
    double w         = 0.0;   // total variance
    double dw_dk     = 0.0;   // first partial w.r.t. log-moneyness
    double d2w_dk2   = 0.0;   // second partial w.r.t. log-moneyness
    double dw_dtheta = 0.0;   // partial w.r.t. ATM variance (calendar arb)
};

/** Compute total variance and all analytical partials at (k, theta, rho, eta). */
SSVIDerivatives derivatives(double k, double theta, double rho, double eta);

/** Vectorised total variance over a grid of log-moneyness values. */
std::vector<double> surface_vec(const std::vector<double>& k_vec,
                                double theta, double rho, double eta);

} // namespace ssvi
