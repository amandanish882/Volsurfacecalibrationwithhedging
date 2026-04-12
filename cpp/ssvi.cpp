
#include "ssvi.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace ssvi {

// ------------------------------------------------------------------ //
//  Internal helpers (not part of the public API)
// ------------------------------------------------------------------ //

/** Power-law mixing function:  phi = eta / theta^gamma */
static double phi_func(double theta, double eta, double gamma = 0.5) noexcept {
    return eta / std::pow(std::max(theta, 1e-12), gamma);
}

// ------------------------------------------------------------------ //
//  Core evaluation
// ------------------------------------------------------------------ //

double total_variance(double k, double theta, double rho, double eta) {
    if (theta <= 0.0) {
        throw std::invalid_argument(
            "ssvi::total_variance: theta must be > 0, got " + std::to_string(theta));
    }
    const double phi  = phi_func(theta, eta);
    const double a    = phi * k + rho;
    const double disc = std::sqrt(a * a + (1.0 - rho * rho));
    return (theta / 2.0) * (1.0 + rho * phi * k + disc);
}

// ------------------------------------------------------------------ //
//  Analytical derivatives
// ------------------------------------------------------------------ //

SSVIDerivatives derivatives(double k, double theta, double rho, double eta) {
    if (theta <= 0.0) {
        throw std::invalid_argument(
            "ssvi::derivatives: theta must be > 0, got " + std::to_string(theta));
    }
    SSVIDerivatives d{};
    const double phi  = phi_func(theta, eta);
    const double a    = phi * k + rho;
    const double b    = 1.0 - rho * rho;
    const double disc = std::sqrt(a * a + b);

    // w(k)
    d.w = (theta / 2.0) * (1.0 + rho * phi * k + disc);

    // dw/dk
    const double ddisc_dk = (phi * a) / disc;
    d.dw_dk = (theta / 2.0) * (rho * phi + ddisc_dk);

    // d2w/dk2
    const double d2disc_dk2 = (phi * phi * b) / (disc * disc * disc);
    d.d2w_dk2 = (theta / 2.0) * d2disc_dk2;

    // dw/dtheta  (needed for calendar arb detection)
    constexpr double gamma       = 0.5;
    const double dphi_dtheta     = -gamma * phi / std::max(theta, 1e-12);
    const double da_dtheta       = dphi_dtheta * k;
    const double ddisc_dtheta    = (a * da_dtheta) / disc;
    d.dw_dtheta = d.w / (2.0 * theta)
                + (theta / 2.0) * (rho * dphi_dtheta * k + ddisc_dtheta);

    return d;
}

// ------------------------------------------------------------------ //
//  Vectorised surface
// ------------------------------------------------------------------ //

std::vector<double> surface_vec(const std::vector<double>& k_vec,
                                double theta, double rho, double eta) {
    std::vector<double> out;
    out.reserve(k_vec.size());
    for (const auto& k : k_vec) {
        out.push_back(total_variance(k, theta, rho, eta));
    }
    return out;
}

} // namespace ssvi
