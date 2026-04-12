
"""
arb_detector.py
===============
Calendar and butterfly arbitrage detection on the SSVI surface.
Violations are structured records — signal for the trader.
Requires the compiled C++ engine (qr_engine).
"""

import numpy as np
from python.types import ArbViolation
from qr_engine.ssvi import derivatives as ssvi_derivs


def check_calendar(params_by_tenor, sorted_tenors):
    """
    Calendar arb: total variance must be non-decreasing in tenor.
    Checks ATM and several wing points.
    """
    violations = []
    k_checks = [0.0, -0.2, 0.2, -0.4, 0.4]

    for i in range(len(sorted_tenors) - 1):
        T1, T2 = sorted_tenors[i], sorted_tenors[i + 1]
        p1 = params_by_tenor[T1]
        p2 = params_by_tenor[T2]

        for k in k_checks:
            d1 = ssvi_derivs(k, p1.theta, p1.rho, p1.eta)

            d2 = ssvi_derivs(k, p2.theta, p2.rho, p2.eta)

            if d2.w < d1.w - 1e-8:
                violations.append(ArbViolation(
                    violation_type="calendar",
                    strike=np.exp(k),
                    tenor=T2,
                    severity=float(d1.w - d2.w),
                    description=(
                        "Total variance decreases from T=%.3f (w=%.6f) "
                        "to T=%.3f (w=%.6f) at k=%.2f. "
                        "Calendar spread arb of %.1f variance bps."
                        % (T1, d1.w, T2, d2.w, k, (d1.w - d2.w) * 1e4)
                    ),
                ))
    return violations


def check_butterfly(k_grid, theta, rho, eta, T):
    """
    Butterfly arb: Gatheral-Jacquier density must be non-negative.
    g(k) = [1 - k*w'/(2w)]^2 - (w')^2/4 * [1/w + 1/4] + w''/2
    """
    violations = []

    for k in k_grid:
        d = ssvi_derivs(float(k), theta, rho, eta)
        w, wp, wpp = d.w, d.dw_dk, d.d2w_dk2

        if w < 1e-12:
            continue

        term1 = (1.0 - k * wp / (2.0 * w)) ** 2
        term2 = (wp ** 2 / 4.0) * (1.0 / w + 0.25)
        term3 = wpp / 2.0
        g = term1 - term2 + term3

        if g < -1e-8:
            violations.append(ArbViolation(
                violation_type="butterfly",
                strike=np.exp(k),
                tenor=T,
                severity=float(-g),
                description=(
                    "Negative density g(k)=%.6f at k=%.3f, T=%.3f. "
                    "Butterfly spread arb; surface concavity violation."
                    % (g, k, T)
                ),
            ))
    return violations
