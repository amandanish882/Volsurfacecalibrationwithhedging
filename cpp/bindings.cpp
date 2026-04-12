
/**
 * pybind11 bindings for the qr_engine C++ library.
 *
 * Exposes SSVI surface evaluation, Black-Scholes pricing,
 * and finite-difference Greeks to Python.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "ssvi.hpp"
#include "greeks.hpp"

#include <string>
#include <sstream>
#include <iomanip>

namespace py = pybind11;

PYBIND11_MODULE(qr_engine, m) {
    m.doc() = "QR Equity Derivatives Flow - C++ pricing engine.\n\n"
              "Submodules:\n"
              "  ssvi   - SSVI volatility surface parameterisation\n"
              "  greeks - Black-Scholes pricing and finite-difference Greeks";

    // -------------------------------------------------------------------
    // SSVI submodule
    // -------------------------------------------------------------------
    auto ssvi_mod = m.def_submodule("ssvi",
        "SSVI (Surface SVI) volatility surface functions.\n\n"
        "Provides total variance evaluation, analytical derivatives,\n"
        "and vectorised surface computation.");

    ssvi_mod.def("total_variance", &ssvi::total_variance,
                 py::arg("k"), py::arg("theta"), py::arg("rho"), py::arg("eta"),
                 "Compute SSVI total variance w(k) at log-moneyness k.\n\n"
                 "  k     : log-moneyness ln(K/F)\n"
                 "  theta : ATM total variance (sigma^2 * T)\n"
                 "  rho   : skew parameter, in (-1, 1)\n"
                 "  eta   : curvature parameter, > 0");

    ssvi_mod.def("surface_vec", &ssvi::surface_vec,
                 py::arg("k_vec"), py::arg("theta"), py::arg("rho"), py::arg("eta"),
                 "Vectorised total variance over a list of log-moneyness values.");

    py::class_<ssvi::SSVIDerivatives>(ssvi_mod, "SSVIDerivatives",
        "Analytical partial derivatives of the SSVI total variance.\n\n"
        "Attributes: w, dw_dk, d2w_dk2, dw_dtheta")
        .def_readonly("w",         &ssvi::SSVIDerivatives::w,
                      "Total variance w(k)")
        .def_readonly("dw_dk",     &ssvi::SSVIDerivatives::dw_dk,
                      "First partial dw/dk")
        .def_readonly("d2w_dk2",   &ssvi::SSVIDerivatives::d2w_dk2,
                      "Second partial d2w/dk2")
        .def_readonly("dw_dtheta", &ssvi::SSVIDerivatives::dw_dtheta,
                      "Partial dw/dtheta (calendar direction)")
        .def("__repr__", [](const ssvi::SSVIDerivatives& d) {
            std::ostringstream os;
            os << std::fixed << std::setprecision(6);
            os << "SSVIDerivatives(w=" << d.w
               << ", dw_dk=" << d.dw_dk
               << ", d2w_dk2=" << d.d2w_dk2
               << ", dw_dtheta=" << d.dw_dtheta << ")";
            return os.str();
        });

    ssvi_mod.def("derivatives", &ssvi::derivatives,
                 py::arg("k"), py::arg("theta"), py::arg("rho"), py::arg("eta"),
                 "Compute total variance and all analytical partials at (k, theta, rho, eta).");

    // -------------------------------------------------------------------
    // Greeks submodule
    // -------------------------------------------------------------------
    auto greeks_mod = m.def_submodule("greeks",
        "Black-Scholes pricing and finite-difference Greeks.\n\n"
        "All functions use the forward price F (not spot S).");

    greeks_mod.def("bs_price", &greeks::bs_price,
                   py::arg("F"), py::arg("K"), py::arg("T"), py::arg("r"),
                   py::arg("sigma"), py::arg("is_call"),
                   "Black-Scholes European option price.\n\n"
                   "  F       : forward price\n"
                   "  K       : strike\n"
                   "  T       : time to expiry (years)\n"
                   "  r       : risk-free rate (continuous)\n"
                   "  sigma   : implied volatility\n"
                   "  is_call : True for call, False for put");

    greeks_mod.def("bs_implied_vol", &greeks::bs_implied_vol,
                   py::arg("price"), py::arg("F"), py::arg("K"), py::arg("T"),
                   py::arg("r"), py::arg("is_call"),
                   py::arg("tol") = 1e-8, py::arg("max_iter") = 100,
                   "Newton-Raphson implied volatility from a market price.\n\n"
                   "  price    : observed option price\n"
                   "  tol      : convergence tolerance (default 1e-8)\n"
                   "  max_iter : iteration cap (default 100)");

    greeks_mod.def("bs_implied_vol_vec", &greeks::bs_implied_vol_vec,
                   py::arg("prices"), py::arg("Fs"), py::arg("Ks"),
                   py::arg("T"), py::arg("r"), py::arg("is_calls"),
                   py::arg("tol") = 1e-8, py::arg("max_iter") = 100,
                   "Vectorised implied vol solver over arrays of prices/strikes.");

    py::class_<greeks::GreekResult>(greeks_mod, "GreekResult",
        "Full set of finite-difference Greeks for one option.\n\n"
        "Attributes: delta, gamma, vega, theta, vanna, volga")
        .def_readonly("delta", &greeks::GreekResult::delta, "dV/dS")
        .def_readonly("gamma", &greeks::GreekResult::gamma, "d2V/dS2")
        .def_readonly("vega",  &greeks::GreekResult::vega,  "dV/dsigma")
        .def_readonly("theta", &greeks::GreekResult::theta, "dV/dT (1-day)")
        .def_readonly("vanna", &greeks::GreekResult::vanna, "d2V/(dS dsigma)")
        .def_readonly("volga", &greeks::GreekResult::volga, "d2V/dsigma2")
        .def("__repr__", [](const greeks::GreekResult& g) {
            std::ostringstream os;
            os << std::fixed << std::setprecision(4);
            os << "GreekResult(delta=" << g.delta
               << ", gamma=" << g.gamma
               << ", vega=" << g.vega
               << ", theta=" << g.theta
               << ", vanna=" << g.vanna
               << ", volga=" << g.volga << ")";
            return os.str();
        });

    greeks_mod.def("compute", &greeks::compute,
                   py::arg("F"), py::arg("K"), py::arg("T"), py::arg("r"),
                   py::arg("sigma"), py::arg("is_call"),
                   py::arg("dS_frac") = 0.005, py::arg("dSigma") = 0.005,
                   py::arg("dT") = 1.0/365.0,
                   "Compute all Greeks via central finite differences.\n\n"
                   "  dS_frac : spot bump as fraction of F (default 0.5%%)\n"
                   "  dSigma  : vol bump in absolute terms (default 50bps)\n"
                   "  dT      : time bump in years (default 1 day)");
}
