"""D1: driftless digital option on the BTC proxy, corrected for the settlement basis."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from models.base import Forecaster


def d1_components(S, b, K, sigma, tau, clip_lo=0.005, clip_hi=0.995, basis_sd=0.0):
    """Return (probability, dp/dS, z) arrays.

    The market settles on a reference price the backtest cannot see, so it is modelled
    as the proxy plus a basis b plus noise with standard deviation basis_sd (in USD):

        z = ln((S + b) / K) / sqrt(sigma^2 * tau + (basis_sd / K)^2),   P = Phi(z)

    With basis_sd = 0 this is the plain digital, and at tau = 0 or sigma = 0 the
    probability is the clip on the sign of ln((S + b) / K).
    """
    S = np.asarray(S, dtype=float)
    b = np.broadcast_to(np.asarray(b, dtype=float), S.shape)
    K = np.broadcast_to(np.asarray(K, dtype=float), S.shape)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), S.shape)
    tau = np.broadcast_to(np.asarray(tau, dtype=float), S.shape)
    noise = np.broadcast_to(np.asarray(basis_sd, dtype=float), S.shape)
    x = S + b
    logm = np.full(S.shape, np.nan)
    priced = np.isfinite(x) & np.isfinite(K) & (x > 0) & (K > 0)
    logm[priced] = np.log(x[priced] / K[priced])
    with np.errstate(invalid="ignore", divide="ignore"):
        var = sigma ** 2 * np.clip(tau, 0.0, None) + np.where(K > 0, noise / K, 0.0) ** 2
        denom = np.sqrt(var)
    live = priced & np.isfinite(denom) & (denom > 0)
    boundary = priced & ~live & ~np.isnan(tau) & (np.isfinite(sigma) | (tau <= 0))

    z = np.full(S.shape, np.nan)
    p = np.full(S.shape, np.nan)
    dp = np.zeros(S.shape, dtype=float)
    z[live] = logm[live] / denom[live]
    p[live] = norm.cdf(z[live])
    dp[live] = norm.pdf(z[live]) / (x[live] * denom[live])
    p[boundary & (logm > 0)] = clip_hi
    p[boundary & (logm < 0)] = clip_lo
    p[boundary & (logm == 0)] = 0.5
    z[boundary & (logm > 0)] = 8.0
    z[boundary & (logm < 0)] = -8.0
    z[boundary & (logm == 0)] = 0.0
    return np.clip(p, clip_lo, clip_hi), dp, z


class D1(Forecaster):
    name = "D1"

    def __init__(self, clip_lo=0.005, clip_hi=0.995):
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi

    def _components(self, grid):
        noise = grid["basis_sd"] if "basis_sd" in grid.columns else 0.0
        return d1_components(
            grid["S"], grid["b"], grid["K"], grid["sigma"], grid["tau"],
            self.clip_lo, self.clip_hi, noise,
        )

    def predict(self, grid):
        return self._series(grid, self._components(grid)[0], self.clip_lo, self.clip_hi)

    def dp_dS(self, grid):
        return self._components(grid)[1]

    def z_score(self, grid):
        return self._components(grid)[2]
