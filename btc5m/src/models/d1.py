"""D1: driftless digital option on the BTC proxy, with the open basis removed."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from models.base import Forecaster


def d1_components(S, b, K, sigma, tau, clip_lo=0.005, clip_hi=0.995):
    """Return (probability, dp/dS, z) arrays.

    P = Phi(ln((S + b) / K) / (sigma * sqrt(tau))), clipped to [clip_lo, clip_hi].
    At tau = 0 or sigma = 0 the probability is the clip on the sign of ln((S + b) / K).
    """
    S = np.asarray(S, dtype=float)
    b = np.asarray(b, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    tau = np.asarray(tau, dtype=float)
    x = S + b
    logm = np.full(S.shape, np.nan)
    priced = np.isfinite(x) & np.isfinite(K) & (x > 0) & (K > 0)
    logm[priced] = np.log(x[priced] / K[priced])
    denom = sigma * np.sqrt(np.clip(tau, 0.0, None))
    live = priced & np.isfinite(denom) & (denom > 0) & np.isfinite(tau) & (tau > 0)
    boundary = priced & ~live

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

    def predict(self, grid):
        p, _, _ = d1_components(
            grid["S"], grid["b"], grid["K"], grid["sigma"], grid["tau"],
            self.clip_lo, self.clip_hi,
        )
        return self._series(grid, p, self.clip_lo, self.clip_hi)

    def dp_dS(self, grid):
        _, dp, _ = d1_components(
            grid["S"], grid["b"], grid["K"], grid["sigma"], grid["tau"],
            self.clip_lo, self.clip_hi,
        )
        return dp

    def z_score(self, grid):
        _, _, z = d1_components(
            grid["S"], grid["b"], grid["K"], grid["sigma"], grid["tau"],
            self.clip_lo, self.clip_hi,
        )
        return z
