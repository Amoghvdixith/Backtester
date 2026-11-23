"""Sharpe-ratio uncertainty, Probabilistic and Deflated Sharpe Ratio.

All Sharpe ratios in this module are **per-period (non-annualised)** unless a
function name says otherwise.  Kurtosis is **Pearson** (normal = 3).

Sources and status
------------------
* PSR / DSR and the expected maximum Sharpe ratio: Bailey, D. H. and
  López de Prado, M. (2014), "The Deflated Sharpe Ratio", Journal of Portfolio
  Management 40(5).  Checked against the paper text: Eq. (1) expected maximum,
  Eq. (2) DSR with sqrt(T - 1) and the denominator
  sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2), g4 = (Pearson) kurtosis; the numerical
  example uses non-annualised SR with 250 observations per year; Appendix A.3
  Eq. (9) implied independent trials N = rho + (1 - rho) * M.
* IID-normal Sharpe standard error sqrt((1 + SR^2 / 2) / T): the asymptotic
  result commonly attributed to Lo (2002).  The original text could not be
  accessed during this build; the formula is the IID-normal special case of the
  non-normal expression below (g3 = 0, g4 = 3).  It is NOT an
  autocorrelation-robust estimator.
* Non-normal IID standard error sqrt((1 - g3*SR + (g4 - 1)/4 * SR^2) / T) (Mertens
  2002 / Opdyke 2007), consistent with the PSR denominator above.
* Serial dependence: a circular block bootstrap is provided instead.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def sharpe_se_iid_normal(sr: float, n: int) -> float:
    """Asymptotic SE of a per-period Sharpe estimate, IID normal returns."""
    if n < 2:
        return float("nan")
    return math.sqrt((1.0 + 0.5 * sr * sr) / n)


def sharpe_se_nonnormal(sr: float, n: int, skew: float, kurt_pearson: float) -> float:
    """Asymptotic SE for IID non-normal returns (Pearson kurtosis)."""
    if n < 2:
        return float("nan")
    v = 1.0 - skew * sr + (kurt_pearson - 1.0) / 4.0 * sr * sr
    if v <= 0:
        return float("nan")
    return math.sqrt(v / n)


def circular_block_bootstrap_sharpe(r: np.ndarray | pd.Series, block: int = 20,
                                    n_boot: int = 2000, seed: int = 0,
                                    periods_per_year: int = 252) -> dict[str, float]:
    """Annualised Sharpe distribution under a circular block bootstrap.

    Blocks of ``block`` consecutive observations (wrapping at the end) preserve
    short-range dependence such as volatility clustering.  The block length is a
    tuning choice; results should be checked for sensitivity to it.
    """
    x = np.asarray(r, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * block or block < 1:
        return {"se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "block": block,
                "n_boot": n_boot}
    rng = np.random.default_rng(seed)
    nb = math.ceil(n / block)
    srs = np.empty(n_boot)
    offs = np.arange(block)
    for b in range(n_boot):
        starts = rng.integers(0, n, nb)
        idx = ((starts[:, None] + offs[None, :]) % n).ravel()[:n]
        s = x[idx]
        sd = s.std(ddof=1)
        srs[b] = s.mean() / sd * math.sqrt(periods_per_year) if sd > 0 else np.nan
    srs = srs[np.isfinite(srs)]
    return {"se": float(srs.std(ddof=1)), "ci_low": float(np.percentile(srs, 2.5)),
            "ci_high": float(np.percentile(srs, 97.5)), "block": block, "n_boot": n_boot}


def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, n: int, skew: float,
                               kurt_pearson: float) -> float:
    """PSR = Phi((SR - SR*) sqrt(n - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2))."""
    if n < 2 or not all(map(math.isfinite, (sr, sr_benchmark, skew, kurt_pearson))):
        return float("nan")
    denom = 1.0 - skew * sr + (kurt_pearson - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(var_sr: float, n_trials: float, mean_sr: float = 0.0) -> float:
    """E[max SR] over N independent trials (Bailey & LdP 2014, Eq. 1).

    ``mean_sr`` is 0 under the null of no skill.  For N <= 1 the expected maximum
    equals the mean (no selection), so ``mean_sr`` is returned.
    """
    if var_sr < 0 or not math.isfinite(var_sr):
        raise ValueError("variance of trial Sharpe ratios must be finite and >= 0")
    if n_trials <= 1:
        return mean_sr
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return mean_sr + math.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def implied_independent_trials(m_trials: int, avg_corr: float) -> float:
    """N_hat = rho + (1 - rho) * M (Bailey & LdP 2014, Appendix A.3, Eq. 9)."""
    if m_trials < 1:
        raise ValueError("need at least one trial")
    rho = float(np.clip(avg_corr, 0.0, 1.0))
    return rho + (1.0 - rho) * m_trials


def average_offdiag_correlation(returns: pd.DataFrame) -> float:
    """Equal-weighted average of off-diagonal pairwise correlations of trial returns."""
    c = returns.corr().to_numpy()
    k = c.shape[0]
    if k < 2:
        return float("nan")
    mask = ~np.eye(k, dtype=bool)
    vals = c[mask]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


def deflated_sharpe_ratio(sr: float, trial_srs: list[float] | np.ndarray, n_obs: int,
                          skew: float, kurt_pearson: float,
                          n_trials: float | None = None) -> dict[str, Any]:
    """Deflated Sharpe Ratio with diagnostics.

    Parameters
    ----------
    sr: per-period Sharpe of the selected strategy.
    trial_srs: per-period Sharpe estimates of ALL trials (failed trials excluded
        because they have no estimate; they are still counted in ``n_trials`` if given).
    n_obs: number of return observations T behind ``sr``.
    n_trials: number of (effective) independent trials; defaults to len(trial_srs).
    """
    srs = np.asarray([s for s in trial_srs if s is not None and math.isfinite(s)], float)
    n_tr = float(n_trials) if n_trials is not None else float(len(srs))
    out: dict[str, Any] = {"sr_per_period": sr, "n_obs": n_obs, "n_trials": n_tr,
                           "skew": skew, "kurtosis_pearson": kurt_pearson, "warnings": []}
    if n_obs < 30:
        out["warnings"].append("fewer than 30 observations: asymptotics unreliable")
    if len(srs) >= 2:
        var = float(np.var(srs, ddof=1))
    else:
        var = 0.0
        out["warnings"].append("fewer than two trial Sharpe ratios: variance set to 0")
    if var == 0 and n_tr > 1:
        out["warnings"].append("zero variance across trials: SR0 = 0")
    out["var_trial_sr"] = var
    sr0 = expected_max_sharpe(var, n_tr) if n_tr >= 1 else float("nan")
    out["sr0_per_period"] = sr0
    out["dsr"] = probabilistic_sharpe_ratio(sr, sr0, n_obs, skew, kurt_pearson)
    out["psr_vs_zero"] = probabilistic_sharpe_ratio(sr, 0.0, n_obs, skew, kurt_pearson)
    if not math.isfinite(out["dsr"]):
        out["warnings"].append("DSR undefined (invalid inputs or non-positive denominator)")
    return out
