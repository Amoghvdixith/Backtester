# Statistical validation and corrections to the build guide

## Conventions used everywhere

* Returns are **daily close-to-close equity returns**. Periods per year *P* = 252.
* Sharpe ratios inside statistical formulas are **per period** (non-annualised).
  Annualised values multiply by √*P*. That assumes IID returns and is only a convention.
* **Kurtosis is Pearson** (normal = 3) in PSR/DSR and the non-normal SE. Tables report
  both Pearson and excess kurtosis.
* Skewness and kurtosis use bias-adjusted sample estimators (`scipy.stats`, `bias=False`).
* Risk-free: an annual rate becomes per-session returns with the same simple-interest,
  ACT/365 convention as the cash account (`rf_from_annual`).
  Training objectives in the walk-forward use rf = 0. Reported OOS metrics use the
  configured rf (2% on synthetic data).

## Sharpe-ratio uncertainty

| Estimator | Formula (per period, *T* obs) | Assumptions | Function |
|---|---|---|---|
| IID normal | √((1 + SR²/2) / T) | IID, normal returns | `sharpe_se_iid_normal` |
| IID non-normal | √((1 − γ₃·SR + (γ₄ − 1)/4 · SR²) / T) | IID, finite 4th moment; γ₄ Pearson | `sharpe_se_nonnormal` |
| Circular block bootstrap | Resample blocks of 20 consecutive days (wrapping), 1,000 draws, seeded | Dependence shorter than the block; stationarity | `circular_block_bootstrap_sharpe` |

The IID-normal formula is the asymptotic result usually attributed to Lo (2002). It is
**not** robust to autocorrelation. Lo's paper also gives an adjustment for serially
correlated returns. That adjustment is **not implemented**, because the original text
could not be checked during this build. For dependence, use the block bootstrap and
check sensitivity to the block length (20 days is a judgement call, not a derived value).

## Probabilistic and Deflated Sharpe Ratio

Source: Bailey, D. H. and López de Prado, M. (2014), *The Deflated Sharpe Ratio*,
Journal of Portfolio Management 40(5). The formulas below were checked against the text
of the paper (Eq. 1, Eq. 2, Appendix A.3 Eq. 9):

```
E[max SR] ≈ E[SR_n] + sqrt(V[SR_n]) * ((1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N e)))     (Eq. 1)
SR0       = sqrt(V[SR_n]) * ((1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N e)))              (null: mean 0)
DSR       = Φ( (SR − SR0) · sqrt(T − 1) / sqrt(1 − γ₃·SR + (γ₄ − 1)/4 · SR²) )        (Eq. 2)
N_implied = ρ̄ + (1 − ρ̄) · M                                                          (Eq. 9)
```

γ ≈ 0.5772 is the Euler–Mascheroni constant. V[SR_n] is the variance of the trials'
per-period Sharpe estimates. N is the number of **independent** trials. M is the
number of correlated trials and ρ̄ their average pairwise correlation.

**Reference test** (`tests/test_metrics_statistics.py`): the paper's numerical example
(annualised SR 2.5, T = 1250 daily obs at 250/yr, N = 100, V = ½ annualised, γ₃ = −3,
γ₄ = 10). The implementation gives SR0 = 0.1132 (per period) and DSR = 0.9004. It also
gives 0.9505 at N = 46, and 0.9505 at N = 88 with normal returns. All four values match
the statements in the paper.

### How DSR is used here

* **Per fold**: the parameter set chosen on the training window is compared with all
  configurations attempted in that fold. Two trial counts are reported. One is the raw
  number attempted, *including invalid configurations*. The other is the
  correlation-implied effective number (Eq. 9, using the average pairwise correlation of
  the trials' training returns).
* **Stitched OOS (conservative diagnostic)**: the stitched OOS Sharpe compared with
  SR0 from all trials in all folds. Out-of-sample returns are less affected by selection
  than the in-sample maximum, so this treats the whole search as the multiple-testing
  burden. It is a stress test, not the canonical use of DSR.
* Special cases: N ≤ 1 gives SR0 = 0 (no selection), so DSR equals PSR against zero.
  Zero trial variance also gives SR0 = 0 (with a warning). A non-positive variance term
  in the denominator returns NaN. T < 30 adds a warning.

### What DSR does **not** establish

* It is not the probability that the strategy will make money.
* It is not proof of genuine alpha. It is conditional on the model assumptions (trial
  Sharpes roughly normal, the effective-N approximation, stationarity).
* It cannot account for experiments run outside the framework. The trial registry
  only knows what was executed through `WalkForward` / experiments. Repeated identical
  runs are also appended, so registry totals can overstate distinct trials.
* "DSR > 0.95" is a convention (a one-sided 95% level), not a law.

## Corrections and qualifications to the supplied guide

| # | Guide statement | Issue | What this project does |
|---|---|---|---|
| 1 | Sample event loop: mark → `fill_pending` → `strategy.on_bar` in one `MarketEvent`, fills queued behind signals | Fills stay unapplied while new orders are sized from stale holdings. Open vs close use is ambiguous | Explicit OPEN/CLOSE phases. Fills applied at the open before the strategy runs at the close (`docs/architecture.md`) |
| 2 | "Live-trading reuse: yes, swap the DataHandler" | Overstated: execution, reconciliation, monitoring and failure handling are missing | Documented as research software; live requirements listed |
| 3 | Buy-and-hold must equal `initial_cash * close / close.iloc[0]` to 1e-10 | Inconsistent with next-open execution (and residual cash) | Reference uses the same sizing close, next-open fill and residual cash: max difference 0.0 on the demo. The naive formula differs (test + figure) |
| 4 | Cost: `commission + ½·spread·|q|·p + impact`, c ≈ 0.5–1 | Spread units not stated (full vs half, bps vs decimal). Coefficient presented as a fact | `spread_bps` = full quoted spread in bps, half charged. Impact uses lagged σ and ADV. c is a configurable assumption. Costs applied once through the fill price |
| 5 | Borrow fee "default 0.5% for liquid large caps"; "cash earns the risk-free rate" | Market claim stated as fact. Day count and short-proceeds treatment unspecified | Rates are config assumptions. ACT/365 or ACT/360 simple accrual. Short proceeds excluded from the interest base by default |
| 6 | Volatility targeting "usually improves Sharpe for trend strategies" | Not a general result. Evidence is mixed and depends on the asset and sample (Moreira & Muir 2017 report gains for volatility-managed factor portfolios; later work questions out-of-sample robustness) | Treated as an empirical question in the robustness table |
| 7 | Kelly f* = μ/σ² | Valid only for a single asset with excess drift and variance at the same frequency. Very sensitive to μ | Excess returns, same-frequency units, shrinkage, fractional Kelly, hard cap; single-asset approximation documented |
| 8 | Sortino: DD = √(mean(min(r − rf, 0)²)) | Fine, but the target and denominator need stating | Target = rf; mean over **all** observations (stated in code and docs) |
| 9 | Turnover = ½ Σ\|Δw\| per year | Δw from trades vs drift not specified; cash treatment unclear | One-way turnover = traded notional / (2 × previous equity), summed and annualised; cash excluded |
| 10 | Sharpe SE ≈ √((1 + ½SR²)/n) (Lo, 2002), annualised | IID-normal special case only. Must use per-period SR, then scale | Per-period formula, annualised by √252 under IID; non-normal SE and block bootstrap added; not labelled autocorrelation-robust |
| 11 | DSR formula with `(kurt − 1)/4` | Correct only if kurt is **Pearson**. With excess kurtosis it should be (κ_excess + 2)/4. SR must be per period. N must be independent trials. N = 1 is degenerate (Φ⁻¹(0) = −∞) | Pearson kurtosis, per-period SR, N = 1 handled, effective N reported, reference case reproduced |
| 12 | "DSR is the probability that the true Sharpe exceeds the expected maximum…"; "DSR > 0.95 is the bar" | Wording invites over-interpretation | Described as a PSR evaluated at a selection-adjusted threshold; 0.95 is a convention |
| 13 | Forward-fill at most one day; drop symbols with large gaps | Forward-filled prices must never be traded. Dropping gappy symbols can add survivorship bias | No forward fill for execution. Stale marks only for valuation and counted. Gappy symbols are not dropped |
| 14 | Momentum sanity test: correlation with French UMD "clearly positive" | Not valid as a unit test (synthetic data, small universes, different construction) | Not a pass/fail test; only an optional contextual diagnostic (not run offline) |
| 15 | Return from t−252 to t−21 days | Endpoint convention and missing-data handling unspecified | MOM = P(t−21)/P(t−252) − 1 on master-calendar sessions, ≤5 sessions stale fallback, deterministic ties, k = max(1, ⌈q·N⌉) |
| 16 | Use yfinance auto-adjusted prices; rebuild membership from Wikipedia | Adjusted prices with share-count accounting is an approximation. Current/Wikipedia lists are not reliable point-in-time data | Convention B documented. Market sample labelled "not point-in-time" |
| 17 | Heatmap of **OOS** Sharpe over fast × slow | Choosing among OOS cells turns OOS into in-sample | Training-window heatmap is primary. The outer-test heatmap is labelled EXPLORATORY and not used for selection |
| 18 | Final hold-out "run once" | Needs enforcement, not intent | Physically removed from research data; lock file; `--confirm`; every evaluation logged with a number |
| 19 | CAGR = (Vn/V0)^(252/n) − 1 | Fine for trading-day counts; undefined when equity ≤ 0 | NaN plus warning for non-positive equity |
| 20 | Monthly rebalancing "on the last trading day" | Needs knowledge of the next session | Uses the exchange calendar (next session date), which is calendar knowledge, not price data |
