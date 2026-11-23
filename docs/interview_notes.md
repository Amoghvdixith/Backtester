# Interview notes (answers tied to this implementation)

Numbers quoted below come from the offline **synthetic** runs in `results/`. They
illustrate the software and are not evidence about real markets.

**1. Five ways a backtest is biased upward, and how this engine prevents each.**
(a) *Look-ahead*: the strategy sees data only through `PointInTimeView`. Orders fill at
the next open, and same-bar eligibility is impossible (spike and future-mutation tests).
(b) *Survivorship / membership look-ahead*: `MembershipUniverse` with `known_from`. Exits
are liquidated. This cannot repair an input file that already drops dead firms.
(c) *Ignored costs*: spread, slippage, square-root impact and commission run on every
fill and are reconciled. Cost scenarios run from 0 to 50 bps.
(d) *Overfitting via parameter search*: chronological walk-forward, a trial log of every
configuration (including invalid ones), DSR with raw and effective N.
(e) *Peeking at the hold-out*: it is physically removed from research data, locked,
needs `--confirm`, and every evaluation is logged.
Also: stale-holdings bugs (fills applied before the strategy runs), forward-filled
execution prices (never used) and same-day volume in impact estimates (lagged).

**2. Why execute on the next bar? What changes intraday?**
The close that produces a signal is not tradable once known. Next-open execution is the
earliest realistic fill for a daily strategy. Intraday you would need time-stamped
quotes/trades, latency, queue position for limit orders, and intraday volume profiles for
impact. The two-phase loop would become a finer event clock (quote/trade events), and
"next bar" would become "next tradable event after decision latency".

**3. Estimating costs for a strategy you have not traded.**
Start from the quoted spread (half per trade), a commission schedule, and an impact model
such as √(q/ADV)·σ with *lagged* ADV and σ, using a conservative coefficient. Then run
cost-sensitivity scenarios and find the break-even cost. On the synthetic SMA sample,
Sharpe falls by more than half between 0 and 50 bps one-way (`tables/cost_sensitivity.csv`).
Calibrate later with your own fills (implementation shortfall vs arrival price, which the
reconciliation table already records).

**4. Sharpe 1.5 over 5 years: significant? After 200 variants?**
With ≈1,260 daily observations, the IID-normal SE of the annualised Sharpe is about
√((1 + SR_d²/2)/T)·√252 ≈ 0.45. So t ≈ 3.3 for a single pre-registered test (less with
fat tails or autocorrelation; check the block bootstrap). After 200 trials, the expected
best Sharpe of skill-less strategies depends on the spread of the trials' Sharpes. With
an annualised trial SR standard deviation of 0.5, Eq. 1 of Bailey & López de Prado gives
SR0 ≈ 1.4 annualised, and the DSR of a 1.5 would be near 0.6, far from 0.95.
`deflated_sharpe_ratio` computes this from the logged trials. Correlated trials reduce
the effective N (`implied_independent_trials`).

**5. Sortino vs Sharpe.**
Sharpe penalises all volatility. Sortino penalises only returns below the target (rf
here), using downside deviation over all observations. The difference matters for skewed
strategies: option selling looks better on Sharpe than its crash risk deserves, while
trend following (positive skew) looks better on Sortino.

**6. Why might volatility targeting help a trend strategy?**
Volatility clusters. Scaling exposure down when trailing volatility is high cuts risk in
the periods when losses are largest, and that can raise risk-adjusted returns. It is not
guaranteed. It adds turnover and depends on the sample. In the synthetic robustness run,
vol targeting had a higher Sharpe than equal weighting, but at much lower average
exposure (`tables/robustness_sma.csv`). That is one synthetic sample, not a general result.

**7. Moving from backtest to live.**
Replace `SimulatedExecution` with a broker adapter. Add order-state and position
reconciliation, partial-fill/reject handling, live and corporate-action data,
calendars/clocks, pre-trade risk limits, persistence and crash recovery, monitoring,
alerting and a kill switch. Then paper-trade and compare live fills with the cost model.
Swapping the data handler is only a small part of the work (`docs/architecture.md`).
