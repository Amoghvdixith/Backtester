# Execution and accounting conventions

## Timing

| Item | Convention |
|---|---|
| Signal time | After the close of session *t* (the strategy sees the full bar *t*) |
| Decision price | Close of *t*; used only to size the order |
| Earliest eligible execution | Open of session *t + d*, with `execution_delay` *d* = 1 by default |
| Fill reference (arrival) price | Open of the first session ≥ *t + d* at which **that symbol** has a tradable bar |
| Same-close fills | Impossible: `OrderEvent` rejects `eligible_from <= created_at` |
| Data used to fill | Session open, plus ADV and volatility computed from bars strictly before the session |
| Missing / halted bar | The order waits; it never fills at a forward-filled price |
| Expiry | After `expires_after` sessions (default 5) with no tradable bar: `EXPIRED` |
| Pending-order replacement | A new order for a symbol with a pending order replaces it (`REPLACED`) |
| Duplicate targets | Target quantity equal to the current holding (or inside `rebalance_band`) creates no order and cancels any pending order |
| Final-bar orders | Not created (logged `order_not_created`); there is no later session |
| Terminal liquidation | Optional `liquidate_on_final_open`: target zero at the second-to-last close, sell with full costs at the final open |
| Universe exit | Default `liquidate`: a holding that leaves the point-in-time universe is targeted to zero at the next close where this is known |

Daily OHLC bars do not show the intraday path. We do not know whether the high came
before the low, or how much volume traded near the open. That is why fills use only the
open and why stop or limit orders are not simulated.

## Rebalancing and cash

1. Strategy signals give a **direction** per symbol. The sizer converts the batch into
   target weights. `PortfolioConstraints` then clip per-position weight
   (`max_position`), scale gross exposure to `max_gross` and net exposure to `max_net`.
   This is one coherent rebalance. Capital is never assigned at 100% to each bullish asset.
2. Target quantity = weight × (1 − `cash_buffer`) × equity at close ÷ close price.
   Quantities are rounded toward zero to whole shares (`fractional=False`, the default).
3. At the next open, **sells and short sales execute first**, then buys, each group in
   symbol order. If buys need more cash than is available (gap up, costs), every buy is
   scaled down pro rata (`insufficient_cash: resize`) or all are rejected (`reject`).
   Without leverage, cash may not go below zero. With `max_gross > 1`, cash may go down
   to −(max_gross − 1) × equity (margin debit, charged at `debit_rate`).
4. Pending orders are not separately reserved. Targets already account for the whole
   portfolio, and at most one live order per symbol exists. Gaps are absorbed by resizing.

**Short selling** is supported when `allow_short=True`. Short-sale proceeds are credited
to cash. By default they are excluded from the interest-bearing balance
(`short_proceeds_earn_interest: false`), so the account earns interest on
`cash − |short market value|`. Borrow fees accrue on the absolute short market value.
Short availability (locates, recalls) is **not** modelled: every short is assumed borrowable.

**Insolvency**: if equity at a close is ≤ 0, all pending orders are cancelled, the
strategy is no longer called, and the run is flagged `insolvent`. Metrics that need
positive equity (CAGR, Calmar) return NaN with a warning.

## Costs (no double counting)

For a trade of signed quantity *q* at arrival price *p* (the open):

```
half_spread = 0.5 * spread_bps * 1e-4                 # spread_bps is the FULL quoted spread
slippage    = slippage_bps * 1e-4 (+ seeded Normal noise if stochastic_sd_bps > 0)
impact      = c * sigma_daily_lagged * sqrt(|q| / ADV_lagged)          # fraction of price
fill_price  = p * (1 + sign(q) * (half_spread + slippage + impact))
commission  = clip(fixed + per_share*|q| + rate_bps*1e-4*|q|*fill_price, minimum, maximum)
cash       -= q * fill_price + commission
```

Spread, slippage and impact are charged **once**, through the fill price. The
`FillEvent` reports them as cash-equivalent amounts (`|q|·p·fraction`) for
reconciliation only. The commission is the only explicit cash fee. The per-fill
reconciliation table (`results/logs/<run>/…_reconciliation.csv`) shows:

* reference value `q·p` and execution value `q·fill_price`;
* execution shortfall `q·(fill_price − p)`, which equals spread + slippage + impact (checked to 1e-6);
* commission;
* delay cost vs the decision close, `q·(p − close_t)`. This is reported but is not a fee: it is the
  overnight price move between decision and execution.

The square-root coefficient *c*, spreads and borrow rates in the configs are
**assumptions**, not calibrated market facts. The empirical literature (e.g. Almgren et
al., 2005) estimates impact functions from proprietary execution data. Their fitted
exponents and coefficients vary by market and period, and they are not universal
constants. With `max_participation` unset, liquidity is unlimited: any size fills at the
open with only modelled impact. Setting `max_participation` caps each fill at a fraction
of *lagged* ADV and cancels the remainder (`PARTIAL`).

### Cost-sensitivity scenarios

`configs/cost_scenarios.yaml` replaces the whole detailed model with a flat one-way
proportional cost (0/5/10/20/50 bps on traded notional), charged as an explicit fee.
Nothing is added on top of the detailed model. Financing is unchanged. Parameters and
dates are fixed.

## Financing

Accrual happens at each close for the calendar days since the previous close (a Monday
accrues three days). It uses the balances held at the previous close:

```
interest_base = cash - |short MV|      (or cash, if short proceeds earn interest)
interest      = interest_base * (credit_rate if base >= 0 else debit_rate) * days / basis
borrow_fee    = |short MV| * borrow_fee_rate * days / basis
```

`basis` is 365 or 360. Using previous-close balances ignores intraday changes from the
day's opening fills (documented simplification). The Sharpe/Sortino risk-free series is
built on exactly the same convention (`performance.rf_from_annual`).

## Accounting identities

Checked at every close (`Portfolio.reconcile`, tolerance 1e-6 + 1e-9 × (|equity| + traded notional)):

```
equity = cash + Σ quantity_i × mark_i                     (shorts have negative quantity)
equity = initial_cash + realised P&L + unrealised P&L − commissions + interest − borrow fees
Σ lot quantities = position quantity, per symbol
```

P&L is measured against **fill prices**, so price-embedded costs are already inside it.

## Lots and round trips

Lots are matched **FIFO**. A sell closes the oldest long lots first; partial closures
split a lot. A fill that crosses zero (a reversal) first closes all existing lots, then
opens a new lot in the opposite direction with the remainder. Each closure produces a
`RoundTrip` with entry/exit dates and prices, gross P&L (at fill prices) and commissions
(entry commission pro rata plus exit commission). It also records net P&L, holding days
and return on entry notional. Trade statistics (hit rate, profit factor, average
win/loss) use these round trips. Fill-level logs (`fills.csv`) are kept separately.

## Prices, corporate actions and the calendar

The framework uses **convention B: an adjusted-price research approximation**. OHLC and
volume are expected on one adjustment basis. Share quantities are "adjusted share
units". Split ratios and dividend cash flows are not modelled as discrete events, so
dividend withholding, ex-date timing and exact share counts are not reproduced. The
synthetic data have no corporate actions. The optional Yahoo data use `auto_adjust=True`.

The master calendar is the union of session dates in the data file. It stands in for the
exchange calendar. Timestamps are tz-naive session dates. The month-end test uses the
next calendar session, which is calendar knowledge, not price knowledge.

## Marks and stale prices

At each close a held position is marked at that session's close. If the symbol has no
bar, it keeps its last observed close (a **stale mark**). The number of stale marks is
recorded in every equity snapshot (`n_stale_marks`). Stale marks are valuation only and
are never used to fill orders. A delisted holding stays at its last close until the
universe-exit order (which cannot fill without a bar) expires. This can **overstate**
value when real delistings come with large losses (see limitations).
