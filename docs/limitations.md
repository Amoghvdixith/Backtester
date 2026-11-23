# Limitations and unresolved biases

This is an educational research framework. It is **not** production trading software,
and no backtest here demonstrates profitability.

## Data
* **Default results use SYNTHETIC data** from a regime-switching factor model with a
  persistent idiosyncratic drift. That drift is a modelling choice and may favour
  trend/momentum rules. Synthetic performance shows that the software works, not that a
  market edge exists.
* The optional market sample (ten US sector ETFs from Yahoo Finance via `yfinance`) is
  a **current-ticker** selection. It is not point-in-time and does not address survivorship
  or selection bias. Yahoo data are unofficial; check the provider's terms before
  redistributing cached files.
* **Corporate actions** use the adjusted-price approximation (convention B). Discrete
  splits, dividend cash flows, withholding taxes and ex-date mechanics are not modelled.
* No real exchange calendar or holiday file: the data's own dates are the calendar.
  The synthetic calendar is Monday–Friday with no holidays.
* Single currency only. No FX conversion is implemented, so do not mix currencies.
* No historical risk-free series is bundled. The synthetic runs use a constant 2%. The
  market config uses 0%.

## Universe and survivorship
* `MembershipUniverse` enforces point-in-time membership *if the input file is
  point-in-time*. Reliable historical constituents and delisting returns are not bundled.
* A holding that stops trading (delisting) keeps its last close as a stale mark. Its exit
  order cannot fill and expires. Real delistings often realise worse prices, so this can
  **overstate** returns. Stale marks are counted (`n_stale_marks`) but not written down.

## Execution
* Fills only at the session open. No intraday path, stops, limits, auctions or queue position.
* Liquidity is unlimited unless `max_participation` is set. There are no partial fills
  from real order-book depth.
* Impact, spread, slippage, borrow and financing parameters are uncalibrated assumptions.
* Every short is assumed borrowable. Locates, recalls and hard-to-borrow fees are not modelled.
* Financing accrues on previous-close balances (intraday cash changes are ignored).
* Pending orders do not reserve cash. Buys are resized at the open instead.
* Terminal liquidation is optional and does not work with `execution_delay` > 1.

## Statistics
* Annualising by √252 assumes IID returns. No autocorrelation-corrected Sharpe
  (Lo-style) is implemented. Use the block bootstrap instead, and vary the block length.
* DSR uses an approximate expected-maximum formula and an approximate effective-N rule.
  It cannot count research done outside the framework. The registry also counts
  repeated identical runs.
* Walk-forward training objectives use rf = 0. Selection by training Sharpe is one
  choice among many.
* The first walk-forward fold has no data before its training window, so long
  indicators use part of the training window as warm-up.

## Not implemented (stretch goals from the guide)
* Vectorised fast mode cross-checked against the event engine.
* Combinatorially purged cross-validation and PBO.
* Intraday bars, limit orders, queue models.
* Futures rolls, FX, multi-currency margin.
* Live or paper trading adapters (deliberately excluded), PyPI packaging, mkdocs site.
* French UMD factor comparison (needs network data; not a valid unit test anyway).

## Moving toward live trading would require
A broker execution adapter, order and position reconciliation against the broker,
partial-fill/reject/cancel handling, real-time and corporate-action data feeds, holiday
calendars and clock management, pre-trade risk checks, persistence and crash recovery,
monitoring and alerting, a kill switch, and separate operational testing. Swapping the
data source alone is not enough.
