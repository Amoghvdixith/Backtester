# Data dictionary

## Input price files (`data/synthetic/*.csv`, `data/market/*.csv`)

Long format, one row per observed symbol-session. Missing sessions are simply absent.

| Column | Type | Meaning |
|---|---|---|
| `date` | date (tz-naive) | Exchange session date |
| `symbol` | str | Instrument identifier |
| `open`, `high`, `low`, `close` | float > 0 | Prices on one adjustment basis; `low ≤ open, close ≤ high` |
| `volume` | float ≥ 0 | Shares traded (same adjustment basis). 0 makes the bar non-tradable |

`<name>.meta.json` holds provenance: generator spec or provider, retrieval date,
period, currency, adjustment convention and missing-data policy.

## Membership (`*_membership.csv`)

| Column | Meaning |
|---|---|
| `symbol` | Instrument |
| `start`, `end` | Membership interval (inclusive). Empty `end` means still a member |
| `known_from` | Date the membership became public. Effective start = max(start, known_from) |

For optional **fundamental data** (not bundled), carry both the economic period end and
the actual availability date. Join on availability, never on period end.

## Output tables (`results/`)

| File | Content |
|---|---|
| `tables/*_metrics.csv` | Metric set for strategy and benchmark (definitions in `performance.py`) |
| `tables/*_sharpe_uncertainty.csv` | IID-normal, non-normal and block-bootstrap Sharpe uncertainty |
| `tables/*_order_fill_audit_example.csv` | First orders joined with final status, fills and cost reconciliation |
| `tables/buy_and_hold_validation*.{csv,json}` | Engine vs reference equity and the summary |
| `tables/<exp>_trial_log.csv` | Every walk-forward trial: params, windows, status (`ok`, `invalid`, `error`, `undefined_objective`), training metrics, runtime |
| `tables/<exp>_fold_summary.csv` | Per-fold windows, selected params, training objective, test metrics |
| `tables/<exp>_oos_metrics.csv` | Stitched OOS metrics vs benchmark on the same dates |
| `tables/<exp>_deflated_sharpe.csv` | DSR inputs and outputs (raw and implied-independent N) |
| `tables/sma_wf_training_sharpe_heatmap.csv` | Mean training Sharpe over folds per (fast, slow) |
| `tables/sma_wf_EXPLORATORY_outer_test_heatmap.csv` | Exploratory outer-test Sharpe per grid point (not used for selection) |
| `tables/cost_sensitivity.csv`, `robustness_sma*.csv` | Cost scenarios and robustness checks |
| `tables/figure_data/<figure>.csv` | Numbers behind each figure |
| `logs/<run>/…_equity.csv` | Per-session cash, long/short value, equity, gross/net exposure, stale marks, cumulative costs, financing |
| `logs/<run>/…_fills.csv` | Each fill: decision price, arrival price, fill price, itemised costs, status |
| `logs/<run>/…_orders.csv` / `…_order_status.csv` | Order events and every status change (submitted, filled, resized, partial, rejected, cancelled, expired, replaced) |
| `logs/<run>/…_round_trips.csv` | FIFO lot closures |
| `logs/<run>/…_events.csv` | Chronological audit trail (signals, orders, fills, exits, liquidation) |
| `logs/<run>/…_reconciliation.csv` | Reference vs execution value, shortfall, components, commission, delay cost |
| `logs/trial_registry.jsonl` | Append-only registry of every trial executed through the framework |
| `logs/holdout_lock.json` | Locked hold-out start per dataset |
| `logs/holdout_runs.jsonl` | Hold-out evaluations (absent if never evaluated) |
| `run_metadata.json` | Environment and package versions, seeds/specs, configs, data provenance, runtimes, headline results |
