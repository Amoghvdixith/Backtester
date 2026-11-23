# Bias controls and the tests that check them

| Bias / error | Safeguard in the code | Test(s) |
|---|---|---|
| Trading on the close used for the signal | Orders become eligible at the next session's open; `OrderEvent` rejects same-bar eligibility | `test_timing_and_lookahead.py::test_fill_at_next_open_never_same_close`, `test_events.py::test_order_validation` |
| Using the day's close/high/low/volume at the open | `PointInTimeView` at `Phase.OPEN` exposes only the open of *t*; `is_tradable_today` raises `LookAheadError` before the close | `test_open_phase_view_hides_current_close` |
| Future data leaking into the past | Strategy reads data only through the view; lagged execution estimates | `test_future_spike_cannot_be_traded_early`, `test_future_data_mutation_does_not_change_past[shuffle, scale]` |
| Stale holdings after fills | Fills are applied at the open before the strategy runs | `test_strategy_sees_post_fill_holdings`, `test_events_processed_in_phase_order` |
| Same-day volume in cost/impact estimates | ADV and sigma are lagged one session | `test_data_universe.py::test_lagged_estimates_ignore_same_day_volume` |
| Filling at forward-filled prices | Orders wait for a tradable bar; expire otherwise | `test_missing_bar_order_waits_and_is_not_filled_at_ffilled_price`, `test_order_expires_without_tradable_bar` |
| Indicator warm-up leakage | SMA emits no active signal before `slow` observations; momentum needs `lookback + 1` | `test_sma_warmup_no_signal_before_slow_window`, `test_momentum_monthly_schedule_and_next_open_execution` |
| Survivorship / look-ahead in membership | `MembershipUniverse` with `known_from`; exits are liquidated | `test_membership_known_from_and_end`, `test_universe_exit_liquidates_holding`, `test_momentum_respects_point_in_time_membership` |
| Over-allocation across signals | Sizer + portfolio-wide constraints on the whole target vector | `test_multi_asset_targets_are_one_coherent_rebalance`, `test_constraints_gross_net_concentration`, `test_leverage_cap_and_margin` |
| Ignored or double-counted costs | Price-embedded costs once; itemised reconciliation | `test_execution_itemised_costs_reconcile`, `test_outputs.py::test_reconciliation_table_closes` |
| Accounting drift | Two identities reconciled at every close | `test_portfolio.py::*`, `test_reconcile_detects_corruption` |
| Incompatible benchmark timing | Buy-and-hold reference uses the same next-open fill and residual cash | `test_buy_and_hold.py::*` (incl. `test_naive_first_close_benchmark_is_not_comparable`) |
| In-sample parameter selection reported as OOS | Chronological folds; training-only selection; one continuous OOS run | `test_walkforward.py::test_folds_are_chronological_and_disjoint`, `test_selection_uses_training_only_with_tie_break`, `test_oos_is_one_continuous_run_with_transition_costs` |
| Peeking at the final hold-out | Hold-out rows physically removed from research data; lock file; `--confirm` required; logged | `test_holdout_is_physically_excluded`, `test_holdout_lock`, `test_holdout_evaluation_requires_confirmation_and_is_logged`, `test_integration_experiments.py` |
| Hiding failed or invalid trials | Every configuration is logged with a status | `test_trial_log_records_every_configuration` |
| Selection bias from many trials | DSR with all attempted trials and a correlation-implied effective N | `test_dsr_reproduces_bailey_lopez_de_prado_2014_example`, `test_effective_trials_and_correlation` |
| Non-reproducibility | Seeds for data, stochastic slippage and bootstrap | `test_reproducibility_with_stochastic_slippage`, `test_synthetic_generator_is_reproducible_and_valid`, `test_block_bootstrap_reproducible` |

## What these controls do not cover

* Research choices made *outside* the framework: strategy ideas tried in a notebook,
  features dropped after a look at the plots, or edits to the grid after seeing results.
  The trial registry cannot know about these.
* Survivorship bias in the input file. A file that already excludes dead companies
  cannot be fixed by the universe interface.
* Data-snooping in the synthetic generator itself. Its persistent drift component was
  chosen by a person (and disclosed). It is not evidence about markets.
