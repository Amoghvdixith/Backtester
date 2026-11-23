"""Transaction-cost and financing models.

Unit conventions (all explicit):

* ``*_bps`` values are basis points (1 bp = 0.0001) of the **reference price**.
* ``spread_bps`` is the FULL quoted bid-ask spread; a market order pays HALF of it.
* Spread, slippage and impact are *price* effects: they move the fill price
  adversely.  They are not deducted from cash again (no double counting).
  ``Commission`` is the only explicit cash fee.
* Square-root impact (fraction of price)::

      impact_frac = coefficient * sigma_daily_lagged * sqrt(|q| / ADV_lagged)

  so the cash cost is ``impact_frac * |q| * p``.  ``sigma`` and ``ADV`` are
  computed from bars strictly before the execution session.  The coefficient is
  an assumption unless calibrated to the user's own executions.
* Financing uses simple interest with an explicit day-count basis:
  ``accrual = balance * annual_rate * calendar_days / basis``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Commission:
    """Commission = clip(fixed + per_share*|q| + rate_bps*notional, minimum, maximum).

    The minimum applies per executed order with non-zero quantity.
    """

    fixed: float = 0.0
    per_share: float = 0.0
    rate_bps: float = 0.0
    minimum: float = 0.0
    maximum: float | None = None

    def __post_init__(self) -> None:
        for name in ("fixed", "per_share", "rate_bps", "minimum"):
            if getattr(self, name) < 0:
                raise ValueError(f"Commission.{name} must be >= 0")
        if self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("Commission.maximum must be >= minimum")

    def __call__(self, quantity: float, price: float) -> float:
        if quantity == 0:
            return 0.0
        q = abs(quantity)
        c = self.fixed + self.per_share * q + self.rate_bps * 1e-4 * q * price
        c = max(c, self.minimum)
        if self.maximum is not None:
            c = min(c, self.maximum)
        return float(c)


@dataclass(frozen=True)
class SpreadCost:
    """Half of the full quoted spread, in price fraction."""

    spread_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.spread_bps < 0:
            raise ValueError("spread_bps must be >= 0")

    def fraction(self) -> float:
        return 0.5 * self.spread_bps * 1e-4


@dataclass
class Slippage:
    """Adverse slippage in bps.  Deterministic by default.

    If ``stochastic_sd_bps > 0`` a Normal draw (seeded) is added; draws may be
    favourable.  The RNG is owned by the instance so runs are reproducible.
    """

    bps: float = 0.0
    stochastic_sd_bps: float = 0.0
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.bps < 0 or self.stochastic_sd_bps < 0:
            raise ValueError("slippage parameters must be >= 0")
        self._rng = np.random.default_rng(self.seed)

    def fraction(self) -> float:
        extra = 0.0
        if self.stochastic_sd_bps > 0:
            extra = float(self._rng.normal(0.0, self.stochastic_sd_bps))
        return (self.bps + extra) * 1e-4


@dataclass(frozen=True)
class SqrtImpact:
    """Square-root temporary impact model (assumption, not a universal fact).

    ``fraction = coefficient * sigma * sqrt(|q| / adv)``.  If lagged sigma or ADV are
    unavailable, ``fallback_bps`` is charged instead and flagged.
    ``max_fraction`` caps the impact so that absurd orders do not create negative prices.
    """

    coefficient: float = 0.0
    fallback_bps: float = 0.0
    max_fraction: float = 0.10

    def __post_init__(self) -> None:
        if self.coefficient < 0 or self.fallback_bps < 0:
            raise ValueError("impact parameters must be >= 0")

    def fraction(self, quantity: float, adv: float, sigma: float) -> tuple[float, bool]:
        if self.coefficient == 0 or quantity == 0:
            return 0.0, False
        if not (math.isfinite(adv) and math.isfinite(sigma)) or adv <= 0:
            return self.fallback_bps * 1e-4, True
        frac = self.coefficient * sigma * math.sqrt(abs(quantity) / adv)
        return min(frac, self.max_fraction), False


@dataclass
class CostModel:
    """Bundle of execution-cost components."""

    commission: Commission = field(default_factory=Commission)
    spread: SpreadCost = field(default_factory=SpreadCost)
    slippage: Slippage = field(default_factory=Slippage)
    impact: SqrtImpact = field(default_factory=SqrtImpact)

    @classmethod
    def zero(cls) -> CostModel:
        return cls()

    @classmethod
    def proportional(cls, one_way_bps: float) -> CostModel:
        """Flat one-way proportional cost charged as commission (replaces all other parts)."""
        return cls(commission=Commission(rate_bps=one_way_bps))

    @classmethod
    def from_dict(cls, d: dict | None, seed: int | None = None) -> CostModel:
        d = dict(d or {})
        if "proportional_bps" in d:
            return cls.proportional(float(d["proportional_bps"]))
        return cls(
            commission=Commission(**d.get("commission", {})),
            spread=SpreadCost(**d.get("spread", {})),
            slippage=Slippage(seed=seed, **d.get("slippage", {})),
            impact=SqrtImpact(**d.get("impact", {})),
        )


@dataclass(frozen=True)
class FinancingModel:
    """Cash interest, debit interest and short borrow fees (annual rates, simple accrual).

    ``credit_rate`` applies to positive interest-bearing cash, ``debit_rate`` to a
    negative balance (margin borrowing), ``borrow_fee`` to the absolute short market
    value.  If ``short_proceeds_earn_interest`` is False, short-sale proceeds are
    excluded from the interest-bearing balance (they are treated as collateral that
    earns no rebate).
    """

    credit_rate: float = 0.0
    debit_rate: float = 0.0
    borrow_fee: float = 0.0
    day_count_basis: int = 365
    short_proceeds_earn_interest: bool = False

    def __post_init__(self) -> None:
        if self.day_count_basis not in (360, 365):
            raise ValueError("day_count_basis must be 360 or 365")
        if self.borrow_fee < 0 or self.debit_rate < 0:
            raise ValueError("borrow_fee and debit_rate must be >= 0")

    def accrue(self, cash: float, short_mv_abs: float, days: int) -> tuple[float, float]:
        """Return (interest, borrow_fee) for ``days`` calendar days.

        interest is signed (+ earned, - paid); borrow fee is returned as a positive cost.
        """
        if days <= 0:
            return 0.0, 0.0
        base = cash if self.short_proceeds_earn_interest else cash - short_mv_abs
        rate = self.credit_rate if base >= 0 else self.debit_rate
        interest = base * rate * days / self.day_count_basis
        fee = short_mv_abs * self.borrow_fee * days / self.day_count_basis
        return float(interest), float(fee)
