"""Position sizing: turn strategy directions into a coherent set of target weights.

The strategy decides *direction/ranking*; a sizer decides *how much*; the
:class:`PortfolioConstraints` then enforce portfolio-wide limits on the whole
target vector at once (one coherent rebalance).

Notes
-----
* ``FixedFraction`` is an *allocation* fraction of equity per position.  It is not
  a fixed fraction of equity *at risk* (that would require a stop distance).
* ``VolTarget`` scales each position by ``target_vol / sigma_hat`` using trailing
  data only.  It ignores correlations between assets (each active position gets
  an equal share of the volatility budget), so realised portfolio volatility will
  generally differ from the target.
* ``KellyCapped`` uses the single-asset continuous-time approximation
  ``f* = mu / sigma^2`` with mu the mean daily simple *excess* return and sigma^2
  the daily variance (same frequency, so the ratio is unit-free).  It is highly
  sensitive to estimation error in mu.
* None of these rules is assumed to improve performance; that is an empirical
  question tested in the robustness experiments.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .data import PointInTimeView
from .events import SignalEvent


@dataclass(frozen=True)
class PortfolioConstraints:
    """Portfolio-wide limits applied to the full target-weight vector.

    max_gross: sum of |w| (gross leverage).  1.0 = no leverage.
    max_net: |sum w| limit.
    max_position: per-instrument |w| cap (concentration).
    allow_short: if False, negative targets are set to zero.
    """

    max_gross: float = 1.0
    max_net: float = 1.0
    max_position: float = 1.0
    allow_short: bool = False

    def __post_init__(self) -> None:
        if self.max_gross <= 0 or self.max_net < 0 or self.max_position <= 0:
            raise ValueError("constraint limits must be positive")

    def apply(self, weights: dict[str, float]) -> dict[str, float]:
        w = {s: float(v) for s, v in weights.items() if math.isfinite(v)}
        if not self.allow_short:
            w = {s: max(0.0, v) for s, v in w.items()}
        w = {s: float(np.clip(v, -self.max_position, self.max_position)) for s, v in w.items()}
        gross = sum(abs(v) for v in w.values())
        if gross > self.max_gross:
            k = self.max_gross / gross
            w = {s: v * k for s, v in w.items()}
        net = sum(w.values())
        if abs(net) > self.max_net and net != 0:
            # scale only the side that creates the excess net exposure
            side = 1.0 if net > 0 else -1.0
            side_sum = sum(v for v in w.values() if np.sign(v) == side)
            excess = abs(net) - self.max_net
            k = max(0.0, 1.0 - excess / abs(side_sum))
            w = {s: (v * k if np.sign(v) == side else v) for s, v in w.items()}
        return w


class Sizer(ABC):
    """Maps a batch of signals to raw target weights (before constraints)."""

    @abstractmethod
    def weights(self, signals: list[SignalEvent], view: PointInTimeView) -> dict[str, float]:
        ...


@dataclass
class EqualWeight(Sizer):
    """Split ``target_gross`` equally among non-zero signals; sign from direction."""

    target_gross: float = 1.0

    def weights(self, signals: list[SignalEvent], view: PointInTimeView) -> dict[str, float]:
        active = [s for s in signals if s.direction != 0]
        out = {s.symbol: 0.0 for s in signals}
        if not active:
            return out
        w = self.target_gross / len(active)
        for s in active:
            out[s.symbol] = w * float(np.sign(s.direction))
        return out


@dataclass
class FixedFraction(Sizer):
    """Allocate ``fraction`` of equity to every active signal (constraints scale down later)."""

    fraction: float = 0.1

    def __post_init__(self) -> None:
        if not 0 < self.fraction <= 10:
            raise ValueError("fraction must be in (0, 10]")

    def weights(self, signals: list[SignalEvent], view: PointInTimeView) -> dict[str, float]:
        return {s.symbol: self.fraction * float(np.sign(s.direction)) for s in signals}


def trailing_vol(view: PointInTimeView, symbol: str, lookback: int,
                 halflife: float | None = None) -> float:
    """Annualised volatility of daily log returns from visible observed closes.

    Returns NaN if fewer than ``lookback // 2 + 1`` observations are available.
    """
    px = view.history(symbol, "close", n=lookback + 1)
    if len(px) < lookback // 2 + 2:
        return float("nan")
    r = np.diff(np.log(px))
    if halflife:
        lam = 0.5 ** (1.0 / halflife)
        wts = lam ** np.arange(len(r))[::-1]
        wts /= wts.sum()
        var = float(np.sum(wts * r ** 2))  # zero-mean EWMA variance
    else:
        var = float(np.var(r, ddof=1))
    return math.sqrt(var * 252.0)


@dataclass
class VolTarget(Sizer):
    """w_i = sign_i * min(max_leverage_per_asset, target_vol / sigma_i) / n_active.

    Assets with insufficient history or vol below ``min_vol`` get zero weight
    (insufficient history) or are capped (near-zero vol), respectively.
    """

    target_vol: float = 0.10
    lookback: int = 60
    halflife: float | None = None
    max_leverage_per_asset: float = 1.0
    min_vol: float = 0.01

    def weights(self, signals: list[SignalEvent], view: PointInTimeView) -> dict[str, float]:
        active = [s for s in signals if s.direction != 0]
        out = {s.symbol: 0.0 for s in signals}
        if not active:
            return out
        n = len(active)
        for s in active:
            vol = trailing_vol(view, s.symbol, self.lookback, self.halflife)
            if not math.isfinite(vol):
                continue
            scale = min(self.max_leverage_per_asset, self.target_vol / max(vol, self.min_vol))
            out[s.symbol] = float(np.sign(s.direction)) * scale / n
        return out


@dataclass
class KellyCapped(Sizer):
    """Capped, shrunk fractional Kelly (single-asset approximation, per asset / n_active).

    f* = mu / sigma^2, with mu the mean daily simple excess return and sigma^2 the
    daily variance (same frequency, so the ratio is unit-free).  The estimate is
    shrunk toward zero (``shrinkage`` in [0, 1], 1 = no shrinkage), multiplied by
    ``kelly_fraction`` and capped at ``cap``.  Only the sign agreeing with the
    strategy direction is used; a negative Kelly estimate for a long signal gives 0.
    """

    lookback: int = 252
    kelly_fraction: float = 0.5
    shrinkage: float = 0.5
    cap: float = 1.0
    daily_rf: float = 0.0
    min_obs: int = 126

    def weights(self, signals: list[SignalEvent], view: PointInTimeView) -> dict[str, float]:
        active = [s for s in signals if s.direction != 0]
        out = {s.symbol: 0.0 for s in signals}
        if not active:
            return out
        n = len(active)
        for s in active:
            px = view.history(s.symbol, "close", n=self.lookback + 1)
            if len(px) < self.min_obs + 1:
                continue
            r = np.diff(px) / px[:-1] - self.daily_rf
            var = float(np.var(r, ddof=1))
            if var <= 1e-12:
                continue
            f = self.shrinkage * float(np.mean(r)) / var * self.kelly_fraction
            f_dir = f * float(np.sign(s.direction))
            f_dir = float(np.clip(f_dir, 0.0, self.cap))
            out[s.symbol] = float(np.sign(s.direction)) * f_dir / n
        return out


def build_sizer(cfg: dict | None) -> Sizer:
    cfg = dict(cfg or {"type": "equal_weight"})
    kind = cfg.pop("type", "equal_weight")
    registry: dict[str, type[Sizer]] = {
        "equal_weight": EqualWeight,
        "fixed_fraction": FixedFraction,
        "vol_target": VolTarget,
        "kelly": KellyCapped,
    }
    if kind not in registry:
        raise ValueError(f"unknown sizer {kind!r}; choose from {sorted(registry)}")
    return registry[kind](**cfg)
