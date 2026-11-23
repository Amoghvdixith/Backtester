"""Strategy interface.

A strategy is called once per session at ``Phase.CLOSE`` with the completed bar
and returns either:

* ``None``  - no change: keep the current targets (no orders are generated), or
* a list of :class:`~backtester.events.SignalEvent` - a *complete* target view.
  Held instruments that are not mentioned are targeted to zero.

Strategies see market data only through :class:`~backtester.data.PointInTimeView`.
They express direction or ranking; sizing and portfolio limits are separate
components injected into the engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .data import PointInTimeView
from .events import MarketEvent, SignalEvent


class Strategy(ABC):
    """Base class for strategies.

    Subclasses set ``name`` and implement :meth:`on_bar`.  Parameters are passed as
    keyword arguments, validated in :meth:`validate_params`, and recorded in results.
    """

    name: str = "strategy"

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)
        self.validate_params()
        self._view: PointInTimeView | None = None

    def validate_params(self) -> None:  # noqa: B027 - optional hook
        """Raise ``ValueError`` for invalid parameter combinations."""

    def bind(self, view: PointInTimeView) -> None:
        self._view = view
        self.reset()

    def reset(self) -> None:  # noqa: B027 - optional hook
        """Clear internal state before a new run."""

    @property
    def view(self) -> PointInTimeView:
        if self._view is None:
            raise RuntimeError("strategy is not bound to a data view")
        return self._view

    def signal(self, event: MarketEvent, symbol: str, direction: float) -> SignalEvent:
        return SignalEvent(event.timestamp, symbol, float(direction), self.name)

    @abstractmethod
    def on_bar(self, event: MarketEvent) -> list[SignalEvent] | None:
        """React to a completed bar; see module docstring for return semantics."""

    @property
    def warmup(self) -> int:
        """Number of observed bars needed before the first meaningful signal."""
        return 0

    def describe(self) -> dict[str, Any]:
        return {"strategy": self.name, **self.params}
