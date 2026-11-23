import dataclasses

import pandas as pd
import pytest

from backtester.events import FillEvent, MarketEvent, OrderEvent, Phase, SignalEvent

T0, T1 = pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05")


def test_events_are_immutable():
    ev = MarketEvent(T0, Phase.CLOSE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.timestamp = T1  # type: ignore[misc]
    o = OrderEvent(1, T0, "A", 10, T0, T1, 100.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.quantity = 20  # type: ignore[misc]


def test_order_validation():
    with pytest.raises(ValueError):
        OrderEvent(1, T0, "A", 0, T0, T1, 100.0)
    with pytest.raises(ValueError, match="eligible"):
        OrderEvent(1, T0, "A", 5, T0, T0, 100.0)  # same-bar eligibility forbidden
    assert OrderEvent(1, T0, "A", -5, T0, T1, 100.0).side == "sell"


def test_signal_validation():
    with pytest.raises(ValueError):
        SignalEvent(T0, "", 1.0)
    with pytest.raises(ValueError):
        SignalEvent(T0, "A", float("nan"))


def test_fill_shortfall_and_notional():
    f = FillEvent(T1, Phase.OPEN, 1, "A", 10, 100.0, 99.0, 100.5, 1.0, 0.5, 0.0, 4.5, T0)
    assert f.notional == pytest.approx(1005.0)
    assert f.shortfall == pytest.approx(5.0)
