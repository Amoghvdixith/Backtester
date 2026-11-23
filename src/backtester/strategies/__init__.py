"""Example strategies."""

from __future__ import annotations

from typing import Any

from ..strategy import Strategy
from .buy_and_hold import BuyAndHold, EqualWeightUniverse
from .momentum_12_1 import Momentum12_1
from .sma_crossover import SMACrossover

REGISTRY: dict[str, type[Strategy]] = {
    "buy_and_hold": BuyAndHold,
    "equal_weight_universe": EqualWeightUniverse,
    "sma_crossover": SMACrossover,
    "momentum_12_1": Momentum12_1,
}


def build_strategy(name: str, **params: Any) -> Strategy:
    if name not in REGISTRY:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name](**params)


__all__ = ["BuyAndHold", "EqualWeightUniverse", "Momentum12_1", "SMACrossover",
           "REGISTRY", "build_strategy"]
