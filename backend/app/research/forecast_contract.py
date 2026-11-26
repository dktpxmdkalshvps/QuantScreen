"""Versioned, provider-independent contract for research forecasts.

This module deliberately does not import experiment runners or train models.
Adapters can wrap existing benchmark implementations in a later PR.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from math import isfinite


class TargetKind(str, Enum):
    TERMINAL_CLOSE = "terminal_close"


@dataclass(frozen=True)
class ForecastRequest:
    instrument_id: str
    market: str
    as_of: date
    horizon_sessions: int
    target: TargetKind = TargetKind.TERMINAL_CLOSE

    def __post_init__(self) -> None:
        if not self.instrument_id.strip() or not self.market.strip():
            raise ValueError("instrument_id and market are required")
        if self.horizon_sessions < 1:
            raise ValueError("horizon_sessions must be positive")
        if self.target != TargetKind.TERMINAL_CLOSE:
            raise ValueError("unsupported forecast target")


@dataclass(frozen=True)
class ForecastResult:
    request: ForecastRequest
    model_id: str
    model_version: str
    predicted_value: float
    trained_until: date
    target_date: date
    available_at: datetime

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.model_version.strip():
            raise ValueError("model_id and model_version are required")
        if not isfinite(self.predicted_value) or self.predicted_value <= 0:
            raise ValueError("terminal close prediction must be positive and finite")
        if self.trained_until > self.request.as_of:
            raise ValueError("training cutoff cannot exceed forecast as_of")
        if self.target_date <= self.request.as_of:
            raise ValueError("target date must follow forecast as_of")
        if self.available_at.tzinfo is None or self.available_at.utcoffset() is None:
            raise ValueError("available_at must be timezone-aware")


def terminal_close_naive(
    request: ForecastRequest,
    *,
    last_close: float,
    trained_until: date,
    target_date: date,
    available_at: datetime,
) -> ForecastResult:
    """No-fit benchmark: repeat the last known close at the forecast horizon."""
    return ForecastResult(
        request=request,
        model_id="naive",
        model_version="1",
        predicted_value=float(last_close),
        trained_until=trained_until,
        target_date=target_date,
        available_at=available_at,
    )
