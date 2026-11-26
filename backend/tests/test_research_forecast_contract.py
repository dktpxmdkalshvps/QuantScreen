"""Contract regression tests; no network, database or optional ML packages."""
import unittest
from datetime import date, datetime, timezone

from app.research.forecast_contract import (
    ForecastRequest, ForecastResult, TargetKind, terminal_close_naive,
)


class ForecastContractTests(unittest.TestCase):
    def setUp(self):
        self.request = ForecastRequest("AAPL", "US", date(2026, 11, 20), 5)
        self.kwargs = dict(
            request=self.request, model_id="naive", model_version="1",
            predicted_value=100.0, trained_until=date(2026, 11, 20),
            target_date=date(2026, 11, 27),
            available_at=datetime(2026, 11, 20, 22, tzinfo=timezone.utc),
        )

    def test_naive_adapter_repeats_last_close(self):
        result = terminal_close_naive(
            self.request, last_close=101.5,
            trained_until=self.kwargs["trained_until"],
            target_date=self.kwargs["target_date"],
            available_at=self.kwargs["available_at"],
        )
        self.assertEqual(result.predicted_value, 101.5)
        self.assertEqual(result.model_id, "naive")
        self.assertEqual(result.request.target, TargetKind.TERMINAL_CLOSE)

    def test_invalid_horizon(self):
        with self.assertRaises(ValueError):
            ForecastRequest("AAPL", "US", date(2026, 11, 20), 0)

    def test_missing_instrument(self):
        with self.assertRaises(ValueError):
            ForecastRequest(" ", "US", date(2026, 11, 20), 5)

    def test_future_training_cutoff_is_rejected(self):
        with self.assertRaises(ValueError):
            ForecastResult(**{**self.kwargs, "trained_until": date(2026, 11, 21)})

    def test_non_future_target_is_rejected(self):
        with self.assertRaises(ValueError):
            ForecastResult(**{**self.kwargs, "target_date": date(2026, 11, 20)})

    def test_nonfinite_or_nonpositive_prediction_is_rejected(self):
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ForecastResult(**{**self.kwargs, "predicted_value": value})

    def test_naive_forecast_rejects_unavailable_close(self):
        with self.assertRaises(ValueError):
            terminal_close_naive(
                self.request, last_close=float("nan"),
                trained_until=self.kwargs["trained_until"],
                target_date=self.kwargs["target_date"],
                available_at=self.kwargs["available_at"],
            )

    def test_timezone_required(self):
        with self.assertRaises(ValueError):
            ForecastResult(**{
                **self.kwargs,
                "available_at": datetime(2026, 11, 20, 22),
            })


if __name__ == "__main__":
    unittest.main()
