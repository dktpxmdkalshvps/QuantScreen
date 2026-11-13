"""Focused temporal and data-integrity checks; fixtures are never benchmark data."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark import (
    HORIZONS, MODELS, build_features, expected_sessions, feature_columns,
    load_asset, origins_for, predict, summarize, training_examples,
)


def fixture(rows=240):
    dates = expected_sessions("2019-01-01", "2020-02-28")[:rows]
    position = np.arange(len(dates), dtype=float)
    close = 100 + position * 0.08 + np.sin(position / 8) * 2
    return pd.DataFrame({"open": close - 0.2, "high": close + 1,
                         "low": close - 1, "close": close,
                         "volume": 1_000_000 + (position % 17) * 1000}, index=dates)


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()
        self.features = build_features(self.data)
        self.origin = 200

    def test_feature_timestamp_alignment(self):
        expected = self.data["close"].iloc[self.origin] / self.data["close"].iloc[self.origin - 5] - 1
        self.assertAlmostEqual(self.features["return_5"].iloc[self.origin], expected)
        self.assertEqual(list(self.features.columns), feature_columns("lgb_technical"))
        self.assertTrue(self.features.iloc[self.origin].notna().all())

    def test_no_lookahead_in_features(self):
        cutoff = self.origin + 1
        truncated = build_features(self.data.iloc[:cutoff])
        pd.testing.assert_frame_equal(self.features.iloc[:cutoff], truncated)
        changed = self.data.copy()
        changed.iloc[cutoff:, changed.columns.get_loc("close")] *= 7
        changed.iloc[cutoff:, changed.columns.get_loc("high")] *= 7
        changed.iloc[cutoff:, changed.columns.get_loc("volume")] *= 7
        pd.testing.assert_series_equal(build_features(changed).iloc[self.origin],
                                       self.features.iloc[self.origin])

    def test_target_timestamp_and_training_cutoff(self):
        horizon = 7
        x, y = training_examples(self.features, self.data["close"], horizon,
                                 self.origin, feature_columns("lgb_technical"))
        self.assertEqual(len(x), len(y))
        self.assertAlmostEqual(y[-1], np.log(self.data["close"].iloc[self.origin] /
                                            self.data["close"].iloc[self.origin - horizon]))
        changed = self.data["close"].copy()
        changed.iloc[self.origin + 1:] *= 8
        _, unchanged_y = training_examples(self.features, changed, horizon,
                                           self.origin, feature_columns("lgb_technical"))
        np.testing.assert_array_equal(y, unchanged_y)
        self.assertGreater(self.data.index[self.origin + horizon], self.data.index[self.origin])

    def test_models_cannot_read_future_prices(self):
        changed = self.data.copy()
        changed.iloc[self.origin + 1:, :] *= 5
        changed_features = build_features(changed)
        for model in MODELS:
            with self.subTest(model=model):
                before = predict(model, self.data, self.features, self.origin, 5)
                after = predict(model, changed, changed_features, self.origin, 5)
                self.assertAlmostEqual(before, after, places=9)

    def test_reproducible_seed(self):
        first = predict("lgb_technical", self.data, self.features, self.origin, 5)
        second = predict("lgb_technical", self.data, self.features, self.origin, 5)
        self.assertEqual(first, second)

    def test_identical_origin_schedule_for_all_models(self):
        for horizon in HORIZONS:
            origins = origins_for(self.data, horizon, 180, 20, "2019-01-01")
            self.assertTrue(all(index + horizon < len(self.data) for index in origins))
            self.assertTrue(all(a < b for a, b in zip(origins, origins[1:])))
        baseline = [{"origin": "2019-01-02", "target": "2019-01-03", "actual": 101.,
                     "prediction": 100., "wall_s": 0.1, "cpu_s": 0.1, "peak_rss_bytes": 1}]
        shifted = [dict(baseline[0], target="2019-01-04")]
        with self.assertRaisesRegex(AssertionError, "origins or targets differ"):
            summarize(shifted, baseline, 1)


class DataTests(unittest.TestCase):
    def test_missing_session_is_reported(self):
        original = fixture()
        missing = str(original.index[100].date())
        data = original.drop(original.index[100]).reset_index(names="date")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "asset.csv"
            data.to_csv(path, index=False)
            _, audit = load_asset(path, "SPY", str(data["date"].iloc[0].date()),
                                  str(data["date"].iloc[-1].date()))
            self.assertEqual(audit["missing_sessions"], [missing])

    def test_duplicate_or_invalid_rows_rejected(self):
        data = fixture().reset_index(names="date")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "asset.csv"
            pd.concat([data, data.iloc[[0]]]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_asset(path, "SPY", "2019-01-01", "2020-02-28")
            data.loc[10, "close"] = np.nan
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing or nonfinite"):
                load_asset(path, "SPY", "2019-01-01", "2020-02-28")


if __name__ == "__main__":
    unittest.main()
