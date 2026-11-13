import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import benchmark as b


def fixture(n=180):
    dates = pd.bdate_range("2021-01-04", periods=n)
    close = 100 * np.exp(np.cumsum(np.sin(np.arange(n) / 7) * .005 + .0002))
    return pd.DataFrame({"open": close, "high": close * 1.001,
                         "low": close * .999, "close": close,
                         "volume": 100000 + np.arange(n)}, index=dates)


class TargetTests(unittest.TestCase):
    def test_target_alignment_and_annualization(self):
        d = fixture()
        y = b.targets(d, 5)
        expected = np.diff(np.log(d.close.iloc[10:16].to_numpy()))
        self.assertAlmostEqual(y.volatility.iloc[10], np.sqrt(252 * np.mean(expected ** 2)))
        self.assertTrue(np.isnan(y.volatility.iloc[-5:]).all())
        self.assertAlmostEqual(b.realized_volatility(np.full(5, .01)), np.sqrt(252) * .01)

    def test_drawdown_includes_origin_and_intermediate_peaks(self):
        self.assertAlmostEqual(b.maximum_drawdown([100, 120, 90, 110]), .25)
        self.assertAlmostEqual(b.maximum_drawdown([100, 90, 95]), .10)
        self.assertEqual(b.maximum_drawdown([100, 110, 120]), 0)

    def test_future_changes_do_not_change_features_or_baselines(self):
        d = fixture()
        altered = d.copy()
        altered.iloc[101:, altered.columns.get_loc("close")] *= 4
        pd.testing.assert_series_equal(b.features(d).iloc[100], b.features(altered).iloc[100])
        y = b.targets(d, 5)
        for model, target in (("historical_volatility", "volatility"),
                              ("ewma_volatility", "volatility"),
                              ("historical_drawdown", "drawdown")):
            self.assertEqual(b.predict(model, target, d, b.features(d), y, 5, 100),
                             b.predict(model, target, altered, b.features(altered), y, 5, 100))
        self.assertNotEqual(b.targets(d, 5).volatility.iloc[100],
                            b.targets(altered, 5).volatility.iloc[100])

    def test_training_label_cutoff(self):
        d = fixture(300)
        x = b.features(d)
        y = b.targets(d, 20).volatility
        a, labels = b.training_examples(x, y, 20, 200)
        mutated = y.copy()
        mutated.iloc[181:] = 999
        a2, labels2 = b.training_examples(x, mutated, 20, 200)
        np.testing.assert_array_equal(a, a2)
        np.testing.assert_array_equal(labels, labels2)

    def test_deterministic_lgb_and_finite_output(self):
        d = fixture()
        x = b.features(d)
        y = b.targets(d, 5)
        first = b.predict("lightgbm", "volatility", d, x, y, 5, 170)
        second = b.predict("lightgbm", "volatility", d, x, y, 5, 170)
        self.assertTrue(np.isfinite(first))
        self.assertEqual(first, second)

    def test_lightgbm_cannot_read_future_prices(self):
        d = fixture(300)
        altered = d.copy()
        altered.iloc[201:, altered.columns.get_loc("close")] *= 3
        first = b.predict("lightgbm", "drawdown", d, b.features(d), b.targets(d, 20), 20, 200)
        second = b.predict("lightgbm", "drawdown", altered, b.features(altered),
                           b.targets(altered, 20), 20, 200)
        self.assertEqual(first, second)

    def test_paired_schedule_and_metrics(self):
        d = fixture(500)
        for h in b.HORIZONS:
            origins = b.origins_for(d, h)
            self.assertEqual(len(origins), len(set(origins)))
            self.assertTrue(all(i + h < len(d) for i in origins))
        ci = b.paired_interval(np.array([1., 2., 3.]), np.array([.8, 1.6, 2.4]), b.SEED)
        self.assertTrue(np.isfinite(ci).all())
        self.assertAlmostEqual(ci[0], 20)
        self.assertAlmostEqual(ci[1], 20)

    def test_duplicate_and_missing_sessions_rejected_by_shared_validator(self):
        validator = b.phase2_module()
        days = validator.expected_sessions("2021-01-04", "2021-01-08")
        d = pd.DataFrame({"date": days, "open": 100, "high": 101,
                          "low": 99, "close": 100, "volume": 1000})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            pd.concat([d, d.iloc[[0]]]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validator.load_asset(path, "SPY", "2021-01-04", "2021-01-08")
            d.iloc[1:].to_csv(path, index=False)
            _, audit = validator.load_asset(path, "SPY", "2021-01-04", "2021-01-08")
            self.assertEqual(audit["missing_sessions"], ["2021-01-04"])


if __name__ == "__main__":
    unittest.main()
