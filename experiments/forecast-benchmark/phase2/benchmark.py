"""Phase 2 terminal-close walk-forward benchmark on cached, real OHLCV files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Easter, Holiday, MO, TH, nearest_workday,
    sunday_to_monday,
)
from pandas.tseries.offsets import DateOffset


TICKERS = ("SPY", "QQQ", "IWM", "AAPL")
HORIZONS = (1, 5, 7, 20)
MODELS = ("naive", "drift", "arima_css", "lgb_lags", "lgb_technical")
LAGS = (1, 5, 10, 20)
FEATURE_SEED = 20261112
BASIC_FEATURES = [f"return_{lag}" for lag in LAGS]
TECHNICAL_FEATURES = BASIC_FEATURES + [
    *(f"price_ma_{window}" for window in (5, 10, 20, 60)),
    "volatility_5", "relative_volume_5", "volatility_20", "relative_volume_20",
    "rsi_14", "distance_high_20",
]


class NyseHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year", month=1, day=1, observance=sunday_to_monday),
        Holiday("Martin Luther King Jr", month=1, day=1, offset=DateOffset(weekday=MO(3))),
        Holiday("Washington Birthday", month=2, day=1, offset=DateOffset(weekday=MO(3))),
        Holiday("Good Friday", month=1, day=1, offset=[Easter(), DateOffset(days=-2)]),
        Holiday("Memorial Day", month=5, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday("Juneteenth", month=6, day=19, observance=nearest_workday,
                start_date="2022-01-01"),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        Holiday("Labor Day", month=9, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("Thanksgiving", month=11, day=1, offset=DateOffset(weekday=TH(4))),
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


EXCEPTIONAL_CLOSURES = pd.DatetimeIndex(["2018-12-05", "2025-01-09"])


def expected_sessions(start: str, end: str) -> pd.DatetimeIndex:
    dates = pd.bdate_range(start, end)
    holidays = NyseHolidays().holidays(start=start, end=end)
    return dates.difference(holidays).difference(EXCEPTIONAL_CLOSURES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_asset(path: Path, ticker: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = ("date", "open", "high", "low", "close", "volume")
    absent = set(required) - set(raw.columns)
    if absent:
        raise ValueError(f"{ticker}: missing columns {sorted(absent)}")
    if "ticker" in raw.columns:
        raw = raw.loc[raw["ticker"] == ticker]
    elif "symbol" in raw.columns:
        raw = raw.loc[raw["symbol"] == ticker]
    raw = raw.loc[:, list(required)].copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="raise", utc=True).dt.tz_localize(None).dt.normalize()
    raw = raw[(raw["date"] >= pd.Timestamp(start)) & (raw["date"] <= pd.Timestamp(end))]
    if raw.empty:
        raise ValueError(f"{ticker}: no rows in target range")
    if raw["date"].duplicated().any():
        raise ValueError(f"{ticker}: duplicate session dates")
    raw = raw.sort_values("date").set_index("date")
    for column in required[1:]:
        raw[column] = pd.to_numeric(raw[column], errors="raise")
    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError(f"{ticker}: missing or nonfinite OHLCV values")
    if (raw[["open", "high", "low", "close"]] <= 0).any().any() or (raw["volume"] <= 0).any():
        raise ValueError(f"{ticker}: nonpositive OHLCV values")
    tolerance = raw["close"] * 0.0001
    bad_bars = ((raw["high"] + tolerance < raw[["open", "close", "low"]].max(axis=1)) |
                (raw["low"] - tolerance > raw[["open", "close", "high"]].min(axis=1)))
    if bad_bars.any():
        raise ValueError(f"{ticker}: {int(bad_bars.sum())} inconsistent OHLC bars")
    sessions = expected_sessions(start, end)
    missing = sessions.difference(raw.index)
    unexpected = raw.index.difference(sessions)
    jumps = raw["close"].pct_change().abs()
    suspicious = raw.index[jumps > 0.40].strftime("%Y-%m-%d").tolist()
    audit = {
        "rows": len(raw), "first": str(raw.index[0].date()), "last": str(raw.index[-1].date()),
        "expected_sessions": len(sessions), "missing_sessions": missing.strftime("%Y-%m-%d").tolist(),
        "unexpected_sessions": unexpected.strftime("%Y-%m-%d").tolist(),
        "close_jumps_over_40pct": suspicious,
    }
    return raw, audit


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    close = data["close"]
    volume = data["volume"]
    output = pd.DataFrame(index=data.index)
    for lag in LAGS:
        output[f"return_{lag}"] = close.div(close.shift(lag)) - 1
    for window in (5, 10, 20, 60):
        output[f"price_ma_{window}"] = close.div(close.rolling(window, min_periods=window).mean()) - 1
    daily_return = np.log(close).diff()
    for window in (5, 20):
        output[f"volatility_{window}"] = daily_return.rolling(window, min_periods=window).std(ddof=0)
        output[f"relative_volume_{window}"] = volume.div(volume.rolling(window, min_periods=window).mean())
    change = close.diff()
    gain = change.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = -change.clip(upper=0).rolling(14, min_periods=14).mean()
    rsi = 100 - 100 / (1 + gain.div(loss.replace(0, np.nan)))
    output["rsi_14"] = rsi.mask((loss == 0) & (gain > 0), 100).mask(
        (loss == 0) & (gain == 0), 50
    )
    output["distance_high_20"] = close.div(data["high"].rolling(20, min_periods=20).max()) - 1
    return output


def feature_columns(model: str) -> list[str]:
    return BASIC_FEATURES if model == "lgb_lags" else TECHNICAL_FEATURES


def training_examples(features: pd.DataFrame, close: pd.Series, horizon: int,
                      origin_position: int, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    # A row s is eligible at origin t only when its target close[s+h] is known: s+h <= t.
    last_training_position = origin_position - horizon
    if last_training_position < 0:
        raise ValueError("Origin precedes the first known training target")
    rows = features.iloc[:last_training_position + 1][columns]
    targets = np.log(close.shift(-horizon).div(close)).iloc[:last_training_position + 1]
    valid = rows.notna().all(axis=1) & np.isfinite(targets)
    if valid.sum() < 80:
        raise ValueError(f"Only {int(valid.sum())} valid training samples")
    return rows.loc[valid].to_numpy(dtype=float), targets.loc[valid].to_numpy(dtype=float)


def origins_for(data: pd.DataFrame, horizon: int, min_train: int, step: int,
                evaluation_start: str) -> list[int]:
    first = max(min_train - 1, int(data.index.searchsorted(pd.Timestamp(evaluation_start))))
    return list(range(first, len(data) - horizon, step))


def arima_css(train: np.ndarray, horizon: int) -> float:
    """Custom conditional least-squares approximation, not statsmodels likelihood ARIMA."""
    from scipy.optimize import minimize
    from scipy.signal import lfilter

    differences = np.diff(train)
    scale = max(float(np.std(differences)), 1e-12)
    normalized = differences / scale

    def objective(parameters: np.ndarray) -> float:
        errors = lfilter([1.0, -parameters[0]], [1.0, parameters[1]], normalized)
        return float(np.dot(errors[1:], errors[1:]))

    fitted = minimize(objective, [0.0, 0.0], method="L-BFGS-B",
                      bounds=[(-0.99, 0.99), (-0.99, 0.99)])
    if not fitted.success:
        raise RuntimeError(f"ARIMA CSS failed: {fitted.message}")
    phi, theta = fitted.x
    last_error = float(lfilter([1.0, -phi], [1.0, theta], normalized)[-1] * scale)
    last_difference = float(differences[-1])
    predicted = float(train[-1])
    for step in range(horizon):
        last_difference = phi * last_difference + (theta * last_error if step == 0 else 0.0)
        predicted += last_difference
    return predicted


def predict(model: str, data: pd.DataFrame, features: pd.DataFrame,
            origin_position: int, horizon: int) -> float:
    close = data["close"]
    origin_close = float(close.iloc[origin_position])
    if model == "naive":
        return origin_close
    if model == "drift":
        return origin_close + horizon * (origin_close - float(close.iloc[0])) / origin_position
    if model == "arima_css":
        return arima_css(close.iloc[:origin_position + 1].to_numpy(dtype=float), horizon)
    import lightgbm as lgb

    columns = feature_columns(model)
    x_train, y_train = training_examples(features, close, horizon, origin_position, columns)
    x_origin = features.iloc[origin_position][columns]
    if not np.isfinite(x_origin.to_numpy(dtype=float)).all():
        raise ValueError("Origin features are incomplete")
    booster = lgb.train(
        {"objective": "regression", "learning_rate": 0.05, "num_leaves": 15,
         "max_depth": 4, "min_data_in_leaf": 20, "verbosity": -1, "num_threads": 1,
         "seed": FEATURE_SEED, "feature_fraction_seed": FEATURE_SEED,
         "data_random_seed": FEATURE_SEED, "deterministic": True},
        lgb.Dataset(x_train, label=y_train, feature_name=columns), num_boost_round=100,
    )
    return origin_close * float(np.exp(booster.predict(x_origin.to_numpy(dtype=float)[None, :])[0]))


def measured_prediction(model: str, data: pd.DataFrame, features: pd.DataFrame,
                        origin_position: int, horizon: int) -> dict:
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(0.01):
            peak[0] = max(peak[0], process.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    cpu_before = process.cpu_times()
    start = time.perf_counter()
    monitor.start()
    try:
        prediction = predict(model, data, features, origin_position, horizon)
    finally:
        stop.set()
        monitor.join()
    wall = time.perf_counter() - start
    cpu_after = process.cpu_times()
    peak[0] = max(peak[0], process.memory_info().rss)
    if not np.isfinite(prediction) or prediction <= 0:
        raise ValueError(f"{model}: invalid prediction")
    return {"prediction": prediction, "wall_s": wall,
            "cpu_s": (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system),
            "peak_rss_bytes": peak[0], "rss_increase_bytes": max(0, peak[0] - baseline)}


def summarize(rows: list[dict], naive: list[dict] | None, seed: int) -> dict:
    prediction = np.array([row["prediction"] for row in rows])
    actual = np.array([row["actual"] for row in rows])
    errors = np.abs(prediction - actual)
    result = {
        "count": len(rows), "mae": float(errors.mean()),
        "rmse": float(np.sqrt(np.mean((prediction - actual) ** 2))),
        "smape_pct": float(np.mean(200 * errors / (np.abs(prediction) + np.abs(actual)))),
        "mean_wall_s": float(np.mean([row["wall_s"] for row in rows])),
        "total_wall_s": float(np.sum([row["wall_s"] for row in rows])),
        "mean_cpu_s": float(np.mean([row["cpu_s"] for row in rows])),
        "total_cpu_s": float(np.sum([row["cpu_s"] for row in rows])),
        "peak_rss_bytes": max(row["peak_rss_bytes"] for row in rows),
    }
    if naive is not None:
        if [(r["origin"], r["target"]) for r in rows] != [(r["origin"], r["target"]) for r in naive]:
            raise AssertionError("Evaluation origins or targets differ across models")
        baseline_errors = np.abs(np.array([row["prediction"] for row in naive]) - actual)
        result["mae_improvement_pct"] = float(100 * (1 - errors.mean() / baseline_errors.mean()))
        result["win_rate_pct"] = float(100 * np.mean(errors < baseline_errors))
        result["ties"] = int(np.sum(errors == baseline_errors))
        # Paired resampling keeps model and Naive errors from each origin together.
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(rows), size=(2000, len(rows)))
        improvement = 100 * (1 - errors[draws].mean(axis=1) / baseline_errors[draws].mean(axis=1))
        result["paired_bootstrap_improvement_95pct"] = np.quantile(improvement, [0.025, 0.975]).tolist()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--min-train", type=int, default=252)
    parser.add_argument("--step", type=int, default=40)
    parser.add_argument("--evaluation-start", default="2019-01-01")
    args = parser.parse_args()
    selected = [name.strip() for name in args.models.split(",")]
    if not selected or len(selected) != len(set(selected)) or any(name not in MODELS for name in selected):
        parser.error(f"models must be unique members of {MODELS}")
    if min(args.min_train, args.step) < 1:
        parser.error("min-train and step must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assets = {}
    audits = {}
    for ticker in TICKERS:
        spec = manifest["assets"][ticker]
        path = args.manifest.parent / spec["path"]
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"{ticker}: SHA-256 mismatch for {path}")
        frame, audit = load_asset(path, ticker, manifest["start"], manifest["end"])
        if audit["missing_sessions"] or audit["unexpected_sessions"] or audit["close_jumps_over_40pct"]:
            raise ValueError(f"{ticker}: data audit failed: {audit}")
        assets[ticker] = frame
        audits[ticker] = audit
    output = {"phase": 2, "python": sys.version.split()[0], "manifest": str(args.manifest),
              "settings": {"horizons": HORIZONS, "min_train": args.min_train,
                           "step": args.step, "evaluation_start": args.evaluation_start,
                           "seed": FEATURE_SEED, "target": "terminal close at origin + horizon sessions"},
              "data_audit": audits, "results": {}}
    for ticker, data in assets.items():
        features = build_features(data)
        output["results"][ticker] = {}
        for horizon in HORIZONS:
            origins = origins_for(data, horizon, args.min_train, args.step, args.evaluation_start)
            if len(origins) < 20:
                raise ValueError(f"{ticker} horizon {horizon}: only {len(origins)} origins")
            group = {}
            for model in dict.fromkeys(["naive", *selected]):
                rows = []
                for origin in origins:
                    measurement = measured_prediction(model, data, features, origin, horizon)
                    rows.append({"origin": str(data.index[origin].date()),
                                 "target": str(data.index[origin + horizon].date()),
                                 "actual": float(data["close"].iloc[origin + horizon]),
                                 **measurement})
                group[model] = {"runs": rows, "aggregate": summarize(rows, group.get("naive", {}).get("runs")
                                                                  if model != "naive" else None,
                                                                  FEATURE_SEED + horizon)}
            output["results"][ticker][str(horizon)] = {name: group[name] for name in selected}
            print(f"{ticker} h={horizon}: " + ", ".join(
                f"{name} MAE={group[name]['aggregate']['mae']:.4f}" for name in selected), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
