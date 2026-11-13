"""Reproducible, local, CPU-only expanding-window forecast comparison."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

MODELS = ("naive", "arima", "prophet", "lightgbm", "chronos_bolt")
REQUIRES = {
    "naive": (),
    "arima": ("scipy",),
    "prophet": ("prophet",),
    "lightgbm": ("lightgbm",),
    "chronos_bolt": ("torch", "chronos"),
}


def load_series(args):
    if args.synthetic:
        rng = np.random.default_rng(20261112)
        dates = pd.bdate_range("2024-01-02", periods=260)
        return {
            name: pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, vol, len(dates)))), index=dates)
            for name, vol in (("synthetic_low_vol", 0.008), ("synthetic_high_vol", 0.025))
        }
    if not args.csv:
        raise ValueError("Provide --csv with historical prices, or --synthetic for a harness check")
    frame = pd.read_csv(args.csv)
    required = [args.date_column, args.value_column] + ([args.series_column] if args.series_column else [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")
    frame[args.date_column] = pd.to_datetime(frame[args.date_column], errors="raise")
    frame[args.value_column] = pd.to_numeric(frame[args.value_column], errors="raise")
    if frame[args.value_column].isna().any() or (frame[args.value_column] <= 0).any():
        raise ValueError("Prices must be positive and nonmissing")
    groups = frame.groupby(args.series_column, sort=True) if args.series_column else [("series", frame)]
    result = {}
    for name, group in groups:
        if group[args.date_column].duplicated().any():
            raise ValueError(f"Duplicate dates in {name}")
        group = group.sort_values(args.date_column)
        result[str(name)] = pd.Series(group[args.value_column].to_numpy(dtype=float), index=group[args.date_column])
    return result


def forecast(model, train, horizon, checkpoint):
    values = train.to_numpy(dtype=float)
    if model == "naive":
        return np.repeat(values[-1], horizon), 0, 0.0
    if model == "arima":
        from scipy.optimize import minimize
        from scipy.signal import lfilter
        start = time.perf_counter()
        differences = np.diff(values)
        scale = max(float(np.std(differences)), 1e-12)
        normalized = differences / scale

        # ARIMA(1,1,1) without drift, fitted by conditional least squares.
        # lfilter computes e[t] = d[t] - phi*d[t-1] - theta*e[t-1].
        def squared_errors(params):
            residuals = lfilter([1.0, -params[0]], [1.0, params[1]], normalized)
            return float(np.dot(residuals[1:], residuals[1:]))

        fitted = minimize(squared_errors, [0.0, 0.0], method="L-BFGS-B",
                          bounds=[(-0.99, 0.99), (-0.99, 0.99)])
        if not fitted.success:
            raise RuntimeError(f"ARIMA conditional fit failed: {fitted.message}")
        phi, theta = fitted.x
        last_error = float(lfilter([1.0, -phi], [1.0, theta], normalized)[-1] * scale)
        last_difference = float(differences[-1])
        fit_s = time.perf_counter() - start
        size = len(pickle.dumps((phi, theta, last_error, last_difference, values[-1]),
                                protocol=pickle.HIGHEST_PROTOCOL))
        predictions = []
        level = float(values[-1])
        for step in range(horizon):
            next_difference = phi * last_difference + (theta * last_error if step == 0 else 0.0)
            level += next_difference
            predictions.append(level)
            last_difference = next_difference
        return np.asarray(predictions), size, fit_s
    if model == "prophet":
        from prophet import Prophet
        history = pd.DataFrame({"ds": train.index, "y": values})
        start = time.perf_counter()
        fitted = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
        fitted.fit(history)
        fit_s = time.perf_counter() - start
        size = len(pickle.dumps(fitted, protocol=pickle.HIGHEST_PROTOCOL))
        # Score exact held-out observation dates, allowing market holidays.
        future_dates = forecast.target_dates
        prediction = fitted.predict(pd.DataFrame({"ds": future_dates}))
        return prediction["yhat"].to_numpy(dtype=float), size, fit_s
    if model == "lightgbm":
        import lightgbm as lgb
        lag = 10
        features = np.asarray([values[i-lag:i][::-1] for i in range(lag, len(values))])
        targets = values[lag:]
        start = time.perf_counter()
        fitted = lgb.train(
            {"objective": "regression", "learning_rate": 0.05, "num_leaves": 7,
             "max_depth": 3, "verbosity": -1, "num_threads": 1, "seed": 20261112},
            lgb.Dataset(features, label=targets), num_boost_round=80,
        )
        fit_s = time.perf_counter() - start
        size = len(fitted.model_to_string().encode("utf-8"))
        context = list(values[-lag:])
        output = []
        for _ in range(horizon):
            prediction = float(fitted.predict(np.asarray(context[-lag:][::-1]).reshape(1, -1))[0])
            output.append(prediction)
            context.append(prediction)
        return np.asarray(output), size, fit_s
    if model == "chronos_bolt":
        import torch
        from chronos import ChronosBoltPipeline
        start = time.perf_counter()
        pipeline = ChronosBoltPipeline.from_pretrained(str(checkpoint), device_map="cpu", local_files_only=True)
        pipeline.model.eval()
        fit_s = time.perf_counter() - start
        size = sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file())
        with torch.inference_mode():
            quantiles = pipeline.predict(torch.tensor(values, dtype=torch.float32), prediction_length=horizon)
        median_index = pipeline.quantiles.index(0.5)
        return quantiles[0, median_index].numpy().astype(float), size, fit_s
    raise ValueError(model)


def measured_call(model, train, test, checkpoint):
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()
    def sample():
        while not stop.wait(0.01):
            peak[0] = max(peak[0], process.memory_info().rss)
    monitor = threading.Thread(target=sample, daemon=True)
    forecast.target_dates = test.index
    cpu_before = process.cpu_times()
    start = time.perf_counter()
    monitor.start()
    try:
        predicted, model_size, fit_s = forecast(model, train, len(test), checkpoint)
    finally:
        stop.set()
        monitor.join()
    elapsed = time.perf_counter() - start
    cpu_after = process.cpu_times()
    peak[0] = max(peak[0], process.memory_info().rss)
    cpu_s = (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)
    if len(predicted) != len(test) or not np.isfinite(predicted).all():
        raise ValueError("Model returned nonfinite predictions or wrong horizon")
    truth = test.to_numpy(dtype=float)
    errors = predicted - truth
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "smape_pct": float(200 * (np.abs(errors) / (np.abs(predicted) + np.abs(truth))).mean()),
        "wall_s": elapsed,
        "fit_or_load_s": fit_s,
        "inference_s": max(0.0, elapsed - fit_s),
        "cpu_s": cpu_s,
        "cpu_pct_one_core": 100 * cpu_s / elapsed if elapsed else 0,
        "peak_rss_bytes": peak[0],
        "incremental_peak_rss_bytes": max(0, peak[0] - baseline),
        "model_bytes": model_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--date-column", default="Date")
    parser.add_argument("--value-column", default="Close")
    parser.add_argument("--series-column")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--origins", type=int, default=3)
    parser.add_argument("--min-train", type=int, default=120)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--chronos-model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()
    selected = [item.strip() for item in args.models.split(",")]
    if any(item not in MODELS for item in selected) or len(set(selected)) != len(selected):
        parser.error(f"--models must be unique names from {MODELS}")
    if min(args.horizon, args.origins, args.min_train) < 1:
        parser.error("horizon, origins, and min-train must be positive")
    data = load_series(args)
    for name, series in data.items():
        if len(series) < args.min_train + args.horizon * args.origins:
            parser.error(f"{name} needs at least {args.min_train + args.horizon * args.origins} rows")
    output = {
        "data_kind": "synthetic_harness_check" if args.synthetic else "historical_csv",
        "source": "deterministic random walk seed 20261112" if args.synthetic else str(args.csv.resolve()),
        "python": sys.version.split()[0], "platform": sys.platform,
        "logical_cpus": os.cpu_count(), "total_ram_bytes": psutil.virtual_memory().total,
        "settings": {"horizon": args.horizon, "origins": args.origins, "min_train": args.min_train, "chronos_checkpoint": str(args.chronos_model) if args.chronos_model else None},
        "series_rows": {name: len(series) for name, series in data.items()}, "models": {},
    }
    for model in selected:
        missing = [dep for dep in REQUIRES[model] if importlib.util.find_spec(dep) is None]
        if missing:
            output["models"][model] = {"status": "unavailable", "reason": "Missing packages: " + ", ".join(missing)}
            continue
        if model == "chronos_bolt" and (args.chronos_model is None or not args.chronos_model.is_dir()):
            output["models"][model] = {"status": "unavailable", "reason": "Local Chronos-Bolt checkpoint directory missing"}
            continue
        rows = []
        try:
            for name, series in data.items():
                for origin in range(args.origins, 0, -1):
                    end = len(series) - origin * args.horizon
                    train = series.iloc[:end]
                    test = series.iloc[end:end + args.horizon]
                    row = measured_call(model, train, test, args.chronos_model)
                    rows.append({"series": name, "train_end": str(train.index[-1].date()), "test_start": str(test.index[0].date()), **row})
            output["models"][model] = {
                "status": "measured", "runs": rows,
                "aggregate": {key: float(np.mean([row[key] for row in rows])) for key in ("mae", "rmse", "smape_pct", "wall_s", "fit_or_load_s", "inference_s", "cpu_s", "cpu_pct_one_core")},
                "max_peak_rss_bytes": max(row["peak_rss_bytes"] for row in rows),
                "max_model_bytes": max(row["model_bytes"] for row in rows),
            }
        except Exception as exc:
            output["models"][model] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "completed_runs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({name: {"status": result["status"], "aggregate": result.get("aggregate"), "reason": result.get("reason")} for name, result in output["models"].items()}, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
