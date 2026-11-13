"""Research-only walk-forward forecasts of forward volatility and drawdown."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parent
PHASE2 = ROOT.parents[1] / "forecast-benchmark" / "phase2"
MANIFEST = PHASE2 / "data" / "manifest.json"
TICKERS = ("SPY", "QQQ", "IWM", "AAPL")
HORIZONS = (5, 20)
TARGETS = ("volatility", "drawdown")
SEED = 20261113
ANNUAL_DAYS = 252
FEATURES = ("return_1", "return_5", "return_10", "return_20", "rv_5", "rv_20",
            "relative_volume_5", "relative_volume_20", "ma_ratio_5", "ma_ratio_20",
            "ma_ratio_60", "rsi_14", "drawdown_20", "drawdown_60")


def phase2_module():
    spec = importlib.util.spec_from_file_location("phase2_data_validation", PHASE2 / "benchmark.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_verified(ticker: str, manifest_path: Path = MANIFEST) -> tuple[pd.DataFrame, dict]:
    phase2 = phase2_module()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["assets"][ticker]
    path = manifest_path.parent / entry["path"]
    if phase2.sha256(path) != entry["sha256"]:
        raise ValueError(f"{ticker}: extracted OHLCV hash differs from Phase 2 manifest")
    frame, audit = phase2.load_asset(path, ticker, manifest["start"], manifest["end"])
    if audit["missing_sessions"] or audit["unexpected_sessions"] or audit["close_jumps_over_40pct"]:
        raise ValueError(f"{ticker}: invalid session audit: {audit}")
    if {key: audit[key] for key in audit} != {key: entry["audit"][key] for key in audit}:
        raise ValueError(f"{ticker}: audit differs from Phase 2 manifest")
    return frame, audit


def realized_volatility(log_returns: np.ndarray) -> float:
    """Root mean square of h daily log returns, annualized to 252 sessions."""
    x = np.asarray(log_returns, dtype=float)
    if len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("invalid return window")
    return float(np.sqrt(ANNUAL_DAYS * np.mean(x * x)))


def maximum_drawdown(prices: np.ndarray) -> float:
    """Positive worst peak-to-trough loss; prices[0] is the origin close."""
    p = np.asarray(prices, dtype=float)
    if len(p) < 2 or not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("invalid close window")
    return float(np.max(1 - p / np.maximum.accumulate(p)))


def targets(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    close = data["close"].to_numpy(dtype=float)
    logret = np.diff(np.log(close))
    vol = np.full(len(close), np.nan)
    dd = np.full(len(close), np.nan)
    for s in range(len(close) - horizon):
        vol[s] = realized_volatility(logret[s:s + horizon])
        dd[s] = maximum_drawdown(close[s:s + horizon + 1])
    return pd.DataFrame({"volatility": vol, "drawdown": dd}, index=data.index)


def features(data: pd.DataFrame) -> pd.DataFrame:
    close, volume = data["close"], data["volume"]
    logret = np.log(close).diff()
    out = pd.DataFrame(index=data.index)
    for lag in (1, 5, 10, 20):
        out[f"return_{lag}"] = close.div(close.shift(lag)) - 1
    for window in (5, 20):
        out[f"rv_{window}"] = np.sqrt(ANNUAL_DAYS * logret.pow(2).rolling(window).mean())
        out[f"relative_volume_{window}"] = volume.div(volume.rolling(window).mean())
    for window in (5, 20, 60):
        out[f"ma_ratio_{window}"] = close.div(close.rolling(window).mean()) - 1
    change = close.diff()
    gain = change.clip(lower=0).rolling(14).mean()
    loss = -change.clip(upper=0).rolling(14).mean()
    out["rsi_14"] = (100 - 100 / (1 + gain.div(loss.replace(0, np.nan)))).mask(
        (loss == 0) & (gain > 0), 100).mask((loss == 0) & (gain == 0), 50)
    for window in (20, 60):
        out[f"drawdown_{window}"] = 1 - close.div(close.rolling(window).max())
    return out


def training_examples(x: pd.DataFrame, y: pd.Series, horizon: int, origin: int):
    # Label for row s uses observations through s+h and is known at t iff s+h <= t.
    last = origin - horizon
    rows = x.iloc[:last + 1].loc[:, FEATURES]
    labels = y.iloc[:last + 1]
    valid = rows.notna().all(axis=1) & np.isfinite(labels)
    if valid.sum() < 100:
        raise ValueError("insufficient fully observed training labels")
    return rows.loc[valid].to_numpy(dtype=float), labels.loc[valid].to_numpy(dtype=float)


def predict(model: str, target: str, data: pd.DataFrame, x: pd.DataFrame,
            y: pd.DataFrame, horizon: int, origin: int) -> float:
    close = data["close"].to_numpy(dtype=float)
    if model == "historical_volatility":
        return realized_volatility(np.diff(np.log(close[origin - horizon:origin + 1])))
    if model == "ewma_volatility":
        returns = np.diff(np.log(close[:origin + 1]))
        alpha = 2 / (2 * horizon + 1)
        variance = returns[0] ** 2
        for value in returns[1:]:
            variance = alpha * value ** 2 + (1 - alpha) * variance
        return float(np.sqrt(ANNUAL_DAYS * variance))
    if model == "historical_drawdown":
        return maximum_drawdown(close[origin - horizon:origin + 1])
    if model != "lightgbm":
        raise ValueError(model)
    import lightgbm as lgb
    train_x, train_y = training_examples(x, y[target], horizon, origin)
    observation = x.iloc[origin].loc[list(FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(observation).all():
        raise ValueError("incomplete origin features")
    booster = lgb.train(
        {"objective": "regression", "learning_rate": 0.05, "num_leaves": 15,
         "max_depth": 4, "min_data_in_leaf": 20, "verbosity": -1,
         "num_threads": 1, "seed": SEED, "feature_fraction_seed": SEED,
         "data_random_seed": SEED, "deterministic": True},
        lgb.Dataset(train_x, label=train_y, feature_name=list(FEATURES)),
        num_boost_round=100,
    )
    return max(0.0, float(booster.predict(observation[None, :])[0]))


def measured_prediction(*args) -> tuple[float, float, float, int]:
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.01):
            peak[0] = max(peak[0], process.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    cpu0 = process.cpu_times()
    wall0 = time.perf_counter()
    monitor.start()
    try:
        value = predict(*args)
    finally:
        stop.set()
        monitor.join()
    cpu1 = process.cpu_times()
    return value, time.perf_counter() - wall0, cpu1.user + cpu1.system - cpu0.user - cpu0.system, max(peak[0], process.memory_info().rss)


def origins_for(data: pd.DataFrame, horizon: int) -> list[int]:
    first = max(251, int(data.index.searchsorted(pd.Timestamp("2019-01-01"))))
    return list(range(first, len(data) - horizon, 20))


def paired_interval(baseline_errors: np.ndarray, model_errors: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(model_errors)
    draws = rng.integers(0, n, size=(2000, n))
    b = baseline_errors[draws].mean(axis=1)
    m = model_errors[draws].mean(axis=1)
    return [float(v) for v in np.quantile(100 * (1 - m / b), [0.025, 0.975])]


def evaluate(manifest_path: Path = MANIFEST) -> tuple[dict, list[dict]]:
    result = {"phase": 3, "manifest": str(manifest_path), "settings": {
        "horizons": HORIZONS, "origin_step_sessions": 20, "min_history_sessions": 252,
        "validation_origin_dates": "2019-01-01 through 2023-12-31",
        "holdout_origin_dates": "2024-01-01 through 2025-12-31",
        "seed": SEED, "annualization_sessions": ANNUAL_DAYS,
        "regime": "high if trailing 63-session RMS annualized volatility >= 0.20"}, "data_audit": {}, "results": {}}
    summary = []
    for ticker in TICKERS:
        data, audit = load_verified(ticker, manifest_path)
        result["data_audit"][ticker] = audit
        x = features(data)
        result["results"][ticker] = {}
        close = data["close"].to_numpy(dtype=float)
        returns = np.diff(np.log(close))
        for horizon in HORIZONS:
            y = targets(data, horizon)
            result["results"][ticker][str(horizon)] = {}
            for target in TARGETS:
                models = (("historical_volatility", "ewma_volatility", "lightgbm")
                          if target == "volatility" else ("historical_drawdown", "lightgbm"))
                baseline = models[0]
                runs = {m: {"validation": [], "holdout": []} for m in models}
                for origin in origins_for(data, horizon):
                    date = str(data.index[origin].date())
                    split = "validation" if date < "2024-01-01" else "holdout"
                    if split == "validation" and data.index[origin + horizon] >= pd.Timestamp("2024-01-01"):
                        continue  # Validation labels cannot consume the untouched holdout period.
                    regime_vol = realized_volatility(returns[origin - 63:origin])
                    regime = "high" if regime_vol >= 0.20 else "low"
                    actual = float(y[target].iloc[origin])
                    for model in models:
                        value, wall, cpu, rss = measured_prediction(model, target, data, x, y, horizon, origin)
                        if not math.isfinite(value) or not math.isfinite(actual):
                            raise ValueError("nonfinite prediction or target")
                        runs[model][split].append({"origin": date,
                            "target_end": str(data.index[origin + horizon].date()),
                            "label_available_through": str(data.index[origin].date()),
                            "regime": regime, "actual": actual, "prediction": value,
                            "wall_s": wall, "cpu_s": cpu, "peak_rss_bytes": rss})
                for split in ("validation", "holdout"):
                    reference = runs[baseline][split]
                    base_error = np.array([abs(r["prediction"] - r["actual"]) for r in reference])
                    for model in models:
                        observed = runs[model][split]
                        if [(r["origin"], r["target_end"], r["actual"]) for r in observed] != [
                            (r["origin"], r["target_end"], r["actual"]) for r in reference]:
                            raise ValueError("unpaired origins or targets")
                        errors = np.array([abs(r["prediction"] - r["actual"]) for r in observed])
                        improvement = None if model == baseline else float(100 * (1 - errors.mean() / base_error.mean()))
                        interval = None if model == baseline else paired_interval(base_error, errors, SEED)
                        for regime in ("all", "low", "high"):
                            indices = [i for i, r in enumerate(observed) if regime == "all" or r["regime"] == regime]
                            if not indices:
                                continue
                            e = errors[indices]
                            be = base_error[indices]
                            row = {"ticker": ticker, "horizon": horizon, "target": target,
                                   "split": split, "regime": regime, "model": model, "origins": len(indices),
                                   "mae": float(e.mean()), "rmse": float(np.sqrt(np.mean(e * e))),
                                   "mae_improvement_pct": improvement if regime == "all" else (
                                       None if model == baseline else float(100 * (1 - e.mean() / be.mean()))),
                                   "paired_ci_low_pct": interval[0] if interval and regime == "all" else None,
                                   "paired_ci_high_pct": interval[1] if interval and regime == "all" else None,
                                   "total_wall_s": float(sum(observed[i]["wall_s"] for i in indices)),
                                   "total_cpu_s": float(sum(observed[i]["cpu_s"] for i in indices)),
                                   "peak_rss_mib": max(observed[i]["peak_rss_bytes"] for i in indices) / 1048576}
                            summary.append(row)
                result["results"][ticker][str(horizon)][target] = runs
    result["pooled_comparison"] = pooled_comparison(result)
    return result, summary


def pooled_comparison(result: dict) -> dict:
    """Resample common origin dates, keeping four asset errors together."""
    output = {}
    for split in ("validation", "holdout"):
        output[split] = {}
        for horizon in HORIZONS:
            output[split][str(horizon)] = {}
            for target in TARGETS:
                baseline = "historical_volatility" if target == "volatility" else "historical_drawdown"
                paired = []
                for ticker in TICKERS:
                    group = result["results"][ticker][str(horizon)][target]
                    b, m = group[baseline][split], group["lightgbm"][split]
                    if [(v["origin"], v["actual"]) for v in b] != [(v["origin"], v["actual"]) for v in m]:
                        raise ValueError("pooled model origin mismatch")
                    paired.append((b, m))
                dates = [v["origin"] for v in paired[0][0]]
                if any([v["origin"] for v in b] != dates for b, _ in paired):
                    raise ValueError("pooled asset origin mismatch")
                base_error = np.array([[abs(v["prediction"] - v["actual"]) for v in b]
                                       for b, _ in paired]).mean(axis=0)
                model_error = np.array([[abs(v["prediction"] - v["actual"]) for v in m]
                                        for _, m in paired]).mean(axis=0)
                output[split][str(horizon)][target] = {
                    "origin_dates": dates, "asset_count": len(TICKERS),
                    "baseline_mae": float(base_error.mean()),
                    "lightgbm_mae": float(model_error.mean()),
                    "improvement_pct": float(100 * (1 - model_error.mean() / base_error.mean())),
                    "paired_date_bootstrap_95pct": paired_interval(base_error, model_error, SEED)}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=ROOT / "results.json")
    args = parser.parse_args()
    result, summary = evaluate(args.manifest)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    with (args.output.parent / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Scored {len(summary)} summary rows; wrote {args.output}")


if __name__ == "__main__":
    main()
