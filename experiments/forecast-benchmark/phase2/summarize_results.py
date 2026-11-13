"""Merge process-isolated Phase 2 runs and verify all paired comparisons."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from benchmark import HORIZONS, MODELS, TICKERS


ROOT = Path(__file__).resolve().parent


def main() -> None:
    files = {model: json.loads((ROOT / f"results-{model}.json").read_text(encoding="utf-8"))
             for model in MODELS}
    settings = files["naive"]["settings"]
    audits = files["naive"]["data_audit"]
    output = {"phase": 2, "manifest": "data/manifest.json", "settings": settings,
              "data_audit": audits, "results": {}}
    summary = []
    regime_cells = {name: [] for name in ("2019", "2020-2021", "2022", "2023-2025")}
    for ticker in TICKERS:
        output["results"][ticker] = {}
        for horizon in HORIZONS:
            key = str(horizon)
            baseline = files["naive"]["results"][ticker][key]["naive"]["runs"]
            expected = [(row["origin"], row["target"], row["actual"]) for row in baseline]
            group = {}
            for model in MODELS:
                run = files[model]
                if run["settings"] != settings or run["data_audit"] != audits:
                    raise ValueError(f"{model}: settings or data audit differ")
                measured = run["results"][ticker][key][model]
                actual = [(row["origin"], row["target"], row["actual"]) for row in measured["runs"]]
                if actual != expected:
                    raise ValueError(f"{ticker} h={horizon} {model}: unpaired origins or targets")
                group[model] = measured
                aggregate = measured["aggregate"]
                ci = aggregate.get("paired_bootstrap_improvement_95pct", [None, None])
                summary.append({
                    "ticker": ticker, "horizon": horizon, "model": model,
                    "origins": aggregate["count"], "mae": aggregate["mae"],
                    "rmse": aggregate["rmse"], "smape_pct": aggregate["smape_pct"],
                    "mae_improvement_pct": aggregate.get("mae_improvement_pct"),
                    "win_rate_pct": aggregate.get("win_rate_pct"),
                    "improvement_ci_low": ci[0], "improvement_ci_high": ci[1],
                    "mean_wall_ms": aggregate["mean_wall_s"] * 1000,
                    "mean_cpu_ms": aggregate["mean_cpu_s"] * 1000,
                    "total_wall_s": aggregate["total_wall_s"],
                    "total_cpu_s": aggregate["total_cpu_s"],
                    "peak_rss_mib": aggregate["peak_rss_bytes"] / (1024 ** 2),
                })
            output["results"][ticker][key] = group
            for regime, years in (("2019", {2019}), ("2020-2021", {2020, 2021}),
                                  ("2022", {2022}), ("2023-2025", {2023, 2024, 2025})):
                indices = [i for i, row in enumerate(baseline) if int(row["target"][:4]) in years]
                if not indices:
                    raise ValueError(f"{ticker} h={horizon}: no origins in {regime}")
                def errors(model):
                    runs = group[model]["runs"]
                    return np.array([abs(runs[i]["prediction"] - runs[i]["actual"]) for i in indices])
                naive_mae = float(errors("naive").mean())
                regime_cells[regime].append({
                    "origins": len(indices),
                    "technical_improvement_pct": 100 * (1 - errors("lgb_technical").mean() / naive_mae),
                    "technical_beats_naive": errors("lgb_technical").mean() < naive_mae,
                    "technical_beats_drift": errors("lgb_technical").mean() < errors("drift").mean(),
                })
    (ROOT / "results.json").write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    with (ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (ROOT / "regime_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["regime", "cells", "min_origins_per_cell",
                                                   "max_origins_per_cell", "technical_wins_vs_naive_cells",
                                                   "technical_wins_vs_drift_cells",
                                                   "median_technical_improvement_pct"])
        writer.writeheader()
        for regime, cells in regime_cells.items():
            writer.writerow({"regime": regime, "cells": len(cells),
                             "min_origins_per_cell": min(cell["origins"] for cell in cells),
                             "max_origins_per_cell": max(cell["origins"] for cell in cells),
                             "technical_wins_vs_naive_cells": sum(cell["technical_beats_naive"] for cell in cells),
                             "technical_wins_vs_drift_cells": sum(cell["technical_beats_drift"] for cell in cells),
                             "median_technical_improvement_pct": float(np.median(
                                 [cell["technical_improvement_pct"] for cell in cells]))})
    print(f"Verified {len(summary)} ticker/horizon/model cells and paired origins; wrote summary.csv")


if __name__ == "__main__":
    main()
