# Phase 2: terminal-close forecast benchmark

This directory is separate from Phase 1. It uses real cached OHLCV and never imports the production app. `REPORT.md` contains the measured results.

Provisional acceptance target: at least 5% out-of-sample MAE improvement over Naive, consistently across assets and horizons. The report records that Phase 2 does not meet it. This target alone does not authorize production deployment.

The large raw archive and extracted CSV caches are local and ignored by Git. Their URLs and SHA-256 hashes are pinned in `prepare_data.py` and `data/manifest.json`.

Run every Python command through `conda run -n qs-forecast`. From the repository root, download the five files named in `REPORT.md` into `experiments/forecast-benchmark/phase2/data/raw/`, then run:

```powershell
conda run -n qs-forecast python experiments/forecast-benchmark/phase2/prepare_data.py
conda run -n qs-forecast python -m unittest discover -s experiments/forecast-benchmark/phase2 -p 'test_*.py' -v
```

Run each model in its own process to compare peak process RSS. The five commands below share the same manifest, horizons, origin spacing, and deterministic seed:

```powershell
conda run -n qs-forecast python experiments/forecast-benchmark/phase2/benchmark.py --manifest experiments/forecast-benchmark/phase2/data/manifest.json --models naive --output experiments/forecast-benchmark/phase2/results-naive.json
conda run -n qs-forecast python experiments/forecast-benchmark/phase2/benchmark.py --manifest experiments/forecast-benchmark/phase2/data/manifest.json --models drift --output experiments/forecast-benchmark/phase2/results-drift.json
conda run -n qs-forecast python experiments/forecast-benchmark/phase2/benchmark.py --manifest experiments/forecast-benchmark/phase2/data/manifest.json --models arima_css --output experiments/forecast-benchmark/phase2/results-arima_css.json
conda run -n qs-forecast python experiments/forecast-benchmark/phase2/benchmark.py --manifest experiments/forecast-benchmark/phase2/data/manifest.json --models lgb_lags --output experiments/forecast-benchmark/phase2/results-lgb_lags.json
conda run -n qs-forecast python experiments/forecast-benchmark/phase2/benchmark.py --manifest experiments/forecast-benchmark/phase2/data/manifest.json --models lgb_technical --output experiments/forecast-benchmark/phase2/results-lgb_technical.json
conda run -n qs-forecast python experiments/forecast-benchmark/phase2/summarize_results.py
```

`summary.csv` has all accuracy and resource metrics for every ticker, horizon, and model. `results.json` preserves every paired forecast origin and target. The runner stops on missing sessions, missing values, nonpositive prices, duplicates, or suspicious close jumps. The reference checks in `prepare_data.py` compare overlapping closes with two other public research caches.
