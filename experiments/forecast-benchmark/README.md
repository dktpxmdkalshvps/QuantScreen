# Forecast benchmark

This is an isolated experiment. It does not import the production forecast route: that route generates formula based sample values rather than fitted forecasts.

## Run

Use Python 3.11 or 3.12 where all model packages have compatible wheels. In the `qs-forecast` Conda environment, the cached historical Apple close benchmark is:

```powershell
conda run -n qs-forecast python experiments/forecast-benchmark/benchmark.py --csv experiments/forecast-benchmark/data/raw/finance-charts-apple.csv --date-column Date --value-column AAPL.Close --horizon 7 --origins 20 --min-train 120 --models naive,arima,lightgbm --output experiments/forecast-benchmark/results.json
```

For another dataset, the CSV must contain dated historical closes. `--series-column` is optional for one series. The default command uses three expanding window origins and a seven observation horizon. Dates are sorted and duplicate dates rejected. No future values enter fitting or lag features. Adjust with `--horizon`, `--origins`, and `--min-train`; each series needs at least `min_train + horizon * origins` observations.

To verify the harness without market data, run `conda run -n qs-forecast python experiments/forecast-benchmark/benchmark.py --synthetic --output experiments/forecast-benchmark/synthetic-results.json`. Synthetic metrics test the code path only and must not be interpreted as stock forecast accuracy.

Chronos-Bolt uses `amazon/chronos-bolt-tiny` by default. Supply a **local** checkpoint directory with `--chronos-model PATH`; the runner uses local files only and never downloads weights. The output records missing dependencies or weights as unavailable. `--models` accepts comma separated names: `naive,arima,prophet,lightgbm,chronos_bolt`.

ARIMA is an ARIMA(1,1,1) without drift, fitted by conditional least squares with SciPy. This avoids the compiled statsmodels state-space extension, which Windows blocked in the measured environment. LightGBM uses the native training API and does not require scikit-learn.

Outputs include per series and origin error, aggregate MAE/RMSE/sMAPE, wall time for fitting/loading and prediction, peak process RSS, process CPU time and CPU percent of one logical core, and serialized model bytes where measurable. Resource measurements are process level, sampled every 10 ms, and include imports; they are indicative rather than isolated deployment benchmarks. Run each model in a separate process to compare peak RSS. Model size excludes shared Python package and runtime size. See `REPORT.md` for the measured historical-data results and limitations.
