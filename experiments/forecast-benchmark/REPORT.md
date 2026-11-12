# QuantScreen forecast benchmark: historical Apple closes

Measured on 2026-11-12 in the local `qs-forecast` Conda environment (Python 3.12.16, Windows, CPU only). This is an isolated experiment. Production code, operational repositories, and deployed services were not changed.

## Data and protocol

- Source: Plotly's [finance-charts-apple.csv](https://github.com/plotly/datasets/blob/master/finance-charts-apple.csv), cached at [`data/raw/finance-charts-apple.csv`](data/raw/finance-charts-apple.csv). SHA-256: `24C7604EDFD5AFE862DDB9F9535E2FD7351F43711BFC442BCA52055E15E37BCD`.
- Target: `AAPL.Close`, 506 dated daily observations from 2015-02-17 through 2017-02-16. This fixed, older sample is real market data, but only one stock and one historical regime.
- Validation: 20 expanding-window origins, each forecasting the next seven observed trading sessions without updating on any of those seven closes. Test windows do not overlap. The first test starts 2016-07-29 and the last ends 2017-02-16. There are 140 held-out daily targets. The earliest fit uses 366 observations; the configured minimum is 120. Dates are sorted and duplicates or invalid prices are rejected by the runner.
- Models: Naive repeats the last close. ARIMA(1,1,1) has no drift and is fitted by conditional least squares with SciPy because Windows blocked the compiled statsmodels state-space extension. LightGBM uses ten close lags, 80 trees, and recursive seven-step prediction through LightGBM's native API. No model selection or tuning used the held-out windows.
- Accuracy entries are means of the 20 window-level MAE, RMSE and sMAPE values. Each window has seven observations. Resource entries come from three separate model processes, each processing its 20 windows serially. `results.json` contains the combined accuracy run; `results-naive.json`, `results-arima.json` and `results-lightgbm.json` contain the process-isolated resource runs.

## Measured results

| Model | Mean MAE (USD) | Mean RMSE (USD) | Mean sMAPE | Mean wall time / origin | Mean process CPU time / origin | Peak process RSS | Maximum per-call RSS increase | Maximum model bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive | 1.9944 | 2.2701 | 1.7401% | 0.428 ms | 0.781 ms | 70.94 MiB | 0.02 MiB | 0 |
| ARIMA(1,1,1), conditional least squares | 1.9961 | 2.2873 | 1.7436% | 52.021 ms | 50.781 ms | 124.00 MiB | 53.08 MiB | 176 |
| LightGBM | 2.5410 | 2.8459 | 2.2134% | 22.838 ms | 32.031 ms | 86.30 MiB | 15.26 MiB | 61,987 |

Naive has the lowest error on this sample. ARIMA is effectively tied on MAE (0.0016 USD higher), while LightGBM's MAE is about 27% higher than Naive. A single stock and historical interval cannot establish general market performance or justify a production deployment decision.

Resource figures are process level and include Python, imported libraries, and model objects. RSS was sampled every 10 ms, so brief peaks may be missed. Wall time covers the `forecast` call, including lazy imports and serialization; the runner's `inference_s` field also includes overhead outside fitting and is not a pure prediction latency. Process CPU time has coarse resolution, especially for Naive, and can exceed wall time when native code uses multiple threads. CPU percentage from very short calls is therefore not useful for sizing. Model bytes are a serialized parameter tuple for ARIMA and a LightGBM text model; they exclude the runtime and dependencies. All measurements are local and should be repeated on the intended host under realistic concurrency.

## Reproduce

From the repository root:

```powershell
conda run -n qs-forecast python experiments/forecast-benchmark/benchmark.py --csv experiments/forecast-benchmark/data/raw/finance-charts-apple.csv --date-column Date --value-column AAPL.Close --horizon 7 --origins 20 --min-train 120 --models naive,arima,lightgbm --output experiments/forecast-benchmark/results.json
```

To reproduce the resource table, rerun the same command once per model, setting `--models naive`, `--models arima`, or `--models lightgbm`, and writing to the corresponding `results-<model>.json` file. The exact cached CSV is identified by the SHA-256 above.
