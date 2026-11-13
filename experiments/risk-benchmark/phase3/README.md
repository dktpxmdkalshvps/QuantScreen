# Phase 3: forward risk benchmark

This isolated research experiment scores forward 5-session and 20-session risk on the four real, cached SPY, QQQ, IWM and AAPL OHLCV series verified by Phase 2. It does not import production application code or score synthetic data. `benchmark.py` imports Phase 2's data validator and checks each extracted CSV against its manifest hash and session audit. Phase 2's terminal-close targets and results are separate.

From the repository root, with the already available `qs-forecast` Conda environment:

```powershell
conda run -n qs-forecast python -m unittest discover -s experiments/risk-benchmark/phase3 -p test_benchmark.py -v
conda run -n qs-forecast python experiments/risk-benchmark/phase3/benchmark.py
```

The input is `experiments/forecast-benchmark/phase2/data/manifest.json` and its pinned extracted CSV files. If absent, follow Phase 2's `README.md` and `prepare_data.py` to reconstruct and verify them from the pinned public archive. `requirements.txt` pins the direct packages used in this local Windows environment; the existing Conda environment is the execution environment. `results.json` stores every scored origin and prediction, and `summary.csv` stores per-asset, per-horizon, split, model and regime metrics.

Both targets use the close at an origin `t` and exactly the next `h` observed sessions. Volatility is `sqrt(252 × mean(r[t+1]^2, ..., r[t+h]^2))`, where `r[s] = log(close[s]/close[s-1])`. It is an annualized decimal, with a zero-mean root-mean-square convention. Maximum drawdown is `max(1 - close[j] / max(close[t], ..., close[j]))` over `j=t+1,...,t+h`, a positive decimal loss. Thus the origin close participates in the initial peak. The same target array is shared by all comparable models.

Forecasts use the information available after the origin close. A row `s` enters model training only if `s+h <= t`. Validation origins run from 2019 through 2023, with any target extending into 2024 excluded. Final holdout origins are in 2024–2025; earlier holdout labels can become available for later predeclared walk-forward refits. Hyperparameters and acceptance criteria were fixed before running the holdout. Origins are spaced by 20 sessions, so same-horizon target windows do not overlap. Results use paired errors on identical origins. Paired bootstrap intervals resample origin pairs 2,000 times and are descriptive, since regimes and assets are dependent.

Read `REPORT.md` for settings, provenance, measured results and interpretation.
