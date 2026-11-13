# Phase 3 risk forecasting report

## Research question and targets

Can past price, volume and volatility features predict future market risk more reliably than simple statistical baselines? This is a research-only CPU benchmark on real cached OHLCV, separated from Phase 2 terminal-close forecasts. Exact target formulas, origin-close treatment and reproduction commands are in `README.md`. Annualization uses 252 trading sessions and root-mean-square daily log returns, without demeaning. Drawdown is a nonnegative peak-to-trough loss fraction over the origin close plus the next 5 or 20 closes. Higher target values mean more risk.

## Data and rights

The four 2018-01-02 through 2025-12-31 series each contain 2,011 validated sessions. The scored cache was extracted from the pinned public [S&P 500 and ETF archive](https://github.com/Johnbrick123/sp500-data/releases/download/data/prices.parquet), identified by SHA-256 in Phase 2's `data/manifest.json`. Phase 2 reports the archive's Yahoo-derived split-adjusted OHLC, reported volume, calendar audit and independent-cache consistency checks. Dividend-adjusted `adj_close` is excluded because adjustment factors may change after a forecast origin. The public research cache is suitable for local comparison; production licensing and redistribution rights have not been established. Upstream caches may share the same vendor. No synthetic records enter scored experiments.

## Fixed models and protocol

- Volatility baselines: trailing `h` log-return root-mean-square annualized by 252, and EWMA squared returns through the origin with `alpha = 2/(2h+1)`, initialized at the first available squared return.
- Drawdown baseline: maximum drawdown over the trailing `h+1` closes, including the current origin close.
- Direct LightGBM regression: separate fits for each target, horizon, asset and origin. Inputs are close returns over 1/5/10/20 sessions, trailing realized volatility over 5/20, relative volume over 5/20, close-to-moving-average ratios over 5/20/60, RSI(14), and distance below trailing 20/60-session close highs. All windows end by their row date. Prediction is floored at zero. Settings: 100 boosting rounds, learning rate 0.05, 15 leaves, max depth 4, minimum leaf 20, one thread, deterministic seed 20261113. These fixed settings were not tuned on the holdout.

The walk-forward schedule starts after 252 sessions, every 20 sessions. A training row `s` is eligible at origin `t` only when its complete label is observed, `s+h <= t`; no future price or target is used in features. Validation origins are in 2019–2023, excluding target windows crossing into 2024. Holdout origins are in 2024–2025. Later holdout refits may use labels from earlier holdout origins once observable; this mimics a predeclared online evaluation. Models for a target/horizon share exact origins and actual values. The regime breakdown labels an origin high risk if its trailing 63-session realized volatility is at least 20% annualized; otherwise low risk. This threshold is descriptive and fixed in advance.

Per asset, horizon, target and split, `summary.csv` reports MAE, RMSE, percentage MAE improvement over the appropriate historical baseline, a paired 95% bootstrap interval for improvement, total wall and process CPU time, and peak sampled process RSS. The paired bootstrap resamples origin pairs 2,000 times with a fixed seed. Intervals do not account fully for regime dependence or shared assets. Resource measures cover model fit plus prediction, exclude initial data loading and feature preparation, and process RSS includes the interpreter and dependencies.

## Risk-specific research criterion

Further study would be justified if LightGBM improves holdout MAE over the matching historical baseline in at least 3 of 4 assets at **both** horizons for a target, with positive pooled paired-improvement intervals and no material deterioration in the high-risk regime. Because high-risk observations are comparatively few and correlated, this is a screening rule rather than a statistical or deployment guarantee. No Phase 2 terminal-price 5% threshold is imported. Licensing and independent forward validation remain prerequisites to any production consideration.

## Results and failure cases

Measured 2026-11-13 on Windows with the `qs-forecast` Conda environment. Each validation cell has 63 origins for 5 days or 62 for 20 days; each holdout cell has 25 or 24. The following table is the **final holdout**. Errors are decimal fractions: volatility MAE `0.05` is five annualized percentage points, and drawdown MAE `0.01` is one percentage point of loss. Improvement compares LightGBM with the matching trailing-window baseline; brackets are paired 95% bootstrap intervals.

| Asset | Days | Target | Baseline MAE | LightGBM MAE | LightGBM RMSE | MAE improvement [95% interval] |
|---|---:|---|---:|---:|---:|---:|
| SPY | 5 | Volatility | 0.0631 | 0.0456 | 0.0636 | 27.7% [-3.5, 43.6] |
| SPY | 20 | Volatility | 0.0670 | 0.0510 | 0.0791 | 23.9% [3.5, 42.0] |
| QQQ | 5 | Volatility | 0.0845 | 0.0714 | 0.0967 | 15.6% [-11.0, 32.4] |
| QQQ | 20 | Volatility | 0.0738 | 0.0650 | 0.0869 | 11.9% [-20.5, 31.6] |
| IWM | 5 | Volatility | 0.0829 | 0.0542 | 0.0734 | 34.6% [2.4, 58.1] |
| IWM | 20 | Volatility | 0.0742 | 0.0573 | 0.0722 | 22.7% [4.4, 36.1] |
| AAPL | 5 | Volatility | 0.1275 | 0.0819 | 0.0961 | 35.8% [-4.6, 58.9] |
| AAPL | 20 | Volatility | 0.0836 | 0.0698 | 0.1190 | 16.5% [-9.2, 37.1] |
| SPY | 5 | Drawdown | 0.0096 | 0.0086 | 0.0108 | 10.6% [-24.4, 33.1] |
| SPY | 20 | Drawdown | 0.0230 | 0.0234 | 0.0300 | -2.1% [-45.0, 24.0] |
| QQQ | 5 | Drawdown | 0.0141 | 0.0125 | 0.0160 | 10.9% [-20.4, 30.9] |
| QQQ | 20 | Drawdown | 0.0300 | 0.0285 | 0.0359 | 5.1% [-41.8, 33.3] |
| IWM | 5 | Drawdown | 0.0170 | 0.0122 | 0.0166 | 28.4% [4.3, 47.5] |
| IWM | 20 | Drawdown | 0.0310 | 0.0271 | 0.0337 | 12.3% [-28.3, 35.7] |
| AAPL | 5 | Drawdown | 0.0172 | 0.0151 | 0.0178 | 12.4% [-36.4, 38.0] |
| AAPL | 20 | Drawdown | 0.0382 | 0.0294 | 0.0467 | 23.1% [-20.2, 51.4] |

Exploratory validation told a different story for volatility: pooled LightGBM MAE was **9.0% worse** at five days and **5.8% worse** at 20 days. Validation drawdown pooled improvements were 10.5% and 4.4%, with intervals crossing zero. The final holdout pooled volatility improvement was **29.3% [1.9, 46.9]** at five days and **18.6% [2.7, 31.4]** at 20 days. Pooled drawdown improvement was **16.4% [-4.1, 32.1]** and **11.2% [-23.9, 34.1]**. These pooled intervals resample dates, keeping all four asset errors on a date together. Assets share market factors, and 24–25 holdout origins remain a small sample.

The volatility holdout passes the stated screening rule: all four assets improve at both horizons, pooled intervals are positive, and no high-risk cell deteriorates. Yet the validation reversal makes the effect unstable across periods. Drawdown fails the rule: SPY at 20 days worsens, high-risk 5-day QQQ and AAPL worsen, and both pooled intervals include zero. High-risk holdout cells range from only 3 SPY origins to 19 AAPL origins, so those breakdowns are especially uncertain. `summary.csv` records every low- and high-risk cell and both baseline models; `results.json` retains every actual and prediction.

Across the 16 holdout LightGBM cells, fit-plus-prediction wall time per cell was roughly 0.7–1.8 seconds, process CPU time roughly 0.9–2.0 seconds, and peak sampled process RSS 107–113 MiB. These are local process measures, not deployable capacity estimates. The simple EWMA baseline sometimes improves on trailing volatility, especially at five days, so beating the historical baseline alone does not establish superiority over every simple comparator.

The measured result justifies **further independent volatility research**, ideally a later prospective paper evaluation and licensed data. It does not justify production integration. Drawdown evidence remains insufficient. No model settings were changed after inspecting holdout results.

## Reproduce

Run the two commands in `README.md`. The runner fails on missing cache files, checksum mismatch, duplicate or missing sessions, suspicious jumps, incomplete training labels, or nonfinite targets or predictions. Unit tests use small synthetic fixtures only for correctness checks. No production API, deployment configuration or live service is used.
