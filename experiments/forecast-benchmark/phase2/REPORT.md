# QuantScreen forecast benchmark — Phase 2

Measured 2026-11-12 with `qs-forecast` (Python 3.12.16, Windows, CPU only). Phase 1 files are unchanged. This is a local research result; no production code, commit, push, or deployment was made.

## Dataset inventory and provenance

The scored data came from the [public S&P 500 and ETF price archive](https://github.com/Johnbrick123/sp500-data). Its release invites direct local queries and labels the 2018–2025 rows for all four assets as Yahoo sourced. The cached archive is [`data/raw/public_prices.parquet`](data/raw/public_prices.parquet), downloaded 2026-11-12 from [this release URL](https://github.com/Johnbrick123/sp500-data/releases/download/data/prices.parquet), SHA-256 `fb846e89dc6cd128411f145bd4f79900c20334a6dd504f33d776cd87d46f560b`. The raw archive contains 12,302,636 rows across many instruments; only the four requested tickers and target dates were extracted. `data/manifest.json` records the hashes of the four extracted CSV files.

| Ticker | Sessions | First–last | Extracted SHA-256 |
|---|---:|---|---|
| SPY | 2,011 | 2018-01-02–2025-12-31 | `17d63e3ef29570acbfd76419976c87523db9bba0c6f8d3bca058295f291a7baf` |
| QQQ | 2,011 | 2018-01-02–2025-12-31 | `99376b3da68a011f6e566f8c54109fdf02bc45a4bad05496e6e4a03823512186` |
| IWM | 2,011 | 2018-01-02–2025-12-31 | `48bf6fdaf4c28d41e42eb78281fef8ff58a84cb6bd9c26efd836e02b77b67169` |
| AAPL | 2,011 | 2018-01-02–2025-12-31 | `0b1428d84b507eda6e05e9bece4e89179dd1c0b83e106d3c97452d09f75bd819` |

The scored `open`, `high`, `low`, and `close` are split-adjusted as the archive stores them. `volume` is the archive's reported volume. The archive's dividend-adjusted `adj_close` is excluded, avoiding dividend factors revised after a forecast origin. The AAPL 4:1 split is recorded on 2020-08-31; the close moves from 124.8075 to 129.0400 across the event, confirming no raw 4:1 jump in the scored series. All 8,044 selected rows have unique dates, finite positive OHLCV, internally consistent bars, and match the expected NYSE calendar including the 2018 and 2025 national days of mourning. There are no missing or unexpected sessions, close jumps over 40%, or daily `adj_close/close` ratio jumps over 5%. The [NYSE holiday calendar](https://www.nyse.com/trade/hours-calendars), [2018 closure notice](https://www.nyse.com/publicdocs/nyse/markets/arca-options/rule-interpretations/2018/NYSE%20Arca%20Options%2018-06.pdf), and [2025 closure notice](https://www.nyse.com/publicdocs/nyse/markets/arca-options/rule-interpretations/2025/National_Day_of_Mourning_20250102.pdf) informed the session check.

Additional raw caches were downloaded 2026-11-12 as consistency checks; none supplies the scored targets. The three ETF files came from the [research release](https://github.com/lambdaclass/options_portfolio_backtester/releases/tag/data-v1), whose [data notice](https://github.com/lambdaclass/options_portfolio_backtester/blob/master/data/DATA_NOTICE.md) describes research reuse. AAPL came from [Pubmarks' public CSV](https://pubmarks.github.io/datasets/stocks/AAPL/ohlcv.csv), whose [repository](https://github.com/Pubmarks/datasets) documents direct file use.

| Reference raw file | Source URL | SHA-256 | Matching close overlap |
|---|---|---|---:|
| `SPY_underlying.parquet` | [SPY release](https://github.com/lambdaclass/options_portfolio_backtester/releases/download/data-v1/SPY_underlying.parquet) | `847e60a441eb10969d87cd4a6da604257b782d9076168f68d5730b84096c79db` | 1,999 rows; max difference 0% |
| `QQQ_underlying.parquet` | [QQQ release](https://github.com/lambdaclass/options_portfolio_backtester/releases/download/data-v1/QQQ_underlying.parquet) | `78c015fd7fa9d493cd798d5ae81b3c2ae6dc3b8176e25e7bb90cd4114b103351` | 2,000 rows; max difference 0% |
| `IWM_underlying.parquet` | [IWM release](https://github.com/lambdaclass/options_portfolio_backtester/releases/download/data-v1/IWM_underlying.parquet) | `e980f50473a8c346ba9ac8b05dcf3da691581dae742f649cb3608d0efe2ef940` | 2,000 rows; max difference 0% |
| `AAPL_pubmarks_ohlcv.csv` | [AAPL CSV](https://pubmarks.github.io/datasets/stocks/AAPL/ohlcv.csv) | `f7782bd98b9ef5088352d125106049b5ecc00c7194681414ebac4f6798b4fa4e` | 2,011 rows; max difference 0.0014% |

The reference `close` columns agree with the scored split-adjusted closes in the overlapping dates. Their reported OHLCV and optional adjusted-close fields were not combined with the scored archive. The ETF references end 2025-12-12/15, so they were rejected as primary data. This agreement is a consistency check, not an independent exchange audit: the archives may share upstream vendors. Direct Yahoo downloads were excluded because [Yahoo's terms restrict automated collection without prior permission](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html). The scored public research release supports this local reproducible comparison; production data rights and redistribution rights remain unverified.

## Experimental methodology

Each prediction is the **terminal close** exactly 1, 5, 7, or 20 observed trading sessions after an origin, not a multi-step path. An origin uses the close, high and volume available after that session's close. Training windows expand from the beginning of 2018. The earliest origin is 2019-01-02 with 252 prior/current sessions. Origins are spaced every 40 sessions, giving 44 paired origins for every ticker and horizon; the last origin is 2025-11-04. The last targets range from 2025-11-05 (one session) to 2025-12-03 (20 sessions). No target window overlaps another at the same horizon.

Naive repeats the last close. Drift extrapolates the average historical close change per session. `arima_css` is explicitly a **custom conditional least-squares approximation** to driftless ARIMA(1,1,1), not the statsmodels likelihood fit: Windows blocked its compiled state-space extension. Both LightGBM variants use the native API with fixed seed `20261112`, 100 trees, learning rate 0.05, maximum depth 4, 15 leaves, and one training thread. They fit a direct horizon-specific log-return target and convert the prediction back to a terminal close. `lgb_lags` uses lagged returns over 1, 5, 10 and 20 sessions. `lgb_technical` adds close/MA ratios over 5, 10, 20 and 60 sessions, return volatility over 5 and 20, relative volume over 5 and 20, RSI(14), and distance from the trailing 20-session high. All rolling calculations end at or before the row's timestamp. A training example at session `s` enters a fit at origin `t` only if its target at `s+h` is known by `t`. No fitted preprocessing or hyperparameter tuning used evaluation targets.

MAE, RMSE and sMAPE are computed on terminal closes. Improvement is `100 × (1 − model MAE / Naive MAE)` on the same 44 origins. Win rate is the fraction of origins with strictly lower absolute error than Naive. The 95% interval resamples paired origin errors 2,000 times with a fixed seed. Intervals are descriptive because training histories overlap and market regimes cluster.

## Accuracy

Mean terminal-close MAE in USD (44 paired origins per cell):

| Ticker | Horizon | Naive | Drift | ARIMA CSS | LightGBM lags | LightGBM technical |
|---|---:|---:|---:|---:|---:|---:|
| SPY | 1 | 3.060 | 3.033 | 3.106 | 3.303 | 3.334 |
| SPY | 5 | 6.605 | 6.487 | 6.701 | 6.320 | 6.589 |
| SPY | 7 | 7.528 | 7.511 | 7.528 | 7.979 | 7.474 |
| SPY | 20 | 16.015 | 15.140 | 16.054 | 15.464 | 14.912 |
| QQQ | 1 | 2.889 | 2.850 | 2.947 | 2.804 | 3.151 |
| QQQ | 5 | 6.736 | 6.631 | 6.734 | 6.759 | 7.366 |
| QQQ | 7 | 7.582 | 7.449 | 7.544 | 7.610 | 7.369 |
| QQQ | 20 | 16.439 | 14.974 | 16.448 | 15.722 | 15.000 |
| IWM | 1 | 2.237 | 2.227 | 2.200 | 2.283 | 2.314 |
| IWM | 5 | 4.785 | 4.773 | 4.770 | 4.962 | 4.919 |
| IWM | 7 | 6.166 | 6.149 | 6.157 | 6.321 | 6.353 |
| IWM | 20 | 9.610 | 9.552 | 9.606 | 9.706 | 10.338 |
| AAPL | 1 | 1.792 | 1.765 | 1.786 | 1.851 | 1.700 |
| AAPL | 5 | 3.707 | 3.682 | 3.714 | 3.837 | 3.733 |
| AAPL | 7 | 4.700 | 4.639 | 4.713 | 4.618 | 4.978 |
| AAPL | 20 | 10.713 | 10.233 | 10.715 | 10.019 | 10.992 |

Technical LightGBM's paired comparison against Naive (improvement, 95% bootstrap interval, per-origin win rate):

| Ticker | Horizon | MAE improvement | 95% interval | Win rate |
|---|---:|---:|---:|---:|
| SPY | 1 | -9.0% | [-24.4%, 2.7%] | 47.7% |
| SPY | 5 | 0.2% | [-24.8%, 21.2%] | 63.6% |
| SPY | 7 | 0.7% | [-18.4%, 18.0%] | 54.5% |
| SPY | 20 | 6.9% | [-7.5%, 21.5%] | 56.8% |
| QQQ | 1 | -9.0% | [-27.3%, 6.2%] | 36.4% |
| QQQ | 5 | -9.4% | [-27.7%, 5.3%] | 43.2% |
| QQQ | 7 | 2.8% | [-12.8%, 18.2%] | 59.1% |
| QQQ | 20 | 8.8% | [-7.3%, 22.9%] | 63.6% |
| IWM | 1 | -3.4% | [-11.1%, 4.2%] | 54.5% |
| IWM | 5 | -2.8% | [-11.9%, 5.5%] | 56.8% |
| IWM | 7 | -3.0% | [-13.8%, 8.8%] | 47.7% |
| IWM | 20 | -7.6% | [-24.2%, 6.2%] | 50.0% |
| AAPL | 1 | 5.2% | [-4.0%, 14.8%] | 54.5% |
| AAPL | 5 | -0.7% | [-15.2%, 13.5%] | 52.3% |
| AAPL | 7 | -5.9% | [-24.1%, 9.5%] | 50.0% |
| AAPL | 20 | -2.6% | [-23.3%, 14.0%] | 56.8% |

Technical LightGBM beats Naive on MAE in 6/16 cells and Drift in 4/16. Every technical-model improvement interval includes zero. Drift has the lowest or near-lowest MAE in many cells and beats Naive on MAE in all 16 cells, although this does not establish a reliable population advantage. [`summary.csv`](summary.csv) contains RMSE, sMAPE, improvement, win rate and interval for all 80 model cells; [`results.json`](results.json) holds every paired origin and prediction.

### Provisional acceptance criterion

The target for a future feature-engineered model is **at least 5% out-of-sample MAE improvement over Naive, with consistent results across assets and horizons**. Technical LightGBM reaches 5% in only **3/16** cells: SPY at 20 sessions (6.9%), QQQ at 20 sessions (8.8%), and AAPL at one session (5.2%). It reaches the target in no IWM cell, no horizon across all four assets, and none of the three qualifying cells has a paired 95% interval excluding zero. Phase 2 therefore **fails the provisional criterion**. Meeting the numeric threshold in a later experiment would still require leakage checks, an untouched holdout, data-use rights, and deployment validation before production approval.

## Market-regime breakdown

The following is descriptive: each of 16 ticker–horizon cells is scored within the target-date periods, with 6–19 origins per cell. These periods were chosen for interpretability and were not used to tune models.

| Target dates | Origins per cell | Technical beats Naive cells | Technical beats Drift cells | Median technical MAE improvement vs Naive |
|---|---:|---:|---:|---:|
| 2019 | 6–7 | 9/16 | 9/16 | 10.0% |
| 2020–2021 | 12–13 | 8/16 | 7/16 | -2.7% |
| 2022 | 6–7 | 6/16 | 6/16 | -12.4% |
| 2023–2025 | 18–19 | 12/16 | 8/16 | 2.6% |

The [regime CSV](regime_summary.csv) retains the exact counts. The 2022 result weakens the claim that technical features improve robustness across market regimes.

## Resource consumption

Each model ran in a separate Python process. The table averages the 16 ticker–horizon mean times (44 forecast calls per cell) and gives the highest process RSS seen in that model's run. These are local process measurements, not deployment capacity guarantees.

| Model | Mean wall time per origin | Mean CPU time per origin | Peak process RSS |
|---|---:|---:|---:|
| Naive | 0.90 ms | 0.55 ms | 89.15 MiB |
| Drift | 0.38 ms | 0.47 ms | 88.75 MiB |
| ARIMA CSS | 9.36 ms | 9.21 ms | 143.27 MiB |
| LightGBM lags | 25.33 ms | 36.55 ms | 108.46 MiB |
| LightGBM technical | 44.64 ms | 55.69 ms | 111.24 MiB |

`summary.csv` gives wall time, CPU time and peak RSS separately for every ticker and horizon. RSS was sampled every 10 ms, so shorter peaks can be missed. Wall and CPU times cover fit plus forecast call but exclude the initial CSV read and feature construction. CPU time can exceed wall time because native libraries may use threads and OS accounting is coarse. Process RSS includes Python and libraries and is not the incremental model memory. Run-to-run timing variation is expected on this workstation.

## Leakage and reproducibility checks

Eight focused tests pass. They cover feature timestamp alignment, invariance to future-row perturbations, target alignment, training-label cutoff (`s+h ≤ t`), deterministic LightGBM output, shared origin/target enforcement, duplicate and missing session handling, and invalid OHLCV rejection. The merged result verifier checked all 80 ticker–horizon–model cells against the exact same target dates and actual closes. Feature warmup rows are omitted from training; no future values are filled or imputed. `prepare_data.py` verifies source labels, the split event, adjustment-ratio jumps, checksums, calendar coverage and cross-source price agreement. The test fixtures are artificial unit-test values only and are never scored as historical data.

## Limitations and failed experiments

- The public research archive is sourced from Yahoo and is not a licensed production market-data feed. Direct Yahoo downloads were excluded under its automated-collection terms. Confirm data-use and redistribution rights before any production integration.
- Reference ETF files stopped in mid-December 2025, so they could only serve as consistency checks. The earlier Stooq endpoint returned a browser challenge during Phase 1 and was not used here.
- The single fixed LightGBM configuration was not tuned. Trying more hyperparameters on these same 44 evaluation origins would contaminate the comparison; a separate later holdout would be needed.
- All assets are U.S. equities or equity ETFs. The result does not cover rates, commodities, other countries, live data changes, transaction costs, or concurrent production traffic.
- Bootstrap intervals use 44 paired origins per cell but do not fully model regime dependence. The across-asset and across-horizon cells are correlated, so 16 cells are not 16 independent experiments.
- The custom ARIMA approximation is not a substitute for a validated statsmodels likelihood fit. Windows application security blocked the compiled state-space extension, and no restriction was bypassed.

## Recommendation for production integration

Do not integrate or deploy feature-engineered LightGBM on this evidence. It fails the provisional 5% and consistency criterion: only 3/16 cells reach 5%, with no positive paired interval excluding zero. Preserve Naive and Drift as simple research baselines. A later integration decision needs a licensed production data source, an untouched later holdout or live paper evaluation, and resource checks on the intended host. The threshold alone is never production approval. No production deployment is recommended here.

## Reproduce

Download the source and reference caches into `experiments/forecast-benchmark/phase2/data/raw/` from the five URLs in the inventory above. The exact source archive is pinned by its SHA-256. From the repository root:

```powershell
conda run -n qs-forecast python experiments/forecast-benchmark/phase2/prepare_data.py
conda run -n qs-forecast python -m unittest discover -s experiments/forecast-benchmark/phase2 -p 'test_*.py' -v
conda run -n qs-forecast python experiments/forecast-benchmark/phase2/benchmark.py --manifest experiments/forecast-benchmark/phase2/data/manifest.json --models naive --output experiments/forecast-benchmark/phase2/results-naive.json
conda run -n qs-forecast python experiments/forecast-benchmark/phase2/benchmark.py --manifest experiments/forecast-benchmark/phase2/data/manifest.json --models drift --output experiments/forecast-benchmark/phase2/results-drift.json
conda run -n qs-forecast python experiments/forecast-benchmark/phase2/benchmark.py --manifest experiments/forecast-benchmark/phase2/data/manifest.json --models arima_css --output experiments/forecast-benchmark/phase2/results-arima_css.json
conda run -n qs-forecast python experiments/forecast-benchmark/phase2/benchmark.py --manifest experiments/forecast-benchmark/phase2/data/manifest.json --models lgb_lags --output experiments/forecast-benchmark/phase2/results-lgb_lags.json
conda run -n qs-forecast python experiments/forecast-benchmark/phase2/benchmark.py --manifest experiments/forecast-benchmark/phase2/data/manifest.json --models lgb_technical --output experiments/forecast-benchmark/phase2/results-lgb_technical.json
conda run -n qs-forecast python experiments/forecast-benchmark/phase2/summarize_results.py
```
