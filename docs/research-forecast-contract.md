# Research forecast contract (PR 1)

This is a deliberately narrow first step toward the QuantScreen Research Agent v0.2 design.

## Verified existing assets

- `experiments/forecast-benchmark/benchmark.py`: isolated Naive, SciPy conditional-least-squares ARIMA approximation, LightGBM benchmark.
- `experiments/forecast-benchmark/phase2/benchmark.py`: Naive, drift, `arima_css`, `lgb_lags`, `lgb_technical` terminal-close predictions and feature generation.
- `backend/app/main.py`: FastAPI entry point; no research endpoints added here.
- `backend/app/models.py`: existing stock/theme/calendar/refresh persistence. Research tables are not yet implemented.

## Contract

`app.research.forecast_contract` defines an immutable request/result and a deterministic naive adapter. All prices are **positive terminal-close price levels**, not returns, probabilities, signals or trading instructions. `horizon_sessions` counts trading sessions; the caller supplies a market-calendar-resolved `target_date`. `as_of` is the signal/reference session; `trained_until <= as_of`; `available_at` is timezone-aware.

The contract rejects invalid values and future training cutoffs, but **cannot prove** that underlying features or market data were actually available as of the requested time. That needs point-in-time dataset validation before live integration. The naive adapter also assumes `last_close` is already known and valid as of the signal time; callers must enforce that separately.

## Run tests

From `backend/`:

```bash
python -m unittest discover -s tests -p 'test_research_forecast_contract.py' -v
```

Uses Python standard library only. Tests cover a valid naive prediction, missing identifiers, invalid horizon, invalid prices, invalid cutoff, invalid target date and timezone requirements.

## Follow-up (not implemented)

1. Wrap existing Phase 2 `predict()` behind this contract without changing benchmark results.
2. Resolve `target_date` using verified exchange sessions; validate `available_at` against feature and model availability.
3. Add data snapshot/version identifiers and model/feature provenance, then persisted run records.
4. Compare all adapters on identical folds with explicit leakage checks before connecting any API/UI.
5. Validate Korean market historical constituents, institutional/foreign flows, source terms and licensing before implementing KOSPI 100 scanning.

No backtest, RAG, market-data ingestion, API endpoint, database migration or model performance claim is part of this PR.
