"""Extract the four pinned research OHLCV series without changing source values."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from benchmark import TICKERS, load_asset, sha256


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw" / "public_prices.parquet"
EXTRACTED = DATA / "extracted"
SOURCE_URL = "https://github.com/Johnbrick123/sp500-data/releases/download/data/prices.parquet"
START, END = "2018-01-01", "2025-12-31"
SOURCE_SHA256 = "fb846e89dc6cd128411f145bd4f79900c20334a6dd504f33d776cd87d46f560b"
REFERENCE_SHA256 = {
    "SPY": "847e60a441eb10969d87cd4a6da604257b782d9076168f68d5730b84096c79db",
    "QQQ": "78c015fd7fa9d493cd798d5ae81b3c2ae6dc3b8176e25e7bb90cd4114b103351",
    "IWM": "e980f50473a8c346ba9ac8b05dcf3da691581dae742f649cb3608d0efe2ef940",
    "AAPL": "f7782bd98b9ef5088352d125106049b5ecc00c7194681414ebac4f6798b4fa4e",
}


def main() -> None:
    if not RAW.is_file():
        raise FileNotFoundError(f"Download the source archive first: {SOURCE_URL}")
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    source_sha = sha256(RAW)
    if source_sha != SOURCE_SHA256:
        raise ValueError("Source archive SHA-256 differs from the measured release")
    manifest = {
        "start": START, "end": END,
        "source": {"url": SOURCE_URL, "downloaded_on": str(date.today()),
                   "path": "raw/public_prices.parquet", "sha256": source_sha,
                   "convention": "Yahoo-derived split-adjusted OHLC, reported volume; adj_close includes dividend adjustment and is not used"},
        "assets": {},
    }
    for ticker in TICKERS:
        # PyArrow filters keep the 12-million-row source archive out of RAM.
        frame = pd.read_parquet(RAW, filters=[("ticker", "==", ticker),
                                               ("date", ">=", pd.Timestamp(START)),
                                               ("date", "<=", pd.Timestamp(END))])
        if set(frame["source"].dropna().unique()) != {"yahoo"}:
            raise ValueError(f"{ticker}: unexpected upstream source labels")
        frame = frame.sort_values("date")
        adjustment_ratio = frame["adj_close"].div(frame["close"])
        ratio_jumps = adjustment_ratio.pct_change().abs()
        suspicious_adjustments = frame.loc[ratio_jumps > 0.05, "date"].dt.strftime("%Y-%m-%d").tolist()
        if suspicious_adjustments:
            raise ValueError(f"{ticker}: suspicious adjustment-ratio jumps: {suspicious_adjustments}")
        split_events = frame.loc[frame["stock_splits"].fillna(0) != 0,
                                 ["date", "stock_splits"]]
        splits = [{"date": str(row.date.date()), "factor": float(row.stock_splits)}
                  for row in split_events.itertuples()]
        output_path = EXTRACTED / f"{ticker}_2018_2025.csv"
        frame = frame.loc[:, ["date", "open", "high", "low", "close", "volume"]]
        frame.to_csv(output_path, index=False, float_format="%.12g")
        _, audit = load_asset(output_path, ticker, START, END)
        audit["adjustment_ratio_jumps_over_5pct"] = suspicious_adjustments
        audit["split_events"] = splits
        if audit["missing_sessions"] or audit["unexpected_sessions"] or audit["close_jumps_over_40pct"]:
            raise ValueError(f"{ticker}: failed audit: {audit}")
        manifest["assets"][ticker] = {
            "path": f"extracted/{output_path.name}", "sha256": sha256(output_path),
            "source_url": SOURCE_URL, "downloaded_on": str(date.today()),
            "convention": manifest["source"]["convention"], "audit": audit,
        }
        print(f"{ticker}: {audit['rows']} rows, {audit['first']} to {audit['last']}")
    manifest["cross_checks"] = {}
    for ticker in TICKERS:
        reference = (RAW.parent / f"{ticker}_underlying.parquet" if ticker != "AAPL"
                     else RAW.parent / "AAPL_pubmarks_ohlcv.csv")
        if not reference.is_file():
            raise FileNotFoundError(f"Missing cross-check cache: {reference}")
        if sha256(reference) != REFERENCE_SHA256[ticker]:
            raise ValueError(f"{ticker}: reference SHA-256 differs from the measured cache")
        reference_frame = (pd.read_parquet(reference) if reference.suffix == ".parquet"
                           else pd.read_csv(reference))
        current = pd.read_csv(EXTRACTED / f"{ticker}_2018_2025.csv")
        joined = current[["date", "close"]].merge(reference_frame[["date", "close"]],
                                                   on="date", suffixes=("_benchmark", "_reference"))
        if len(joined) < 1900:
            raise ValueError(f"{ticker}: cross-check overlap too short: {len(joined)}")
        difference = (joined["close_benchmark"].div(joined["close_reference"]) - 1).abs()
        if difference.max() > 0.01:
            raise ValueError(f"{ticker}: cross-check price deviation exceeds 1%")
        url = (f"https://github.com/lambdaclass/options_portfolio_backtester/releases/download/data-v1/{ticker}_underlying.parquet"
               if ticker != "AAPL" else "https://pubmarks.github.io/datasets/stocks/AAPL/ohlcv.csv")
        manifest["cross_checks"][ticker] = {
            "url": url, "path": f"raw/{reference.name}", "downloaded_on": str(date.today()),
            "sha256": sha256(reference), "overlap_rows": len(joined),
            "first_overlap": str(joined["date"].iloc[0])[:10],
            "last_overlap": str(joined["date"].iloc[-1])[:10],
            "max_close_difference_pct": float(100 * difference.max()),
            "note": "ETF release stops in mid-December 2025; reference is a consistency check only",
        }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Source SHA-256: {source_sha}")


if __name__ == "__main__":
    main()
