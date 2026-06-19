from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

random.seed(2)
np.random.seed(2)
torch.manual_seed(2)
torch.use_deterministic_algorithms(True, warn_only=True)

_HERE = Path(__file__).resolve().parent          # asset_selector/
sys.path.insert(0, str(_HERE.parent))            # Portfolio_Personalization/

from asset_selector.asset_selector import classify_assets, get_profile_tickers
from asset_selector.visualizer import plot_all

_CONFIG_PATH = _HERE / "assets_egx30.json"
_PRICES_DIR  = _HERE.parent / "data" / "raw" / "prices"

logger = logging.getLogger(__name__)

# Column name mapping 

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]

def _find_col(columns: list[str], name: str) -> Optional[str]:
    """Case-insensitive column lookup. Returns None if not found."""
    for c in columns:
        if c.lower() == name:
            return c
    return None


# Preprocessing 

def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three-rule preprocessing applied to a single price-type DataFrame
    (open, high, low, or close — NOT volume).

    Rule 3 (applied FIRST):
        For each calendar year strictly after a stock's first valid
        observation, if every value in that year is NaN (delisted /
        inactive), replace with 0.0 (delisting marker).
        Years before first listing are left as NaN.

    Rule 2 (applied SECOND):
        Forward-fill from first_valid_index onward. The 0.0 blocks
        produced by Rule 3 are non-NaN so ffill stops at them correctly.

    Rule 1 (implicit):
        Leading NaN rows before a stock's first listing date are never
        touched — they represent genuine absence from the market.
    """
    df = df.copy()
    for col in df.columns:
        series   = df[col]
        fvi      = series.first_valid_index()
        if fvi is None:
            continue
        fvi_year = fvi.year

        for yr in sorted(df.index.year.unique()):
            if yr <= fvi_year:
                continue
            yr_mask = df.index.year == yr
            if df.loc[yr_mask, col].isna().all():
                df.loc[yr_mask, col] = 0.0
                logger.debug("%s: year %d all-NaN after listing → 0.0", col, yr)

        df.loc[fvi:, col] = df.loc[fvi:, col].ffill()

    return df


# Price loading 

def _load_ohlcv(
    universe:   list[str],
    start:      date,
    end:        date,
    prices_dir: Path,
) -> pd.DataFrame:
    """
    Read per-ticker CSVs and return a single MultiIndex DataFrame.

    Structure
    ---------
    Columns : MultiIndex (ticker, price_type)
              price_type ∈ {open, high, low, close, volume}
    Index   : DatetimeIndex (daily, normalised to midnight)

    Preprocessing
    -------------
    open, high, low, close : three-rule preprocessing applied
                             (delisting markers + forward fill).
    volume                 : zeros replaced with NaN, no forward fill.
                             Missing dates filled with 0.0 after alignment.

    All price types are outer-joined on the date index and filtered
    to [start, end].
    """
    # Accumulate per-price-type frames: {price_type: {ticker: Series}}
    frames: dict[str, dict[str, pd.Series]] = {c: {} for c in _OHLCV_COLS}

    for ticker in universe:
        csv_path = prices_dir / f"{ticker}.csv"
        if not csv_path.exists():
            logger.warning(
                "No CSV for %s — skipping. "
                "Run asset_selector/download_egx_prices.py first.",
                ticker,
            )
            continue

        raw = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        raw.index = pd.to_datetime(raw.index).normalize()
        raw       = raw[~raw.index.duplicated(keep="first")].sort_index()
        cols      = raw.columns.tolist()

        for price_type in _OHLCV_COLS:
            src_col = _find_col(cols, price_type)
            if src_col is None:
                logger.debug("%s: column '%s' not found — will be NaN", ticker, price_type)
                continue
            s = raw[src_col].copy()
            if not s.empty:
                frames[price_type][ticker] = s

    if not frames["close"]:
        raise FileNotFoundError(
            f"No close-price CSVs found in {prices_dir}. "
            "Run asset_selector/download_egx_prices.py first."
        )

    # Build a reference date index from the close prices
    close_df = pd.DataFrame(frames["close"]).sort_index()
    close_df = close_df.loc[~close_df.index.duplicated(keep="first")]
    close_df = _apply_preprocessing(close_df)
    close_df = close_df.loc[str(start) : str(end)]
    date_index = close_df.index          # canonical index all others align to
    tickers    = sorted(close_df.columns.tolist())

    # Process each price type
    processed: dict[str, pd.DataFrame] = {}

    for price_type in _OHLCV_COLS:
        if not frames[price_type]:
            logger.warning(
                "No data for price type '%s' — filling with NaN.", price_type
            )
            processed[price_type] = pd.DataFrame(
                np.nan, index=date_index, columns=tickers
            )
            continue

        df = pd.DataFrame(frames[price_type]).sort_index()
        df = df.loc[~df.index.duplicated(keep="first")]

        if price_type == "volume":
            # Volume: no three-rule preprocessing, zeros → NaN, missing → 0
            df = df.reindex(index=date_index, columns=tickers)
            df = df.replace(0.0, np.nan)
            # Leave NaN — env handles it; downstream fillna(0) if needed
        else:
            # open / high / low: same three-rule preprocessing as close
            df = _apply_preprocessing(df)
            df = df.loc[str(start) : str(end)]
            df = df.reindex(index=date_index, columns=tickers)

        processed[price_type] = df

    # Assemble MultiIndex DataFrame: columns = (ticker, price_type)
    ticker_dfs = {}
    for ticker in tickers:
        ticker_dfs[ticker] = pd.DataFrame(
            {pt: processed[pt][ticker] for pt in _OHLCV_COLS},
            index=date_index,
        )

    ohlcv = pd.concat(ticker_dfs, axis=1)   # columns: (ticker, price_type)
    ohlcv.columns.names = ["ticker", "price_type"]

    logger.info(
        "OHLCV loaded: %d tickers  %d rows  (%s → %s)",
        len(tickers), len(date_index),
        date_index[0].date(), date_index[-1].date(),
    )
    return ohlcv


# Pipeline 

def run_pipeline(
    output_dir:      str            = "asset_selector/output",
    generate_plots:  bool           = True,
    n_episodes:      int            = 250,
    start:           date           = date(2018, 1, 1),
    end:             Optional[date] = None,
    train_end:       Optional[str]  = "2023-12-31",
    prices_dir:      Optional[str]  = None,
    lookback:        int            = 126,
    forward:         int            = 21,
    step_size:       int            = 21,
    composition_csv: Optional[str]  = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdir = Path(prices_dir) if prices_dir else _PRICES_DIR

    if end is None:
        end = date.today()

    # Stage 1: Universe
    logger.info("=" * 60)
    logger.info("Stage 1 / 4 — Loading universe from config")
    logger.info("=" * 60)
    config   = _load_config()
    universe = config["historical_tickers"]
    cfg_start = config.get("start_date")
    if cfg_start and start == date(2018, 1, 1):
        start = date.fromisoformat(cfg_start)
    logger.info("Universe: %d tickers  start=%s", len(universe), start)

    # Stage 2: OHLCV data
    logger.info("=" * 60)
    logger.info("Stage 2 / 4 — Loading OHLCV from %s", pdir)
    logger.info("=" * 60)
    ohlcv = _load_ohlcv(universe, start=start, end=end, prices_dir=pdir)

    # Cache so run_rl.py can skip this stage
    ohlcv_path = out / "ohlcv.parquet"
    ohlcv.to_parquet(ohlcv_path)
    logger.info("OHLCV cached to %s", ohlcv_path)

    # Stage 3: RL risk profiling
    logger.info("=" * 60)
    logger.info("Stage 3 / 4 — RL risk profiling")
    logger.info("=" * 60)
    semi_annual_df, eval_df, static_df = classify_assets(
        ohlcv,
        n_episodes      = n_episodes,
        train_end       = train_end,
        output_dir      = output_dir,
        lookback        = lookback,
        forward         = forward,
        step_size       = step_size,
        composition_csv = composition_csv,
    )

    # Stage 4: Visualisation
    if generate_plots:
        logger.info("=" * 60)
        logger.info("Stage 4 / 4 — Generating plots")
        logger.info("=" * 60)
        plot_all(static_df, output_dir=output_dir, eval_df=eval_df)

    # Final summary
    logger.info("=" * 60)
    logger.info("Pipeline complete — %d tickers classified", len(static_df))
    logger.info("=" * 60)
    for profile in ("conservative", "balanced", "aggressive"):
        tickers = get_profile_tickers(static_df, profile)
        logger.info("  %-12s (%2d): %s", profile, len(tickers), tickers)

    return semi_annual_df, eval_df, static_df


# CLI 

def _cli() -> None:
    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt  = "%H:%M:%S",
        handlers = [logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(
        description="EGX30 Asset Selector — RL risk profiling pipeline",
    )
    parser.add_argument(
        "--output-dir",
        default = "asset_selector/output",
        metavar = "DIR",
        help    = "Directory for all output files (default: asset_selector/output)",
    )
    parser.add_argument(
        "--prices-dir",
        default = None,
        metavar = "DIR",
        help    = "Directory of per-ticker price CSVs (default: data/raw/prices)",
    )
    parser.add_argument(
        "--no-plots",
        action  = "store_true",
        help    = "Skip generating visualisation plots",
    )
    parser.add_argument(
        "--n-episodes",
        type    = int,
        default = 250,
        metavar = "N",
        help    = "PPO training episodes (default: 250)",
    )
    parser.add_argument(
        "--train-end",
        default = "2023-12-31",
        metavar = "DATE",
        help    = (
            "Last day of the training period in ISO format "
            "(default: 2023-12-31). "
            "Pass 'none' to disable the split and train on all data."
        ),
    )
    parser.add_argument(
        "--composition-csv",
        default = None,
        metavar = "FILE",
        help    = (
            "Path to EGX30 composition CSV (period_date, ticker). "
            "When provided, tertile splits are restricted to index-active "
            "tickers for each period (Option A filtering)."
        ),
    )
    args = parser.parse_args()

    train_end_arg = (
        None if args.train_end.lower() == "none" else args.train_end
    )

    run_pipeline(
        output_dir      = args.output_dir,
        generate_plots  = not args.no_plots,
        n_episodes      = args.n_episodes,
        train_end       = train_end_arg,
        prices_dir      = args.prices_dir,
        composition_csv = args.composition_csv,
    )


if __name__ == "__main__":
    _cli()