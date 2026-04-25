"""
main.py

Entry point for the EGX30 Asset Selector pipeline.

Pipeline stages
---------------
1. Universe + config   (assets_egx30.json)
2. Price loading       (data/raw/prices/*.csv — produced by download_egx_prices.py)
3. RL classification   (asset_selector.py)
4. Visualisation       (visualizer.py)
5. Output              (CSV + JSON)

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

_HERE = Path(__file__).resolve().parent          # asset_selector/
sys.path.insert(0, str(_HERE.parent))            # Portfolio_Personalization/

from asset_selector.asset_selector import classify_assets, get_profile_tickers
from asset_selector.visualizer import plot_all

_CONFIG_PATH = _HERE / "assets_egx30.json"
_PRICES_DIR  = _HERE.parent / "data" / "raw" / "prices"

logger = logging.getLogger(__name__)

# Config + price loading

def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three-rule preprocessing applied to the combined close-price DataFrame.

    Rule 3 (applied FIRST):
        For each calendar year strictly after a stock's first valid
        observation, if every value in that year is NaN (delisted /
        inactive), replace with 0.0 (delisting marker).
        Years before first listing are left as NaN.

    Rule 2 (applied SECOND):
        Forward-fill from first_valid_index onward.  The 0.0 blocks
        produced by Rule 3 are non-NaN, so ffill stops at them correctly.

    Rule 1 (implicit):
        Leading NaN rows before a stock's first listing date are never
        touched — they represent genuine absence from the market.

    """
    df = df.copy()
    for col in df.columns:
        series = df[col]
        fvi    = series.first_valid_index()
        if fvi is None:
            continue

        fvi_year: int = fvi.year

        # Rule 3: mark whole-year NaN gaps after first listing as 0.0
        for yr in sorted(df.index.year.unique()):
            if yr <= fvi_year:
                continue
            yr_mask = df.index.year == yr
            if df.loc[yr_mask, col].isna().all():
                df.loc[yr_mask, col] = 0.0
                logger.debug(
                    "%s: year %d all-NaN after listing — set to 0.0", col, yr
                )

        # Rule 2: forward-fill from first valid observation
        df.loc[fvi:, col] = df.loc[fvi:, col].ffill()

    return df


def _load_prices(
    universe:   list[str],
    start:      date,
    end:        date,
    prices_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read per-ticker CSVs, apply preprocessing rules, filter to [start, end].

    Returns
    -------
    close_df  : pd.DataFrame  DatetimeIndex, columns = tickers, close prices
    volume_df : pd.DataFrame  DatetimeIndex, columns = tickers, raw volume
                Volume is NOT preprocessed with the three rules — NaN where
                missing, 0.0 where the stock was inactive. Forward-fill is
                not applied to volume because zero volume is meaningful.
    """
    close_frames:  dict[str, pd.Series] = {}
    volume_frames: dict[str, pd.Series] = {}

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

        close_col = next(
            (c for c in raw.columns if c.lower() == "close"), None
        )
        if close_col is None:
            close_col = raw.columns[-1]

        vol_col = next(
            (c for c in raw.columns if c.lower() == "volume"), None
        )

        s_close = raw[close_col].sort_index()
        s_close.index = pd.to_datetime(s_close.index).normalize()
        if not s_close.empty:
            close_frames[ticker] = s_close

        if vol_col is not None:
            s_vol = raw[vol_col].sort_index()
            s_vol.index = pd.to_datetime(s_vol.index).normalize()
            if not s_vol.empty:
                volume_frames[ticker] = s_vol

    if not close_frames:
        raise FileNotFoundError(
            f"No price CSVs found in {prices_dir}. "
            "Run asset_selector/download_egx_prices.py first."
        )

    # Close prices — outer join, apply preprocessing rules
    close_df = pd.DataFrame(close_frames).sort_index()
    close_df = close_df.loc[~close_df.index.duplicated(keep="first")]
    close_df = _apply_preprocessing(close_df)
    close_df = close_df.loc[str(start) : str(end)]

    # Volume — outer join, no preprocessing rules applied
    # Replace NaN with 0 so downstream code can safely check vol > 0
    if volume_frames:
        volume_df = pd.DataFrame(volume_frames).sort_index()
        volume_df = volume_df.loc[~volume_df.index.duplicated(keep="first")]
        volume_df = volume_df.loc[str(start) : str(end)]
        # Align to close index
        volume_df = volume_df.reindex(close_df.index).fillna(0.0)
    else:
        logger.warning("No volume data found — volume features will be NaN.")
        volume_df = pd.DataFrame(
            0.0, index=close_df.index, columns=close_df.columns
        )

    logger.info(
        "Prices loaded: %d tickers  %d rows  (%s → %s)",
        close_df.shape[1], close_df.shape[0],
        close_df.index[0].date(), close_df.index[-1].date(),
    )
    return close_df, volume_df

# Pipeline

def run_pipeline(
    output_dir:     str            = "asset_selector/output",
    generate_plots: bool           = True,
    n_episodes:     int            = 250,
    start:          date           = date(2020, 1, 1),
    end:            date           = date(2026, 4, 22),
    train_end:      Optional[str]  = "2024-12-31",
    prices_dir:     Optional[str]  = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute the full EGX30 Asset Selector pipeline.

    Returns
    -------
    quarterly_df : pd.DataFrame
        Dynamic quarterly classifications (train + test, split-labelled).
    eval_df : pd.DataFrame
        Evaluation results with actual Sharpe, fwd vol, and accuracy flags.
    static_df : pd.DataFrame
        One row per ticker — risk_profile is the most recent quarterly label.
    """
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdir = Path(prices_dir) if prices_dir else _PRICES_DIR

    # Stage 1: Universe 
    logger.info("=" * 60)
    logger.info("Stage 1 / 4 — Loading universe from config")
    logger.info("=" * 60)
    config   = _load_config()
    universe = config["historical_tickers"]
    logger.info("Universe: %d tickers", len(universe))

    # Stage 2: Price data
    logger.info("=" * 60)
    logger.info("Stage 2 / 4 — Loading prices from %s", pdir)
    logger.info("=" * 60)
    prices, volume = _load_prices(universe, start=start, end=end, prices_dir=pdir)

    # Cache both so run_rl.py can skip this stage
    prices_path = out / "prices.parquet"
    volume_path = out / "volume.parquet"
    prices.to_parquet(prices_path)
    volume.to_parquet(volume_path)
    logger.info("Prices cached to %s  Volume cached to %s", 
                prices_path, volume_path,
    )


    # Stage 3: RL risk profiling 
    logger.info("=" * 60)
    logger.info("Stage 3 / 4 — RL risk profiling")
    logger.info("=" * 60)
    quarterly_df, eval_df, static_df = classify_assets(
        prices,
        volume     = volume,
        n_episodes = n_episodes,
        train_end  = train_end,
        output_dir = output_dir,
    )

    # Log file locations for the downstream team
    logger.info("=" * 60)
    logger.info("Output files written to %s", out)
    logger.info("  risk_profiles.json             ← downstream portfolio models")
    logger.info("  quarterly_classifications.csv  ← full dynamic history")
    logger.info("  evaluation.csv                 ← Sharpe / accuracy per quarter")
    logger.info("  quarterly_spearman.csv         ← per-quarter Spearman ρ")
    logger.info("  asset_classification.csv       ← current labels (visualiser)")

    # Stage 4: Visualisation
    if generate_plots:
        logger.info("=" * 60)
        logger.info("Stage 4 / 4 — Generating plots")
        logger.info("=" * 60)
        plot_all(static_df, output_dir=output_dir, eval_df=eval_df)

    #  Final summary 
    logger.info("=" * 60)
    logger.info("Pipeline complete — %d tickers classified", len(static_df))
    logger.info("=" * 60)
    for profile in ("conservative", "balanced", "aggressive"):
        tickers = get_profile_tickers(static_df, profile)
        logger.info("  %-12s (%2d): %s", profile, len(tickers), tickers)

    return quarterly_df, eval_df, static_df

# CLI
def _cli() -> None:
    import random
    import numpy as np
    import torch
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.use_deterministic_algorithms(True)
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt = "%H:%M:%S",
        handlers= [logging.StreamHandler(sys.stdout)],
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
        default = "2024-12-31",
        metavar = "DATE",
        help    = (
            "Last day of the training period in ISO format "
            "(default: 2023-12-31). "
            "Pass 'none' to disable the split and train on all data."
        ),
    )
    args = parser.parse_args()

    train_end_arg = (
        None if args.train_end.lower() == "none" else args.train_end
    )

    run_pipeline(
        output_dir     = args.output_dir,
        generate_plots = not args.no_plots,
        n_episodes     = args.n_episodes,
        train_end      = train_end_arg,
        prices_dir     = args.prices_dir,
    )


if __name__ == "__main__":
    _cli()