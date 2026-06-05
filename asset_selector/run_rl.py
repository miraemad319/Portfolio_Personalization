"""
run_rl.py

Runs only the RL classification stage using OHLCV data cached from a prior
full pipeline run. Skips the price download and preprocessing stages.
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

random.seed(2)
np.random.seed(2)
torch.manual_seed(2)
torch.use_deterministic_algorithms(True, warn_only=True)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from asset_selector.asset_selector import classify_assets, get_profile_tickers
from asset_selector.visualizer import plot_all

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%H:%M:%S",
    handlers= [logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_rl_only(
    output_dir:     str           = "asset_selector/output",
    n_episodes:     int           = 250,
    train_end:      Optional[str] = "2023-12-31",
    generate_plots: bool          = True,
    lookback:       int           = 126,
    forward:        int           = 126,
    step_size:      int           = 21,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = Path(output_dir)

    # Load cached OHLCV
    ohlcv_path = out / "ohlcv.parquet"
    if not ohlcv_path.exists():
        raise FileNotFoundError(
            f"ohlcv.parquet not found at {ohlcv_path}. "
            "Run python -m asset_selector.main first to download "
            "and cache the OHLCV data."
        )

    logger.info("Loading cached OHLCV from %s", ohlcv_path)
    ohlcv = pd.read_parquet(ohlcv_path)

    logger.info(
        "OHLCV loaded: %d tickers  %d rows  (%s → %s)",
        ohlcv.columns.get_level_values("ticker").nunique(),
        len(ohlcv),
        ohlcv.index[0].date(),
        ohlcv.index[-1].date(),
    )

    # RL classification
    logger.info(
        "Running RL classification: %d episodes  train_end=%s",
        n_episodes, train_end,
    )
    semi_annual_df, eval_df, static_df = classify_assets(
        ohlcv,
        n_episodes = n_episodes,
        train_end  = train_end,
        output_dir = output_dir,
        lookback   = lookback,
        forward    = forward,
        step_size  = step_size,
    )

    # Visualisation
    if generate_plots:
        logger.info("Generating plots …")
        plot_all(static_df, output_dir=output_dir, eval_df=eval_df)

    # Final summary
    logger.info("=" * 60)
    logger.info("RL stage complete — %d tickers classified", len(static_df))
    logger.info("=" * 60)
    for profile in ("conservative", "balanced", "aggressive"):
        tickers = get_profile_tickers(static_df, profile)
        logger.info("  %-12s (%2d): %s", profile, len(tickers), tickers)

    return semi_annual_df, eval_df, static_df


# CLI 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "EGX30 Asset Selector — RL stage only (uses cached ohlcv.parquet)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default = "asset_selector/output",
        metavar = "DIR",
        help    = "Directory containing ohlcv.parquet and for output files "
                  "(default: asset_selector/output)",
    )
    parser.add_argument(
        "--lookback",
        type    = int,
        default = 126,
        metavar = "N",
        help    = "Lookback window in trading days (default: 126)",
    )
    parser.add_argument(
        "--forward",
        type    = int,
        default = 126,
        metavar = "N",
        help    = "Forward window in trading days (default: 126)",
    )
    parser.add_argument(
        "--step-size",
        type    = int,
        default = 21,
        metavar = "N",
        help    = "Step size between observation windows (default: 21)",
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
            "Pass 'none' to train on the full dataset without a split."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action  = "store_true",
        help    = "Skip generating visualisation plots",
    )
    args = parser.parse_args()

    train_end_arg = (
        None if args.train_end.lower() == "none" else args.train_end
    )

    run_rl_only(
        output_dir     = args.output_dir,
        n_episodes     = args.n_episodes,
        train_end      = train_end_arg,
        generate_plots = not args.no_plots,
        lookback       = args.lookback,
        forward        = args.forward,
        step_size      = args.step_size,
    )