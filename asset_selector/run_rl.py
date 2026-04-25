"""
run_rl.py

Runs only the RL classification stage using prices cached from a prior
full pipeline run.  Skips the price download and preprocessing stages.

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

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.use_deterministic_algorithms(True)

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
    n_episodes:     int           = 300,
    train_end:      Optional[str] = "2024-12-31",
    generate_plots: bool          = True,
    lookback:       int           = 126,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = Path(output_dir)

    # Load cached prices 
    prices_path = out / "prices.parquet"
    volume_path = out / "volume.parquet"
    if not prices_path.exists():
        raise FileNotFoundError(
            f"prices.parquet not found at {prices_path}. "
            "Run python -m asset_selector.main first to download "
            "and cache the price data."
        )
    
    logger.info("Loading cached prices from %s", prices_path)
    prices = pd.read_parquet(prices_path)
    
    if volume_path.exists():
        logger.info("Loading cached volume from %s", volume_path)
        volume = pd.read_parquet(volume_path)
    else:
        logger.warning(
            "volume.parquet not found — volume features will be NaN. "
            "Run python -m asset_selector.main to regenerate."
        )
        volume = pd.DataFrame(
            0.0, index=prices.index, columns=prices.columns
        )

    logger.info(
        "Prices loaded: %d tickers  %d rows  (%s → %s)",
        prices.shape[1], prices.shape[0],
        prices.index[0].date(), prices.index[-1].date(),
    )

    # RL classification 
    logger.info(
        "Running RL classification: %d episodes  train_end=%s",
        n_episodes, train_end,
    )
    quarterly_df, eval_df, static_df = classify_assets(
        prices,
        volume = volume,
        n_episodes = n_episodes,
        train_end  = train_end,
        output_dir = output_dir,
        lookback   = lookback,
    )
    

    # Log output file locations 
    logger.info("=" * 60)
    logger.info("Output files written to %s", out)
    logger.info("  risk_profiles.json             ← downstream portfolio models")
    logger.info("  quarterly_classifications.csv  ← full dynamic history")
    logger.info("  evaluation.csv                 ← Sharpe / accuracy per quarter")
    logger.info("  quarterly_spearman.csv         ← per-quarter Spearman ρ")
    logger.info("  asset_classification.csv       ← current labels (visualiser)")

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

    return quarterly_df, eval_df, static_df

# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "EGX30 Asset Selector — RL stage only (uses cached prices.parquet)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default = "asset_selector/output",
        metavar = "DIR",
        help    = "Directory containing prices.parquet and for output files "
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
        "--n-episodes",
        type    = int,
        default = 250,
        metavar = "N",
        help    = "PPO training episodes (default: 300)",
    )
    parser.add_argument(
        "--train-end",
        default = "2024-12-31",
        metavar = "DATE",
        help    = (
            "Last day of the training period in ISO format "
            "(default: 2024-12-31). "
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
    )