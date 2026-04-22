"""
run_rl.py
=========
Runs only the RL classification stage using cached prices from a prior
full pipeline run — skips data fetching.

Usage
-----
    venv/Scripts/python asset_selector/run_rl.py
    venv/Scripts/python asset_selector/run_rl.py --n-episodes 100
    venv/Scripts/python asset_selector/run_rl.py --output-dir asset_selector/output
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from asset_selector.asset_selector import classify_assets, get_profile_tickers
from asset_selector.visualizer import plot_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_rl_only(
    output_dir: str = "asset_selector/output",
    n_episodes: int = 150,
    generate_plots: bool = True,
    train_end: str = "2024-12-31",
) -> pd.DataFrame:
    out = Path(output_dir)

    logger.info("Loading cached prices from %s/prices.parquet", out)
    prices = pd.read_parquet(out / "prices.parquet")

    logger.info("Running RL classification (%d episodes) ...", n_episodes)
    quarterly_df, eval_df, classification = classify_assets(
        prices,
        n_clusters=3,
        n_episodes=n_episodes,
        output_dir=output_dir,
        train_end=train_end,
    )

    # Static summary (dominant quarterly label per ticker)
    clf_path = out / "asset_classification.csv"
    classification.to_csv(clf_path, index=False)
    logger.info("Classification saved to %s", clf_path)

    # quarterly_classifications.csv and evaluation.csv already saved by classify_assets
    logger.info(
        "Quarterly classifications: %d rows saved to %s",
        len(quarterly_df),
        out / "quarterly_classifications.csv",
    )
    logger.info(
        "Evaluation results: %d rows saved to %s",
        len(eval_df),
        out / "evaluation.csv",
    )

    profile_map = {
        p: get_profile_tickers(classification, p)
        for p in ("conservative", "balanced", "aggressive")
    }
    json_path = out / "risk_profiles.json"
    existing = json.loads(json_path.read_text()) if json_path.exists() else {}
    profile_map["benchmarking_map"] = existing.get("benchmarking_map", {})
    json_path.write_text(json.dumps(profile_map, indent=2), encoding="utf-8")
    logger.info("Risk profiles saved to %s", json_path)

    if generate_plots:
        plot_all(classification, output_dir=output_dir, eval_df=eval_df)

    for p in ("conservative", "balanced", "aggressive"):
        tickers = profile_map[p]
        logger.info("  %-12s (%2d): %s", p, len(tickers), tickers)

    return classification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL stage only (uses cached prices)")
    parser.add_argument("--output-dir", default="asset_selector/output", metavar="DIR")
    parser.add_argument("--n-episodes", type=int, default=150, metavar="N")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--train-end", default="2024-12-31", metavar="DATE",
        help="End of training period (default: 2024-12-31). Pass 'none' to disable split.",
    )
    args = parser.parse_args()

    train_end_arg = None if args.train_end.lower() == "none" else args.train_end
    run_rl_only(
        output_dir=args.output_dir,
        n_episodes=args.n_episodes,
        generate_plots=not args.no_plots,
        train_end=train_end_arg,
    )
