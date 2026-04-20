"""
main.py
=======
Entry point for the EGX30 Asset Selector pipeline.

Pipeline stages
---------------
1. Universe + config  (assets_egx30.json)
2. Price loading      (data/raw/prices/*.csv — produced by download_egx_prices.py)
3. RL classification  (asset_selector.py)
4. Visualisation      (visualizer.py)
5. Output             (CSV + JSON)

Usage
-----
    # From project root, activate venv then:
    python -m asset_selector.main

    # Or directly:
    python asset_selector/main.py [options]

Options
-------
    --output-dir PATH   Directory for output files        [asset_selector/output]
    --no-plots          Skip generating visualisation plots
    --n-episodes N      PPO training episodes             [60]  (150 recommended)
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

# ── Make the module importable both as a script and as a package ──────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from asset_selector.asset_selector import classify_assets, get_profile_tickers
from asset_selector.visualizer import plot_all

_CONFIG_PATH = _HERE / "assets_egx30.json"
_PRICES_DIR  = _HERE.parent / "data" / "raw" / "prices"

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


def _apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three-rule preprocessing applied to the combined close-price DataFrame.

    Rule 3 (FIRST):
        For each calendar year strictly after a stock's first valid observation,
        if every value in that year is NaN (delisted/inactive), replace with 0.0.
        Years before first listing remain NaN.

    Rule 2 (SECOND):
        Forward-fill from first_valid_index onward. The 0.0 blocks from Rule 3
        are non-NaN, so ffill stops correctly at them.

    Rule 1 (implicit):
        Leading NaN rows before first listing are never touched.

    Before log-return computation downstream, replace 0.0 with NaN to avoid
    ±inf at delisting transitions.
    """
    df = df.copy()
    for col in df.columns:
        series = df[col]
        fvi = series.first_valid_index()
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
                logger.debug("%s: year %d all-NaN after listing — set to 0.0", col, yr)

        # Rule 2: forward-fill from first valid observation
        df.loc[fvi:, col] = df.loc[fvi:, col].ffill()

    return df


def _load_prices(
    universe: list[str],
    start: date,
    end: date,
    prices_dir: Path,
) -> pd.DataFrame:
    """
    Read per-ticker CSVs produced by download_egx_prices.py, apply the
    three preprocessing rules, then filter to [start, end].

    Returns a close-price DataFrame (DatetimeIndex, columns = tickers).
    Tickers without a CSV on disk are silently skipped.
    """
    frames: dict[str, pd.Series] = {}
    for ticker in universe:
        csv = prices_dir / f"{ticker}.csv"
        if not csv.exists():
            logger.warning("No CSV for %s — skipping (run download_egx_prices.py first)", ticker)
            continue
        raw = pd.read_csv(csv, index_col=0, parse_dates=True)
        # egxpy returns columns like Open/High/Low/Close/Volume
        close_col = next((c for c in raw.columns if c.lower() == "close"), None)
        if close_col is None:
            close_col = raw.columns[-1]
        s = raw[close_col].sort_index()
        s.index = pd.to_datetime(s.index).normalize()
        if not s.empty:
            frames[ticker] = s

    if not frames:
        raise FileNotFoundError(
            f"No price CSVs found in {prices_dir}. "
            "Run asset_selector/download_egx_prices.py first."
        )

    # Outer join — preserves per-ticker NaN structure needed for preprocessing
    combined = pd.DataFrame(frames).sort_index()
    combined = combined.loc[~combined.index.duplicated(keep="first")]

    # Apply the three preprocessing rules before date filtering
    combined = _apply_preprocessing(combined)

    # Filter to requested date range
    combined = combined.loc[str(start) : str(end)]

    return combined


def run_pipeline(
    output_dir: str = "asset_selector/output",
    generate_plots: bool = True,
    n_episodes: int = 150,
    start: date = date(2020, 1, 1),
    end: date = date(2025, 12, 31),
    prices_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Execute the full EGX30 Asset Selector pipeline.

    Returns
    -------
    classification : pd.DataFrame
        Static summary — columns: ticker | volatility | mean_return |
        cluster_id | risk_profile | rl_risk_score
        risk_profile is the dominant quarterly label.
    benchmarking_map : dict
        Always empty — historical EGX30 composition is not tracked here.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pdir = Path(prices_dir) if prices_dir else _PRICES_DIR

    # ── Stage 1: Universe ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 1 / 4 — Loading universe from config")
    logger.info("=" * 60)
    config   = _load_config()
    universe = config["historical_tickers"]
    benchmarking_map: dict = {}
    logger.info("Universe: %d tickers", len(universe))

    # ── Stage 2: Price data ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 2 / 4 — Loading prices from %s", pdir)
    logger.info("=" * 60)
    prices = _load_prices(universe, start=start, end=end, prices_dir=pdir)
    prices_path = out / "prices.parquet"
    prices.to_parquet(prices_path)
    logger.info("Prices saved to %s  (%d tickers, %d rows)", prices_path, prices.shape[1], prices.shape[0])

    # ── Stage 3: RL risk profiling ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 3 / 4 — RL risk profiling")
    logger.info("=" * 60)
    quarterly_df, eval_df, classification = classify_assets(
        prices,
        n_clusters=3,
        n_episodes=n_episodes,
        output_dir=output_dir,
    )

    # Save static classification table (dominant quarterly label per ticker)
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

    # Save profile → tickers JSON (consumed by downstream portfolio models)
    profile_map = {
        profile: get_profile_tickers(classification, profile)
        for profile in ("conservative", "balanced", "aggressive")
    }
    profile_map["benchmarking_map"] = benchmarking_map
    json_path = out / "risk_profiles.json"
    json_path.write_text(json.dumps(profile_map, indent=2), encoding="utf-8")
    logger.info("Risk profiles saved to %s", json_path)

    # ── Stage 4: Visualisation ─────────────────────────────────────────────
    if generate_plots:
        logger.info("=" * 60)
        logger.info("Stage 4 / 4 — Generating plots")
        logger.info("=" * 60)
        plot_all(classification, output_dir=output_dir, eval_df=eval_df)

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Pipeline complete  (%d tickers classified)", len(classification))
    logger.info("=" * 60)
    for profile in ("conservative", "balanced", "aggressive"):
        tickers = profile_map[profile]
        logger.info("  %-12s (%2d): %s", profile, len(tickers), tickers)

    return classification, benchmarking_map


def _cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    parser = argparse.ArgumentParser(
        description="EGX30 Asset Selector — RL risk profiling pipeline"
    )
    parser.add_argument(
        "--output-dir",
        default="asset_selector/output",
        metavar="DIR",
        help="Output directory (default: asset_selector/output)",
    )
    parser.add_argument(
        "--prices-dir",
        default=None,
        metavar="DIR",
        help="Directory containing per-ticker price CSVs (default: data/raw/prices)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating visualisation plots",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=60,
        metavar="N",
        help="RL training episodes (default: 60; 150 recommended for best convergence)",
    )
    args = parser.parse_args()

    run_pipeline(
        output_dir=args.output_dir,
        generate_plots=not args.no_plots,
        n_episodes=args.n_episodes,
        prices_dir=args.prices_dir,
    )


if __name__ == "__main__":
    _cli()
