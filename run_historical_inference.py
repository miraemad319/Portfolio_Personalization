"""
run_historical_inference.py

Runs the saved asset selector model (rl_agent.pt) on OHLCV data from
2012 to today to produce daily risk classifications for all 31 tickers.

Does NOT retrain. Uses collect_all_scores() for per-window inference,
then applies our own tertile split per window and forward-fills to daily.

This bypasses collect_period_scores() which requires >= 3 valid scores
per semi-annual bucket — a check that silently drops most of 2015-2023.

Output:
    outputs/volatility_clustering/daily_risk_classifications.csv

    Format: DatetimeIndex rows, ticker columns,
    values = "conservative" | "balanced" | "aggressive" | NaN

Usage:
    python run_historical_inference.py --start 2012-01-01
    python run_historical_inference.py --start 2012-01-01 \\
        --output-path outputs/volatility_clustering/daily_risk_classifications.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from asset_selector.rl_agent import RLAssetSelectorAgent
from asset_selector.rl_environment import AssetSelectorEnv

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
    handlers= [logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION — must match training settings in classify_assets()
# =============================================================================

AGENT_HIDDEN_DIMS = [128, 64, 32]
LOOKBACK          = 126
FORWARD           = 21
STEP_SIZE         = 21
N_CLUSTERS        = 3
MIN_VALID_TICKERS = 3   # minimum tickers with finite scores to classify a window

OUR_TICKERS = [
    "ABUK", "ADIB", "AMOC", "ARCC", "BTFH", "CCAP", "COMI", "EAST",
    "EFID", "EFIH", "EGAL", "EGCH", "EMFD", "ETEL", "FWRY", "GBCO",
    "HELI", "HRHO", "ISPH", "JUFO", "MCQE", "OIH",  "ORAS", "ORHD",
    "ORWE", "PHDC", "RAYA", "RMDA", "TMGH", "VLMR", "VLMRA",
]

DEFAULT_PRICES_DIR  = _HERE / "data/raw/OHLCV"
DEFAULT_AGENT_PATH  = _HERE / "asset_selector/output/rl_agent.pt"
DEFAULT_OUTPUT_PATH = _HERE / "outputs/volatility_clustering/daily_risk_classifications.csv"
DEFAULT_START       = date(2012, 1, 1)

# =============================================================================
# OHLCV LOADER — identical to main.py
# =============================================================================

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _find_col(columns, name):
    for c in columns:
        if c.lower() == name:
            return c
    return None


def _apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        series = df[col]
        fvi    = series.first_valid_index()
        if fvi is None:
            continue
        fvi_year = fvi.year
        for yr in sorted(df.index.year.unique()):
            if yr <= fvi_year:
                continue
            yr_mask = df.index.year == yr
            if df.loc[yr_mask, col].isna().all():
                df.loc[yr_mask, col] = 0.0
        df.loc[fvi:, col] = df.loc[fvi:, col].ffill()
    return df


def load_ohlcv(tickers, start, end, prices_dir):
    frames = {c: {} for c in _OHLCV_COLS}
    missing = []

    for ticker in tickers:
        csv_path = prices_dir / f"{ticker}.csv"
        if not csv_path.exists():
            csv_path = prices_dir / f"{ticker.lower()}.csv"
        if not csv_path.exists():
            missing.append(ticker)
            continue

        raw = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        raw.index = pd.to_datetime(raw.index).normalize()
        raw = raw[~raw.index.duplicated(keep="first")].sort_index()

        for price_type in _OHLCV_COLS:
            src_col = _find_col(raw.columns.tolist(), price_type)
            if src_col is not None:
                frames[price_type][ticker] = raw[src_col].copy()

    if missing:
        log.warning(f"Missing CSVs for: {missing}")

    close_df   = pd.DataFrame(frames["close"]).sort_index()
    close_df   = close_df.loc[~close_df.index.duplicated(keep="first")]
    close_df   = _apply_preprocessing(close_df)
    close_df   = close_df.loc[str(start):str(end)]
    date_index = close_df.index
    available  = sorted(close_df.columns.tolist())

    log.info(f"Price panel: {len(available)} tickers, {len(date_index)} rows, "
             f"{date_index[0].date()} → {date_index[-1].date()}")

    processed = {}
    for price_type in _OHLCV_COLS:
        if not frames[price_type]:
            processed[price_type] = pd.DataFrame(
                np.nan, index=date_index, columns=available
            )
            continue
        df = pd.DataFrame(frames[price_type]).sort_index()
        df = df.loc[~df.index.duplicated(keep="first")]
        if price_type == "volume":
            df = df.reindex(index=date_index, columns=available).replace(0.0, np.nan)
        else:
            df = _apply_preprocessing(df)
            df = df.loc[str(start):str(end)]
            df = df.reindex(index=date_index, columns=available)
        processed[price_type] = df

    ticker_dfs = {
        t: pd.DataFrame(
            {pt: processed[pt][t] for pt in _OHLCV_COLS},
            index=date_index,
        )
        for t in available
    }
    ohlcv = pd.concat(ticker_dfs, axis=1)
    ohlcv.columns.names = ["ticker", "price_type"]
    return ohlcv


# =============================================================================
# TERTILE CLASSIFICATION
# Per-window: rank valid scores into thirds → conservative/balanced/aggressive
# Matches the logic in AssetSelectorEnv.assign_clusters()
# =============================================================================

def scores_to_profiles(scores: np.ndarray, tickers: list) -> dict[str, str]:
    """
    Convert a per-ticker score array into risk profile labels using
    tertile split on valid (finite) scores only.

    Returns dict {ticker: profile} for tickers with valid scores.
    Tickers with NaN scores are omitted (will remain NaN in output).
    """
    valid_mask = np.isfinite(scores)
    n_valid    = valid_mask.sum()

    if n_valid < MIN_VALID_TICKERS:
        return {}

    valid_scores  = scores[valid_mask]
    valid_tickers = [t for t, v in zip(tickers, valid_mask) if v]

    # Tertile thresholds on this window's scores
    t33 = np.percentile(valid_scores, 33.33)
    t67 = np.percentile(valid_scores, 66.67)

    result = {}
    for ticker, score in zip(valid_tickers, valid_scores):
        if score <= t33:
            result[ticker] = "conservative"
        elif score <= t67:
            result[ticker] = "balanced"
        else:
            result[ticker] = "aggressive"

    return result


# =============================================================================
# MAIN INFERENCE RUNNER
# =============================================================================

def run_historical_inference(
    prices_dir:  Path = DEFAULT_PRICES_DIR,
    agent_path:  Path = DEFAULT_AGENT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    start:       date = DEFAULT_START,
    end:         Optional[date] = None,
    dry_run:     bool = False,
) -> pd.DataFrame:

    if end is None:
        end = date.today()

    log.info("=" * 60)
    log.info("EGXAI Asset Selector — Historical Inference")
    log.info(f"  Date range : {start} → {end}")
    log.info(f"  Prices dir : {prices_dir}")
    log.info(f"  Agent path : {agent_path}")
    log.info(f"  Output     : {output_path}")
    log.info("=" * 60)

    # ── Step 1: OHLCV ─────────────────────────────────────────────────────────
    ohlcv      = load_ohlcv(OUR_TICKERS, start=start, end=end, prices_dir=prices_dir)
    tickers    = sorted(ohlcv.columns.get_level_values("ticker").unique().tolist())
    date_index = ohlcv.index
    log.info(f"Tickers: {len(tickers)}")

    # ── Step 2: Environment ────────────────────────────────────────────────────
    log.info("Building environment...")
    env = AssetSelectorEnv(
        ohlcv         = ohlcv,
        lookback      = LOOKBACK,
        forward       = FORWARD,
        step_size     = STEP_SIZE,
        n_clusters    = N_CLUSTERS,
        train_end_idx = None,
    )
    log.info(f"  feature_dim={env.feature_dim}, "
             f"start_idx={env._start_idx}, "
             f"full_end_idx={env._full_end_idx}")

    # ── Step 3: Load agent ────────────────────────────────────────────────────
    if not agent_path.exists():
        raise FileNotFoundError(f"rl_agent.pt not found: {agent_path}")
    agent = RLAssetSelectorAgent(
        feature_dim   = env.feature_dim,
        hidden_dims   = AGENT_HIDDEN_DIMS,
        lr            = 3e-4,
        gamma         = 0.0,
        gae_lambda    = 0.0,
        entropy_coeff = 0.0,
        device        = "cpu",
    )
    agent.load(agent_path)
    agent.actor.eval()
    agent.critic.eval()
    log.info(f"Agent loaded from {agent_path}")

    # ── Step 4: Per-window inference using collect_all_scores() ───────────────
    # This runs inference on every 21-day window individually without any
    # semi-annual bucket validity check — so early sparse windows are included.
    log.info("Running per-window inference (collect_all_scores)...")

    (_, score_matrix, _, _) = agent.collect_all_scores(
        env,
        start_idx = env._start_idx,
        end_idx   = env._full_end_idx,
    )
    # score_matrix shape: (n_windows, n_tickers)
    # Each row = one 21-day window's scores for all tickers

    # Build window date index: the date at the START of each window
    window_dates = [
        env.prices.index[i]
        for i in range(env._start_idx, env._full_end_idx, env.step_size)
        if i < len(env.prices)
    ]
    n_windows = min(len(window_dates), score_matrix.shape[0])
    window_dates  = window_dates[:n_windows]
    score_matrix  = score_matrix[:n_windows]

    log.info(f"Windows: {n_windows} total, "
             f"{window_dates[0].date()} → {window_dates[-1].date()}")

    # ── Step 5: Classify each window independently via tertile split ──────────
    log.info("Applying per-window tertile classification...")

    window_classifications = []   # list of (date, {ticker: profile})
    n_skipped = 0

    for i, (w_date, scores) in enumerate(zip(window_dates, score_matrix)):
        profiles = scores_to_profiles(scores, tickers)
        if not profiles:
            n_skipped += 1
            continue
        window_classifications.append((w_date, profiles))

    log.info(f"  Classified: {len(window_classifications)} windows, "
             f"skipped: {n_skipped} (< {MIN_VALID_TICKERS} valid scores)")

    if not window_classifications:
        raise ValueError(
            "No windows could be classified. Check OHLCV data quality."
        )

    # ── Step 6: Build sparse window-level DataFrame ───────────────────────────
    # One row per classified window, then forward-fill to daily
    sparse_index = pd.DatetimeIndex([d for d, _ in window_classifications])
    sparse_df    = pd.DataFrame(
        index   = sparse_index,
        columns = tickers,
        dtype   = object,
    )

    for w_date, profiles in window_classifications:
        for ticker, profile in profiles.items():
            sparse_df.loc[w_date, ticker] = profile

    log.info(f"Sparse window DataFrame: {sparse_df.shape}")

    # ── Step 7: Forward-fill to daily ─────────────────────────────────────────
    # Reindex to full daily date range, forward-fill within each ticker.
    # Dates before the first classified window remain NaN.
    daily_df = sparse_df.reindex(date_index)
    daily_df = daily_df.ffill()

    # Dates before first valid window stay NaN (correct — no history yet)
    first_window_date = sparse_index[0]
    daily_df.loc[daily_df.index < first_window_date] = np.nan

    # Reorder columns to match OUR_TICKERS order
    available_cols = [t for t in OUR_TICKERS if t in daily_df.columns]
    daily_df = daily_df[available_cols]

    # ── Step 8: Report coverage ────────────────────────────────────────────────
    n_valid = daily_df.notna().sum().sum()
    n_total = daily_df.size
    log.info(f"Daily coverage: {n_valid/n_total*100:.1f}% "
             f"({n_valid:,} / {n_total:,} stock-days classified)")
    log.info(f"First classified date: {first_window_date.date()}")

    for profile in ("conservative", "balanced", "aggressive"):
        count = (daily_df.values == profile).sum()
        log.info(f"  {profile:<14}: {count:,} stock-days ({count/n_total*100:.1f}%)")

    log.info("Coverage by year:")
    for yr in sorted(daily_df.index.year.unique()):
        yr_mask    = daily_df.index.year == yr
        yr_df      = daily_df.loc[yr_mask]
        classified = yr_df.notna().sum().sum()
        total      = yr_df.size
        log.info(f"  {yr}: {classified:,}/{total:,} ({classified/total*100:.1f}%)")

    # ── Step 9: Save ──────────────────────────────────────────────────────────
    if dry_run:
        log.info("DRY RUN — not writing output")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(output_path)
        log.info(f"Saved: {output_path}  ({daily_df.shape})")

    # ── Step 10: Validate for notebook 01 ────────────────────────────────────
    log.info("=" * 60)
    log.info("VALIDATION")
    log.info(f"  Shape         : {daily_df.shape}")
    log.info(f"  Index dtype   : {daily_df.index.dtype}")
    log.info(f"  Columns       : {daily_df.columns[:5].tolist()}...")
    first_valid_row = daily_df.dropna(how="all").iloc[0]
    log.info(f"  First valid row ({daily_df.dropna(how='all').index[0].date()}):")
    for t, v in first_valid_row.items():
        log.info(f"    {t}: {v}")
    log.info("=" * 60)

    return daily_df


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Asset selector historical inference (no retraining)"
    )
    parser.add_argument("--prices-dir",  type=Path, default=DEFAULT_PRICES_DIR)
    parser.add_argument("--agent-path",  type=Path, default=DEFAULT_AGENT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--end",   type=date.fromisoformat, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug",   action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        df = run_historical_inference(
            prices_dir  = args.prices_dir,
            agent_path  = args.agent_path,
            output_path = args.output_path,
            start       = args.start,
            end         = args.end,
            dry_run     = args.dry_run,
        )
        print(f"\nDone. Shape: {df.shape}")
        print(f"\nFirst 3 classified rows (first 5 tickers):")
        first3 = df.dropna(how="all").head(3)
        print(first3.iloc[:, :5].to_string())
    except Exception as e:
        log.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)