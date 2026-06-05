"""
run_inference.py

Loads a saved rl_agent.pt and runs inference on full historical data
from 2015, producing daily risk classifications for all stocks.

No retraining — pure inference only.
Scheduled via task scheduler alongside other ingestion scripts.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from asset_selector.rl_agent import RLAssetSelectorAgent
from asset_selector.rl_environment import AssetSelectorEnv

# ── Configuration ─────────────────────────────────────────────────────────────
AGENT_PATH   = _ROOT / "asset_selector/output/rl_agent.pt"
OHLCV_DIR    = _ROOT / "data/raw/OHLCV"
OUTPUT_PATH  = _ROOT / "outputs/volatility_clustering/daily_risk_classifications.csv"
METADATA_DIR = _ROOT / "data/processed/metadata"

START_DATE   = "2015-01-01"
END_DATE     = "2026-03-31"

# Must match exactly what the agent was trained with
HIDDEN_DIMS  = [128, 64, 32]
LOOKBACK     = 126
FORWARD      = 126
STEP_SIZE    = 21
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt  = "%H:%M:%S",
    handlers = [logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_ohlcv_from_csvs(ohlcv_dir: Path, start: str, end: str) -> pd.DataFrame:
    """
    Loads all ticker CSVs from ohlcv_dir and returns a MultiIndex DataFrame
    with columns (ticker, price_type) as expected by AssetSelectorEnv.
    """
    price_types = ["open", "high", "low", "close", "volume"]
    ticker_dfs  = {}

    csv_files = sorted(ohlcv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {ohlcv_dir}")

    logger.info("Loading %d ticker CSVs from %s...", len(csv_files), ohlcv_dir)

    for fp in csv_files:
        ticker = fp.stem.upper()
        try:
            df = pd.read_csv(fp)
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.dropna(subset=["datetime"])
            df = df.sort_values("datetime").drop_duplicates(
                subset=["datetime"], keep="last"
            )
            df = df.set_index("datetime")
            df.index = df.index.normalize()

            # Filter to date range
            df = df.loc[start:end]
            if df.empty:
                logger.warning("%s: no data in range %s-%s, skipping", ticker, start, end)
                continue

            # Keep only needed columns
            available = {col.lower(): col for col in df.columns}
            ticker_data = {}
            for pt in price_types:
                if pt in available:
                    ticker_data[pt] = pd.to_numeric(
                        df[available[pt]], errors="coerce"
                    )
                else:
                    ticker_data[pt] = pd.Series(np.nan, index=df.index)

            ticker_dfs[ticker] = pd.DataFrame(ticker_data, index=df.index)

        except Exception as e:
            logger.warning("Failed to load %s: %s", ticker, e)
            continue

    if not ticker_dfs:
        raise ValueError("No ticker data loaded successfully.")

    # Build common date index from union of all dates
    all_dates = sorted(set().union(*[set(df.index) for df in ticker_dfs.values()]))
    date_index = pd.DatetimeIndex(all_dates)

    # Reindex all tickers to common date index
    for ticker in ticker_dfs:
        ticker_dfs[ticker] = ticker_dfs[ticker].reindex(date_index)

    # Assemble MultiIndex DataFrame: columns = (ticker, price_type)
    ohlcv = pd.concat(ticker_dfs, axis=1)
    ohlcv.columns.names = ["ticker", "price_type"]

    logger.info(
        "OHLCV built: %d tickers  %d rows  (%s → %s)",
        len(ticker_dfs), len(date_index),
        date_index[0].date(), date_index[-1].date(),
    )
    return ohlcv


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load OHLCV from CSVs
    ohlcv = load_ohlcv_from_csvs(OHLCV_DIR, start=START_DATE, end=END_DATE)

    # 2. Build environment
    logger.info("Building environment...")
    env = AssetSelectorEnv(
        ohlcv         = ohlcv,
        lookback      = LOOKBACK,
        forward       = FORWARD,
        step_size     = STEP_SIZE,
        n_clusters    = 3,
        train_end_idx = None,
    )
    logger.info(
        "Env: %d tickers  feature_dim=%d  windows from idx %d to %d",
        env.n_tickers, env.feature_dim, env._start_idx, env._full_end_idx,
    )

    # 3. Load saved agent
    logger.info("Loading agent from %s...", AGENT_PATH)
    agent = RLAssetSelectorAgent(
        feature_dim   = env.feature_dim,
        hidden_dims   = HIDDEN_DIMS,
        lr            = 3e-4,
        gamma         = 0.0,
        gae_lambda    = 0.0,
        entropy_coeff = 0.0,
        device        = "cpu",
    )
    agent.load(AGENT_PATH)
    agent.actor.eval()
    logger.info("Agent loaded successfully.")

    # 4. Run inference over full date range
    logger.info("Running semi-annual inference over full history...")
    period_data = agent.collect_period_scores(
        env,
        start_idx  = env._start_idx,
        end_idx    = env._full_end_idx,
        thresholds = None,
    )
    logger.info("Got %d semi-annual periods.", len(period_data))

    # 5. Build semi-annual DataFrame
    rows = []
    for q in period_data:
        for i, ticker in enumerate(env.tickers):
            rp = q["risk_profiles"][i]
            rows.append({
                "semiannual_start": q["semiannual_start"],
                "ticker":           ticker,
                "rl_risk_score":    float(q["mean_scores"][i])
                                    if np.isfinite(q["mean_scores"][i]) else np.nan,
                "risk_profile":     rp if rp != "" else None,
            })

    semi_df = pd.DataFrame(rows)
    semi_df["semiannual_start"] = pd.to_datetime(semi_df["semiannual_start"])
    logger.info(
        "Semi-annual classifications: %d rows  periods: %s → %s",
        len(semi_df),
        semi_df["semiannual_start"].min().date(),
        semi_df["semiannual_start"].max().date(),
    )

    # Save semi-annual for reference
    semi_out = OUTPUT_PATH.parent / "semi_annual_classifications_full.csv"
    semi_df.to_csv(semi_out, index=False)
    logger.info("Semi-annual classifications saved to %s", semi_out)

    # 6. Load RL date index
    logger.info("Loading RL date index...")
    with open(METADATA_DIR / "date_index.json") as f:
        date_index = pd.to_datetime(json.load(f))
    logger.info(
        "RL date index: %s → %s  (%d days)",
        date_index[0].date(), date_index[-1].date(), len(date_index),
    )

    # 7. Pivot and forward-fill to daily
    logger.info("Forward-filling to daily...")
    semi_pivot = (
        semi_df
        .dropna(subset=["risk_profile"])
        .pivot(index="semiannual_start", columns="ticker", values="risk_profile")
    )

    daily_classifications = (
        semi_pivot
        .reindex(date_index)
        .ffill()
        .bfill()
    )

    # 8. Sanity check
    n_nan = daily_classifications.isna().sum().sum()
    if n_nan > 0:
        logger.warning(
            "%d NaN cells remaining after ffill+bfill — "
            "tickers with insufficient data in some periods.",
            n_nan,
        )
    else:
        logger.info("Full coverage — no NaN cells.")

    logger.info(
        "Daily classifications shape: %s  (days × tickers)",
        daily_classifications.shape,
    )

    # 9. Save
    daily_classifications.to_csv(OUTPUT_PATH)
    logger.info("Saved to %s", OUTPUT_PATH)

    # 10. Summary
    logger.info("\n── Profile distribution (most recent date) ──")
    latest = daily_classifications.iloc[-1]
    for profile in ("conservative", "balanced", "aggressive"):
        tickers_in_profile = latest[latest == profile].index.tolist()
        logger.info(
            "  %-12s (%2d): %s",
            profile, len(tickers_in_profile), tickers_in_profile,
        )

    return daily_classifications


if __name__ == "__main__":
    main()