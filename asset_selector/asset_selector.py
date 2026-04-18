"""
asset_selector.py
=================
Classifies EGX30 tickers into three risk profiles — conservative, balanced,
aggressive — using a Reinforcement Learning agent trained on rolling OHLCV
feature windows.

Why RL over GMM / K-Means
--------------------------
Static clustering on a single snapshot of risk features ignores the fact
that a stock's volatility regime can change multiple times over 2020-2025
(COVID crash, recovery rally, inflation shock, etc.).

The RL agent addresses this by:
  1. Operating on rolling windows — each window captures the *current*
     volatility regime rather than a fixed historical average.
  2. Learning from delayed feedback — the reward signal is the Spearman
     rank correlation between the agent's risk ranking and the *actual*
     forward realised volatility, so the agent learns what feature patterns
     *predict* future risk, not just describe historical risk.
  3. Producing a dynamic time-series of classifications (saved separately)
     alongside the final aggregated static assignment.

Pipeline integration
--------------------
classify_assets() returns a DataFrame consumed by the visualiser and JSON
export — columns: ticker | volatility | mean_return | cluster_id |
risk_profile | rl_risk_score.

Public API
----------
    result = classify_assets(
        prices,                 # pd.DataFrame from load_price_data()
        n_clusters=3,
        n_episodes=60,
        output_dir=None,        # optional: save model + dynamic scores
    )
    # Returns pd.DataFrame:  ticker | volatility | mean_return |
    #                         cluster_id | risk_profile | rl_risk_score

    tickers = get_profile_tickers(result, 'aggressive')
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from asset_selector.rl_environment import AssetSelectorEnv
from asset_selector.rl_agent import RLAssetSelectorAgent

logger = logging.getLogger(__name__)

RISK_LABELS: Dict[int, str] = {0: "conservative", 1: "balanced", 2: "aggressive"}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def classify_assets(
    prices: pd.DataFrame,
    n_clusters: int = 3,
    n_episodes: int = 150,
    lookback: int = 126,
    forward: int = 63,
    step_size: int = 21,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Train an RL agent on rolling OHLCV windows then classify each ticker
    into one of three risk profiles.

    Parameters
    ----------
    prices : pd.DataFrame
        Preprocessed close prices from load_price_data().
        DatetimeIndex, columns = ticker symbols.
    n_clusters : int
        Number of risk tiers.  Currently only 3 is supported.
    n_episodes : int
        Number of training episodes (full chronological sweeps).
    lookback : int
        Lookback window in trading days (~126 = 6 months).
    forward : int
        Forward window in trading days used to compute the reward signal
        (~63 = 3 months).
    step_size : int
        Days to advance between windows during training (~21 = 1 month).
    output_dir : str | None
        If given, saves:
          • <output_dir>/rl_agent.pt           – trained model weights
          • <output_dir>/rl_dynamic_scores.csv – per-window risk scores
          • <output_dir>/rl_training_history.csv

    Returns
    -------
    pd.DataFrame
        Columns: ticker | volatility | mean_return | cluster_id |
                 risk_profile | rl_risk_score
        Sorted by volatility ascending.
        Tickers excluded from RL (< 3 valid windows) receive risk_profile=None.
    """
    if n_clusters != 3:
        raise ValueError("RL asset selector currently supports n_clusters=3 only.")

    all_tickers = prices.columns.tolist()
    if not all_tickers:
        raise ValueError("No tickers in prices DataFrame.")

    # Separate tickers with sufficient price data from those with none.
    # Tickers with zero valid price rows cannot be scored by the RL agent —
    # including them as score=0 would silently pollute the conservative cluster.
    min_rows = lookback + forward + 1
    prices_clean = prices.replace(0.0, np.nan)
    data_tickers  = [
        t for t in all_tickers
        if prices_clean[t].notna().sum() >= min_rows
    ]
    no_data_tickers = [t for t in all_tickers if t not in data_tickers]

    if no_data_tickers:
        logger.warning(
            "%d tickers excluded from RL (insufficient price data): %s",
            len(no_data_tickers), no_data_tickers,
        )

    if not data_tickers:
        raise ValueError("No tickers have sufficient price data for RL training.")

    prices_aligned = prices[sorted(data_tickers)]

    logger.info(
        "RL asset selector: %d tickers  %d rows  "
        "lookback=%d forward=%d step=%d episodes=%d",
        len(data_tickers), len(prices_aligned),
        lookback, forward, step_size, n_episodes,
    )

    # ── Build environment ─────────────────────────────────────────────────────
    env = AssetSelectorEnv(
        prices    = prices_aligned,
        lookback  = lookback,
        forward   = forward,
        step_size = step_size,
        n_clusters= n_clusters,
    )

    # ── Build agent ───────────────────────────────────────────────────────────
    agent = RLAssetSelectorAgent(
        feature_dim   = env.feature_dim,
        hidden_dims   = [128, 64, 32],
        lr            = 3e-4,
        gamma         = 0.99,    # 0.995 → 0.99: tighter credit assignment
        entropy_coeff = 0.05,    # 0.01 → 0.05: maintain exploration longer
        device        = "cpu",
    )

    # ── Supervised pre-training ───────────────────────────────────────────────
    agent.pretrain(env, n_epochs=150, lr_pretrain=1e-3)

    # ── PPO fine-tuning ───────────────────────────────────────────────────────
    logger.info("PPO fine-tuning for %d episodes …", n_episodes)
    history = agent.train(env, n_episodes=n_episodes, log_every=10)

    mean_reward = np.mean([h["mean_reward"] for h in history[-10:]])
    logger.info(
        "Training complete. Mean reward (last 10 episodes): %.4f", mean_reward
    )

    # ── Inference: collect time-averaged risk scores and forward stats ───────
    logger.info("Collecting risk scores and forward stats across all windows …")
    mean_scores, score_matrix, fwd_vol_matrix, fwd_ret_matrix = agent.collect_all_scores(env)

    # ── Assign clusters ───────────────────────────────────────────────────────
    cluster_ids, risk_profiles = env.assign_clusters(mean_scores)

    # ── Build result DataFrame ────────────────────────────────────────────────
    # Per-ticker mean forward vol/return across all windows — these are the
    # exact quantities the RL model was trained to predict, so comparing
    # rl_risk_score against them is a meaningful evaluation.
    scored_arr = np.array(sorted(data_tickers))   # matches env.tickers order
    vol_values = np.nanmean(fwd_vol_matrix, axis=0)   # (n_tickers,)
    ret_values = np.nanmean(fwd_ret_matrix, axis=0)   # (n_tickers,)

    # Sort scored tickers by volatility ascending
    sort_order           = np.argsort(np.where(np.isfinite(vol_values), vol_values, np.inf))
    tickers_sorted       = scored_arr[sort_order]
    vol_sorted           = vol_values[sort_order]
    ret_sorted           = ret_values[sort_order]
    cluster_ids_sorted   = cluster_ids[sort_order]
    risk_profiles_sorted = risk_profiles[sort_order]
    mean_scores_sorted   = mean_scores[sort_order]

    risk_profile_final: List[Optional[str]] = [
        rp if rp != "" else None for rp in risk_profiles_sorted
    ]
    cluster_id_final: List[Optional[int]] = [
        int(cid) if cid != -1 else None for cid in cluster_ids_sorted
    ]

    result = pd.DataFrame({"ticker": tickers_sorted})
    result["volatility"]    = vol_sorted
    result["mean_return"]   = ret_sorted
    result["cluster_id"]    = cluster_id_final
    result["risk_profile"]  = risk_profile_final
    result["rl_risk_score"] = mean_scores_sorted

    # ── Append no-data tickers with risk_profile=None ─────────────────────────
    if no_data_tickers:
        nd_rows = pd.DataFrame({"ticker": no_data_tickers})
        nd_rows["volatility"]    = np.nan
        nd_rows["mean_return"]   = np.nan
        nd_rows["cluster_id"]    = None
        nd_rows["risk_profile"]  = None
        nd_rows["rl_risk_score"] = np.nan
        result = pd.concat([result, nd_rows], ignore_index=True)

    result = result.reset_index(drop=True)

    # ── Logging summary ───────────────────────────────────────────────────────
    for profile in ("conservative", "balanced", "aggressive"):
        grp = result[result["risk_profile"] == profile]
        if not grp.empty:
            logger.info(
                "  %-12s: %2d tickers  vol [%.3f, %.3f]  "
                "return [%.3f, %.3f]  rl_score [%.3f, %.3f]",
                profile, len(grp),
                grp["volatility"].min(), grp["volatility"].max(),
                grp["mean_return"].min(), grp["mean_return"].max(),
                grp["rl_risk_score"].min(), grp["rl_risk_score"].max(),
            )

    unclassified = result["risk_profile"].isna().sum()
    if unclassified:
        logger.warning(
            "%d tickers unclassified (insufficient data windows): %s",
            unclassified,
            result.loc[result["risk_profile"].isna(), "ticker"].tolist(),
        )

    # ── Optional outputs ──────────────────────────────────────────────────────
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Model weights
        agent.save(out / "rl_agent.pt")

        # Dynamic per-window scores and forward vols
        window_dates = [
            env.prices.index[min(i, len(env.prices) - 1)]
            for i in range(env._start_idx, env._end_idx, env.step_size)
        ]
        dynamic_df = pd.DataFrame(
            score_matrix,
            index   = window_dates[: len(score_matrix)],
            columns = sorted(data_tickers),
        )
        dynamic_path = out / "rl_dynamic_scores.csv"
        dynamic_df.to_csv(dynamic_path)
        logger.info("Dynamic risk scores saved to %s", dynamic_path)

        # Forward realized volatility matrix (actual vol the RL was trained to predict)
        fwd_vol_df = pd.DataFrame(
            fwd_vol_matrix,
            index   = window_dates[: len(fwd_vol_matrix)],
            columns = sorted(data_tickers),
        )
        fwd_vol_path = out / "rl_dynamic_fwd_vol.csv"
        fwd_vol_df.to_csv(fwd_vol_path)
        logger.info("Dynamic forward vol saved to %s", fwd_vol_path)

        # Training history
        history_df = pd.DataFrame(history)
        history_path = out / "rl_training_history.csv"
        history_df.to_csv(history_path, index=False)
        logger.info("Training history saved to %s", history_path)

    return result


def get_profile_tickers(
    classification: pd.DataFrame,
    profile: str,
) -> list[str]:
    """
    Return tickers belonging to a given risk profile.

    Parameters
    ----------
    classification : pd.DataFrame
        Output of classify_assets().
    profile : str
        One of 'conservative', 'balanced', 'aggressive'.

    Returns
    -------
    list[str]
        Sorted list of ticker symbols.
    """
    valid_profiles = {"conservative", "balanced", "aggressive"}
    if profile not in valid_profiles:
        raise ValueError(f"profile must be one of {valid_profiles}")
    mask = classification["risk_profile"] == profile
    return sorted(classification.loc[mask, "ticker"].tolist())
