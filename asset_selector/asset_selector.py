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
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
    train_end: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train an RL agent on rolling OHLCV windows then classify each ticker
    into one of three risk profiles, evaluated quarterly.

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
          • <output_dir>/rl_agent.pt                    – trained model weights
          • <output_dir>/rl_dynamic_scores.csv          – per-window risk scores
          • <output_dir>/rl_dynamic_fwd_vol.csv         – per-window forward vol
          • <output_dir>/rl_training_history.csv        – training history
          • <output_dir>/quarterly_classifications.csv  – quarterly labels
          • <output_dir>/evaluation.csv                 – Sharpe/vol evaluation
    train_end : str | None
        ISO date string (e.g. "2022-12-31") for a temporal train/test split.
        Windows whose lookback ends at or before this date are used for
        training (pre-training + PPO).  Windows after this date are held out
        as a test set: the agent runs inference on them but never trains on
        them.  quarterly_df and eval_df gain a ``split`` column ("train" /
        "test").  When None (default) the full dataset is used for training
        (backward-compatible behaviour).

    Returns
    -------
    quarterly_df : pd.DataFrame
        Columns: quarter_start | ticker | rl_risk_score | risk_profile
        One row per (quarter, ticker) for all RL-scored tickers.
    eval_df : pd.DataFrame
        Columns: quarter_start | ticker | risk_profile | actual_sharpe |
                 actual_fwd_vol | classification_correct
        classification_correct (float 0/1/NaN): whether the ticker's actual
        forward volatility rank was consistent with its label tertile.
    static_df : pd.DataFrame
        Columns: ticker | volatility | mean_return | cluster_id |
                 risk_profile | rl_risk_score
        One row per ticker, sorted by volatility ascending.
        risk_profile is the dominant (most frequent) quarterly label.
        For use by the downstream visualiser.
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

    # ── Temporal train/test split ─────────────────────────────────────────────
    train_end_idx: Optional[int] = None
    if train_end is not None:
        split_ts  = pd.Timestamp(train_end)
        split_pos = prices_aligned.index.searchsorted(split_ts, side="right") - 1
        if 0 <= split_pos < len(prices_aligned):
            train_end_idx = int(split_pos)
            train_date    = prices_aligned.index[train_end_idx].strftime("%Y-%m-%d")
            test_start_pos = min(train_end_idx + 1, len(prices_aligned) - 1)
            test_date      = prices_aligned.index[test_start_pos].strftime("%Y-%m-%d")
            logger.info(
                "Temporal split: TRAIN through %s (row %d) | TEST from %s onward",
                train_date, train_end_idx, test_date,
            )
        else:
            logger.warning(
                "train_end %s is outside the price data range — ignoring split.", train_end
            )

    # ── Build environment ─────────────────────────────────────────────────────
    env = AssetSelectorEnv(
        prices        = prices_aligned,
        lookback      = lookback,
        forward       = forward,
        step_size     = step_size,
        n_clusters    = n_clusters,
        train_end_idx = train_end_idx,
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

    # ── Inference 1: per-window scores (train period) ─────────────────────────
    logger.info("Collecting risk scores on TRAIN windows …")
    (train_mean_scores,
     train_score_matrix,
     train_fwd_vol_matrix,
     train_fwd_ret_matrix) = agent.collect_all_scores(
        env, start_idx=env._start_idx, end_idx=env._end_idx
    )

    # ── Inference 1b: per-window scores (test period, if split set) ───────────
    if env._test_start_idx is not None and env._test_start_idx < env._full_end_idx:
        logger.info("Collecting risk scores on TEST windows (held-out) …")
        (test_mean_scores,
         test_score_matrix,
         test_fwd_vol_matrix,
         test_fwd_ret_matrix) = agent.collect_all_scores(
            env, start_idx=env._test_start_idx, end_idx=env._full_end_idx
        )
        # Combine for dynamic CSV outputs and static_df aggregation
        score_matrix   = np.concatenate([train_score_matrix,   test_score_matrix],   axis=0)
        fwd_vol_matrix = np.concatenate([train_fwd_vol_matrix, test_fwd_vol_matrix], axis=0)
        fwd_ret_matrix = np.concatenate([train_fwd_ret_matrix, test_fwd_ret_matrix], axis=0)
        mean_scores    = np.nanmean(score_matrix, axis=0)
        valid_counts   = np.isfinite(score_matrix).sum(axis=0)
        mean_scores[valid_counts < 3] = np.nan
    else:
        score_matrix   = train_score_matrix
        fwd_vol_matrix = train_fwd_vol_matrix
        fwd_ret_matrix = train_fwd_ret_matrix
        mean_scores    = train_mean_scores

    # ── Inference 2: quarterly scores (train + test separately) ──────────────
    logger.info("Collecting quarterly risk scores (TRAIN) …")
    train_quarterly = agent.collect_quarterly_scores(
        env, start_idx=env._start_idx, end_idx=env._end_idx
    )
    for q in train_quarterly:
        q["split"] = "train"

    if env._test_start_idx is not None and env._test_start_idx < env._full_end_idx:
        logger.info("Collecting quarterly risk scores (TEST) …")
        test_quarterly = agent.collect_quarterly_scores(
            env, start_idx=env._test_start_idx, end_idx=env._full_end_idx
        )
        for q in test_quarterly:
            q["split"] = "test"
    else:
        test_quarterly = []

    quarterly_data = train_quarterly + test_quarterly

    # ── Build quarterly_df ────────────────────────────────────────────────────
    quarterly_rows: List[Dict] = []
    for q_data in quarterly_data:
        for i, ticker in enumerate(env.tickers):
            score = q_data["mean_scores"][i]
            rp    = q_data["risk_profiles"][i]
            quarterly_rows.append({
                "quarter_start":  q_data["quarter_start"],
                "ticker":         ticker,
                "rl_risk_score":  float(score) if np.isfinite(score) else np.nan,
                "risk_profile":   rp if rp != "" else None,
                "split":          q_data.get("split", "train"),
            })
    quarterly_df = pd.DataFrame(quarterly_rows)

    logger.info(
        "Quarterly classifications: %d quarters × %d tickers = %d rows",
        len(quarterly_data), env.n_tickers, len(quarterly_df),
    )

    # ── Build eval_df (actual Sharpe, forward vol, classification_correct) ────
    eval_rows: List[Dict] = []
    spearman_stats: List[Dict] = []
    _label_to_rank = {"conservative": 0, "balanced": 1, "aggressive": 2}

    for q_data in quarterly_data:
        q_start       = q_data["quarter_start"]
        q_idx         = q_data["quarter_idx"]
        risk_profiles = q_data["risk_profiles"]   # (n_tickers,) str array

        actual_sharpe  = env.compute_sharpe(q_idx, env.forward)           # (n_tickers,)
        actual_fwd_vol = env._compute_forward_vol(q_idx)                   # (n_tickers,)

        has_profile = np.array([rp != "" for rp in risk_profiles])
        has_vol     = np.isfinite(actual_fwd_vol)
        valid_mask  = has_profile & has_vol
        n_valid     = int(valid_mask.sum())

        # ── Per-quarter Spearman ρ ─────────────────────────────────────────
        if n_valid >= 3:
            label_ranks = np.array(
                [_label_to_rank[risk_profiles[i]] for i in range(env.n_tickers) if valid_mask[i]],
                dtype=float,
            )
            vol_vals = actual_fwd_vol[valid_mask]
            rho_val, _ = spearmanr(label_ranks, vol_vals)
            rho = float(rho_val) if np.isfinite(rho_val) else np.nan
        else:
            rho = np.nan

        spearman_stats.append({
            "quarter_start":    q_start,
            "spearman_rho":     rho,
            "n_valid_tickers":  n_valid,
            "split":            q_data.get("split", "train"),
        })

        # ── classification_correct (vol-rank tertile match) ────────────────
        if n_valid >= 3:
            valid_indices = np.where(valid_mask)[0]
            vol_for_valid = actual_fwd_vol[valid_indices]
            # Rank from 1 (lowest vol) to n_valid (highest vol)
            vol_ranks_local = pd.Series(vol_for_valid).rank(method="average").values
            third = n_valid / 3.0
            rank_map: Dict[int, float] = {
                int(valid_indices[j]): vol_ranks_local[j] for j in range(n_valid)
            }

            def _expected(rank: float) -> str:
                if rank <= third:
                    return "conservative"
                elif rank <= 2 * third:
                    return "balanced"
                else:
                    return "aggressive"
        else:
            rank_map = {}

        for i, ticker in enumerate(env.tickers):
            if valid_mask[i] and rank_map:
                exp = _expected(rank_map[i])
                correct: float = 1.0 if risk_profiles[i] == exp else 0.0
            else:
                correct = np.nan

            eval_rows.append({
                "quarter_start":          q_start,
                "ticker":                 ticker,
                "risk_profile":           risk_profiles[i] if risk_profiles[i] != "" else None,
                "actual_sharpe":          float(actual_sharpe[i])  if np.isfinite(actual_sharpe[i])  else np.nan,
                "actual_fwd_vol":         float(actual_fwd_vol[i]) if np.isfinite(actual_fwd_vol[i]) else np.nan,
                "classification_correct": correct,
                "split":                  q_data.get("split", "train"),
            })

    eval_df = pd.DataFrame(eval_rows)

    # ── Summary statistics ────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("QUARTERLY CLASSIFICATION EVALUATION SUMMARY")
    logger.info("=" * 55)
    logger.info("Per-quarter Spearman ρ  (label rank vs actual forward vol):")
    for row in spearman_stats:
        rho_str = f"{row['spearman_rho']:.3f}" if np.isfinite(row["spearman_rho"]) else "N/A"
        logger.info(
            "  %s  n_tickers=%2d  Spearman ρ = %s",
            row["quarter_start"].strftime("%Y-%m-%d"),
            row["n_valid_tickers"],
            rho_str,
        )

    rho_vals = [r["spearman_rho"] for r in spearman_stats if np.isfinite(r["spearman_rho"])]
    if rho_vals:
        logger.info(
            "Spearman ρ across all quarters:  mean=%.3f  std=%.3f  (n=%d quarters)",
            float(np.mean(rho_vals)), float(np.std(rho_vals)), len(rho_vals),
        )

    train_rhos = [
        r["spearman_rho"] for r in spearman_stats
        if r.get("split") == "train" and np.isfinite(r["spearman_rho"])
    ]
    test_rhos = [
        r["spearman_rho"] for r in spearman_stats
        if r.get("split") == "test" and np.isfinite(r["spearman_rho"])
    ]
    if train_rhos:
        logger.info(
            "  TRAIN Spearman ρ: mean=%.3f  std=%.3f  (n=%d quarters)",
            float(np.mean(train_rhos)), float(np.std(train_rhos)), len(train_rhos),
        )
    if test_rhos:
        logger.info(
            "  TEST  Spearman ρ: mean=%.3f  std=%.3f  (n=%d quarters)  ← held-out",
            float(np.mean(test_rhos)), float(np.std(test_rhos)), len(test_rhos),
        )

    logger.info("Mean actual Sharpe and forward vol per risk profile (all quarters):")
    for profile in ("conservative", "balanced", "aggressive"):
        grp = eval_df[eval_df["risk_profile"] == profile]
        if not grp.empty:
            logger.info(
                "  %-12s: mean_sharpe=%+.3f  mean_fwd_vol=%.3f",
                profile,
                float(grp["actual_sharpe"].mean()),
                float(grp["actual_fwd_vol"].mean()),
            )

    valid_correct = eval_df["classification_correct"].dropna()
    if not valid_correct.empty:
        logger.info(
            "Overall classification accuracy (vol-rank tertile): %.1f%%",
            float(valid_correct.mean()) * 100,
        )
    if "split" in eval_df.columns:
        for sp in ("train", "test"):
            sp_correct = eval_df.loc[eval_df["split"] == sp, "classification_correct"].dropna()
            if not sp_correct.empty:
                label = "TRAIN" if sp == "train" else "TEST (held-out)"
                logger.info(
                    "  %s accuracy: %.1f%%", label, float(sp_correct.mean()) * 100
                )

    # ── Build static_df (backward-compat with visualiser) ─────────────────────
    # risk_profile = dominant (most-frequent) quarterly label per ticker.
    scored_arr = np.array(sorted(data_tickers))   # matches env.tickers order

    ticker_to_quarterly_profiles: Dict[str, List[str]] = {t: [] for t in env.tickers}
    for q_data in quarterly_data:
        for i, t in enumerate(env.tickers):
            rp = q_data["risk_profiles"][i]
            if rp != "":
                ticker_to_quarterly_profiles[t].append(rp)

    _profile_to_id = {"conservative": 0, "balanced": 1, "aggressive": 2}
    dominant_cluster_ids   = np.full(env.n_tickers, -1,  dtype=int)
    dominant_risk_profiles = np.full(env.n_tickers, "",  dtype=object)

    for i, t in enumerate(env.tickers):
        profiles = ticker_to_quarterly_profiles[t]
        if profiles:
            dominant = Counter(profiles).most_common(1)[0][0]
            dominant_risk_profiles[i] = dominant
            dominant_cluster_ids[i]   = _profile_to_id[dominant]

    vol_values = np.nanmean(fwd_vol_matrix, axis=0)   # (n_tickers,)
    ret_values = np.nanmean(fwd_ret_matrix, axis=0)   # (n_tickers,)

    sort_order           = np.argsort(np.where(np.isfinite(vol_values), vol_values, np.inf))
    tickers_sorted       = scored_arr[sort_order]
    vol_sorted           = vol_values[sort_order]
    ret_sorted           = ret_values[sort_order]
    cluster_ids_sorted   = dominant_cluster_ids[sort_order]
    risk_profiles_sorted = dominant_risk_profiles[sort_order]
    mean_scores_sorted   = mean_scores[sort_order]

    static_df = pd.DataFrame({"ticker": tickers_sorted})
    static_df["volatility"]    = vol_sorted
    static_df["mean_return"]   = ret_sorted
    static_df["cluster_id"]    = [int(c) if c != -1 else None for c in cluster_ids_sorted]
    static_df["risk_profile"]  = [rp if rp != "" else None for rp in risk_profiles_sorted]
    static_df["rl_risk_score"] = mean_scores_sorted

    if no_data_tickers:
        nd_rows = pd.DataFrame({"ticker": no_data_tickers})
        nd_rows["volatility"]    = np.nan
        nd_rows["mean_return"]   = np.nan
        nd_rows["cluster_id"]    = None
        nd_rows["risk_profile"]  = None
        nd_rows["rl_risk_score"] = np.nan
        static_df = pd.concat([static_df, nd_rows], ignore_index=True)

    static_df = static_df.reset_index(drop=True)

    # ── Logging summary (static view) ─────────────────────────────────────────
    for profile in ("conservative", "balanced", "aggressive"):
        grp = static_df[static_df["risk_profile"] == profile]
        if not grp.empty:
            logger.info(
                "  %-12s: %2d tickers  vol [%.3f, %.3f]  "
                "return [%.3f, %.3f]  rl_score [%.3f, %.3f]",
                profile, len(grp),
                grp["volatility"].min(), grp["volatility"].max(),
                grp["mean_return"].min(), grp["mean_return"].max(),
                grp["rl_risk_score"].min(), grp["rl_risk_score"].max(),
            )

    unclassified = static_df["risk_profile"].isna().sum()
    if unclassified:
        logger.warning(
            "%d tickers unclassified (insufficient data windows): %s",
            unclassified,
            static_df.loc[static_df["risk_profile"].isna(), "ticker"].tolist(),
        )

    # ── Optional outputs ──────────────────────────────────────────────────────
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Model weights
        agent.save(out / "rl_agent.pt")

        # Dynamic per-window scores and forward vols (full range: train + test)
        window_dates = [
            env.prices.index[min(i, len(env.prices) - 1)]
            for i in range(env._start_idx, env._full_end_idx, env.step_size)
        ]
        dynamic_df = pd.DataFrame(
            score_matrix,
            index   = window_dates[: len(score_matrix)],
            columns = sorted(data_tickers),
        )
        dynamic_path = out / "rl_dynamic_scores.csv"
        dynamic_df.to_csv(dynamic_path)
        logger.info("Dynamic risk scores saved to %s", dynamic_path)

        fwd_vol_file = pd.DataFrame(
            fwd_vol_matrix,
            index   = window_dates[: len(fwd_vol_matrix)],
            columns = sorted(data_tickers),
        )
        fwd_vol_path = out / "rl_dynamic_fwd_vol.csv"
        fwd_vol_file.to_csv(fwd_vol_path)
        logger.info("Dynamic forward vol saved to %s", fwd_vol_path)

        # Training history
        history_df = pd.DataFrame(history)
        history_path = out / "rl_training_history.csv"
        history_df.to_csv(history_path, index=False)
        logger.info("Training history saved to %s", history_path)

        # Quarterly classifications
        q_path = out / "quarterly_classifications.csv"
        quarterly_df.to_csv(q_path, index=False)
        logger.info("Quarterly classifications saved to %s", q_path)

        # Evaluation results
        e_path = out / "evaluation.csv"
        eval_df.to_csv(e_path, index=False)
        logger.info("Evaluation results saved to %s", e_path)

        # Per-quarter Spearman summary
        spearman_df = pd.DataFrame(spearman_stats)
        if rho_vals:
            spearman_df.attrs["mean_spearman"] = float(np.mean(rho_vals))
            spearman_df.attrs["std_spearman"]  = float(np.std(rho_vals))
        spearman_path = out / "quarterly_spearman.csv"
        spearman_df.to_csv(spearman_path, index=False)
        logger.info("Per-quarter Spearman saved to %s", spearman_path)

    return quarterly_df, eval_df, static_df


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
