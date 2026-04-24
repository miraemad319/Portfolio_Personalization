"""
asset_selector.py

Classifies EGX30 tickers into three risk tiers — conservative, balanced,
aggressive — using a PPO agent trained on rolling OHLCV feature windows.

Design:
The RL agent observes 11 features computed from the past 126 trading days
for every stock and outputs a continuous risk score per stock.  The reward
signal is the Spearman rank correlation between the agent's scores and the
actual forward 63-day realised volatility + max drawdown.  The agent learns
to rank stocks by future risk without ever seeing future data in its features.

Output files

quarterly_classifications.csv  — one row per (quarter × ticker)
                                  columns: quarter_start | ticker |
                                  rl_risk_score | risk_profile | split
evaluation.csv                 — same + actual_sharpe | actual_fwd_vol |
                                  classification_correct | split
quarterly_spearman.csv         — per-quarter Spearman ρ with split label
risk_profiles.json             — CURRENT labels (most recent quarter),
                                  consumed by downstream portfolio models
asset_classification.csv       — same current labels, for the visualiser
rl_dynamic_scores.csv          — per-window RL scores (full history)
rl_dynamic_fwd_vol.csv         — per-window forward vol (full history)
rl_training_history.csv        — per-episode training stats
rl_agent.pt                    — trained model weights

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from asset_selector.rl_environment import AssetSelectorEnv
from asset_selector.rl_agent import RLAssetSelectorAgent

logger = logging.getLogger(__name__)

# Public API

def classify_assets(
    prices:     pd.DataFrame,
    n_clusters: int            = 3,
    n_episodes: int            = 150,
    lookback:   int            = 126,
    forward:    int            = 63,
    step_size:  int            = 21,
    train_end:  Optional[str]  = "2023-12-31",
    output_dir: Optional[str]  = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if n_clusters != 3:
        raise ValueError("classify_assets currently supports n_clusters=3 only.")

    all_tickers = prices.columns.tolist()
    if not all_tickers:
        raise ValueError("prices DataFrame has no columns.")

    # filter tickers with sufficient data 
    # A ticker needs at least lookback + forward + 1 valid rows to contribute
    min_rows        = lookback + forward + 1
    prices_clean    = prices.replace(0.0, np.nan)
    data_tickers    = [
        t for t in all_tickers
        if prices_clean[t].notna().sum() >= min_rows
    ]
    no_data_tickers = [t for t in all_tickers if t not in data_tickers]

    if no_data_tickers:
        logger.warning(
            "%d tickers excluded (insufficient price data, need >= %d rows): %s",
            len(no_data_tickers), min_rows, no_data_tickers,
        )
    if not data_tickers:
        raise ValueError("No tickers have sufficient price data for RL training.")

    # Sort alphabetically so ticker order is deterministic across runs
    prices_aligned = prices[sorted(data_tickers)]

    logger.info(
        "classify_assets: %d tickers  %d rows  "
        "lookback=%d  forward=%d  step=%d  episodes=%d  train_end=%s",
        len(data_tickers), len(prices_aligned),
        lookback, forward, step_size, n_episodes, train_end,
    )

    train_end_idx: Optional[int] = None
    if train_end is not None:
        split_ts  = pd.Timestamp(train_end)
        split_pos = prices_aligned.index.searchsorted(split_ts, side="right") - 1
        if 0 <= split_pos < len(prices_aligned):
            train_end_idx = int(split_pos)
            train_date    = prices_aligned.index[train_end_idx].strftime("%Y-%m-%d")
            test_row      = min(train_end_idx + 1, len(prices_aligned) - 1)
            test_date     = prices_aligned.index[test_row].strftime("%Y-%m-%d")
            logger.info(
                "Temporal split — TRAIN through %s (row %d) | "
                "TEST from %s onward",
                train_date, train_end_idx, test_date,
            )
        else:
            logger.warning(
                "train_end '%s' is outside the price data range — "
                "split ignored, training on full dataset.",
                train_end,
            )

    # Build environment 
    env = AssetSelectorEnv(
        prices        = prices_aligned,
        lookback      = lookback,
        forward       = forward,
        step_size     = step_size,
        n_clusters    = n_clusters,
        train_end_idx = train_end_idx,
    )

    # Build agent 
    agent = RLAssetSelectorAgent(
        feature_dim   = env.feature_dim,
        hidden_dims   = [128, 64, 32],
        lr            = 3e-4,
        gamma         = 0.99,
        entropy_coeff = 0.05,
        device        = "cpu",
    )

    # Supervised pretraining 
    agent.pretrain(env, n_epochs=150, lr_pretrain=1e-3)

    # PPO fine-tuning 
    logger.info("PPO fine-tuning for %d episodes …", n_episodes)
    history    = agent.train(env, n_episodes=n_episodes, log_every=10)
    mean_rew   = float(np.mean([h["mean_reward"] for h in history[-10:]]))
    logger.info("Training complete. Mean reward (last 10 episodes): %.4f", mean_rew)

    # Inference — collect per-window scores 
    logger.info("Inference on TRAIN windows …")
    (train_mean_scores,
     train_score_matrix,
     train_fwd_vol_matrix,
     train_fwd_ret_matrix) = agent.collect_all_scores(
        env,
        start_idx = env._start_idx,
        end_idx   = env._end_idx,
    )

    # Test windows 
    if env._test_start_idx is not None and env._test_start_idx < env._full_end_idx:
        logger.info("Inference on TEST windows (held-out) …")
        (test_mean_scores,
         test_score_matrix,
         test_fwd_vol_matrix,
         test_fwd_ret_matrix) = agent.collect_all_scores(
            env,
            start_idx = env._test_start_idx,
            end_idx   = env._full_end_idx,
        )
        # Combine for dynamic CSV outputs and static_df
        score_matrix   = np.concatenate([train_score_matrix,   test_score_matrix],   axis=0)
        fwd_vol_matrix = np.concatenate([train_fwd_vol_matrix, test_fwd_vol_matrix], axis=0)
        fwd_ret_matrix = np.concatenate([train_fwd_ret_matrix, test_fwd_ret_matrix], axis=0)
        # Recompute mean_scores over the full dataset for static_df vol/return cols
        mean_scores  = np.nanmean(score_matrix, axis=0)
        valid_counts = np.isfinite(score_matrix).sum(axis=0)
        mean_scores[valid_counts < 3] = np.nan
    else:
        score_matrix   = train_score_matrix
        fwd_vol_matrix = train_fwd_vol_matrix
        fwd_ret_matrix = train_fwd_ret_matrix
        mean_scores    = train_mean_scores

    # Inference — quarterly classifications 
    logger.info("Quarterly inference on TRAIN windows …")
    train_quarterly = agent.collect_quarterly_scores(
        env,
        start_idx = env._start_idx,
        end_idx   = env._end_idx,
    )
    for q in train_quarterly:
        q["split"] = "train"

    if env._test_start_idx is not None and env._test_start_idx < env._full_end_idx:
        logger.info("Quarterly inference on TEST windows …")
        test_quarterly = agent.collect_quarterly_scores(
            env,
            start_idx = env._test_start_idx,
            end_idx   = env._full_end_idx,
        )
        for q in test_quarterly:
            q["split"] = "test"
    else:
        test_quarterly = []

    quarterly_data = train_quarterly + test_quarterly

    logger.info(
        "Total quarters: %d train + %d test = %d",
        len(train_quarterly), len(test_quarterly), len(quarterly_data),
    )

    # Build quarterly_df 
    quarterly_rows: List[Dict] = []
    for q_data in quarterly_data:
        for i, ticker in enumerate(env.tickers):
            score = q_data["mean_scores"][i]
            rp    = q_data["risk_profiles"][i]
            quarterly_rows.append({
                "quarter_start": q_data["quarter_start"],
                "ticker":        ticker,
                "rl_risk_score": float(score) if np.isfinite(score) else np.nan,
                "risk_profile":  rp if rp != "" else None,
                "split":         q_data.get("split", "train"),
            })
    quarterly_df = pd.DataFrame(quarterly_rows)

    logger.info(
        "quarterly_df: %d quarters × %d tickers = %d rows",
        len(quarterly_data), env.n_tickers, len(quarterly_df),
    )

    # Build eval_df 
    eval_rows:      List[Dict] = []
    spearman_stats: List[Dict] = []
    _label_to_rank = {"conservative": 0, "balanced": 1, "aggressive": 2}

    for q_data in quarterly_data:
        q_start       = q_data["quarter_start"]
        q_idx         = q_data["quarter_idx"]
        risk_profiles = q_data["risk_profiles"]   # np.ndarray of str

        actual_sharpe  = env.compute_sharpe(q_idx, env.forward)
        actual_fwd_vol = env._compute_forward_vol(q_idx)

        has_profile = np.array([rp != "" for rp in risk_profiles])
        has_vol     = np.isfinite(actual_fwd_vol)
        valid_mask  = has_profile & has_vol
        n_valid     = int(valid_mask.sum())

        # Per-quarter Spearman ρ
        if n_valid >= 3:
            label_ranks = np.array(
                [
                    _label_to_rank[risk_profiles[i]]
                    for i in range(env.n_tickers)
                    if valid_mask[i]
                ],
                dtype=float,
            )
            rho_val, _ = spearmanr(label_ranks, actual_fwd_vol[valid_mask])
            rho = float(rho_val) if np.isfinite(rho_val) else np.nan
        else:
            rho = np.nan

        spearman_stats.append({
            "quarter_start":   q_start,
            "spearman_rho":    rho,
            "n_valid_tickers": n_valid,
            "split":           q_data.get("split", "train"),
        })

        # classification_correct 
        if n_valid >= 3:
            valid_indices   = np.where(valid_mask)[0]
            vol_for_valid   = actual_fwd_vol[valid_indices]
            vol_ranks_local = pd.Series(vol_for_valid).rank(method="average").values
            third           = n_valid / 3.0

            def _expected_profile(rank: float) -> str:
                if rank <= third:
                    return "conservative"
                elif rank <= 2 * third:
                    return "balanced"
                else:
                    return "aggressive"

            rank_map: Dict[int, float] = {
                int(valid_indices[j]): vol_ranks_local[j]
                for j in range(n_valid)
            }
        else:
            rank_map = {}

        for i, ticker in enumerate(env.tickers):
            if valid_mask[i] and rank_map:
                correct: float = (
                    1.0
                    if risk_profiles[i] == _expected_profile(rank_map[i])
                    else 0.0
                )
            else:
                correct = np.nan

            eval_rows.append({
                "quarter_start":         q_start,
                "ticker":                ticker,
                "risk_profile":          risk_profiles[i] if risk_profiles[i] != "" else None,
                "actual_sharpe":         float(actual_sharpe[i])  if np.isfinite(actual_sharpe[i])  else np.nan,
                "actual_fwd_vol":        float(actual_fwd_vol[i]) if np.isfinite(actual_fwd_vol[i]) else np.nan,
                "classification_correct": correct,
                "split":                 q_data.get("split", "train"),
            })

    eval_df = pd.DataFrame(eval_rows)

    # Summary statistics 
    _log_summary(spearman_stats, eval_df)

    # Build static_df 
    # risk_profile = label from the most recent quarter that produced a valid label for each ticker.
    static_df = _build_static_df(
        env            = env,
        quarterly_data = quarterly_data,
        mean_scores    = mean_scores,
        fwd_vol_matrix = fwd_vol_matrix,
        fwd_ret_matrix = fwd_ret_matrix,
        no_data_tickers= no_data_tickers,
        data_tickers   = data_tickers,
    )

    # Save outputs 
    if output_dir:
        _save_outputs(
            out_path       = Path(output_dir),
            agent          = agent,
            env            = env,
            history        = history,
            quarterly_df   = quarterly_df,
            eval_df        = eval_df,
            static_df      = static_df,
            spearman_stats = spearman_stats,
            score_matrix   = score_matrix,
            fwd_vol_matrix = fwd_vol_matrix,
            data_tickers   = data_tickers,
        )

    return quarterly_df, eval_df, static_df

# Internal helpers

def _log_summary(
    spearman_stats: List[Dict],
    eval_df:        pd.DataFrame,
) -> None:
    logger.info("=" * 60)
    logger.info("QUARTERLY CLASSIFICATION EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info("Per-quarter Spearman ρ (label rank vs actual forward vol):")

    for row in spearman_stats:
        rho_str = (
            f"{row['spearman_rho']:.3f}"
            if np.isfinite(row["spearman_rho"])
            else "N/A"
        )
        logger.info(
            "  [%-5s]  %s  n=%2d  ρ = %s",
            row.get("split", "train").upper(),
            row["quarter_start"].strftime("%Y-%m-%d"),
            row["n_valid_tickers"],
            rho_str,
        )

    rho_vals = [
        r["spearman_rho"] for r in spearman_stats
        if np.isfinite(r["spearman_rho"])
    ]
    if rho_vals:
        logger.info(
            "All quarters:  mean ρ = %.3f  std = %.3f  (n=%d)",
            float(np.mean(rho_vals)), float(np.std(rho_vals)), len(rho_vals),
        )

    for split_label in ("train", "test"):
        split_rhos = [
            r["spearman_rho"] for r in spearman_stats
            if r.get("split") == split_label and np.isfinite(r["spearman_rho"])
        ]
        if split_rhos:
            tag = "TRAIN" if split_label == "train" else "TEST  (held-out)"
            logger.info(
                "  %s: mean ρ = %.3f  std = %.3f  (n=%d quarters)",
                tag,
                float(np.mean(split_rhos)),
                float(np.std(split_rhos)),
                len(split_rhos),
            )

    logger.info("Mean actual Sharpe and forward vol per risk profile:")
    for profile in ("conservative", "balanced", "aggressive"):
        grp = eval_df[eval_df["risk_profile"] == profile]
        if not grp.empty:
            logger.info(
                "  %-12s  mean_sharpe=%+.3f  mean_fwd_vol=%.3f",
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
            sp_data = eval_df.loc[
                eval_df["split"] == sp, "classification_correct"
            ].dropna()
            if not sp_data.empty:
                tag = "TRAIN" if sp == "train" else "TEST  (held-out)"
                logger.info(
                    "  %s accuracy: %.1f%%",
                    tag, float(sp_data.mean()) * 100,
                )


def _build_static_df(
    env:             AssetSelectorEnv,
    quarterly_data:  List[Dict],
    mean_scores:     np.ndarray,
    fwd_vol_matrix:  np.ndarray,
    fwd_ret_matrix:  np.ndarray,
    no_data_tickers: List[str],
    data_tickers:    List[str],
) -> pd.DataFrame:
   
    # Walk quarterly_data in reverse to find each ticker's most recent label
    ticker_to_latest: Dict[str, Tuple[str, pd.Timestamp]] = {}

    for q_data in reversed(quarterly_data):
        q_start = q_data["quarter_start"]
        for i, ticker in enumerate(env.tickers):
            if ticker in ticker_to_latest:
                continue  
            rp = q_data["risk_profiles"][i]
            if rp != "":
                ticker_to_latest[ticker] = (rp, q_start)

    _profile_to_id = {"conservative": 0, "balanced": 1, "aggressive": 2}
    scored_arr = np.array(sorted(data_tickers))   # matches env.tickers order

    latest_profiles  = []
    latest_cluster_ids = []
    latest_quarters  = []
    for ticker in env.tickers:
        if ticker in ticker_to_latest:
            rp, qs = ticker_to_latest[ticker]
            latest_profiles.append(rp)
            latest_cluster_ids.append(_profile_to_id[rp])
            latest_quarters.append(qs)
        else:
            latest_profiles.append("")
            latest_cluster_ids.append(-1)
            latest_quarters.append(pd.NaT)

    latest_profiles_arr   = np.array(latest_profiles,    dtype=object)
    latest_cluster_arr    = np.array(latest_cluster_ids, dtype=int)
    latest_quarters_arr   = np.array(latest_quarters,    dtype=object)

    vol_values = np.nanmean(fwd_vol_matrix, axis=0)
    ret_values = np.nanmean(fwd_ret_matrix, axis=0)

    sort_order = np.argsort(
        np.where(np.isfinite(vol_values), vol_values, np.inf)
    )

    tickers_sorted        = scored_arr[sort_order]
    vol_sorted            = vol_values[sort_order]
    ret_sorted            = ret_values[sort_order]
    cluster_ids_sorted    = latest_cluster_arr[sort_order]
    risk_profiles_sorted  = latest_profiles_arr[sort_order]
    mean_scores_sorted    = mean_scores[sort_order]
    quarters_sorted       = latest_quarters_arr[sort_order]

    static_df = pd.DataFrame({"ticker": tickers_sorted})
    static_df["volatility"]         = vol_sorted
    static_df["mean_return"]        = ret_sorted
    static_df["cluster_id"]         = [
        int(c) if c != -1 else None for c in cluster_ids_sorted
    ]
    static_df["risk_profile"]       = [
        rp if rp != "" else None for rp in risk_profiles_sorted
    ]
    static_df["rl_risk_score"]      = mean_scores_sorted
    static_df["most_recent_quarter"] = quarters_sorted

    # Append no-data tickers as unclassified rows
    if no_data_tickers:
        nd = pd.DataFrame({"ticker": no_data_tickers})
        nd["volatility"]          = np.nan
        nd["mean_return"]         = np.nan
        nd["cluster_id"]          = None
        nd["risk_profile"]        = None
        nd["rl_risk_score"]       = np.nan
        nd["most_recent_quarter"] = pd.NaT
        static_df = pd.concat([static_df, nd], ignore_index=True)

    static_df = static_df.reset_index(drop=True)

    # Log the static summary
    logger.info("Current risk profile (most recent quarter) summary:")
    for profile in ("conservative", "balanced", "aggressive"):
        grp = static_df[static_df["risk_profile"] == profile]
        if not grp.empty:
            logger.info(
                "  %-12s: %2d tickers  vol [%.3f, %.3f]  "
                "return [%.3f, %.3f]  rl_score [%.3f, %.3f]",
                profile, len(grp),
                grp["volatility"].min(),    grp["volatility"].max(),
                grp["mean_return"].min(),   grp["mean_return"].max(),
                grp["rl_risk_score"].min(), grp["rl_risk_score"].max(),
            )

    unclassified = static_df["risk_profile"].isna().sum()
    if unclassified:
        logger.warning(
            "%d tickers unclassified (no valid quarterly label): %s",
            unclassified,
            static_df.loc[static_df["risk_profile"].isna(), "ticker"].tolist(),
        )

    return static_df


def _save_outputs(
    out_path:       Path,
    agent:          RLAssetSelectorAgent,
    env:            AssetSelectorEnv,
    history:        List[Dict],
    quarterly_df:   pd.DataFrame,
    eval_df:        pd.DataFrame,
    static_df:      pd.DataFrame,
    spearman_stats: List[Dict],
    score_matrix:   np.ndarray,
    fwd_vol_matrix: np.ndarray,
    data_tickers:   List[str],
) -> None:
    out_path.mkdir(parents=True, exist_ok=True)

    # Model weights
    agent.save(out_path / "rl_agent.pt")

    # Per-window dynamic scores (full dataset: train + test)
    window_dates = [
        env.prices.index[min(i, len(env.prices) - 1)]
        for i in range(env._start_idx, env._full_end_idx, env.step_size)
    ]
    sorted_tickers = sorted(data_tickers)

    pd.DataFrame(
        score_matrix,
        index   = window_dates[: len(score_matrix)],
        columns = sorted_tickers,
    ).to_csv(out_path / "rl_dynamic_scores.csv")
    logger.info("Dynamic scores saved to %s", out_path / "rl_dynamic_scores.csv")

    pd.DataFrame(
        fwd_vol_matrix,
        index   = window_dates[: len(fwd_vol_matrix)],
        columns = sorted_tickers,
    ).to_csv(out_path / "rl_dynamic_fwd_vol.csv")
    logger.info("Dynamic fwd vol saved to %s", out_path / "rl_dynamic_fwd_vol.csv")

    # Training history
    pd.DataFrame(history).to_csv(
        out_path / "rl_training_history.csv", index=False
    )
    logger.info("Training history saved.")

    # Quarterly classifications
    quarterly_df.to_csv(
        out_path / "quarterly_classifications.csv", index=False
    )
    logger.info(
        "Quarterly classifications saved (%d rows).", len(quarterly_df)
    )

    # Evaluation
    eval_df.to_csv(out_path / "evaluation.csv", index=False)
    logger.info("Evaluation saved (%d rows).", len(eval_df))

    # Per-quarter Spearman
    pd.DataFrame(spearman_stats).to_csv(
        out_path / "quarterly_spearman.csv", index=False
    )
    logger.info("Per-quarter Spearman saved.")

    # risk_profiles.json — CURRENT labels (most recent quarter per ticker)
    # This is what the downstream portfolio models consume.
    profile_map: Dict[str, List[str]] = {
        "conservative": [],
        "balanced":     [],
        "aggressive":   [],
    }
    for _, row in static_df.dropna(subset=["risk_profile"]).iterrows():
        profile_map[row["risk_profile"]].append(row["ticker"])
    for k in profile_map:
        profile_map[k] = sorted(profile_map[k])

  
    latest_quarter_dates = (
        static_df["most_recent_quarter"]
        .dropna()
        .sort_values()
    )
    profile_map["as_of_quarter"] = (
        str(latest_quarter_dates.iloc[-1].date())
        if not latest_quarter_dates.empty
        else "unknown"
    )

    import json
    (out_path / "risk_profiles.json").write_text(
        json.dumps(profile_map, indent=2), encoding="utf-8"
    )
    logger.info(
        "risk_profiles.json saved (as_of_quarter: %s).",
        profile_map["as_of_quarter"],
    )

    # asset_classification.csv 
    static_df.to_csv(out_path / "asset_classification.csv", index=False)
    logger.info("asset_classification.csv saved.")

# Utility

def get_profile_tickers(
    classification: pd.DataFrame,
    profile:        str,
) -> List[str]:
    """
    Return sorted list of tickers with the given risk profile.

    Parameters
    ----------
    classification : pd.DataFrame
        Output static_df from classify_assets().
    profile : str
        One of 'conservative', 'balanced', 'aggressive'.
    """
    valid = {"conservative", "balanced", "aggressive"}
    if profile not in valid:
        raise ValueError(f"profile must be one of {valid}, got '{profile}'")
    mask = classification["risk_profile"] == profile
    return sorted(classification.loc[mask, "ticker"].tolist())