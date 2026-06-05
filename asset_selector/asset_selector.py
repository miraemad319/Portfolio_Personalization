"""
asset_selector.py

Classifies EGX30 tickers into three risk tiers — conservative, balanced,
aggressive — using a PPO agent trained on rolling OHLCV feature windows.

Design:
The RL agent observes 20 features computed from the past 126 trading days
for every stock and outputs a continuous risk score per stock. The reward
signal is the Spearman rank correlation between the agent's scores and the
actual forward 126-day realised volatility + max drawdown. Classifications
are produced every 126 days (semi-annual), with each period's label based
on averaging 6 window scores within that period.

Output files

semi_annual_classifications.csv — one row per (period × ticker)
                                   columns: period_start | ticker |
                                   rl_risk_score | risk_profile | split
evaluation.csv                  — same + actual_sharpe | actual_fwd_vol |
                                   classification_correct | split
semi_annual_spearman.csv        — per-period Spearman ρ with split label
risk_profiles.json              — CURRENT labels (most recent period),
                                   consumed by downstream portfolio models
asset_classification.csv        — same current labels, for the visualiser
rl_dynamic_scores.csv           — per-window RL scores (full history)
rl_dynamic_fwd_vol.csv          — per-window forward vol (full history)
rl_training_history.csv         — per-episode training stats
rl_agent.pt                     — trained model weights
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr

from asset_selector.rl_environment import AssetSelectorEnv
from asset_selector.rl_agent import RLAssetSelectorAgent

logger = logging.getLogger(__name__)

#  Public API 
def classify_assets(
    ohlcv:      pd.DataFrame,
    n_clusters: int           = 3,
    n_episodes: int           = 250,
    lookback:   int           = 126,
    forward:    int           = 126,
    step_size:  int           = 21,
    train_end:  Optional[str] = "2023-12-31",
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
   
    if n_clusters != 3:
        raise ValueError("classify_assets supports n_clusters=3 only.")

    all_tickers = ohlcv.columns.get_level_values("ticker").unique().tolist()
    if not all_tickers:
        raise ValueError("ohlcv DataFrame has no tickers.")

    # Filter tickers with sufficient close-price data
    min_rows    = lookback + forward + 1
    close_df    = ohlcv.xs("close", axis=1, level="price_type").replace(0.0, np.nan)
    data_tickers = [
        t for t in all_tickers
        if close_df[t].notna().sum() >= min_rows
    ]
    no_data_tickers = [t for t in all_tickers if t not in data_tickers]

    if no_data_tickers:
        logger.warning(
            "%d tickers excluded (insufficient data, need >= %d rows): %s",
            len(no_data_tickers), min_rows, no_data_tickers,
        )
    if not data_tickers:
        raise ValueError("No tickers have sufficient data for RL training.")

    # Reindex ohlcv to sorted data_tickers 
    sorted_tickers = sorted(data_tickers)
    ohlcv_aligned  = ohlcv.loc[:, (sorted_tickers, slice(None))]

    logger.info(
        "classify_assets: %d tickers  %d rows  "
        "lookback=%d  forward=%d  step=%d  episodes=%d  train_end=%s",
        len(sorted_tickers), len(ohlcv_aligned),
        lookback, forward, step_size, n_episodes, train_end,
    )

    # Temporal split
    train_end_idx: Optional[int] = None
    if train_end is not None:
        split_ts  = pd.Timestamp(train_end)
        split_pos = ohlcv_aligned.index.searchsorted(split_ts, side="right") - 1
        if 0 <= split_pos < len(ohlcv_aligned):
            train_end_idx = int(split_pos)
            train_date    = ohlcv_aligned.index[train_end_idx].strftime("%Y-%m-%d")
            test_row      = min(train_end_idx + 1, len(ohlcv_aligned) - 1)
            test_date     = ohlcv_aligned.index[test_row].strftime("%Y-%m-%d")
            logger.info(
                "Temporal split — TRAIN through %s (row %d) | "
                "TEST from %s onward",
                train_date, train_end_idx, test_date,
            )
        else:
            logger.warning(
                "train_end '%s' is outside the data range — "
                "split ignored, training on full dataset.",
                train_end,
            )

    # Build environment
    env = AssetSelectorEnv(
        ohlcv         = ohlcv_aligned,
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
        gamma         = 0.0,
        gae_lambda    = 0.0,
        entropy_coeff = 0.0,
        device        = "cpu",
    )

    # Supervised pretraining
    agent.pretrain(env, n_epochs=150, lr_pretrain=1e-3)

    # PPO fine-tuning with early stopping
    es_window      = 20
    es_check_every = 10
    es_patience    = 2
    es_min_delta   = 5e-4

    logger.info(
        "PPO fine-tuning (max %d episodes, early stopping active) …", n_episodes
    )

    history:          list  = []
    es_counter:       int   = 0
    es_best_smoothed: float = -np.inf
    best_reward:      float = -np.inf
    best_state:       dict  = {
        "actor":  {k: v.clone() for k, v in agent.actor.state_dict().items()},
        "critic": {k: v.clone() for k, v in agent.critic.state_dict().items()},
    }

    for ep in range(1, n_episodes + 1):
        ep_stats = agent.train_one_episode(env)
        history.append(ep_stats)

        if ep_stats["mean_reward"] > best_reward:
            best_reward = ep_stats["mean_reward"]
            best_state = {
                "actor":  {k: v.clone() for k, v in agent.actor.state_dict().items()},
                "critic": {k: v.clone() for k, v in agent.critic.state_dict().items()},
            }

        if ep % 10 == 0 or ep == 1:
            logger.info(
                "Episode %3d/%d  mean_reward=%.4f  "
                "actor_loss=%.4f  value_loss=%.4f",
                ep, n_episodes,
                ep_stats["mean_reward"],
                ep_stats["actor_loss"],
                ep_stats["value_loss"],
            )

        if ep >= es_window and ep % es_check_every == 0:
            recent_rewards = [h["mean_reward"] for h in history[-es_window:]]
            smoothed       = float(np.mean(recent_rewards))
            improvement    = smoothed - es_best_smoothed

            if improvement > es_min_delta:
                es_best_smoothed = smoothed
                es_counter       = 0
            else:
                es_counter += 1
                logger.info(
                    "  Early stopping: no improvement (smoothed=%.4f, best=%.4f, "
                    "patience %d/%d)",
                    smoothed, es_best_smoothed, es_counter, es_patience,
                )
                if es_counter >= es_patience:
                    logger.info(
                        "Early stopping triggered at episode %d "
                        "(smoothed reward plateau: %.4f)",
                        ep, es_best_smoothed,
                    )
                    break

    # Restore best weights — PPO may have degraded since peak episode
    agent.actor.load_state_dict(best_state["actor"])
    agent.critic.load_state_dict(best_state["critic"])
    logger.info(
        "Restored best weights (peak episode reward: %.4f)", best_reward
    )

    mean_rew = float(np.mean([h["mean_reward"] for h in history[-10:]]))
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


    # Semi annual inference — per-semi annual period tertile thresholds, matching
    # the evaluation which also uses per-semi annual tertiles of actual vol.
    logger.info("Semi-annual inference on TRAIN windows …")
    train_period = agent.collect_period_scores(
        env,
        start_idx  = env._start_idx,
        end_idx    = env._end_idx,
        thresholds = None,
    )
    for q in train_period:
        q["split"] = "train"

    test_period=[]

    if env._test_start_idx is not None and env._test_start_idx < env._full_end_idx:
        logger.info("Semi annual inference on TEST windows …")
        test_period = agent.collect_period_scores(
            env,
            start_idx  = env._test_start_idx,
            end_idx    = env._full_end_idx,
            thresholds = None,
        )
        for q in test_period:
            q["split"] = "test"

    period_data = train_period + test_period

    logger.info(
        "Total periods: %d train + %d test = %d",
        len(train_period), len(test_period), len(period_data),
    )

    # Build semiannual_df
    period_rows: List[Dict] = []
    for q_data in period_data:
        for i, ticker in enumerate(env.tickers):
            score = q_data["mean_scores"][i]
            rp    = q_data["risk_profiles"][i]
            period_rows.append({
                "semiannual_start": q_data["semiannual_start"],
                "ticker":        ticker,
                "rl_risk_score": float(score) if np.isfinite(score) else np.nan,
                "risk_profile":  rp if rp != "" else None,
                "split":         q_data.get("split", "train"),
            })
    semi_annual_df = pd.DataFrame(period_rows)

    logger.info(
        "semi_annual_df: %d periods × %d tickers = %d rows",
        len(period_data), env.n_tickers, len(semi_annual_df),
    )

    # Build eval_df
    eval_rows:      List[Dict] = []
    spearman_stats: List[Dict] = []
    _label_to_rank = {"conservative": 0, "balanced": 1, "aggressive": 2}

    ticker_to_latest_vol:   Dict[str, float] = {}
    ticker_to_latest_ret:   Dict[str, float] = {}
    ticker_to_latest_score: Dict[str, float] = {}

    for q_data in period_data:
        q_start       = q_data["period_start"]
        q_idx         = q_data["period_idx"]
        risk_profiles = q_data["risk_profiles"]

        actual_sharpe  = env.compute_sharpe(q_idx, env.forward)
        actual_fwd_vol = env._compute_forward_vol(q_idx)
        actual_fwd_ret = env._compute_forward_ret(q_idx)

        has_profile = np.array([rp != "" for rp in risk_profiles])
        has_vol     = np.isfinite(actual_fwd_vol)
        valid_mask  = has_profile & has_vol
        n_valid     = int(valid_mask.sum())

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
            "period_start":   q_start,
            "spearman_rho":    rho,
            "n_valid_tickers": n_valid,
            "split":           q_data.get("split", "train"),
        })

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
                "period_start":          q_start,
                "ticker":                 ticker,
                "risk_profile":           risk_profiles[i] if risk_profiles[i] != "" else None,
                "actual_sharpe":          float(actual_sharpe[i])  if np.isfinite(actual_sharpe[i])  else np.nan,
                "actual_fwd_vol":         float(actual_fwd_vol[i]) if np.isfinite(actual_fwd_vol[i]) else np.nan,
                "classification_correct": correct,
                "split":                  q_data.get("split", "train"),
            })

    eval_df = pd.DataFrame(eval_rows)

    # Walk semi annaul data in reverse for most-recent-period lookups
    for q_data in reversed(period_data):
        q_idx          = q_data["period_idx"]
        actual_fwd_vol = env._compute_forward_vol(q_idx)
        actual_fwd_ret = env._compute_forward_ret(q_idx)
        for i, ticker in enumerate(env.tickers):
            rp = q_data["risk_profiles"][i]
            if rp == "":
                continue
            if ticker not in ticker_to_latest_vol and np.isfinite(actual_fwd_vol[i]):
                ticker_to_latest_vol[ticker] = float(actual_fwd_vol[i])
                ticker_to_latest_ret[ticker] = (
                    float(actual_fwd_ret[i]) if np.isfinite(actual_fwd_ret[i]) else np.nan
                )
            if ticker not in ticker_to_latest_score and np.isfinite(q_data["mean_scores"][i]):
                ticker_to_latest_score[ticker] = float(q_data["mean_scores"][i])

    # Summary + static_df
    _log_summary(spearman_stats, eval_df)

    static_df = _build_static_df(
        env                    = env,
        period_data         = period_data,
        ticker_to_latest_vol   = ticker_to_latest_vol,
        ticker_to_latest_ret   = ticker_to_latest_ret,
        ticker_to_latest_score = ticker_to_latest_score,
        no_data_tickers        = no_data_tickers,
        data_tickers           = data_tickers,
    )

    if output_dir:
        _save_outputs(
            out_path       = Path(output_dir),
            agent          = agent,
            env            = env,
            history        = history,
            semi_annual_df   = semi_annual_df,
            eval_df        = eval_df,
            static_df      = static_df,
            spearman_stats = spearman_stats,
            score_matrix   = score_matrix,
            fwd_vol_matrix = fwd_vol_matrix,
            data_tickers   = data_tickers,
        )

    return semi_annual_df, eval_df, static_df


#  Internal helpers 

def _log_summary(
    spearman_stats: List[Dict],
    eval_df:        pd.DataFrame,
) -> None:
    logger.info("=" * 60)
    logger.info("SEMI ANNUAL CLASSIFICATION EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info("Per-semi-annual period Spearman ρ (label rank vs actual forward vol):")

    for row in spearman_stats:
        rho_str = (
            f"{row['spearman_rho']:.3f}"
            if np.isfinite(row["spearman_rho"])
            else "N/A"
        )
        logger.info(
            "  [%-5s]  %s  n=%2d  ρ = %s",
            row.get("split", "train").upper(),
            row["period_start"].strftime("%Y-%m-%d"),
            row["n_valid_tickers"],
            rho_str,
        )

    rho_vals = [
        r["spearman_rho"] for r in spearman_stats
        if np.isfinite(r["spearman_rho"])
    ]
    if rho_vals:
        logger.info(
            "All periods:  mean ρ = %.3f  std = %.3f  (n=%d)",
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
                "  %s: mean ρ = %.3f  std = %.3f  (n=%d periods)",
                tag,
                float(np.mean(split_rhos)),
                float(np.std(split_rhos)),
                len(split_rhos),
            )

    logger.info("Mean actual Sharpe and forward vol per risk profile:")
    profile_mean_vol: dict = {}
    for profile in ("conservative", "balanced", "aggressive"):
        grp = eval_df[eval_df["risk_profile"] == profile]
        if not grp.empty:
            mean_vol = float(grp["actual_fwd_vol"].mean())
            profile_mean_vol[profile] = mean_vol
            logger.info(
                "  %-12s  mean_sharpe=%+.3f  mean_fwd_vol=%.3f",
                profile,
                float(grp["actual_sharpe"].mean()),
                mean_vol,
            )

    if "aggressive" in profile_mean_vol and "conservative" in profile_mean_vol:
        spread = profile_mean_vol["aggressive"] - profile_mean_vol["conservative"]
        logger.info(
            "top-bottom vol spread (aggressive − conservative): %.4f", spread
        )
    else:
        logger.info(
            "top-bottom vol spread: N/A "
            "(one or both boundary tiers have no valid rows in eval_df)"
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
    env:                    AssetSelectorEnv,
    period_data:         List[Dict],
    ticker_to_latest_vol:   Dict[str, float],
    ticker_to_latest_ret:   Dict[str, float],
    ticker_to_latest_score: Dict[str, float],
    no_data_tickers:        List[str],
    data_tickers:           List[str],
) -> pd.DataFrame:

    ticker_to_latest: Dict[str, Tuple[str, pd.Timestamp]] = {}

    for q_data in reversed(period_data):
        q_start = q_data["period_start"]
        for i, ticker in enumerate(env.tickers):
            if ticker in ticker_to_latest:
                continue
            rp = q_data["risk_profiles"][i]
            if rp != "":
                ticker_to_latest[ticker] = (rp, q_start)

    _profile_to_id = {"conservative": 0, "balanced": 1, "aggressive": 2}
    scored_arr = np.array(sorted(data_tickers))

    latest_profiles    = []
    latest_cluster_ids = []
    latest_periods    = []
    latest_scores      = []
    latest_vols        = []
    latest_rets        = []

    for ticker in env.tickers:
        if ticker in ticker_to_latest:
            rp, qs = ticker_to_latest[ticker]
            latest_profiles.append(rp)
            latest_cluster_ids.append(_profile_to_id[rp])
            latest_periods.append(qs)
        else:
            latest_profiles.append("")
            latest_cluster_ids.append(-1)
            latest_periods.append(pd.NaT)

        latest_scores.append(ticker_to_latest_score.get(ticker, np.nan))
        latest_vols.append(ticker_to_latest_vol.get(ticker, np.nan))
        latest_rets.append(ticker_to_latest_ret.get(ticker, np.nan))

    latest_profiles_arr  = np.array(latest_profiles,    dtype=object)
    latest_cluster_arr   = np.array(latest_cluster_ids, dtype=int)
    latest_periods_arr  = np.array(latest_periods,    dtype=object)
    latest_scores_arr    = np.array(latest_scores,      dtype=np.float64)
    latest_vols_arr      = np.array(latest_vols,        dtype=np.float64)
    latest_rets_arr      = np.array(latest_rets,        dtype=np.float64)

    sort_order = np.argsort(
        np.where(np.isfinite(latest_vols_arr), latest_vols_arr, np.inf)
    )

    static_df = pd.DataFrame({
        "ticker":              scored_arr[sort_order],
        "volatility":          latest_vols_arr[sort_order],
        "mean_return":         latest_rets_arr[sort_order],
        "cluster_id":          [
            int(c) if c != -1 else None
            for c in latest_cluster_arr[sort_order]
        ],
        "risk_profile":        [
            rp if rp != "" else None
            for rp in latest_profiles_arr[sort_order]
        ],
        "rl_risk_score":       latest_scores_arr[sort_order],
        "most_recent_period": latest_periods_arr[sort_order],
    })

    if no_data_tickers:
        nd = pd.DataFrame({"ticker": no_data_tickers})
        nd["volatility"]          = np.nan
        nd["mean_return"]         = np.nan
        nd["cluster_id"]          = None
        nd["risk_profile"]        = None
        nd["rl_risk_score"]       = np.nan
        nd["most_recent_period"] = pd.NaT
        static_df = pd.concat([static_df, nd], ignore_index=True)

    static_df = static_df.reset_index(drop=True)

    logger.info("Current risk profile (most recent semi annual period) summary:")
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
            "%d tickers unclassified (no valid semi annual label): %s",
            unclassified,
            static_df.loc[static_df["risk_profile"].isna(), "ticker"].tolist(),
        )

    return static_df


def _save_outputs(
    out_path:       Path,
    agent:          RLAssetSelectorAgent,
    env:            AssetSelectorEnv,
    history:        List[Dict],
    semi_annual_df:   pd.DataFrame,
    eval_df:        pd.DataFrame,
    static_df:      pd.DataFrame,
    spearman_stats: List[Dict],
    score_matrix:   np.ndarray,
    fwd_vol_matrix: np.ndarray,
    data_tickers:   List[str],
) -> None:
    out_path.mkdir(parents=True, exist_ok=True)

    agent.save(out_path / "rl_agent.pt")

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

    pd.DataFrame(history).to_csv(
        out_path / "rl_training_history.csv", index=False
    )
    logger.info("Training history saved.")

    semi_annual_df.to_csv(out_path / "semi_annual_classifications.csv", index=False)
    logger.info("Semi-annual classifications saved (%d rows).", len(semi_annual_df))

    eval_df.to_csv(out_path / "evaluation.csv", index=False)
    logger.info("Evaluation saved (%d rows).", len(eval_df))

    pd.DataFrame(spearman_stats).to_csv(
        out_path / "semi_annual_spearman.csv", index=False
    )
    logger.info("Per-period Spearman saved.")

    profile_map: Dict[str, List[str]] = {
        "conservative": [],
        "balanced":     [],
        "aggressive":   [],
    }
    for _, row in static_df.dropna(subset=["risk_profile"]).iterrows():
        profile_map[row["risk_profile"]].append(row["ticker"])
    for k in profile_map:
        profile_map[k] = sorted(profile_map[k])

    latest_period_dates = (
        static_df["most_recent_period"].dropna().sort_values()
    )
    profile_map["as_of_period"] = (
        str(latest_period_dates.iloc[-1].date())
        if not latest_period_dates.empty
        else "unknown"
    )

    (out_path / "risk_profiles.json").write_text(
        json.dumps(profile_map, indent=2), encoding="utf-8"
    )
    logger.info(
        "risk_profiles.json saved (as_of_period: %s).",
        profile_map["as_of_period"],
    )

    static_df.to_csv(out_path / "asset_classification.csv", index=False)
    logger.info("asset_classification.csv saved.")

    logger.info("=" * 60)
    logger.info("Output files written to %s", out_path)
    logger.info("  risk_profiles.json                  ← downstream portfolio models")
    logger.info("  semi_annual_classifications.csv     ← full dynamic history")
    logger.info("  evaluation.csv                      ← Sharpe / accuracy per period")
    logger.info("  semi_annual_spearman.csv            ← per-period Spearman ρ")
    logger.info("  asset_classification.csv            ← current labels (visualiser)")

#  Utility 

def get_profile_tickers(
    classification: pd.DataFrame,
    profile:        str,
) -> List[str]:
    """Return sorted list of tickers with the given risk profile."""
    valid = {"conservative", "balanced", "aggressive"}
    if profile not in valid:
        raise ValueError(f"profile must be one of {valid}, got '{profile}'")
    mask = classification["risk_profile"] == profile
    return sorted(classification.loc[mask, "ticker"].tolist())