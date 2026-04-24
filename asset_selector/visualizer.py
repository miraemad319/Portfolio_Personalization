"""
visualizer.py

Plots generated

1. cluster_assignments.png
       Horizontal bar chart — one bar per ticker ranked by mean forward
       volatility, coloured by current risk profile.  The primary "what
       did the model decide" plot.

2. actual_risk_return.png
       Scatter: actual mean forward volatility (x) vs actual mean forward
       return (y), coloured by current risk profile.  Shows whether the
       three tiers are genuinely separated in risk-return space.

3. rl_accuracy_scatter.png
       Scatter: RL risk score (x) vs actual forward volatility (y) for
       every (window, ticker) pair across the full dataset.  The best-fit
       line and Spearman ρ show how well the model's continuous score
       tracks real risk.  Train and test points are plotted in different
       shades to make the held-out period visible.

4. classification_accuracy.png
       Bar chart — per-quarter Spearman ρ between label rank and actual
       forward volatility.  Green = positive (correct ordering), red =
       negative (inverted).  Train quarters and test quarters are
       visually distinguished.  This is the primary model evaluation plot.

5. quarterly_sharpe.png
       Grouped bar chart — mean actual Sharpe per risk profile per quarter.
       Shows whether conservative / balanced / aggressive buckets had
       meaningfully different risk-adjusted returns in reality.
       Train and test quarters are separated by a vertical divider.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

PROFILE_COLOURS: dict[str, str] = {
    "conservative": "#2ecc71",
    "balanced":     "#f39c12",
    "aggressive":   "#e74c3c",
}
PROFILE_ORDER = ["conservative", "balanced", "aggressive"]

# Train / test shading
TRAIN_ALPHA = 0.55
TEST_ALPHA  = 0.90

def _legend_patches() -> list[mpatches.Patch]:
    return [
        mpatches.Patch(color=PROFILE_COLOURS[p], label=p.capitalize())
        for p in PROFILE_ORDER
    ]

def _save(fig: plt.Figure, path: Optional[str]) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Plot saved to %s", path)
        plt.close(fig)
    else:
        plt.show()


def _path(output_dir: Optional[str], name: str) -> Optional[str]:
    return f"{output_dir}/{name}" if output_dir else None

# Plot 1 — Cluster assignments

def plot_cluster_assignments(
    result_df:   pd.DataFrame,
    output_path: Optional[str] = None,
) -> plt.Figure:
    valid = (
        result_df.dropna(subset=["volatility", "risk_profile"])
        .sort_values("volatility")
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(12, max(6, len(valid) * 0.32)))
    sns.set_style("whitegrid")

    colours = valid["risk_profile"].map(PROFILE_COLOURS).fillna("#95a5a6")
    bars    = ax.barh(
        valid["ticker"], valid["volatility"],
        color=colours, edgecolor="white", linewidth=0.4,
    )

    for bar, vol in zip(bars, valid["volatility"]):
        ax.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{vol:.1%}",
            va="center", ha="left", fontsize=7.5,
        )

    ax.set_xlabel("Mean Forward 63-Day Annualised Volatility", fontsize=12)
    ax.set_title(
        "EGX30 Risk Classification — Current Labels\n"
        "(volatility = mean across all scored quarters)",
        fontsize=13, fontweight="bold",
    )
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )
    ax.legend(handles=_legend_patches(), loc="lower right", fontsize=10)

    fig.tight_layout()
    _save(fig, output_path)
    return fig

# Plot 2 — Actual risk-return scatter

def plot_actual_risk_return(
    result_df:   pd.DataFrame,
    output_path: Optional[str] = None,
) -> plt.Figure:
    valid = result_df.dropna(subset=["volatility", "mean_return", "risk_profile"])

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.set_style("whitegrid")

    for profile in PROFILE_ORDER:
        grp = valid[valid["risk_profile"] == profile]
        if grp.empty:
            continue
        ax.scatter(
            grp["volatility"],
            grp["mean_return"],
            color      = PROFILE_COLOURS[profile],
            label      = profile.capitalize(),
            s          = 90,
            alpha      = 0.85,
            edgecolors = "white",
            linewidths = 0.6,
            zorder     = 3,
        )
        for _, row in grp.iterrows():
            ax.annotate(
                row["ticker"],
                (row["volatility"], row["mean_return"]),
                fontsize   = 7,
                ha         = "center",
                va         = "bottom",
                xytext     = (0, 5),
                textcoords = "offset points",
                color      = PROFILE_COLOURS[profile],
            )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean Forward 63-Day Annualised Volatility", fontsize=12)
    ax.set_ylabel("Mean Forward 63-Day Annualised Return",     fontsize=12)
    ax.set_title(
        "Actual Risk-Return by Risk Profile\n"
        "(values averaged across all scored quarters)",
        fontsize=13, fontweight="bold",
    )
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(handles=_legend_patches(), fontsize=10)

    fig.tight_layout()
    _save(fig, output_path)
    return fig

# Plot 3 — RL score vs actual forward volatility scatter

def plot_rl_accuracy_scatter(
    scores_df:   pd.DataFrame,
    fwd_vol_df:  pd.DataFrame,
    result_df:   pd.DataFrame,
    output_path: Optional[str] = None,
) -> plt.Figure:
    ticker_profile = (
        result_df.dropna(subset=["risk_profile"])
        .set_index("ticker")["risk_profile"]
        .to_dict()
    )

    # Identify the train/test boundary from the index
    all_dates  = scores_df.index.sort_values()
    split_date = None

    # Try to infer split from a gap larger than 2× the typical step
    if len(all_dates) > 2:
        diffs = np.diff(all_dates.astype(np.int64))
        median_diff = np.median(diffs)
        gap_idx = np.where(diffs > 2 * median_diff)[0]
        if len(gap_idx) > 0:
            split_date = all_dates[gap_idx[0] + 1]

    all_scores:   list[float] = []
    all_vols:     list[float] = []
    all_profiles: list[str]   = []
    all_splits:   list[str]   = []

    common_tickers = [
        t for t in scores_df.columns
        if t in fwd_vol_df.columns and t in ticker_profile
    ]

    for date in all_dates:
        if date not in fwd_vol_df.index:
            continue
        split = (
            "test"
            if split_date is not None and date >= split_date
            else "train"
        )
        for t in common_tickers:
            s = scores_df.loc[date, t]
            v = fwd_vol_df.loc[date, t]
            if pd.isna(s) or pd.isna(v):
                continue
            all_scores.append(float(s))
            all_vols.append(float(v))
            all_profiles.append(ticker_profile[t])
            all_splits.append(split)

    if not all_scores:
        logger.warning("plot_rl_accuracy_scatter: no data — skipping.")
        fig, ax = plt.subplots()
        ax.set_title("No data")
        _save(fig, output_path)
        return fig

    all_scores_arr   = np.array(all_scores)
    all_vols_arr     = np.array(all_vols)
    all_profiles_arr = np.array(all_profiles)
    all_splits_arr   = np.array(all_splits)

    rho, _ = spearmanr(all_scores_arr, all_vols_arr)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.set_style("whitegrid")

    for profile in PROFILE_ORDER:
        for split, alpha in [("train", TRAIN_ALPHA), ("test", TEST_ALPHA)]:
            mask = (all_profiles_arr == profile) & (all_splits_arr == split)
            if mask.sum() == 0:
                continue
            label = f"{profile.capitalize()} ({'test' if split == 'test' else 'train'})"
            ax.scatter(
                all_scores_arr[mask],
                all_vols_arr[mask],
                color      = PROFILE_COLOURS[profile],
                alpha      = alpha,
                s          = 14,
                edgecolors = "none",
                label      = label,
            )

    # Best-fit line
    m, b    = np.polyfit(all_scores_arr, all_vols_arr, 1)
    x_line  = np.linspace(all_scores_arr.min(), all_scores_arr.max(), 200)
    ax.plot(
        x_line, m * x_line + b,
        color="black", linewidth=1.6, linestyle="--",
        label="Best-fit line",
    )

    ax.set_xlabel("RL Risk Score (model output)",          fontsize=12)
    ax.set_ylabel("Actual Forward 63-Day Realised Vol",    fontsize=12)
    ax.set_title(
        f"RL Predicted Risk vs Actual Forward Volatility\n"
        f"All windows × all tickers   |   Spearman ρ = {rho:.3f}",
        fontsize=13, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=9, ncol=2)

    fig.tight_layout()
    _save(fig, output_path)
    return fig

# Plot 4 — Classification accuracy (Spearman ρ per quarter)

def plot_classification_accuracy(
    eval_df:     pd.DataFrame,
    output_path: Optional[str] = None,
) -> plt.Figure:
    _ltr = {"conservative": 0, "balanced": 1, "aggressive": 2}

    quarters = sorted(eval_df["quarter_start"].dropna().unique())
    rho_vals: list[float] = []
    splits:   list[str]   = []

    for q in quarters:
        q_data = eval_df[
            (eval_df["quarter_start"] == q)
            & eval_df["actual_fwd_vol"].notna()
            & eval_df["risk_profile"].notna()
        ].copy()
        q_data["label_rank"] = q_data["risk_profile"].map(_ltr)
        q_data = q_data.dropna(subset=["label_rank"])

        if len(q_data) < 3:
            rho_vals.append(np.nan)
        else:
            rho, _ = spearmanr(
                q_data["label_rank"].values,
                q_data["actual_fwd_vol"].values,
            )
            rho_vals.append(float(rho) if np.isfinite(rho) else np.nan)

        # Determine split for this quarter
        q_split = eval_df.loc[
            eval_df["quarter_start"] == q, "split"
        ].iloc[0] if "split" in eval_df.columns else "train"
        splits.append(q_split)

    x      = np.arange(len(quarters))
    colours = [
        "#2ecc71" if (v is not None and not np.isnan(v) and v > 0)
        else "#e74c3c"
        for v in rho_vals
    ]
    heights = [v if (v is not None and not np.isnan(v)) else 0.0 for v in rho_vals]

    fig, ax = plt.subplots(figsize=(max(12, len(quarters) * 1.1), 5))
    sns.set_style("whitegrid")

    bars = ax.bar(x, heights, color=colours, alpha=0.82, edgecolor="white")

    # Hatch test bars to distinguish them visually
    for i, (bar, split) in enumerate(zip(bars, splits)):
        if split == "test":
            bar.set_hatch("//")
            bar.set_edgecolor("#555555")

    # Vertical divider between train and test
    if "test" in splits:
        first_test = next(i for i, s in enumerate(splits) if s == "test")
        ax.axvline(
            first_test - 0.5,
            color="navy", linewidth=1.5, linestyle="--", alpha=0.7,
        )
        ax.text(
            first_test - 0.4, ax.get_ylim()[1] * 0.92,
            "← TRAIN    TEST →",
            fontsize=9, color="navy", va="top",
        )

    ax.axhline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(pd.Timestamp(q).date()) for q in quarters],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Quarter Start",  fontsize=12)
    ax.set_ylabel("Spearman ρ",     fontsize=12)
    ax.set_title(
        "Quarterly Classification Accuracy\n"
        "Spearman ρ: predicted label rank vs actual forward volatility  "
        "(hatched = held-out test)",
        fontsize=13, fontweight="bold",
    )

    # Annotate ρ value on each bar
    for i, v in enumerate(rho_vals):
        if v is not None and not np.isnan(v):
            ax.text(
                i, v + (0.03 if v >= 0 else -0.07),
                f"{v:.2f}",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=7, color="black",
            )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#2ecc71", label="ρ > 0  (correct ordering)"),
        mpatches.Patch(facecolor="#e74c3c", label="ρ < 0  (inverted ordering)"),
        mpatches.Patch(
            facecolor="white", edgecolor="#555555",
            hatch="//", label="Test quarter (held-out)",
        ),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower left")

    fig.tight_layout()
    _save(fig, output_path)
    return fig

# Plot 5 — Quarterly Sharpe by risk profile

def plot_quarterly_sharpe(
    eval_df:     pd.DataFrame,
    output_path: Optional[str] = None,
) -> plt.Figure:
   
    grp = (
        eval_df.dropna(subset=["actual_sharpe", "risk_profile"])
        .groupby(["quarter_start", "risk_profile"])["actual_sharpe"]
        .mean()
        .reset_index()
    )

    if grp.empty:
        logger.warning("plot_quarterly_sharpe: no data — skipping.")
        fig, ax = plt.subplots()
        ax.set_title("No data")
        _save(fig, output_path)
        return fig

    quarters = sorted(grp["quarter_start"].unique())
    x        = np.arange(len(quarters))
    width    = 0.25

    # Determine which quarters are test
    quarter_splits: dict = {}
    if "split" in eval_df.columns:
        for q in quarters:
            sp = eval_df.loc[
                eval_df["quarter_start"] == q, "split"
            ].iloc[0]
            quarter_splits[q] = sp

    fig, ax = plt.subplots(figsize=(max(12, len(quarters) * 1.2), 5))
    sns.set_style("whitegrid")

    for offset, profile in enumerate(PROFILE_ORDER):
        pgrp = grp[grp["risk_profile"] == profile]
        vals = []
        for q in quarters:
            row = pgrp[pgrp["quarter_start"] == q]
            vals.append(
                float(row["actual_sharpe"].values[0]) if not row.empty else 0.0
            )
        bar_objects = ax.bar(
            x + (offset - 1) * width,
            vals,
            width      = width,
            color      = PROFILE_COLOURS[profile],
            alpha      = 0.85,
            label      = profile.capitalize(),
            edgecolor  = "white",
        )
        # Hatch test quarter bars
        for bar, q in zip(bar_objects, quarters):
            if quarter_splits.get(q) == "test":
                bar.set_hatch("//")
                bar.set_edgecolor("#555555")

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)

    # Vertical divider between train and test
    if quarter_splits:
        test_quarters = [q for q, s in quarter_splits.items() if s == "test"]
        if test_quarters:
            first_test_x = list(quarters).index(min(test_quarters))
            ax.axvline(
                first_test_x - 0.5,
                color="navy", linewidth=1.5, linestyle="--", alpha=0.7,
            )
            ax.text(
                first_test_x - 0.4,
                ax.get_ylim()[1] * 0.95,
                "← TRAIN    TEST →",
                fontsize=9, color="navy", va="top",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(pd.Timestamp(q).date()) for q in quarters],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_xlabel("Quarter Start",          fontsize=12)
    ax.set_ylabel("Mean Actual Sharpe",     fontsize=12)
    ax.set_title(
        "Quarterly Mean Sharpe Ratio by Risk Profile\n"
        "(hatched bars = held-out test quarters)",
        fontsize=13, fontweight="bold",
    )

    legend_elements = _legend_patches() + [
        mpatches.Patch(
            facecolor="white", edgecolor="#555555",
            hatch="//", label="Test quarter (held-out)",
        ),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    fig.tight_layout()
    _save(fig, output_path)
    return fig

# plot_all — entry point called by main.py and run_rl.py

def plot_all(
    result_df:  pd.DataFrame,
    output_dir: Optional[str]       = None,
    eval_df:    Optional[pd.DataFrame] = None,
) -> None:
    
    plot_cluster_assignments(
        result_df,
        output_path=_path(output_dir, "cluster_assignments.png"),
    )

    plot_actual_risk_return(
        result_df,
        output_path=_path(output_dir, "actual_risk_return.png"),
    )

    if output_dir:
        scores_path  = Path(output_dir) / "rl_dynamic_scores.csv"
        fwd_vol_path = Path(output_dir) / "rl_dynamic_fwd_vol.csv"

        if scores_path.exists() and fwd_vol_path.exists():
            scores_df  = pd.read_csv(
                scores_path,  index_col=0, parse_dates=True
            )
            fwd_vol_df = pd.read_csv(
                fwd_vol_path, index_col=0, parse_dates=True
            )
            
            plot_rl_accuracy_scatter(
                scores_df, fwd_vol_df, result_df,
                output_path=_path(output_dir, "rl_accuracy_scatter.png"),
            )
        else:
            logger.warning(
                "rl_dynamic_scores.csv or rl_dynamic_fwd_vol.csv not found "
                "— skipping rl_accuracy_scatter.png"
            )

    if eval_df is not None and not eval_df.empty:
        plot_classification_accuracy(
            eval_df,
            output_path=_path(output_dir, "classification_accuracy.png"),
        )

        plot_quarterly_sharpe(
            eval_df,
            output_path=_path(output_dir, "quarterly_sharpe.png"),
        )
    else:
        logger.warning(
            "eval_df not provided — skipping classification_accuracy.png "
            "and quarterly_sharpe.png"
        )