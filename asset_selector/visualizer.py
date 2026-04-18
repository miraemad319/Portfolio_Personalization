"""
visualizer.py
=============
Visualisation utilities for the EGX30 Asset Selector pipeline.

Functions
---------
plot_volatility_distribution(result_df, output_path=None)
    Histogram of annualised realised volatility, colour-coded by risk profile.

plot_cluster_assignments(result_df, output_path=None)
    Horizontal bar chart of tickers ranked by volatility with colour bands.

plot_cluster_scatter(result_df, output_path=None)
    Scatter plot (1-D projection) showing cluster separation.

plot_actual_risk_return(result_df, output_path=None)
    Risk-return scatter: actual annualised volatility (x) vs actual
    annualised return (y), coloured by RL-assigned risk profile.

plot_predicted_risk_return(result_df, output_path=None)
    Risk-return scatter: RL risk score (x, the model's predicted risk rank)
    vs actual annualised return (y), coloured by risk profile.  Illustrates
    how the model's risk ordering aligns with realised return outcomes.

save_or_show(fig, output_path)
    Save to file if path is given, otherwise display interactively.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Colour palette for the three risk profiles
PROFILE_COLOURS: dict[str, str] = {
    "conservative": "#2ecc71",   # green
    "balanced":     "#f39c12",   # orange
    "aggressive":   "#e74c3c",   # red
}
PROFILE_ORDER = ["conservative", "balanced", "aggressive"]


def _legend_patches() -> list[mpatches.Patch]:
    return [
        mpatches.Patch(color=PROFILE_COLOURS[p], label=p.capitalize())
        for p in PROFILE_ORDER
    ]


def save_or_show(fig: plt.Figure, output_path: Optional[str]) -> None:
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Plot saved to %s", path)
        plt.close(fig)
    else:
        plt.show()


def plot_volatility_distribution(
    result_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Histogram of annualised realised volatility, colour-coded by risk profile.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output of asset_selector.classify_assets().
    output_path : str, optional
        If given, save the figure to this path instead of displaying it.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_style("whitegrid")

    valid = result_df.dropna(subset=["volatility", "risk_profile"])
    bins = np.linspace(valid["volatility"].min() * 0.9, valid["volatility"].max() * 1.1, 20)

    for profile in PROFILE_ORDER:
        grp = valid[valid["risk_profile"] == profile]["volatility"]
        if grp.empty:
            continue
        ax.hist(
            grp,
            bins=bins,
            color=PROFILE_COLOURS[profile],
            alpha=0.7,
            label=profile.capitalize(),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Mean Forward 63-Day Annualised Volatility", fontsize=12)
    ax.set_ylabel("Number of Tickers", fontsize=12)
    ax.set_title("EGX30 Forward Volatility Distribution by Risk Profile", fontsize=14, fontweight="bold")
    ax.legend(handles=_legend_patches(), fontsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    fig.tight_layout()
    save_or_show(fig, output_path)
    return fig


def plot_cluster_assignments(
    result_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart: one bar per ticker, ranked by volatility,
    coloured by risk profile.
    """
    valid = (
        result_df.dropna(subset=["volatility", "risk_profile"])
        .sort_values("volatility")
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(12, max(6, len(valid) * 0.3)))
    sns.set_style("whitegrid")

    colours = valid["risk_profile"].map(PROFILE_COLOURS).fillna("#95a5a6")
    bars = ax.barh(valid["ticker"], valid["volatility"], color=colours, edgecolor="white")

    # Annotate values
    for bar, vol in zip(bars, valid["volatility"]):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{vol:.1%}",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_xlabel("Mean Forward 63-Day Annualised Volatility", fontsize=12)
    ax.set_title("EGX30 Ticker Risk Classification", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(handles=_legend_patches(), loc="lower right", fontsize=10)

    fig.tight_layout()
    save_or_show(fig, output_path)
    return fig


def plot_cluster_scatter(
    result_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    1-D scatter plot (jittered) showing cluster separation along the
    volatility axis.
    """
    valid = result_df.dropna(subset=["volatility", "risk_profile"])

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.set_style("whitegrid")

    rng = np.random.default_rng(0)
    for profile in PROFILE_ORDER:
        grp = valid[valid["risk_profile"] == profile]
        if grp.empty:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(grp))
        ax.scatter(
            grp["volatility"],
            jitter,
            color=PROFILE_COLOURS[profile],
            label=profile.capitalize(),
            s=80,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        # Label ticker symbols
        for _, row in grp.iterrows():
            ax.annotate(
                row["ticker"],
                (row["volatility"], jitter[grp.index.get_loc(row.name)]),
                fontsize=6,
                ha="center",
                va="bottom",
                xytext=(0, 6),
                textcoords="offset points",
                color=PROFILE_COLOURS[profile],
            )

    ax.set_xlabel("Mean Forward 63-Day Annualised Volatility", fontsize=12)
    ax.set_yticks([])
    ax.set_title("EGX30 Risk Cluster Scatter", fontsize=14, fontweight="bold")
    ax.legend(handles=_legend_patches(), fontsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    fig.tight_layout()
    save_or_show(fig, output_path)
    return fig


def plot_actual_risk_return(
    result_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Risk-return scatter using actual (realised) values.

    X-axis : mean forward 63-day annualised volatility, averaged across all
             rolling windows  (the actual risk the RL model was trained to predict)
    Y-axis : mean forward 63-day annualised return, averaged across all windows
    Colour : RL-assigned risk profile

    Each point is labelled with its ticker symbol.  A horizontal zero-return
    line is drawn for reference.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output of asset_selector.classify_assets().
        Must contain columns: ticker, volatility, mean_return, risk_profile.
    output_path : str, optional
        If given, save the figure to this path instead of displaying it.
    """
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
            color=PROFILE_COLOURS[profile],
            label=profile.capitalize(),
            s=90,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        for _, row in grp.iterrows():
            ax.annotate(
                row["ticker"],
                (row["volatility"], row["mean_return"]),
                fontsize=7,
                ha="left",
                va="bottom",
                xytext=(4, 3),
                textcoords="offset points",
                color=PROFILE_COLOURS[profile],
            )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_xlabel("Mean Forward 63-Day Volatility (Actual Risk)", fontsize=12)
    ax.set_ylabel("Mean Forward 63-Day Return (Actual Return)", fontsize=12)
    ax.set_title(
        "Actual Forward Risk-Return by RL Risk Profile",
        fontsize=14, fontweight="bold",
    )
    ax.legend(handles=_legend_patches(), fontsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    fig.tight_layout()
    save_or_show(fig, output_path)
    return fig


def plot_predicted_risk_return(
    result_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    RL predicted risk vs actual return scatter.

    X-axis : RL risk score (model's predicted risk ranking, higher = riskier)
    Y-axis : mean forward 63-day annualised return, averaged across all windows
    Colour : RL-assigned risk profile

    This plot evaluates whether the RL model's risk ordering aligns with the
    realised forward risk-return tradeoff.  Both axes are averaged over the
    same rolling windows used during training, so the comparison is consistent:
    stocks with higher predicted risk should cluster toward higher forward
    volatility and potentially more dispersed forward returns.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output of asset_selector.classify_assets().
        Must contain columns: ticker, rl_risk_score, mean_return, risk_profile.
    output_path : str, optional
        If given, save the figure to this path instead of displaying it.
    """
    valid = result_df.dropna(subset=["rl_risk_score", "mean_return", "risk_profile"])

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.set_style("whitegrid")

    for profile in PROFILE_ORDER:
        grp = valid[valid["risk_profile"] == profile]
        if grp.empty:
            continue
        ax.scatter(
            grp["rl_risk_score"],
            grp["mean_return"],
            color=PROFILE_COLOURS[profile],
            label=profile.capitalize(),
            s=90,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        for _, row in grp.iterrows():
            ax.annotate(
                row["ticker"],
                (row["rl_risk_score"], row["mean_return"]),
                fontsize=7,
                ha="left",
                va="bottom",
                xytext=(4, 3),
                textcoords="offset points",
                color=PROFILE_COLOURS[profile],
            )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_xlabel("RL Risk Score (Predicted Risk)", fontsize=12)
    ax.set_ylabel("Mean Forward 63-Day Return (Actual Return)", fontsize=12)
    ax.set_title(
        "RL Predicted Risk vs Actual Forward Return",
        fontsize=14, fontweight="bold",
    )
    ax.legend(handles=_legend_patches(), fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    fig.tight_layout()
    save_or_show(fig, output_path)
    return fig


def plot_rl_vs_actual_vol_scatter(
    scores_df: pd.DataFrame,
    fwd_vol_df: pd.DataFrame,
    result_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot of RL risk score (x) vs actual forward realised volatility (y)
    across all rolling windows and all tickers.

    Each dot is one (ticker, window) observation.  If the RL model is correct,
    higher scores should map to higher realised forward volatility — we expect
    a positive slope.  The Spearman correlation is shown in the title.

    Parameters
    ----------
    scores_df  : pd.DataFrame  — rl_dynamic_scores.csv  (windows × tickers)
    fwd_vol_df : pd.DataFrame  — rl_dynamic_fwd_vol.csv (windows × tickers)
    result_df  : pd.DataFrame  — asset_classification.csv
    output_path : str, optional
    """
    from scipy.stats import spearmanr

    profile_map = (
        result_df.dropna(subset=["risk_profile"])
        .set_index("ticker")["risk_profile"]
        .to_dict()
    )

    all_scores, all_vols, all_profiles = [], [], []

    for ticker in scores_df.columns:
        if ticker not in fwd_vol_df.columns:
            continue
        s = scores_df[ticker].dropna().values
        v = fwd_vol_df[ticker].dropna().values
        # Align by index (same windows)
        s_idx = scores_df[ticker].dropna().index
        v_idx = fwd_vol_df[ticker].dropna().index
        common = s_idx.intersection(v_idx)
        if len(common) < 3:
            continue
        s_vals = scores_df.loc[common, ticker].values
        v_vals = fwd_vol_df.loc[common, ticker].values
        all_scores.extend(s_vals)
        all_vols.extend(v_vals)
        all_profiles.extend([profile_map.get(ticker, "balanced")] * len(common))

    all_scores  = np.array(all_scores)
    all_vols    = np.array(all_vols)
    all_profiles = np.array(all_profiles)

    rho, _ = spearmanr(all_scores, all_vols)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.set_style("whitegrid")

    for profile in PROFILE_ORDER:
        mask = all_profiles == profile
        if mask.sum() == 0:
            continue
        ax.scatter(
            all_scores[mask],
            all_vols[mask],
            color   = PROFILE_COLOURS[profile],
            label   = profile.capitalize(),
            s       = 18,
            alpha   = 0.45,
            edgecolors = "none",
        )

    # Best-fit line
    m, b = np.polyfit(all_scores, all_vols, 1)
    x_line = np.linspace(all_scores.min(), all_scores.max(), 100)
    ax.plot(x_line, m * x_line + b, color="black", linewidth=1.5,
            linestyle="--", label="Best-fit line")

    ax.set_xlabel("RL Risk Score (model's predicted risk)", fontsize=12)
    ax.set_ylabel("Actual Forward 63-Day Realised Volatility", fontsize=12)
    ax.set_title(
        f"RL Predicted Risk vs Actual Forward Volatility\n"
        f"(all windows × all tickers)   Spearman ρ = {rho:.3f}",
        fontsize=13, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(handles=_legend_patches() + [
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.5,
                   label="Best-fit line")
    ], fontsize=10)

    fig.tight_layout()
    save_or_show(fig, output_path)
    return fig


def plot_rl_vs_actual_vol_timeseries(
    scores_df: pd.DataFrame,
    fwd_vol_df: pd.DataFrame,
    result_df: pd.DataFrame,
    n_per_bucket: int = 3,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grid of per-ticker time-series plots: RL risk score vs actual forward vol.

    Selects n_per_bucket representative stocks from each risk profile and
    plots two lines for each:
      • Orange/solid  — RL risk score (right Y-axis, normalised to [0,1])
      • Blue/dashed   — Actual forward realised volatility (left Y-axis, %)

    If the model is correct, the two lines should rise and fall together.

    Parameters
    ----------
    scores_df     : pd.DataFrame  — rl_dynamic_scores.csv
    fwd_vol_df    : pd.DataFrame  — rl_dynamic_fwd_vol.csv
    result_df     : pd.DataFrame  — asset_classification.csv
    n_per_bucket  : int           — stocks sampled per risk profile (default 3)
    output_path   : str, optional
    """
    # Pick representative tickers: median-vol stock from each profile
    picks: list[str] = []
    for profile in PROFILE_ORDER:
        grp = (
            result_df[result_df["risk_profile"] == profile]
            .dropna(subset=["volatility"])
            .sort_values("volatility")
        )
        if grp.empty:
            continue
        # Spread across low / mid / high within the profile
        indices = np.linspace(0, len(grp) - 1, min(n_per_bucket, len(grp)), dtype=int)
        picks.extend(grp.iloc[indices]["ticker"].tolist())

    n_plots = len(picks)
    n_cols  = n_per_bucket
    n_rows  = len(PROFILE_ORDER)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        constrained_layout=True,
    )
    axes = np.array(axes).reshape(n_rows, n_cols)

    profile_order_map = {p: i for i, p in enumerate(PROFILE_ORDER)}

    ticker_profile = (
        result_df.dropna(subset=["risk_profile"])
        .set_index("ticker")["risk_profile"]
        .to_dict()
    )

    for ticker in picks:
        profile = ticker_profile.get(ticker, "balanced")
        row     = profile_order_map[profile]

        # find column slot within this row
        col = next(
            (j for j in range(n_cols)
             if axes[row, j].get_title() == ""),
            None,
        )
        if col is None:
            continue

        ax = axes[row, col]

        if ticker not in scores_df.columns or ticker not in fwd_vol_df.columns:
            ax.set_visible(False)
            continue

        s = scores_df[ticker].dropna()
        v = fwd_vol_df[ticker].dropna()
        common = s.index.intersection(v.index)
        if len(common) < 3:
            ax.set_visible(False)
            continue

        s_vals = s.loc[common]
        v_vals = v.loc[common]

        # Normalise RL score to [0, 1] so both lines fit on a readable scale
        s_min, s_max = s_vals.min(), s_vals.max()
        s_norm = (s_vals - s_min) / (s_max - s_min + 1e-8)

        colour = PROFILE_COLOURS[profile]

        ax2 = ax.twinx()
        ax.plot(
            common, v_vals * 100,
            color="steelblue", linewidth=1.8, linestyle="--",
            label="Actual fwd vol (%)",
        )
        ax2.plot(
            common, s_norm,
            color=colour, linewidth=1.8, linestyle="-",
            label="RL score (normalised)",
        )

        ax.set_title(f"{ticker}  [{profile}]", fontsize=10,
                     color=colour, fontweight="bold")
        ax.set_ylabel("Actual Vol (%)", fontsize=8, color="steelblue")
        ax2.set_ylabel("RL Score (0–1)", fontsize=8, color=colour)
        ax.tick_params(axis="y", labelcolor="steelblue", labelsize=7)
        ax2.tick_params(axis="y", labelcolor=colour, labelsize=7)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax2.set_ylim(-0.05, 1.05)

    # Hide any unused axes
    for r in range(n_rows):
        for c in range(n_cols):
            if axes[r, c].get_title() == "":
                axes[r, c].set_visible(False)

    # Row labels
    for profile, row_idx in profile_order_map.items():
        fig.text(
            0.01, 1 - (row_idx + 0.5) / n_rows,
            profile.upper(),
            va="center", ha="left",
            fontsize=11, fontweight="bold",
            color=PROFILE_COLOURS[profile],
            rotation=90,
        )

    # Shared legend
    handles = [
        plt.Line2D([0], [0], color="steelblue", linestyle="--",
                   linewidth=1.8, label="Actual forward vol"),
        plt.Line2D([0], [0], color="grey", linestyle="-",
                   linewidth=1.8, label="RL risk score (norm. 0–1)"),
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "RL Risk Score vs Actual Forward Volatility — Per Stock Time Series\n"
        "(lines moving together = model tracking real risk correctly)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    save_or_show(fig, output_path)
    return fig


def plot_all(
    result_df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> None:
    """
    Render all plots.  If output_dir is set, save to that directory.

    Plots generated
    ---------------
    vol_distribution.png       – volatility histogram by risk profile
    cluster_assignments.png    – horizontal bar chart ranked by volatility
    cluster_scatter.png        – 1-D jitter scatter of cluster separation
    actual_risk_return.png     – actual vol vs actual return, coloured by profile
    predicted_risk_return.png  – RL risk score vs actual return, coloured by profile
    """
    def _path(name: str) -> Optional[str]:
        return f"{output_dir}/{name}" if output_dir else None

    plot_volatility_distribution(result_df, output_path=_path("vol_distribution.png"))
    plot_cluster_assignments(result_df, output_path=_path("cluster_assignments.png"))
    plot_cluster_scatter(result_df, output_path=_path("cluster_scatter.png"))
    plot_actual_risk_return(result_df, output_path=_path("actual_risk_return.png"))
    plot_predicted_risk_return(result_df, output_path=_path("predicted_risk_return.png"))

    # RL accuracy plots — only if the dynamic files exist
    if output_dir:
        import pandas as pd
        from pathlib import Path
        scores_path  = Path(output_dir) / "rl_dynamic_scores.csv"
        fwd_vol_path = Path(output_dir) / "rl_dynamic_fwd_vol.csv"
        if scores_path.exists() and fwd_vol_path.exists():
            scores_df  = pd.read_csv(scores_path,  index_col=0, parse_dates=True)
            fwd_vol_df = pd.read_csv(fwd_vol_path, index_col=0, parse_dates=True)
            plot_rl_vs_actual_vol_scatter(
                scores_df, fwd_vol_df, result_df,
                output_path=_path("rl_accuracy_scatter.png"),
            )
            plot_rl_vs_actual_vol_timeseries(
                scores_df, fwd_vol_df, result_df,
                output_path=_path("rl_accuracy_timeseries.png"),
            )
