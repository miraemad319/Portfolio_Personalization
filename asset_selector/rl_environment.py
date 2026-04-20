"""
rl_environment.py
=================
Gymnasium-compatible rolling-window environment for RL-based asset
risk-tier classification.

Episode structure
-----------------
Each episode is a single chronological pass through the price data:

  t = lookback, lookback + step, lookback + 2*step, …, T - forward - 1

At each step t the agent receives a feature observation built from the
*lookback* window ending at t, then submits a continuous risk score per
ticker.  The reward measures how well those scores rank tickers by their
*forward* realised volatility and max drawdown (composite Spearman).

State  : (n_tickers, 11) float32 array — per-ticker features, z-score
         normalised across tickers at each step.
Action : (n_tickers,)    float32 array — continuous risk score per ticker.
         Higher score = agent believes ticker is riskier.
Reward : 0.6 × Spearman(scores, fwd_vol) + 0.4 × Spearman(scores, fwd_max_dd)

Features (column order)
-----------------------
0  realised_vol    – annualised daily σ over lookback window (decimal)
1  mean_return     – annualised mean log-return
2  sharpe          – annualised Sharpe (zero risk-free rate)
3  max_drawdown    – peak-to-trough drawdown (positive decimal)
4  skewness        – return distribution skewness
5  excess_kurtosis – Fisher excess kurtosis (fat-tail proxy)
6  momentum_21     – 21-day price return (short-term momentum)
7  momentum_63     – 63-day price return (medium-term momentum)
8  vol_trend       – recent 21-day vol / full-window vol (vol acceleration)
9  downside_vol    – semi-deviation of negative returns (annualised)
10 beta            – covariance with cross-sectional mean return / market var
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
FEATURE_NAMES = [
    "realised_vol",
    "mean_return",
    "sharpe",
    "max_drawdown",
    "skewness",
    "excess_kurtosis",
    "momentum_21",
    "momentum_63",
    "vol_trend",
    "downside_vol",
    "beta",
]


# ──────────────────────────────────────────────────────────────────────────────
# Feature helpers (vectorised over a window)
# ──────────────────────────────────────────────────────────────────────────────

def _window_features(
    ret_window: pd.DataFrame,
    px_window: pd.DataFrame,
) -> np.ndarray:
    """
    Compute a (n_tickers, 11) feature matrix for one time window.

    Parameters
    ----------
    ret_window : pd.DataFrame
        Daily log returns for the lookback window.
        DatetimeIndex, columns = tickers.
    px_window : pd.DataFrame
        Close prices for the same window.

    Returns
    -------
    np.ndarray, shape (n_tickers, 11), dtype float32.
        NaN where a ticker has insufficient data.
    """
    n_features = len(FEATURE_NAMES)
    tickers = ret_window.columns.tolist()

    rows: List[List[float]] = []

    for ticker in tickers:
        ret = ret_window[ticker].dropna()
        px  = px_window[ticker].dropna()
        n   = len(ret)

        if n < 5:
            rows.append([np.nan] * n_features)
            continue

        # ── Original 6 features ──────────────────────────────────────────────
        vol = float(ret.std() * np.sqrt(TRADING_DAYS))
        mean_ret = float(ret.mean() * TRADING_DAYS)
        sharpe = float(mean_ret / vol) if vol > 1e-8 else 0.0

        # Bug fix 3: use exp(cumsum) for correct log-return drawdown.
        # (1 + log_ret).cumprod() is a first-order approximation that
        # underestimates cumulative returns during large daily moves (e.g.
        # COVID crash).  exp(cumsum) is exact for log returns.
        if len(px) > 2:
            cum = np.exp(ret.cumsum())
            roll_max = cum.cummax()
            dd = (cum - roll_max) / roll_max
            max_dd = float(abs(dd.min()))
        else:
            max_dd = np.nan

        skew = float(ret.skew()) if n > 4 else np.nan
        kurt = float(ret.kurt()) if n > 4 else np.nan

        # ── 5 new features ───────────────────────────────────────────────────

        # momentum_21: 21-day price return (short-term)
        if len(px) >= 22 and px.iloc[-22] > 0:
            mom_21 = float((px.iloc[-1] / px.iloc[-22]) - 1.0)
        else:
            mom_21 = np.nan

        # momentum_63: 63-day price return (medium-term)
        if len(px) >= 64 and px.iloc[-64] > 0:
            mom_63 = float((px.iloc[-1] / px.iloc[-64]) - 1.0)
        else:
            mom_63 = np.nan

        # vol_trend: recent 21-day vol / full-window vol
        # > 1 means volatility is accelerating (rising risk regime)
        if n >= 21:
            recent_vol = float(ret.iloc[-21:].std() * np.sqrt(TRADING_DAYS))
            vol_trend = float(recent_vol / vol) if vol > 1e-8 else np.nan
        else:
            vol_trend = np.nan

        # downside_vol: semi-deviation of negative returns (annualised)
        neg_ret = ret[ret < 0]
        downside_vol = (
            float(neg_ret.std() * np.sqrt(TRADING_DAYS))
            if len(neg_ret) >= 5 else np.nan
        )

        # Bug fix 1 & 2: beta using pure covariance matrix (no ddof mismatch)
        # and excluding the ticker itself from the market proxy to remove
        # the self-inclusion bias (~1/n_tickers per ticker).
        other_cols = [c for c in ret_window.columns if c != ticker]
        market_ret = ret_window[other_cols].mean(axis=1)
        common_idx = ret.index.intersection(market_ret.dropna().index)
        if len(common_idx) >= 10:
            r_t = ret.loc[common_idx].values
            m_t = market_ret.loc[common_idx].values
            cov_mat = np.cov(r_t, m_t)          # ddof=1 for both entries
            m_var = float(cov_mat[1, 1])
            beta = float(cov_mat[0, 1] / m_var) if m_var > 1e-10 else np.nan
        else:
            beta = np.nan

        rows.append([
            vol, mean_ret, sharpe, max_dd, skew, kurt,
            mom_21, mom_63, vol_trend, downside_vol, beta,
        ])

    return np.array(rows, dtype=np.float32)


def _zscore_normalise(X: np.ndarray) -> np.ndarray:
    """
    Z-score normalise each feature column across tickers.
    Columns that are entirely NaN or have zero std are left as 0.
    """
    out = np.zeros_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        valid = col[np.isfinite(col)]
        if len(valid) < 2:
            continue
        mu, sigma = valid.mean(), valid.std()
        if sigma < 1e-10:
            continue
        out[:, j] = np.where(np.isfinite(col), (col - mu) / sigma, 0.0)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────

class AssetSelectorEnv(gym.Env):
    """
    Gymnasium environment for rolling-window risk-tier RL training.

    Parameters
    ----------
    prices : pd.DataFrame
        Preprocessed close prices. DatetimeIndex, columns = tickers.
        0.0 = delisting marker (treated as NaN for returns).
    lookback : int
        Number of trading days in the observation window.
    forward : int
        Number of trading days ahead used to compute the reward.
    step_size : int
        Number of trading days to advance per step.
    n_clusters : int
        Number of risk tiers (default 3).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: pd.DataFrame,
        lookback: int = 126,
        forward: int = 63,
        step_size: int = 21,
        n_clusters: int = 3,
    ) -> None:
        super().__init__()

        self.prices      = prices.replace(0.0, np.nan)
        self.tickers     = list(prices.columns)
        self.n_tickers   = len(self.tickers)
        self.lookback    = lookback
        self.forward     = forward
        self.step_size   = step_size
        self.n_clusters  = n_clusters
        self.feature_dim = len(FEATURE_NAMES)

        # Pre-compute log returns once for speed
        self.log_returns: pd.DataFrame = np.log(
            self.prices / self.prices.shift(1)
        )

        # Valid step range
        self._start_idx = lookback
        self._end_idx   = len(prices) - forward - 1

        if self._end_idx <= self._start_idx:
            raise ValueError(
                f"Not enough data: need lookback={lookback} + forward={forward} "
                f"< {len(prices)} rows."
            )

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.n_tickers, self.feature_dim),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_tickers,),
            dtype=np.float32,
        )

        self._current_idx: int = self._start_idx
        self._step_count:  int = 0

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._current_idx = self._start_idx
        self._step_count  = 0
        obs = self._get_observation(self._current_idx)
        return obs, {"date": self._current_date()}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Apply action (risk scores), compute reward, advance one step.

        Parameters
        ----------
        action : np.ndarray, shape (n_tickers,)

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        risk_scores = np.asarray(action, dtype=np.float32)
        reward = self._compute_reward(self._current_idx, risk_scores)

        self._current_idx += self.step_size
        self._step_count  += 1

        terminated = self._current_idx >= self._end_idx
        truncated  = False

        if not terminated:
            obs = self._get_observation(self._current_idx)
        else:
            obs = np.zeros(
                (self.n_tickers, self.feature_dim), dtype=np.float32
            )

        info: Dict = {
            "date":       self._current_date(),
            "step":       self._step_count,
            "reward":     reward,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _current_date(self) -> pd.Timestamp:
        idx = min(self._current_idx, len(self.prices) - 1)
        return self.prices.index[idx]

    def _get_valid_mask(self, idx: int) -> np.ndarray:
        """
        Returns bool array (n_tickers,) — True where ticker has >= 5 returns
        in the lookback window ending at idx.  Used to exclude tickers whose
        observation is all-zero (nan→0 fill) from reward and score aggregation.
        """
        ret_win = self.log_returns.iloc[idx - self.lookback : idx]
        return np.array(
            [len(ret_win[t].dropna()) >= 5 for t in self.tickers],
            dtype=bool,
        )

    def _get_observation(self, idx: int) -> np.ndarray:
        """
        Build the (n_tickers, feature_dim) observation for window ending at idx.
        Features are z-score normalised across tickers.
        """
        ret_win = self.log_returns.iloc[idx - self.lookback : idx]
        px_win  = self.prices.iloc[idx - self.lookback : idx]

        raw = _window_features(ret_win, px_win)
        return _zscore_normalise(raw).astype(np.float32)

    def _compute_forward_vol(self, idx: int) -> np.ndarray:
        """
        Compute forward annualised realised vol for each ticker starting at idx.
        Returns np.ndarray shape (n_tickers,), NaN where data is insufficient.
        Extracted so pre-training can reuse it without running a full step.
        """
        fwd_end = min(idx + self.forward, len(self.prices))
        fwd_ret = self.log_returns.iloc[idx:fwd_end]
        return np.array(
            [
                float(fwd_ret[t].dropna().std() * np.sqrt(TRADING_DAYS))
                if len(fwd_ret[t].dropna()) >= 5   # Bug fix 4: >= 5, consistent with lookback
                else np.nan
                for t in self.tickers
            ],
            dtype=np.float32,
        )

    def _compute_forward_ret(self, idx: int) -> np.ndarray:
        """
        Compute forward annualised mean log-return for each ticker starting at idx.
        Returns np.ndarray shape (n_tickers,), NaN where data is insufficient.
        """
        fwd_end = min(idx + self.forward, len(self.prices))
        fwd_ret = self.log_returns.iloc[idx:fwd_end]
        return np.array(
            [
                float(fwd_ret[t].dropna().mean() * TRADING_DAYS)
                if len(fwd_ret[t].dropna()) >= 5   # Bug fix 4: >= 5, consistent with lookback
                else np.nan
                for t in self.tickers
            ],
            dtype=np.float32,
        )

    def _compute_forward_max_dd(self, idx: int) -> np.ndarray:
        """
        Compute forward peak-to-trough max drawdown for each ticker starting
        at idx.  Returns np.ndarray shape (n_tickers,), NaN where insufficient.
        """
        fwd_end = min(idx + self.forward, len(self.prices))
        fwd_px  = self.prices.iloc[idx:fwd_end]
        results = []
        for t in self.tickers:
            px = fwd_px[t].replace(0.0, np.nan).dropna()
            if len(px) >= 5:   # Bug fix 4: >= 5, consistent with lookback
                roll_max = px.cummax()
                dd = (px - roll_max) / roll_max
                results.append(float(abs(dd.min())))
            else:
                results.append(np.nan)
        return np.array(results, dtype=np.float32)

    def _compute_reward(self, idx: int, risk_scores: np.ndarray) -> float:
        """
        Composite reward:
            0.6 × Spearman(scores, fwd_vol) + 0.4 × Spearman(scores, fwd_max_dd)

        Using both forward volatility and forward max drawdown gives the agent
        a richer risk signal — it must learn to rank tickers by both dimensions
        simultaneously, which produces more robust risk scores.

        Tickers with NaN in either dimension OR with insufficient lookback data
        (all-zero observation) are excluded per component.
        Returns 0.0 if fewer than 3 valid tickers for every component.
        """
        fwd_vol  = self._compute_forward_vol(idx)
        fwd_dd   = self._compute_forward_max_dd(idx)
        has_data = self._get_valid_mask(idx)   # exclude phantom all-zero obs

        reward = 0.0

        valid_vol = has_data & np.isfinite(risk_scores) & np.isfinite(fwd_vol)
        if valid_vol.sum() >= 3:
            corr_vol, _ = spearmanr(risk_scores[valid_vol], fwd_vol[valid_vol])
            if np.isfinite(corr_vol):
                reward += 0.6 * corr_vol

        valid_dd = has_data & np.isfinite(risk_scores) & np.isfinite(fwd_dd)
        if valid_dd.sum() >= 3:
            corr_dd, _ = spearmanr(risk_scores[valid_dd], fwd_dd[valid_dd])
            if np.isfinite(corr_dd):
                reward += 0.4 * corr_dd

        return reward

    # ── Inference helpers ─────────────────────────────────────────────────────

    def iter_all_windows(self):
        """
        Iterate through every valid window in chronological order.

        Yields
        ------
        (date: pd.Timestamp, obs: np.ndarray, idx: int, valid_mask: np.ndarray)
            valid_mask is bool (n_tickers,) — True where ticker had real data.
        """
        idx = self._start_idx
        while idx < self._end_idx:
            yield (
                self.prices.index[idx],
                self._get_observation(idx),
                idx,
                self._get_valid_mask(idx),
            )
            idx += self.step_size

    def assign_clusters(
        self, mean_risk_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map per-ticker mean risk scores to cluster IDs and profile strings.

        Stocks are split by tertiles of *valid* scores into:
            cluster 0 → 'conservative'
            cluster 1 → 'balanced'
            cluster 2 → 'aggressive'

        Parameters
        ----------
        mean_risk_scores : np.ndarray, shape (n_tickers,)
            Mean risk scores across training windows (higher = riskier).

        Returns
        -------
        cluster_ids   : np.ndarray (int),  -1 for tickers with NaN scores
        risk_profiles : np.ndarray (str),  '' for NaN tickers
        """
        cluster_ids   = np.full(self.n_tickers, -1, dtype=int)
        risk_profiles = np.full(self.n_tickers, "", dtype=object)

        valid = np.isfinite(mean_risk_scores)
        if valid.sum() < self.n_clusters:
            logger.warning(
                "Only %d valid scores; cannot form %d clusters",
                valid.sum(), self.n_clusters,
            )
            return cluster_ids, risk_profiles

        scores = mean_risk_scores[valid]
        n = len(scores)

        # Find natural cluster boundaries from the RL score distribution.
        # Search all valid pairs of cut points (i, j) and pick the pair that
        # maximises the sum of the two gap sizes, subject to each cluster
        # containing at least min_size tickers.
        #
        # min_size = max(2, n // 10) — roughly 10% per cluster.
        # The previous n // 6 (~16%) was too large and forced everything into
        # the middle "balanced" bucket when the score distribution is dense.
        min_size = max(2, n // 10)
        sorted_scores = np.sort(scores)
        gaps = np.diff(sorted_scores)

        best_t1, best_t2, best_gap_sum = None, None, -np.inf
        for i in range(min_size - 1, n - min_size - 1):
            for j in range(i + min_size, n - 1):
                gap_sum = gaps[i] + gaps[j]
                if gap_sum > best_gap_sum:
                    best_gap_sum = gap_sum
                    best_t1 = (sorted_scores[i] + sorted_scores[i + 1]) / 2
                    best_t2 = (sorted_scores[j] + sorted_scores[j + 1]) / 2

        ids_candidate = np.where(
            scores > best_t2, 2,
            np.where(scores > best_t1, 1, 0),
        )

        # Sanity check: if any single cluster holds > 55% of tickers, the
        # gap-search found poor boundaries — fall back to tertile split.
        cluster_sizes = np.bincount(ids_candidate, minlength=3)
        if cluster_sizes.max() > 0.55 * n:
            logger.warning(
                "Gap-search produced imbalanced clusters %s (>55%% in one bin). "
                "Falling back to tertile split.",
                cluster_sizes.tolist(),
            )
            q33, q67 = np.percentile(sorted_scores, [33.3, 66.7])
            best_t1, best_t2 = float(q33), float(q67)
            ids_candidate = np.where(
                scores > best_t2, 2,
                np.where(scores > best_t1, 1, 0),
            )

        ids = ids_candidate

        cluster_ids[valid]   = ids
        risk_profiles[valid] = np.where(
            ids == 2, "aggressive",
            np.where(ids == 1, "balanced", "conservative"),
        )
        return cluster_ids, risk_profiles

    # ── Quarterly helpers ─────────────────────────────────────────────────────

    def collect_quarterly_windows(self) -> List[Dict]:
        """
        Group rolling-window step indices into non-overlapping 63-day quarters.

        The first quarter spans [_start_idx, _start_idx + 63), the second
        [_start_idx + 63, _start_idx + 126), etc.  Because step_size=21 divides
        63 evenly, every quarter boundary is also a valid step index.

        Returns
        -------
        List of dicts, each containing:
            quarter_start  : pd.Timestamp — date at quarter_idx in prices
            quarter_idx    : int          — row index of the quarter's first window
            window_indices : List[int]    — all step idx values in [q_start, q_start+63)
        """
        quarter_size = 63
        quarters: List[Dict] = []

        all_step_indices = list(range(self._start_idx, self._end_idx, self.step_size))
        if not all_step_indices:
            return quarters

        q_num = 0
        while True:
            q_start_idx = self._start_idx + q_num * quarter_size
            q_end_idx   = q_start_idx + quarter_size

            if q_start_idx >= self._end_idx:
                break

            window_indices = [i for i in all_step_indices if q_start_idx <= i < q_end_idx]

            if window_indices:
                quarters.append({
                    "quarter_start":  self.prices.index[q_start_idx],
                    "quarter_idx":    q_start_idx,
                    "window_indices": window_indices,
                })

            q_num += 1

        return quarters

    def compute_sharpe(self, idx: int, forward: int) -> np.ndarray:
        """
        Compute per-ticker annualised Sharpe ratio over log_returns[idx:idx+forward].

        Sharpe = (mean_log_return × 252) / (std_log_return × √252)
        Zero risk-free rate.  Returns NaN for tickers with fewer than 5 valid
        log returns in the window.

        Parameters
        ----------
        idx     : int — starting row index in the prices/log_returns DataFrame
        forward : int — number of trading days to look forward

        Returns
        -------
        np.ndarray shape (n_tickers,), dtype float64
        """
        fwd_end = min(idx + forward, len(self.prices))
        fwd_ret = self.log_returns.iloc[idx:fwd_end]
        results: List[float] = []
        for t in self.tickers:
            ret = fwd_ret[t].dropna()
            if len(ret) < 5:
                results.append(np.nan)
                continue
            mean_r = float(ret.mean() * TRADING_DAYS)
            std_r  = float(ret.std()  * np.sqrt(TRADING_DAYS))
            results.append(mean_r / std_r if std_r > 1e-8 else np.nan)
        return np.array(results, dtype=np.float64)
