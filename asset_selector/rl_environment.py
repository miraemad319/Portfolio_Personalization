from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Optional, Tuple

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

# Feature helpers


def _window_features(
    ret_window: pd.DataFrame,
    px_window: pd.DataFrame,
) -> np.ndarray:
    """
    Compute a (n_tickers, 11) feature matrix for one time window.

    Parameters
    ret_window : pd.DataFrame
        Daily log returns for the lookback window.
    px_window : pd.DataFrame
        Close prices for the same window.

    Returns
    np.ndarray shape (n_tickers, 11), dtype float32.
    NaN for tickers with fewer than 5 valid returns.
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

        # Core risk features 
        vol      = float(ret.std() * np.sqrt(TRADING_DAYS))
        mean_ret = float(ret.mean() * TRADING_DAYS)
        sharpe   = float(mean_ret / vol) if vol > 1e-8 else 0.0

        # Drawdown: exp(cumsum) is exact for log returns
        # Guard checks len(ret) not len(px) — ret drives the calculation
        if n > 2:
            cum      = np.exp(ret.cumsum())
            roll_max = cum.cummax()
            dd       = (cum - roll_max) / roll_max
            max_dd   = float(abs(dd.min()))
        else:
            max_dd = np.nan

        skew = float(ret.skew()) if n > 4 else np.nan
        kurt = float(ret.kurt()) if n > 4 else np.nan

        # Momentum features 
        if len(px) >= 22 and px.iloc[-22] > 0:
            mom_21 = float((px.iloc[-1] / px.iloc[-22]) - 1.0)
        else:
            mom_21 = np.nan

        if len(px) >= 64 and px.iloc[-64] > 0:
            mom_63 = float((px.iloc[-1] / px.iloc[-64]) - 1.0)
        else:
            mom_63 = np.nan

        # Volatility trend 
        if n >= 21:
            recent_vol = float(ret.iloc[-21:].std() * np.sqrt(TRADING_DAYS))
            vol_trend  = float(recent_vol / vol) if vol > 1e-8 else np.nan
        else:
            vol_trend = np.nan

        # Downside volatility 
        neg_ret      = ret[ret < 0]
        downside_vol = (
            float(neg_ret.std() * np.sqrt(TRADING_DAYS))
            if len(neg_ret) >= 5 else np.nan
        )

        # Beta (self-exclusion to remove ~1/N bias) 
        other_cols = [c for c in ret_window.columns if c != ticker]
        market_ret = ret_window[other_cols].mean(axis=1)
        common_idx = ret.index.intersection(market_ret.dropna().index)
        if len(common_idx) >= 10:
            r_t     = ret.loc[common_idx].values
            m_t     = market_ret.loc[common_idx].values
            cov_mat = np.cov(r_t, m_t)          # ddof=1 for both
            m_var   = float(cov_mat[1, 1])
            beta    = float(cov_mat[0, 1] / m_var) if m_var > 1e-10 else np.nan
        else:
            beta = np.nan

        rows.append([
            vol, mean_ret, sharpe, max_dd, skew, kurt,
            mom_21, mom_63, vol_trend, downside_vol, beta,
        ])

    return np.array(rows, dtype=np.float32)


def _zscore_normalise(X: np.ndarray) -> np.ndarray:
    """
    Cross-sectional z-score: normalise each feature column across tickers.
    Columns that are entirely NaN or have zero std are set to 0.
    No cross-time information is introduced — only the ~60 tickers at
    this single time step are used to compute mean and std.
    """
    out = np.zeros_like(X)
    for j in range(X.shape[1]):
        col   = X[:, j]
        valid = col[np.isfinite(col)]
        if len(valid) < 2:
            continue
        mu, sigma = valid.mean(), valid.std()
        if sigma < 1e-10:
            continue
        out[:, j] = np.where(np.isfinite(col), (col - mu) / sigma, 0.0)
    return out

# Environment

class AssetSelectorEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices:        pd.DataFrame,
        lookback:      int           = 126,
        forward:       int           = 63,
        step_size:     int           = 21,
        n_clusters:    int           = 3,
        train_end_idx: Optional[int] = None,
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

        # Pre-compute log returns once — 0.0 already replaced above
        self.log_returns: pd.DataFrame = np.log(
            self.prices / self.prices.shift(1)
        )

        # Index landmarks 
        # First valid step: need a full lookback window
        self._start_idx = lookback

        # Last valid step overall: need a full forward window after it
        self._full_end_idx = len(prices) - forward - 1

        if train_end_idx is not None:
            # _end_idx is the PPO episode boundary.
            # A step at idx uses forward data [idx, idx+forward).
            # To avoid leaking test data into training, the last training step must satisfy: idx + forward <= train_end_idx
            # Therefore: idx <= train_end_idx - forward
            self._train_end_idx  = int(train_end_idx)
            self._end_idx        = min(
                self._train_end_idx - self.forward,
                self._full_end_idx,
            )
            # Test period starts immediately after the training boundary
            self._test_start_idx: Optional[int] = self._train_end_idx + 1
        else:
            self._train_end_idx  = self._full_end_idx + self.forward  # sentinel
            self._end_idx        = self._full_end_idx
            self._test_start_idx = None

        if self._end_idx <= self._start_idx:
            raise ValueError(
                f"Training period too short after applying lookback={lookback} "
                f"and forward={forward} constraints. "
                f"Got _start_idx={self._start_idx}, _end_idx={self._end_idx}."
            )

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.n_tickers, self.feature_dim),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_tickers,),
            dtype=np.float32,
        )

        self._current_idx: int = self._start_idx
        self._step_count:  int = 0

        logger.info(
            "AssetSelectorEnv: %d tickers  %d rows  "
            "lookback=%d  forward=%d  step=%d",
            self.n_tickers, len(prices), lookback, forward, step_size,
        )
        logger.info(
            "Index landmarks: start=%d  train_end=%d  "
            "test_start=%s  full_end=%d",
            self._start_idx,
            self._end_idx,
            str(self._test_start_idx),
            self._full_end_idx,
        )

    # Gymnasium interface 

    def reset(
        self,
        *,
        seed:    Optional[int]  = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._current_idx = self._start_idx
        self._step_count  = 0
        obs = self._get_observation(self._current_idx)
        return obs, {"date": self._current_date()}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Apply action (risk scores), compute reward, advance one step.
        Episodes are bounded by _end_idx (training boundary only).
        """
        risk_scores = np.asarray(action, dtype=np.float32)
        reward      = self._compute_reward(self._current_idx, risk_scores)

        self._current_idx += self.step_size
        self._step_count  += 1

        terminated = self._current_idx >= self._end_idx
        truncated  = False

        obs = (
            self._get_observation(self._current_idx)
            if not terminated
            else np.zeros((self.n_tickers, self.feature_dim), dtype=np.float32)
        )

        return obs, reward, terminated, truncated, {
            "date":   self._current_date(),
            "step":   self._step_count,
            "reward": reward,
        }

    def render(self) -> None:
        pass

    # Observation and masking 

    def _current_date(self) -> pd.Timestamp:
        idx = min(self._current_idx, len(self.prices) - 1)
        return self.prices.index[idx]

    def _get_valid_mask(self, idx: int) -> np.ndarray:
        """
        Bool array (n_tickers,) — True where ticker has >= 5 log returns
        in the lookback window ending at idx.
        Excludes tickers whose observation is all-zero (delisting marker).
        """
        ret_win = self.log_returns.iloc[idx - self.lookback : idx]
        return np.array(
            [len(ret_win[t].dropna()) >= 5 for t in self.tickers],
            dtype=bool,
        )

    def _get_observation(self, idx: int) -> np.ndarray:
        """
        Build the (n_tickers, feature_dim) observation for window ending at idx.
        Uses only data in [idx-lookback, idx) — strictly past-only.
        """
        ret_win = self.log_returns.iloc[idx - self.lookback : idx]
        px_win  = self.prices.iloc[idx - self.lookback : idx]
        raw     = _window_features(ret_win, px_win)
        return _zscore_normalise(raw).astype(np.float32)

    # Forward metrics (reward and evaluation) 

    def _compute_forward_vol(self, idx: int) -> np.ndarray:
        """
        Annualised realised vol over [idx, idx+forward).
        Returns (n_tickers,) float32, NaN where < 5 valid returns.
        """
        fwd_end = min(idx + self.forward, len(self.prices))
        fwd_ret = self.log_returns.iloc[idx:fwd_end]
        return np.array(
            [
                float(fwd_ret[t].dropna().std() * np.sqrt(TRADING_DAYS))
                if len(fwd_ret[t].dropna()) >= 5 else np.nan
                for t in self.tickers
            ],
            dtype=np.float32,
        )

    def _compute_forward_ret(self, idx: int) -> np.ndarray:
        """
        Annualised mean log-return over [idx, idx+forward).
        Returns (n_tickers,) float32, NaN where < 5 valid returns.
        """
        fwd_end = min(idx + self.forward, len(self.prices))
        fwd_ret = self.log_returns.iloc[idx:fwd_end]
        return np.array(
            [
                float(fwd_ret[t].dropna().mean() * TRADING_DAYS)
                if len(fwd_ret[t].dropna()) >= 5 else np.nan
                for t in self.tickers
            ],
            dtype=np.float32,
        )

    def _compute_forward_max_dd(self, idx: int) -> np.ndarray:
        """
        Forward peak-to-trough max drawdown over [idx, idx+forward).
        Returns (n_tickers,) float32, NaN where < 5 valid prices.
        """
        fwd_end = min(idx + self.forward, len(self.prices))
        fwd_px  = self.prices.iloc[idx:fwd_end]
        results = []
        for t in self.tickers:
            px = fwd_px[t].replace(0.0, np.nan).dropna()
            if len(px) >= 5:
                roll_max = px.cummax()
                dd       = (px - roll_max) / roll_max
                results.append(float(abs(dd.min())))
            else:
                results.append(np.nan)
        return np.array(results, dtype=np.float32)

    def compute_sharpe(self, idx: int, forward: int) -> np.ndarray:
        """
        Annualised Sharpe ratio over [idx, idx+forward).
        Sharpe = (mean_log_return × 252) / (std × √252), zero risk-free rate.
        Returns (n_tickers,) float64, NaN where < 5 valid returns.
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

    def _compute_reward(self, idx: int, risk_scores: np.ndarray) -> float:
        """
        Composite Spearman reward:
            0.6 × Spearman(scores, fwd_vol) + 0.4 × Spearman(scores, fwd_max_dd)

        Tickers with NaN in either forward metric or with insufficient lookback
        data are excluded per component.
        Returns 0.0 if fewer than 3 valid tickers for every component.

        """
        fwd_vol  = self._compute_forward_vol(idx)
        fwd_dd   = self._compute_forward_max_dd(idx)
        has_data = self._get_valid_mask(idx)

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

    # Inference iteration 

    def iter_all_windows(
        self,
        start_idx: Optional[int] = None,
        end_idx:   Optional[int] = None,
    ) -> Iterator[Tuple[pd.Timestamp, np.ndarray, int, np.ndarray]]:
        
        idx_s = start_idx if start_idx is not None else self._start_idx
        idx_e = end_idx   if end_idx   is not None else self._end_idx
        idx   = idx_s
        while idx < idx_e:
            yield (
                self.prices.index[min(idx, len(self.prices) - 1)],
                self._get_observation(idx),
                idx,
                self._get_valid_mask(idx),
            )
            idx += self.step_size

    # Quarterly grouping 

    def collect_quarterly_windows(
        self,
        start_idx: Optional[int] = None,
        end_idx:   Optional[int] = None,
    ) -> List[Dict]:
        """
        Group step indices into non-overlapping 63-day quarters.

        Quarters are defined relative to start_idx so that the training
        and test periods each produce a self-contained quarter sequence
        with no gaps or overlaps between them.

        """
        idx_s        = start_idx if start_idx is not None else self._start_idx
        idx_e        = end_idx   if end_idx   is not None else self._end_idx
        quarter_size = self.forward   # 63 trading days = 1 quarter

        all_steps = list(range(idx_s, idx_e, self.step_size))
        if not all_steps:
            return []

        quarters: List[Dict] = []
        q_num = 0
        while True:
            q_start_idx = idx_s + q_num * quarter_size
            q_end_idx   = q_start_idx + quarter_size
            if q_start_idx >= idx_e:
                break
            window_indices = [i for i in all_steps if q_start_idx <= i < q_end_idx]
            if window_indices:
                quarters.append({
                    "quarter_start":  self.prices.index[
                        min(q_start_idx, len(self.prices) - 1)
                    ],
                    "quarter_idx":    q_start_idx,
                    "window_indices": window_indices,
                })
            q_num += 1

        return quarters

    # Cluster assignment 

    def assign_clusters(
        self,
        mean_risk_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map per-ticker mean risk scores → cluster IDs and profile strings.

        Scores are cross-sectionally standardised before gap-search so that
        compressed actor outputs (common early in training or with a narrow
        score range) still produce meaningful cluster boundaries.

            cluster 0 → 'conservative'
            cluster 1 → 'balanced'
            cluster 2 → 'aggressive'
        """
        cluster_ids   = np.full(self.n_tickers, -1, dtype=int)
        risk_profiles = np.full(self.n_tickers, "",  dtype=object)

        valid = np.isfinite(mean_risk_scores)
        if valid.sum() < self.n_clusters:
            logger.warning(
                "Only %d valid scores; cannot form %d clusters — "
                "all tickers left unclassified for this window.",
                valid.sum(), self.n_clusters,
            )
            return cluster_ids, risk_profiles

        scores = mean_risk_scores[valid]
        n      = len(scores)

        # Standardise scores to zero mean and unit std before gap-search.
        # This ensures the gap-search operates on a consistent scale
        # regardless of how compressed the raw actor outputs are.
        # The standardisation is purely for boundary-finding — the original
        # scores are preserved in mean_risk_scores for all downstream uses.
        s_std = scores.std()
        if s_std > 1e-8:
            scores_normed = (scores - scores.mean()) / s_std
        else:
            # All scores identical — fall straight through to tertile split
            scores_normed = scores.copy()

        min_size      = max(2, n // 10)
        sorted_normed = np.sort(scores_normed)
        gaps          = np.diff(sorted_normed)

        best_t1, best_t2, best_gap_sum = None, None, -np.inf
        for i in range(min_size - 1, n - min_size - 1):
            for j in range(i + min_size, n - 1):
                gap_sum = gaps[i] + gaps[j]
                if gap_sum > best_gap_sum:
                    best_gap_sum = gap_sum
                    best_t1 = (sorted_normed[i]     + sorted_normed[i + 1]) / 2
                    best_t2 = (sorted_normed[j]     + sorted_normed[j + 1]) / 2

        ids_candidate = np.where(
            scores_normed > best_t2, 2,
            np.where(scores_normed > best_t1, 1, 0),
        )

        cluster_sizes = np.bincount(ids_candidate, minlength=3)
        if cluster_sizes.max() > 0.55 * n:
            logger.warning(
                "Gap-search produced imbalanced clusters %s (>55%% in one bin). "
                "Falling back to tertile split.",
                cluster_sizes.tolist(),
            )
            q33, q67 = np.percentile(sorted_normed, [33.3, 66.7])
            best_t1, best_t2 = float(q33), float(q67)
            ids_candidate = np.where(
                scores_normed > best_t2, 2,
                np.where(scores_normed > best_t1, 1, 0),
            )

        cluster_ids[valid]   = ids_candidate
        risk_profiles[valid] = np.where(
            ids_candidate == 2, "aggressive",
            np.where(ids_candidate == 1, "balanced", "conservative"),
        )
        return cluster_ids, risk_profiles