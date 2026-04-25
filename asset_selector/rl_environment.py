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
    # --- original 13 features ---
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
    "volume_trend",
    "volume_shock",
    # --- 5 new features ---
    "vol_of_vol",         # std of rolling 21d vol — measures vol stability
    "vol_autocorr",       # lag-1 autocorr of squared returns — GARCH persistence signal
    "market_stress",      # ticker vol / cross-sectional median vol — regime position
    "pain_index",         # mean drawdown depth over window — smoother than max_dd
    "return_consistency", # fraction of positive-return days — orthogonal behavioural signal
]

# Feature helpers


def _window_features(
    ret_window: pd.DataFrame,
    px_window: pd.DataFrame,
    vol_window: pd.DataFrame,
) -> np.ndarray:
    """
    Compute a (n_tickers, 18) feature matrix for one time window.

    Parameters
    ret_window : pd.DataFrame
        Daily log returns for the lookback window.
    px_window : pd.DataFrame
        Close prices for the same window.
    vol_window : pd.DataFrame  Daily volume for the same window.

    Returns
    np.ndarray shape (n_tickers, 18), dtype float32.
    NaN for tickers with fewer than 5 valid returns.
    Original 13 features + 5 new: vol_of_vol, vol_autocorr,
    market_stress, pain_index, return_consistency.
    """
    n_features = len(FEATURE_NAMES)
    tickers = ret_window.columns.tolist()
    rows: List[List[float]] = []

    for ticker in tickers:
        ret = ret_window[ticker].dropna()
        px  = px_window[ticker].dropna()
        volume_series = vol_window[ticker].replace(0.0, np.nan).dropna()
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

        #vol trends
        if len(volume_series) >= 21:
            recent_mean_vol = float(volume_series.iloc[-21:].mean())
            full_mean_vol   = float(volume_series.mean())
            volume_trend    = (
                float(recent_mean_vol / full_mean_vol)
                if full_mean_vol > 1e-8 else np.nan
            )
        else:
            volume_trend = np.nan

        #vol shock
        if len(volume_series) >= 10:
            median_vol   = float(volume_series.median())
            max_vol      = float(volume_series.max())
            volume_shock = (
                float(max_vol / median_vol)
                if median_vol > 1e-8 else np.nan
            )
        else:
            volume_shock = np.nan

        rows.append([
            vol, mean_ret, sharpe, max_dd, skew, kurt,
            mom_21, mom_63, vol_trend, downside_vol, beta, volume_trend, volume_shock,
            np.nan, np.nan, np.nan, np.nan, np.nan,  # placeholders for new features
        ])

    feat_matrix = np.array(rows, dtype=np.float32)  # shape (n_tickers, 18)

    # ── Compute the 5 new features ────────────────────────────────────────────
    # Index offsets for the placeholder columns:
    IDX_VOL_OF_VOL    = 13
    IDX_VOL_AUTOCORR  = 14
    IDX_MARKET_STRESS = 15
    IDX_PAIN_INDEX    = 16
    IDX_CONSISTENCY   = 17

    # Cross-sectional median realised_vol (column 0) for market_stress
    all_vols = feat_matrix[:, 0]  # realised_vol per ticker
    valid_vols = all_vols[np.isfinite(all_vols)]
    cross_median_vol = float(np.median(valid_vols)) if len(valid_vols) >= 3 else np.nan

    for i, ticker in enumerate(tickers):
        ret = ret_window[ticker].dropna()
        n   = len(ret)

        if n < 5:
            # All new features stay NaN — already set above
            continue

        # vol_of_vol: std of rolling 21-day annualised vol
        # Requires at least 42 observations to get 2+ non-NaN rolling windows
        if n >= 42:
            roll_vol = (
                ret.rolling(21).std().dropna() * np.sqrt(TRADING_DAYS)
            )
            feat_matrix[i, IDX_VOL_OF_VOL] = float(roll_vol.std()) if len(roll_vol) >= 2 else np.nan
        # else: stays NaN

        # vol_autocorr: lag-1 autocorrelation of squared returns
        # Squared returns are the standard proxy for variance in GARCH literature
        if n >= 10:
            sq_ret = ret.values ** 2
            if sq_ret.std() > 1e-10:
                autocorr = float(np.corrcoef(sq_ret[:-1], sq_ret[1:])[0, 1])
                feat_matrix[i, IDX_VOL_AUTOCORR] = autocorr if np.isfinite(autocorr) else np.nan

        # market_stress: this ticker's vol relative to cross-sectional median
        # > 1.0 means this stock is more volatile than the median today
        ticker_vol = feat_matrix[i, 0]  # realised_vol already computed
        if np.isfinite(ticker_vol) and np.isfinite(cross_median_vol) and cross_median_vol > 1e-8:
            feat_matrix[i, IDX_MARKET_STRESS] = ticker_vol / cross_median_vol

        # pain_index: mean of the drawdown series (not just the worst point)
        # Gives a smoother picture of how much time the stock spent underwater
        if n > 2:
            cum      = np.exp(ret.cumsum())
            roll_max = cum.cummax()
            dd_series = (cum - roll_max) / roll_max  # always <= 0
            feat_matrix[i, IDX_PAIN_INDEX] = float(abs(dd_series.mean()))

        # return_consistency: fraction of trading days with positive returns
        feat_matrix[i, IDX_CONSISTENCY] = float((ret > 0).sum() / n)

    return feat_matrix


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
        volume:        Optional[pd.DataFrame] = None,
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

        # Volume: replace 0.0 and NaN with NaN for clean ratio computation
        if volume is not None:
            self.volume: pd.DataFrame = (
                volume.replace(0.0, np.nan)
                .reindex(index=self.prices.index, columns=self.tickers)
            )
        else:
            self.volume = pd.DataFrame(
                np.nan, index=self.prices.index, columns=self.tickers
            )

        # Pre-compute log returns once — 0.0 already replaced above
        self.log_returns: pd.DataFrame = np.log(
            self.prices / self.prices.shift(1)
        )

        # Index landmarks 
        # First valid step: need a full lookback window
        self._start_idx = lookback

        # Last valid step overall: a step at idx reads [idx, idx+forward).
        # The slice is valid as long as idx + forward <= len(prices),
        # i.e. idx <= len(prices) - forward.
        self._full_end_idx = len(prices) - forward

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
        vol_win = self.volume.iloc[idx - self.lookback : idx]
        raw     = _window_features(ret_win, px_win, vol_win)
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
        Composite Spearman reward.

        Target rank = 0.7 × rank(fwd_vol) + 0.3 × rank(fwd_max_dd),
        computed over the intersection of tickers where BOTH fwd_vol AND
        fwd_max_dd are finite.  Using the intersection (rather than separate
        masks per component) means the reward is always measured on the same
        population, so the composite rank is coherent and the gradient signal
        is consistent across episodes.

        Returns 0.0 if fewer than 3 tickers satisfy the intersection mask.
        """
        fwd_vol  = self._compute_forward_vol(idx)
        fwd_dd   = self._compute_forward_max_dd(idx)
        has_data = self._get_valid_mask(idx)

        # Intersection: ticker must have valid features AND both forward labels
        valid = (
            has_data
            & np.isfinite(risk_scores)
            & np.isfinite(fwd_vol)
            & np.isfinite(fwd_dd)
        )
        if valid.sum() < 3:
            return 0.0

        # Percentile ranks within this window (ties broken by average)
        vol_rank = pd.Series(fwd_vol[valid]).rank(pct=True).values.astype(np.float64)
        dd_rank  = pd.Series(fwd_dd[valid]).rank(pct=True).values.astype(np.float64)
        composite_rank = 0.7 * vol_rank + 0.3 * dd_rank

        rho, _ = spearmanr(risk_scores[valid], composite_rank)
        return float(rho) if np.isfinite(rho) else 0.0

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
        thresholds: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map per-ticker mean risk scores to cluster IDs and profile strings.

            cluster 0 → 'conservative'
            cluster 1 → 'balanced'
            cluster 2 → 'aggressive'

        """
        cluster_ids   = np.full(self.n_tickers, -1, dtype=int)
        risk_profiles = np.full(self.n_tickers, "",  dtype=object)

        valid = np.isfinite(mean_risk_scores)
        if valid.sum() < self.n_clusters:
            logger.warning(
                "Only %d valid scores; cannot form %d clusters.",
                valid.sum(), self.n_clusters,
            )
            return cluster_ids, risk_profiles

        scores = mean_risk_scores[valid]

        if thresholds is not None:
            t_low, t_high = thresholds
        else:
            # Fallback only — callers should always supply thresholds
            logger.warning(
                "assign_clusters called without thresholds "
                "falling back to tertile split of current scores. "
                "This should only happen during testing or debugging."
            )
            t_low, t_high = float(np.percentile(scores, 33.3)), \
                            float(np.percentile(scores, 66.7))

        ids = np.where(
            scores >= t_high, 2,
            np.where(scores >= t_low, 1, 0),
        )

        # Log the resulting distribution so imbalances are visible
        counts = np.bincount(ids, minlength=3)
        logger.debug(
            "assign_clusters: conservative=%d  balanced=%d  aggressive=%d  "
            "(thresholds: %.4f / %.4f)",
            counts[0], counts[1], counts[2], t_low, t_high,
        )

        cluster_ids[valid]   = ids
        risk_profiles[valid] = np.where(
            ids == 2, "aggressive",
            np.where(ids == 1, "balanced", "conservative"),
        )
        return cluster_ids, risk_profiles