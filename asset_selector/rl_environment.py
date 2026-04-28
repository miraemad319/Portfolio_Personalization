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
    "realised_vol",        # annualised historical vol — HAR monthly proxy
    "momentum_63",         # 3-month price momentum
    "vol_trend",           # recent 21d vol / full-window vol
    "downside_vol",        # semi-deviation — asymmetric risk
    "beta",                # systematic risk vs equal-weight market proxy
    "volume_trend",        # recent 21d mean volume / full-window mean
    "volume_shock",        # max volume / median volume — liquidity spike
    "vol_of_vol",          # std of rolling 21d vol — vol stability
    "vol_autocorr",        # lag-1 autocorr of squared returns — GARCH persistence
    "market_stress",       # ticker vol / cross-sectional median vol
    "pain_index",          # mean drawdown depth
    "return_consistency",  # fraction of positive-return days
    "rv_daily",            # HAR daily: last day squared return annualised
    "rv_weekly",           # HAR weekly: mean of last 5 daily RVs annualised
    "cvar_95",             # CVaR 95%: mean return in worst 5% of days annualised
    "rolling_mdd_21",      # 21-day rolling max drawdown
    "downside_beta",       # beta on market-down days only
    "vol_percentile_rank", # cross-sectional percentile rank of realised vol
    "amihud_illiquidity",  # mean |return| / volume
    "atr_ratio",           # ATR / close — normalised average true range
]


#  Feature computation 

def _window_features(
    ohlcv_window: pd.DataFrame,
    ret_window:   pd.DataFrame,
) -> np.ndarray:
    
    n_features = len(FEATURE_NAMES)
    tickers    = ret_window.columns.tolist()

    # Pre-compute equal-weight market return once per window
    market_ret_all: Dict[str, pd.Series] = {
        ticker: ret_window[[c for c in tickers if c != ticker]].mean(axis=1)
        for ticker in tickers
    }

    rows: List[List[float]] = []

    for ticker in tickers:
        ret = ret_window[ticker].dropna()
        n   = len(ret)

        if n < 5:
            rows.append([np.nan] * n_features)
            continue

        # Extract per-ticker OHLCV series from the MultiIndex window
        t_data = ohlcv_window[ticker]
        px     = t_data["close"].dropna()
        high   = t_data["high"].dropna()
        low    = t_data["low"].dropna()
        volume = t_data["volume"].replace(0.0, np.nan).dropna()

        #  1. Realised vol (HAR monthly proxy) 
        vol = float(ret.std() * np.sqrt(TRADING_DAYS))

        #  2. 3-month momentum 
        mom_63 = (
            float((px.iloc[-1] / px.iloc[-64]) - 1.0)
            if len(px) >= 64 and px.iloc[-64] > 0 else np.nan
        )

        #  3. Vol trend: recent 21d vol / full-window vol 
        vol_trend = (
            float(ret.iloc[-21:].std() * np.sqrt(TRADING_DAYS) / vol)
            if n >= 21 and vol > 1e-8 else np.nan
        )

        #  4. Downside volatility 
        neg_ret      = ret[ret < 0]
        downside_vol = (
            float(neg_ret.std() * np.sqrt(TRADING_DAYS))
            if len(neg_ret) >= 5 else np.nan
        )

        #  5 & 17. Beta and downside beta 
        market_ret = market_ret_all[ticker]
        common_idx = ret.index.intersection(market_ret.dropna().index)
        beta = downside_beta = np.nan
        if len(common_idx) >= 10:
            r_t     = ret.loc[common_idx].values
            m_t     = market_ret.loc[common_idx].values
            cov_mat = np.cov(r_t, m_t)
            m_var   = float(cov_mat[1, 1])
            if m_var > 1e-10:
                beta = float(cov_mat[0, 1] / m_var)
            down_mask = m_t < 0
            if down_mask.sum() >= 10:
                cov_d   = np.cov(r_t[down_mask], m_t[down_mask])
                m_var_d = float(cov_d[1, 1])
                if m_var_d > 1e-10:
                    downside_beta = float(cov_d[0, 1] / m_var_d)

        #  6. Volume trend 
        volume_trend = np.nan
        if len(volume) >= 21:
            full_mean = float(volume.mean())
            if full_mean > 1e-8:
                volume_trend = float(volume.iloc[-21:].mean() / full_mean)

        #  7. Volume shock 
        volume_shock = np.nan
        if len(volume) >= 10:
            median_v = float(volume.median())
            if median_v > 1e-8:
                volume_shock = float(volume.max() / median_v)

        #  8. Vol of vol 
        vol_of_vol = np.nan
        if n >= 42:
            roll_vol = ret.rolling(21).std().dropna() * np.sqrt(TRADING_DAYS)
            if len(roll_vol) >= 2:
                vol_of_vol = float(roll_vol.std())

        #  9. Vol autocorr 
        vol_autocorr = np.nan
        if n >= 10:
            sq_ret = ret.values ** 2
            if sq_ret.std() > 1e-10:
                ac = float(np.corrcoef(sq_ret[:-1], sq_ret[1:])[0, 1])
                vol_autocorr = ac if np.isfinite(ac) else np.nan

        #  11. Pain index 
        pain_index = np.nan
        if n > 2:
            cum       = np.exp(ret.cumsum())
            rollmax   = cum.cummax()
            pain_index = float(abs(((cum - rollmax) / rollmax).mean()))

        #  12. Return consistency 
        return_consistency = float((ret > 0).sum() / n)

        #  13. HAR daily 
        rv_daily = float(ret.iloc[-1] ** 2 * TRADING_DAYS)

        #  14. HAR weekly 
        rv_weekly = float((ret.iloc[-5:] ** 2).mean() * TRADING_DAYS)

        #  15. CVaR 95% 
        cvar_95 = np.nan
        if n >= 20:
            tail = ret[ret <= ret.quantile(0.05)]
            if len(tail) >= 1:
                cvar_95 = float(tail.mean() * TRADING_DAYS)

        #  16. Rolling 21d max drawdown 
        rolling_mdd_21 = np.nan
        if n >= 21:
            r21     = ret.iloc[-21:]
            cum21   = np.exp(r21.cumsum())
            dd21    = (cum21 - cum21.cummax()) / cum21.cummax()
            rolling_mdd_21 = float(abs(dd21.min()))

        #  19. Amihud illiquidity 
        amihud = np.nan
        if len(volume) >= 10:
            common_v = ret.index.intersection(volume.index)
            if len(common_v) >= 10:
                illiq = (
                    ret.loc[common_v].abs() / volume.loc[common_v]
                ).replace([np.inf, -np.inf], np.nan).dropna()
                if len(illiq) >= 5:
                    amihud = float(illiq.mean())

        #  20. ATR ratio: mean true range / close 
        atr_ratio = np.nan
        common_hl = high.index.intersection(low.index).intersection(px.index)
        if len(common_hl) >= 5:
            h  = high.loc[common_hl]
            l  = low.loc[common_hl]
            c  = px.loc[common_hl]
            c_prev = c.shift(1).dropna()
            common_atr = c_prev.index.intersection(h.index)
            if len(common_atr) >= 5:
                hl  = h.loc[common_atr] - l.loc[common_atr]
                hpc = (h.loc[common_atr] - c_prev.loc[common_atr]).abs()
                lpc = (l.loc[common_atr] - c_prev.loc[common_atr]).abs()
                tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
                last_close = float(c.iloc[-1])
                if last_close > 1e-8:
                    atr_ratio = float(tr.mean() / last_close)

        rows.append([
            vol, mom_63, vol_trend, downside_vol,
            beta, volume_trend, volume_shock,
            vol_of_vol, vol_autocorr,
            np.nan,             # market_stress — filled post-loop
            pain_index, return_consistency,
            rv_daily, rv_weekly, cvar_95,
            rolling_mdd_21, downside_beta,
            np.nan,             # vol_percentile_rank — filled post-loop
            amihud, atr_ratio,
        ])

    feat_matrix = np.array(rows, dtype=np.float32)  # (n_tickers, 20)

    #  Cross-sectional features (require all tickers to be scored first) 
    IDX_MARKET_STRESS  = 9
    IDX_VOL_PERCENTILE = 17

    all_vols        = feat_matrix[:, 0]
    finite_vol_mask = np.isfinite(all_vols)
    valid_vols      = all_vols[finite_vol_mask]

    if len(valid_vols) >= 3:
        cross_median_vol = float(np.median(valid_vols))

        # market_stress: ticker vol / cross-sectional median
        if cross_median_vol > 1e-8:
            feat_matrix[:, IDX_MARKET_STRESS] = np.where(
                finite_vol_mask,
                all_vols / cross_median_vol,
                np.nan,
            )

        # vol_percentile_rank: cross-sectional percentile rank [0, 1]
        vol_ranks = pd.Series(all_vols).rank(pct=True).values.astype(np.float32)
        feat_matrix[:, IDX_VOL_PERCENTILE] = np.where(
            finite_vol_mask, vol_ranks, np.nan
        )

    return feat_matrix


def _zscore_normalise(X: np.ndarray) -> np.ndarray:
    """
    Cross-sectional z-score: normalise each feature column across tickers.
    Columns that are entirely NaN or have zero std are set to 0.
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


#  Environment 

class AssetSelectorEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        ohlcv:         pd.DataFrame,
        lookback:      int           = 126,
        forward:       int           = 126,
        step_size:     int           = 21,
        n_clusters:    int           = 3,
        train_end_idx: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        ohlcv : pd.DataFrame
            MultiIndex columns (ticker, price_type).
            price_type ∈ {open, high, low, close, volume}.
            DatetimeIndex rows, daily frequency.
        """
        super().__init__()

        #  Extract tickers and validate structure 
        self.tickers   = sorted(ohlcv.columns.get_level_values("ticker").unique().tolist())
        self.n_tickers = len(self.tickers)
        self.lookback  = lookback
        self.forward   = forward
        self.step_size = step_size
        self.n_clusters  = n_clusters
        self.feature_dim = len(FEATURE_NAMES)

        # Store the full OHLCV MultiIndex DataFrame 
        self.ohlcv = ohlcv

        #  Derive close prices and log returns 
        # Close is used for forward vol/return/drawdown evaluation and
        # for the date index. All other price types are used only in features.
        self.prices: pd.DataFrame = (
            ohlcv.xs("close", axis=1, level="price_type")
            .replace(0.0, np.nan)
            .reindex(columns=self.tickers)
        )

        self.log_returns: pd.DataFrame = np.log(
            self.prices / self.prices.shift(1)
        )

        # Index landmarks 
        self._start_idx    = lookback
        self._full_end_idx = len(self.prices) - forward

        if train_end_idx is not None:
            self._train_end_idx        = int(train_end_idx)
            self._end_idx              = min(
                self._train_end_idx - self.forward,
                self._full_end_idx,
            )
            self._test_start_idx: Optional[int] = self._train_end_idx + 1
        else:
            self._train_end_idx  = self._full_end_idx + self.forward
            self._end_idx        = self._full_end_idx
            self._test_start_idx = None

        if self._end_idx <= self._start_idx:
            raise ValueError(
                f"Training period too short: lookback={lookback} forward={forward} "
                f"→ _start_idx={self._start_idx} _end_idx={self._end_idx}."
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
            self.n_tickers, len(self.prices), lookback, forward, step_size,
        )
        logger.info(
            "Index landmarks: start=%d  train_end=%d  "
            "test_start=%s  full_end=%d",
            self._start_idx, self._end_idx,
            str(self._test_start_idx), self._full_end_idx,
        )

    #  Gymnasium interface 

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
        Bool array (n_tickers,) — True where ticker has >= 5 valid log
        returns in the lookback window [idx-lookback, idx).
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
        ohlcv_win = self.ohlcv.iloc[idx - self.lookback : idx]
        ret_win   = self.log_returns.iloc[idx - self.lookback : idx]
        raw       = _window_features(ohlcv_win, ret_win)
        return _zscore_normalise(raw).astype(np.float32)

    # Forward metrics 

    def _compute_forward_vol(self, idx: int) -> np.ndarray:
        """
        Annualised realised vol over [idx, idx+forward).
        Returns (n_tickers,) float32, NaN where < 5 valid returns.
        """
        fwd_end = min(idx + self.forward, len(self.prices))
        fwd_ret = self.log_returns.iloc[idx:fwd_end]
        results = []
        for t in self.tickers:
            ret = fwd_ret[t].dropna()
            results.append(
                float(ret.std() * np.sqrt(TRADING_DAYS))
                if len(ret) >= 5 else np.nan
            )
        return np.array(results, dtype=np.float32)

    def _compute_forward_ret(self, idx: int) -> np.ndarray:
        """
        Annualised mean log-return over [idx, idx+forward).
        Returns (n_tickers,) float32, NaN where < 5 valid returns.
        """
        fwd_end = min(idx + self.forward, len(self.prices))
        fwd_ret = self.log_returns.iloc[idx:fwd_end]
        results = []
        for t in self.tickers:
            ret = fwd_ret[t].dropna()
            results.append(
                float(ret.mean() * TRADING_DAYS)
                if len(ret) >= 5 else np.nan
            )
        return np.array(results, dtype=np.float32)

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
        Returns (n_tickers,) float64, NaN where < 5 valid returns.
        """
        fwd_end = min(idx + self.forward, len(self.prices))
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
        computed over the intersection of tickers where BOTH forward
        labels are finite. Returns 0.0 if fewer than 3 valid tickers.
        """
        fwd_vol  = self._compute_forward_vol(idx)
        fwd_dd   = self._compute_forward_max_dd(idx)
        has_data = self._get_valid_mask(idx)

        valid = (
            has_data
            & np.isfinite(risk_scores)
            & np.isfinite(fwd_vol)
            & np.isfinite(fwd_dd)
        )
        if valid.sum() < 3:
            return 0.0

        vol_rank       = pd.Series(fwd_vol[valid]).rank(pct=True).values.astype(np.float64)
        dd_rank        = pd.Series(fwd_dd[valid]).rank(pct=True).values.astype(np.float64)
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

    # semi annual grouping 

    def collect_period_windows(
        self,
        start_idx: Optional[int] = None,
        end_idx:   Optional[int] = None,
    ) -> List[Dict]:
        """
        Group step indices into non-overlapping 126-day semi-annual periods.
        Each period contains approximately 6 windows at step_size=21.
        """
        idx_s        = start_idx if start_idx is not None else self._start_idx
        idx_e        = end_idx   if end_idx   is not None else self._end_idx
        quarter_size = self.forward

        all_steps = list(range(idx_s, idx_e, self.step_size))
        if not all_steps:
            return []

        quarters: List[Dict] = []
        q_num = 0
        while True:
            q_start_idx    = idx_s + q_num * quarter_size
            q_end_idx      = q_start_idx + quarter_size
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

    #  Cluster assignment 

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
            # Per-semi annual period tertile split for threshold computation
           
            t_low  = float(np.percentile(scores, 33.3))
            t_high = float(np.percentile(scores, 66.7))

        ids = np.where(
            scores >= t_high, 2,
            np.where(scores >= t_low, 1, 0),
        )

        counts = np.bincount(ids, minlength=3)
        logger.info(
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