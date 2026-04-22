"""
rl_agent.py
===========
PPO-based agent for RL-based asset risk-tier classification.

Why PPO over REINFORCE
----------------------
REINFORCE has very high gradient variance — it updates on a single Monte
Carlo return per episode with no baseline beyond a simple running mean.
In our setup (49 tickers × ~61 steps per episode) the gradient signal
averages over too many noisy samples and the policy never meaningfully
improves beyond the supervised pre-training starting point.

PPO fixes this with three mechanisms:
  1. A learned value function (critic) that provides a per-step baseline,
     dramatically reducing variance via Generalized Advantage Estimation.
  2. Clipped surrogate objective — prevents large destructive updates that
     undo good pre-training.
  3. K update epochs per rollout — reuses each episode's data multiple
     times, extracting more gradient signal per environment interaction.

Architecture
------------
RiskScorerNet (actor)
    Shared-weight MLP: (n_tickers, feature_dim) → (n_tickers,) risk scores.
    Same weights for every ticker — generalises across universe size.
    Stochastic during training (Gaussian), deterministic at inference.

ValueNet (critic)
    MLP: mean-pooled observation (feature_dim,) → scalar V(s).
    Mean-pooling is permutation-invariant, matching the actor's property.

RLAssetSelectorAgent
    Wraps both networks with a joint Adam optimiser.
    pretrain()  — supervised warm-start on forward-vol targets.
    train()     — PPO fine-tuning for N episodes.
    collect_all_scores() / predict_mean_scores() — deterministic inference.
    save() / load() — checkpoint both networks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Actor
# ──────────────────────────────────────────────────────────────────────────────

class RiskScorerNet(nn.Module):
    """
    Shared-weight actor MLP: feature_dim → scalar risk score per ticker.

    Handles inputs of shape:
        (feature_dim,)              — single ticker
        (n_tickers, feature_dim)    — full universe at one time step
        (T, n_tickers, feature_dim) — batched PPO update
    """

    def __init__(
        self,
        feature_dim: int = 6,
        hidden_dims: List[int] = [128, 64, 32],
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        in_dim = feature_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        # Global learnable log-std — 0.0 init; pre-training already anchors the
        # policy so we don't need aggressive cold-start exploration
        self.log_std = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        last_linear = [m for m in self.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last_linear.weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : (..., feature_dim) — any leading batch dims.
        Returns mean scores (...,) and log_std scalar.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)          # (1, feature_dim)
        mean = self.net(x).squeeze(-1)  # (..., n_tickers)
        return mean, self.log_std

    def sample_scores(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step stochastic forward pass (used during rollout collection).
        x : (n_tickers, feature_dim)
        Returns (scores, sum_log_prob, entropy) — all scalar-reducible.
        """
        mean, log_std = self.forward(x)
        std  = torch.exp(log_std).clamp(1e-4, 2.0)
        dist = torch.distributions.Normal(mean, std.expand_as(mean))
        scores   = dist.rsample()
        log_prob = dist.log_prob(scores).sum()   # joint log_prob across tickers
        entropy  = dist.entropy().mean()
        return scores, log_prob, entropy

    def evaluate_actions(
        self,
        obs_batch:     torch.Tensor,
        actions_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched evaluation for PPO update.
        obs_batch     : (T, n_tickers, feature_dim)
        actions_batch : (T, n_tickers)
        Returns new_log_probs (T,) and mean entropy scalar.
        """
        mean, log_std = self.forward(obs_batch)       # (T, n_tickers)
        std  = torch.exp(log_std).clamp(1e-4, 2.0)
        dist = torch.distributions.Normal(mean, std.expand_as(mean))
        log_probs = dist.log_prob(actions_batch).sum(dim=-1)  # (T,)
        entropy   = dist.entropy().mean()
        return log_probs, entropy


# ──────────────────────────────────────────────────────────────────────────────
# Critic
# ──────────────────────────────────────────────────────────────────────────────

class ValueNet(nn.Module):
    """
    State-value critic.  Takes the mean-pooled observation as state
    representation (permutation-invariant, matches actor's shared weights).

    Input  : (n_tickers, feature_dim)    or  (T, n_tickers, feature_dim)
    Output : scalar V(s)                 or  (T,) V(s_t) for each step
    """

    def __init__(
        self,
        feature_dim: int = 6,
        hidden_dims: List[int] = [64, 32],
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        in_dim = feature_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Mean-pool across tickers, then pass through MLP."""
        if obs.dim() == 2:
            state = obs.mean(dim=0)        # (feature_dim,)
        else:
            state = obs.mean(dim=1)        # (T, feature_dim)
        return self.net(state).squeeze(-1) # scalar or (T,)


# ──────────────────────────────────────────────────────────────────────────────
# PPO agent
# ──────────────────────────────────────────────────────────────────────────────

class RLAssetSelectorAgent:
    """
    PPO agent for rolling-window asset risk-tier classification.

    Parameters
    ----------
    feature_dim : int
    hidden_dims : list[int]   — actor hidden layer sizes
    lr : float                — joint Adam learning rate
    gamma : float             — discount factor
    gae_lambda : float        — GAE smoothing (0 = TD, 1 = MC)
    clip_eps : float          — PPO clipping range
    ppo_epochs : int          — update passes per rollout
    value_coeff : float       — critic loss weight
    entropy_coeff : float     — entropy bonus weight
    device : str
    """

    def __init__(
        self,
        feature_dim:   int             = 6,
        hidden_dims:   Optional[List[int]] = None,
        lr:            float           = 3e-4,
        gamma:         float           = 0.99,
        gae_lambda:    float           = 0.95,
        clip_eps:      float           = 0.2,
        ppo_epochs:    int             = 10,
        value_coeff:   float           = 0.5,
        entropy_coeff: float           = 0.05,
        device:        str             = "cpu",
    ) -> None:
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.device        = torch.device(device)
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_eps      = clip_eps
        self.ppo_epochs    = ppo_epochs
        self.value_coeff   = value_coeff
        self.entropy_coeff = entropy_coeff

        self.actor  = RiskScorerNet(feature_dim, hidden_dims).to(self.device)
        self.critic = ValueNet(feature_dim).to(self.device)

        # Separate learning rates: critic learns 3× faster than actor.
        # This stabilises the value baseline and reduces diverging value loss.
        self.optimizer = optim.Adam([
            {"params": self.actor.parameters(),  "lr": lr},
            {"params": self.critic.parameters(), "lr": lr * 3},
        ])
        self._scheduler: Optional[optim.lr_scheduler.CosineAnnealingLR] = None

    # ── Supervised pre-training ───────────────────────────────────────────────

    def pretrain(
        self,
        env,
        n_epochs:    int   = 150,
        lr_pretrain: float = 1e-3,
    ) -> None:
        """
        Warm-start the actor to predict forward realised vol from features.

        Collects all (obs, forward_vol) pairs from every rolling window,
        then trains the actor with MSE loss.  This ensures the policy starts
        from a meaningful risk ranking rather than near-zero random outputs,
        giving PPO a good initialisation to refine.
        """
        logger.info("Pre-training actor on forward realised-vol targets ...")

        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        for _date, obs, idx, valid_mask in env.iter_all_windows():
            fwd_vol = env._compute_forward_vol(idx)
            # Exclude tickers with no lookback data (all-zero obs would give
            # conflicting gradient signals and bias the actor's zero-input output)
            valid   = valid_mask & np.isfinite(fwd_vol)
            if valid.sum() < 3:
                continue
            X_list.append(obs[valid])
            y_list.append(fwd_vol[valid])

        if not X_list:
            logger.warning("Pre-training: no valid pairs found — skipping.")
            return

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        y_norm = (y - y_min) / (y_max - y_min + 1e-8)

        X_t = torch.nan_to_num(
            torch.tensor(X, dtype=torch.float32).to(self.device),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        y_t = torch.tensor(y_norm, dtype=torch.float32).to(self.device)

        pre_opt = optim.Adam(self.actor.parameters(), lr=lr_pretrain)
        mse     = nn.MSELoss()

        self.actor.train()
        for epoch in range(1, n_epochs + 1):
            mean_scores, _ = self.actor(X_t)
            loss = mse(mean_scores, y_t)
            pre_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            pre_opt.step()

            if epoch % 30 == 0 or epoch == 1:
                logger.info(
                    "  Pre-train epoch %3d/%d  MSE=%.5f", epoch, n_epochs, loss.item()
                )

        logger.info("Pre-training complete.")

    # ── PPO training ──────────────────────────────────────────────────────────

    def train(
        self,
        env,
        n_episodes: int = 60,
        log_every:  int = 10,
    ) -> List[Dict]:
        """
        Fine-tune with PPO for n_episodes episodes.

        Each episode = one full chronological sweep through the data.
        After each sweep, run `ppo_epochs` update passes over the rollout.

        Returns
        -------
        history : list of per-episode dicts with mean_reward, loss, etc.
        """
        self._scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_episodes, eta_min=1e-5,
        )
        history: List[Dict] = []

        for ep in range(1, n_episodes + 1):
            rollout = self._collect_rollout(env)
            loss, actor_loss, value_loss = self._ppo_update(rollout)
            self._scheduler.step()

            rewards = rollout["rewards"]
            stats = {
                "episode":      ep,
                "mean_reward":  float(np.mean(rewards)),
                "total_reward": float(np.sum(rewards)),
                "loss":         loss,
                "actor_loss":   actor_loss,
                "value_loss":   value_loss,
            }
            history.append(stats)

            if ep % log_every == 0 or ep == 1:
                logger.info(
                    "Episode %3d/%d  mean_reward=%.4f  "
                    "actor_loss=%.4f  value_loss=%.4f",
                    ep, n_episodes,
                    stats["mean_reward"],
                    stats["actor_loss"],
                    stats["value_loss"],
                )

        return history

    def _collect_rollout(self, env) -> Dict:
        """
        Run one full episode and collect all transitions.

        Returns dict with keys:
            obs          : list of np.ndarray (n_tickers, feature_dim)
            actions      : list of np.ndarray (n_tickers,)
            log_probs    : list of float  (old policy, for PPO ratio)
            rewards      : list of float
            values       : list of float  (critic estimates)
            dones        : list of bool
        """
        self.actor.train()
        self.critic.train()

        obs_list:      List[np.ndarray] = []
        actions_list:  List[np.ndarray] = []
        log_probs_list: List[float]     = []
        rewards_list:  List[float]      = []
        values_list:   List[float]      = []
        dones_list:    List[bool]       = []

        obs_np, _ = env.reset()
        done = False

        while not done:
            obs_t = torch.nan_to_num(
                torch.tensor(obs_np, dtype=torch.float32).to(self.device),
                nan=0.0, posinf=0.0, neginf=0.0,
            )

            with torch.no_grad():
                scores_t, log_prob_t, _ = self.actor.sample_scores(obs_t)
                value_t = self.critic(obs_t)

            actions_np = scores_t.cpu().numpy()
            obs_np_next, reward, terminated, truncated, _ = env.step(actions_np)
            done = terminated or truncated

            obs_list.append(obs_np)
            actions_list.append(actions_np)
            log_probs_list.append(float(log_prob_t))
            rewards_list.append(float(reward))
            values_list.append(float(value_t))
            dones_list.append(done)

            obs_np = obs_np_next

        return {
            "obs":       obs_list,
            "actions":   actions_list,
            "log_probs": log_probs_list,
            "rewards":   rewards_list,
            "values":    values_list,
            "dones":     dones_list,
        }

    def _compute_gae(
        self,
        rewards: List[float],
        values:  List[float],
        dones:   List[bool],
    ) -> Tuple[List[float], List[float]]:
        """
        Generalized Advantage Estimation (Schulman et al. 2016).
        Returns (advantages, value_targets).
        """
        advantages = []
        gae = 0.0
        values_ext = values + [0.0]   # bootstrap terminal value = 0

        for t in reversed(range(len(rewards))):
            next_val = 0.0 if dones[t] else values_ext[t + 1]
            delta    = rewards[t] + self.gamma * next_val - values_ext[t]
            gae      = delta + self.gamma * self.gae_lambda * (0.0 if dones[t] else gae)
            advantages.insert(0, gae)

        targets = [a + v for a, v in zip(advantages, values)]
        return advantages, targets

    def _ppo_update(self, rollout: Dict) -> Tuple[float, float, float]:
        """
        Run ppo_epochs update passes over the collected rollout.
        Returns (mean_total_loss, mean_actor_loss, mean_value_loss).
        """
        advantages, returns = self._compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"]
        )

        # Stack into tensors
        obs_batch = torch.nan_to_num(
            torch.tensor(np.stack(rollout["obs"], axis=0), dtype=torch.float32).to(self.device),
            nan=0.0, posinf=0.0, neginf=0.0,
        )   # (T, n_tickers, feature_dim)
        actions_batch = torch.tensor(
            np.stack(rollout["actions"], axis=0), dtype=torch.float32
        ).to(self.device)   # (T, n_tickers)
        old_log_probs = torch.tensor(
            rollout["log_probs"], dtype=torch.float32
        ).to(self.device)   # (T,)
        old_values_t = torch.tensor(
            rollout["values"], dtype=torch.float32
        ).to(self.device)   # (T,)  — needed for value clipping
        adv_t = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        ret_t = torch.tensor(returns,    dtype=torch.float32).to(self.device)

        # Normalise advantages
        if adv_t.std() > 1e-8:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_losses: List[float] = []
        actor_losses: List[float] = []
        value_losses: List[float] = []

        for _ in range(self.ppo_epochs):
            # Actor
            new_log_probs, entropy = self.actor.evaluate_actions(obs_batch, actions_batch)
            ratio    = torch.exp(new_log_probs - old_log_probs)
            clipped  = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
            actor_loss = -torch.min(ratio * adv_t, clipped * adv_t).mean()

            # Critic — clipped value loss prevents large critic updates that
            # cause the value baseline to drift (fixes diverging value_loss).
            values_pred   = self.critic(obs_batch)           # (T,)
            values_clipped = old_values_t + torch.clamp(
                values_pred - old_values_t, -self.clip_eps, self.clip_eps
            )
            value_loss = torch.max(
                F.mse_loss(values_pred,   ret_t),
                F.mse_loss(values_clipped, ret_t),
            )

            loss = (
                actor_loss
                + self.value_coeff   * value_loss
                - self.entropy_coeff * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                max_norm=0.5,
            )
            self.optimizer.step()

            total_losses.append(float(loss.detach()))
            actor_losses.append(float(actor_loss.detach()))
            value_losses.append(float(value_loss.detach()))

        return (
            float(np.mean(total_losses)),
            float(np.mean(actor_losses)),
            float(np.mean(value_losses)),
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_mean_scores(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic actor forward pass for a single time-step."""
        self.actor.eval()
        x = torch.nan_to_num(
            torch.tensor(obs, dtype=torch.float32).to(self.device),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        with torch.no_grad():
            mean, _ = self.actor(x)
        return mean.cpu().numpy()

    def collect_all_scores(
        self,
        env,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Deterministic pass over windows in [start_idx, end_idx) → time-averaged
        scores and the corresponding forward realised vol / return.

        Parameters
        ----------
        start_idx : int | None
            First window index. Defaults to env._start_idx.
        end_idx : int | None
            One-past-last window index. Defaults to env._end_idx.

        Returns
        -------
        mean_scores      : (n_tickers,)           — mean RL score across windows
        score_matrix     : (n_windows, n_tickers) — per-window RL scores
        fwd_vol_matrix   : (n_windows, n_tickers) — forward 63-day realised vol
        fwd_ret_matrix   : (n_windows, n_tickers) — forward 63-day mean return
        """
        all_scores:    List[np.ndarray] = []
        all_fwd_vols:  List[np.ndarray] = []
        all_fwd_rets:  List[np.ndarray] = []

        for _date, obs, idx, valid_mask in env.iter_all_windows(start_idx=start_idx, end_idx=end_idx):
            scores = self.predict_mean_scores(obs).astype(np.float64)
            # Set score to NaN for tickers that had no data in this window.
            # Without this, the actor's constant ~0.485 output for all-zero obs
            # would pollute the time-averaged mean_scores and cause high-vol
            # late-entrant tickers (e.g. TALM, ASPI) to be severely under-scored.
            scores[~valid_mask] = np.nan
            all_scores.append(scores)
            all_fwd_vols.append(env._compute_forward_vol(idx))
            all_fwd_rets.append(env._compute_forward_ret(idx))

        if not all_scores:
            empty = np.empty((0, env.n_tickers))
            return np.full(env.n_tickers, np.nan), empty, empty, empty

        score_matrix   = np.stack(all_scores,   axis=0)   # (n_windows, n_tickers)
        fwd_vol_matrix = np.stack(all_fwd_vols, axis=0)
        fwd_ret_matrix = np.stack(all_fwd_rets, axis=0)

        mean_scores = np.nanmean(score_matrix, axis=0)    # (n_tickers,)

        valid_counts = np.isfinite(score_matrix).sum(axis=0)
        mean_scores[valid_counts < 3] = np.nan

        return mean_scores, score_matrix, fwd_vol_matrix, fwd_ret_matrix

    def collect_quarterly_scores(
        self,
        env,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> List[Dict]:
        """
        Run deterministic inference grouped into non-overlapping 63-day quarters.

        For each quarter returned by env.collect_quarterly_windows(), collects
        RL risk scores for every window in that quarter using predict_mean_scores(),
        nanmean-averages them across windows (masking tickers with no data), then
        calls env.assign_clusters() to get conservative/balanced/aggressive labels.

        Parameters
        ----------
        env : AssetSelectorEnv
        start_idx : int | None
            First window index. Defaults to env._start_idx.
        end_idx : int | None
            One-past-last window index. Defaults to env._end_idx.

        Returns
        -------
        List of dicts, one per quarter:
            quarter_start : pd.Timestamp
            quarter_idx   : int
            mean_scores   : np.ndarray (n_tickers,)  — quarterly-averaged RL scores
            risk_profiles : np.ndarray (n_tickers,)  — label strings (''/invalid)
            cluster_ids   : np.ndarray (n_tickers,)  — 0/1/2, or -1 for invalid
        """
        quarters = env.collect_quarterly_windows(start_idx=start_idx, end_idx=end_idx)
        result: List[Dict] = []

        for q in quarters:
            scores_in_q: List[np.ndarray] = []

            for idx in q["window_indices"]:
                obs    = env._get_observation(idx)
                valid  = env._get_valid_mask(idx)
                scores = self.predict_mean_scores(obs).astype(np.float64)
                scores[~valid] = np.nan
                scores_in_q.append(scores)

            if not scores_in_q:
                continue

            score_arr   = np.stack(scores_in_q, axis=0)      # (w, n_tickers)
            mean_scores = np.nanmean(score_arr, axis=0)       # (n_tickers,)
            # Require at least one valid window per ticker
            mean_scores[np.isfinite(score_arr).sum(axis=0) < 1] = np.nan

            cluster_ids, risk_profiles = env.assign_clusters(mean_scores)

            result.append({
                "quarter_start":  q["quarter_start"],
                "quarter_idx":    q["quarter_idx"],
                "mean_scores":    mean_scores,
                "risk_profiles":  risk_profiles,
                "cluster_ids":    cluster_ids,
            })

        logger.info("Quarterly scoring complete: %d quarters", len(result))
        return result

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "actor":     self.actor.state_dict(),
                "critic":    self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info("Agent saved to %s", path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        logger.info("Agent loaded from %s", path)
