"""
rl_agent.py

PPO agent for RL-based asset risk-tier classification on the EGX30 index.

Architecture: 

RiskScorerNet (actor)
    Shared-weight MLP: (n_tickers, feature_dim) → (n_tickers,) risk scores.
    The same weights are applied to every ticker's feature vector.
    This is the correct design for cross-sectional ranking — it generalises
    across universe size and does not overfit to specific ticker positions.
    Stochastic during training (Gaussian), deterministic at inference.

ValueNet (critic)
    MLP: mean-pooled observation (feature_dim,) → scalar V(s).
    Mean-pooling is permutation-invariant, matching the actor's property.
    A simpler critic is appropriate here given the small dataset size.

RLAssetSelectorAgent
    Wraps both networks with a joint Adam optimiser (critic lr = 3× actor lr).
    pretrain()              — supervised warm-start on forward-vol targets,
                              restricted strictly to training-period windows.
    train()                 — PPO fine-tuning for N episodes.
    collect_all_scores()    — deterministic inference over a window range.
    collect_quarterly_scores() — inference grouped into 63-day quarters.
    save() / load()         — checkpoint both networks.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)

# Actor

class RiskScorerNet(nn.Module):
    """
    Shared-weight actor MLP: feature_dim → scalar risk score per ticker.

    Accepts inputs of shape:
        (feature_dim,)              — single ticker
        (n_tickers, feature_dim)    — full universe at one time step
        (T, n_tickers, feature_dim) — batched PPO update
    """

    def __init__(
        self,
        feature_dim:  int,
        hidden_dims:  List[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers: List[nn.Module] = []
        in_dim = feature_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        # Global learnable log-std — scalar shared across all tickers.
        # Initialised at 0.0 (std=1.0)
        self.log_std = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Small output gain keeps initial scores near zero
        last_linear = [m for m in self.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last_linear.weight, gain=0.01)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : (..., feature_dim)
        Returns mean scores (...,) and log_std scalar.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        mean = self.net(x).squeeze(-1)
        return mean, self.log_std

    def sample_scores(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
        mean, log_std = self.forward(x)
        std  = torch.exp(log_std).clamp(1e-4, 2.0)
        dist = torch.distributions.Normal(mean, std.expand_as(mean))
        scores   = dist.rsample()
        log_prob = dist.log_prob(scores).sum()
        entropy  = dist.entropy().mean()
        return scores, log_prob, entropy

    def evaluate_actions(
        self,
        obs_batch:     torch.Tensor,
        actions_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs_batch)
        std       = torch.exp(log_std).clamp(1e-4, 2.0)
        dist      = torch.distributions.Normal(mean, std.expand_as(mean))
        log_probs = dist.log_prob(actions_batch).sum(dim=-1)
        entropy   = dist.entropy().mean()
        return log_probs, entropy

# Critic

class ValueNet(nn.Module):
    """
    State-value critic.
    Mean-pools the (n_tickers, feature_dim) observation across tickers,
    then passes through a small MLP to produce a scalar V(s).
    Mean-pooling is permutation-invariant, matching the actor.
    """

    def __init__(
        self,
        feature_dim:  int,
        hidden_dims:  List[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

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
        """
        obs : (n_tickers, feature_dim)  or  (T, n_tickers, feature_dim)
        Returns scalar  or  (T,).
        """
        if obs.dim() == 2:
            state = obs.mean(dim=0)       # (feature_dim,)
        else:
            state = obs.mean(dim=1)       # (T, feature_dim)
        return self.net(state).squeeze(-1)

# PPO agent

class RLAssetSelectorAgent:
    """
    PPO agent for rolling-window asset risk-tier classification.

    Parameters
    ----------
    feature_dim   : int
    hidden_dims   : list[int]   actor hidden layer sizes
    lr            : float       base learning rate (actor)
    gamma         : float       discount factor
    gae_lambda    : float       GAE smoothing (0=TD, 1=MC)
    clip_eps      : float       PPO ratio clip range
    ppo_epochs    : int         update passes per rollout
    value_coeff   : float       critic loss weight in total loss
    entropy_coeff : float       entropy bonus weight
    device        : str
    """

    def __init__(
        self,
        feature_dim:   int,
        hidden_dims:   Optional[List[int]] = None,
        lr:            float = 3e-4,
        gamma:         float = 0.0,   # FIX 2: was 0.99. Each env step is an
                                      # independent ranking problem — the action
                                      # at step t does not causally affect the
                                      # state at t+1, so discounting future
                                      # rewards creates false long-range
                                      # dependencies the critic cannot model.
                                      # gamma=0 → advantage = r_t - V(s_t).
        gae_lambda:    float = 0.0,   # FIX 2: was 0.95. Irrelevant when
                                      # gamma=0 but set explicitly for clarity.
        clip_eps:      float = 0.2,
        ppo_epochs:    int   = 10,
        value_coeff:   float = 0.5,
        entropy_coeff: float = 0.0,   # FIX 4: was 0.05. See _ppo_update for
                                      # explanation — entropy is replaced by
                                      # the diversity_loss on mean scores.
        device:        str   = "cpu",
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

        # Critic learns 3× faster than actor to stabilise the value baseline
        self.optimizer = optim.Adam([
            {"params": self.actor.parameters(),  "lr": lr},
            {"params": self.critic.parameters(), "lr": lr * 3},
        ])

    # Supervised pretraining

    def pretrain(
        self,
        env,
        n_epochs:    int   = 150,
        lr_pretrain: float = 1e-3,
    ) -> None:
       
        logger.info(
            "Pretraining actor on composite rank targets "
            "(0.7 × rank(fwd_vol) + 0.3 × rank(fwd_max_dd), "
            "train windows only: idx in [%d, %d)) …",
            env._start_idx, env._end_idx,
        )

        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        for _date, obs, idx, valid_mask in env.iter_all_windows():
            fwd_vol = env._compute_forward_vol(idx)
            fwd_dd  = env._compute_forward_max_dd(idx)

            # Intersection mask: both forward labels must be finite
            valid = valid_mask & np.isfinite(fwd_vol) & np.isfinite(fwd_dd)
            if valid.sum() < 3:
                continue

            # Composite rank — same formula as _compute_reward
            vol_rank = pd.Series(fwd_vol[valid]).rank(pct=True).values.astype(np.float32)
            dd_rank  = pd.Series(fwd_dd[valid]).rank(pct=True).values.astype(np.float32)
            composite_rank = (0.7 * vol_rank + 0.3 * dd_rank).astype(np.float32)

            X_list.append(obs[valid])
            y_list.append(composite_rank)

        if not X_list:
            logger.warning("Pretraining: no valid pairs found — skipping.")
            return

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        # y is already in [0, 1] — no further normalisation needed

        X_t = torch.nan_to_num(
            torch.tensor(X, dtype=torch.float32).to(self.device),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)

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
                    "  Pretrain epoch %3d/%d  MSE=%.5f",
                    epoch, n_epochs, loss.item(),
                )

        logger.info("Pretraining complete.")

    # PPO training 

    def train_one_episode(self, env) -> Dict:
        """
        Run a single PPO episode and return its stats dict.

        """
        rollout                      = self._collect_rollout(env)
        loss, actor_loss, value_loss = self._ppo_update(rollout)

        rewards = rollout["rewards"]
        return {
            "mean_reward":  float(np.mean(rewards)),
            "total_reward": float(np.sum(rewards)),
            "loss":         loss,
            "actor_loss":   actor_loss,
            "value_loss":   value_loss,
        }


    def _collect_rollout(self, env) -> Dict:
        self.actor.train()
        self.critic.train()

        obs_list:       List[np.ndarray] = []
        actions_list:   List[np.ndarray] = []
        log_probs_list: List[float]      = []
        rewards_list:   List[float]      = []
        values_list:    List[float]      = []
        dones_list:     List[bool]       = []

        obs_np, _ = env.reset()
        done = False

        while not done:
            obs_t = torch.nan_to_num(
                torch.tensor(obs_np, dtype=torch.float32).to(self.device),
                nan=0.0, posinf=0.0, neginf=0.0,
            )

            with torch.no_grad():
                scores_t, log_prob_t, _ = self.actor.sample_scores(obs_t)
                value_t                 = self.critic(obs_t)

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
        Generalized Advantage Estimation:

        delta_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
        A_t     = delta_t + γλ · (1 - done_t) · A_{t+1}

        Terminal bootstrap value is 0 (episode ends naturally).
        Returns (advantages, value_targets).
        """
        advantages  = []
        gae         = 0.0
        values_ext  = values + [0.0]   # bootstrap terminal

        for t in reversed(range(len(rewards))):
            next_val = 0.0 if dones[t] else values_ext[t + 1]
            delta    = rewards[t] + self.gamma * next_val - values_ext[t]
            gae      = delta + self.gamma * self.gae_lambda * (0.0 if dones[t] else gae)
            advantages.insert(0, gae)

        targets = [a + v for a, v in zip(advantages, values)]
        return advantages, targets

    def _ppo_update(self, rollout: Dict) -> Tuple[float, float, float]:
        advantages, returns = self._compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"]
        )

        obs_batch = torch.nan_to_num(
            torch.tensor(
                np.stack(rollout["obs"], axis=0), dtype=torch.float32
            ).to(self.device),
            nan=0.0, posinf=0.0, neginf=0.0,
        )   # (T, n_tickers, feature_dim)

        actions_batch = torch.tensor(
            np.stack(rollout["actions"], axis=0), dtype=torch.float32
        ).to(self.device)   # (T, n_tickers)

        old_log_probs = torch.tensor(
            rollout["log_probs"], dtype=torch.float32
        ).to(self.device)   # (T,)

        adv_t = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        ret_t = torch.tensor(returns,    dtype=torch.float32).to(self.device)

        # Normalise advantages for stable gradient scale
        if adv_t.std() > 1e-8:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_losses: List[float] = []
        actor_losses: List[float] = []
        value_losses: List[float] = []

        for epoch_idx in range(self.ppo_epochs):
            # FIX 5 (minibatch shuffling) is applied inside this loop — see below.
            # We re-shuffle each epoch so the model never sees the same
            # temporal ordering twice within a single PPO update.
            T = obs_batch.shape[0]
            mb_size = min(32, T)
            indices = torch.randperm(T, device=self.device)

            epoch_total:  List[float] = []
            epoch_actor:  List[float] = []
            epoch_value:  List[float] = []

            for start in range(0, T, mb_size):
                mb_idx = indices[start : start + mb_size]

                new_log_probs, _ = self.actor.evaluate_actions(
                    obs_batch[mb_idx], actions_batch[mb_idx]
                )

                ratio   = torch.exp(new_log_probs - old_log_probs[mb_idx])
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                actor_loss = -torch.min(ratio * adv_t[mb_idx], clipped * adv_t[mb_idx]).mean()

        
                mean_scores_mb, _ = self.actor(obs_batch[mb_idx])
                score_var      = mean_scores_mb.var(dim=-1).mean()
                diversity_loss = -0.05 * score_var  

                values_pred = self.critic(obs_batch[mb_idx])
                value_loss  = F.mse_loss(values_pred, ret_t[mb_idx])

              
                loss = actor_loss + self.value_coeff * value_loss + diversity_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    max_norm=0.5,
                )
                self.optimizer.step()

                epoch_total.append(float(loss.detach()))
                epoch_actor.append(float(actor_loss.detach()))
                epoch_value.append(float(value_loss.detach()))

            total_losses.append(float(np.mean(epoch_total)))
            actor_losses.append(float(np.mean(epoch_actor)))
            value_losses.append(float(np.mean(epoch_value)))

        return (
            float(np.mean(total_losses)),
            float(np.mean(actor_losses)),
            float(np.mean(value_losses)),
        )

    # Inference 

    def predict_mean_scores(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic actor forward pass for a single time step."""
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
        end_idx:   Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Deterministic pass over windows in [start_idx, end_idx).

        Returns
        -------
        mean_scores    : (n_tickers,)
        score_matrix   : (n_windows, n_tickers)
        fwd_vol_matrix : (n_windows, n_tickers)
        fwd_ret_matrix : (n_windows, n_tickers)
        """
        all_scores:   List[np.ndarray] = []
        all_fwd_vols: List[np.ndarray] = []
        all_fwd_rets: List[np.ndarray] = []

        for _date, obs, idx, valid_mask in env.iter_all_windows(
            start_idx=start_idx, end_idx=end_idx
        ):
            scores = self.predict_mean_scores(obs).astype(np.float64)
            scores[~valid_mask] = np.nan
            all_scores.append(scores)
            all_fwd_vols.append(env._compute_forward_vol(idx))
            all_fwd_rets.append(env._compute_forward_ret(idx))

        if not all_scores:
            empty = np.empty((0, env.n_tickers))
            return np.full(env.n_tickers, np.nan), empty, empty, empty

        score_matrix   = np.stack(all_scores,   axis=0)
        fwd_vol_matrix = np.stack(all_fwd_vols, axis=0)
        fwd_ret_matrix = np.stack(all_fwd_rets, axis=0)

        mean_scores  = np.nanmean(score_matrix, axis=0)
        valid_counts = np.isfinite(score_matrix).sum(axis=0)
        mean_scores[valid_counts < 3] = np.nan

        return mean_scores, score_matrix, fwd_vol_matrix, fwd_ret_matrix

    def collect_quarterly_scores(
        self,
        env,
        start_idx:  Optional[int]                  = None,
        end_idx:    Optional[int]                  = None,
        thresholds: Optional[Tuple[float, float]]  = None,
    ) -> List[Dict]:
        """
        Deterministic inference grouped into non-overlapping 63-day quarters.

        Parameters
        env        : AssetSelectorEnv
        start_idx  : int | None
        end_idx    : int | None
        thresholds : (t_low, t_high) | None
            Fixed score thresholds derived from training data.
            Passed through to env.assign_clusters() for every quarter.
        """
        quarters = env.collect_quarterly_windows(
            start_idx=start_idx, end_idx=end_idx
        )
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

            score_arr = np.stack(scores_in_q, axis=0)
            with np.errstate(all="ignore"):  # suppress Mean of empty slice
                mean_scores = np.nanmean(score_arr, axis=0)
                mean_scores[np.isfinite(score_arr).sum(axis=0) < 3] = np.nan

            cluster_ids, risk_profiles = env.assign_clusters(
                mean_scores, thresholds=thresholds
            )

            result.append({
                "quarter_start":  q["quarter_start"],
                "quarter_idx":    q["quarter_idx"],
                "mean_scores":    mean_scores,
                "risk_profiles":  risk_profiles,
                "cluster_ids":    cluster_ids,
            })

        logger.info(
            "Quarterly scoring complete: %d quarters "
            "(start=%s, end=%s)",
            len(result),
            result[0]["quarter_start"].strftime("%Y-%m-%d") if result else "N/A",
            result[-1]["quarter_start"].strftime("%Y-%m-%d") if result else "N/A",
        )
        return result

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
        ckpt = torch.load(path, map_location=self.device,weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        logger.info("Agent loaded from %s", path)