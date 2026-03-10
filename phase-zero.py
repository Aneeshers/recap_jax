#!/usr/bin/env python3
"""
Notes:
─────────────────────────────────────────
1. Flow matching policy (Section V-A, Eq. 4)
   The policy generates actions via a learned velocity field f_θ, trained with the
   CFM.

2. conditioning dropout (Section V-B, Appendix F)
   During training, 30% of the time we drop the advantage indicator — the model
   sees cond_id=0 (uncond) instead of the true label. This trains a SINGLE model
   that implicitly represents both π(a|s) and π(a|I,s), exactly like CFG in diffusion models. 

3. CFG inference (Section V-B, Appendix E, Eq. 13)
   At sampling time, each Euler step combines two velocity predictions:
   v_guided = v_uncond + Beta·(v_pos - v_uncond)
   Beta=1 recovers the positive-conditioned policy. Beta>1 sharpens.
   This operates on the velocity field (score), not on means, it's the correct
   analogue of CFG for flow matching.

4. Rewards (Section V-C, Eq. 5)
   We use RECAP's time-to-completion reward: -1 per step, 0 at success, -C_fail
   at failure. OGBench already does this perfectly.
   
5. Distributional Value Func (Section IV-A, Eq. 1)
   A categorical distribution over B=201 return bins, trained with cross-entropy
   on MC returns. The expected value is used to compute advantages.

draft workflow:
  1. Load OGBench dataset.
  2. Transform rewards to RECAP format (-1 per step, 0 at success).
  3. Compute RTG
  4. Train distributional value function V(s) -> RTG bins.
  5. Compute advantages A = RTG - E[V(s)].
  6. Set threshold ε so ~pos_frac of transitions have A > ε.
  7. Train flow matching policy with conditioning dropout.
  8. Evaluate with CFG inference at various Beta.
  9. Save checkpoint.


metrics:
value/train_loss: Cross-entropy on training batch (every log_interval)
value/grad_norm: Global gradient L2 norm (every log_interval)
value/val_loss: Cross-entropy on held-out data (every diag_interval)
value/val_mae: |E[V(s)] - RTG| on val (every diag_interval)
value/val_rank_corr       Spearman ρ between V(s) and RTG on val (every diag_interval)
value/val_mean_bias       mean(V) - mean(RTG) on val — positive = overestimates (every diag_interval)
value/val_entropy         Mean entropy of predicted bin distribution (every diag_interval)

advantage/epsilon         Threshold for positive labeling
advantage/std             Std of advantage distribution
advantage/pos_frac        Actual fraction labeled positive
advantage/histogram       Full advantage distribution (wandb histogram)

actor/train_loss          Flow matching MSE on training batch (every log_interval)
actor/grad_norm           Global gradient L2 norm (every log_interval)
actor/loss_pos            FM loss on pos samples queried as cond=2 (every diag_interval)
actor/loss_neg            FM loss on neg samples queried as cond=1 (every diag_interval)
actor/loss_uncond         FM loss on all samples queried as cond=0 (every diag_interval)
actor/guidance_gap        mean ‖v_pos − v_uncond‖₂ — THE key RECAP metric (every diag_interval)
actor/velocity_r2         Explained variance of velocity prediction (every diag_interval)
actor/v_magnitude         Mean ‖v_pred‖₂ — sanity check (every diag_interval)

eval/return_beta0         Mean episode return with Beta=0 (unconditional)
eval/success_beta0        Mean success rate with Beta=0
eval/return_beta1         Mean episode return with Beta=1 (positive-conditioned)
eval/success_beta1        Mean success rate with Beta=1
eval/guidance_lift        return(Beta=1) - return(Beta=0) — should be positive if RECAP works

Usage:
  python recap_phase0_ogbench.py --env cube-single-play-singletask-v0
  python recap_phase0_ogbench.py --env cube-single-play-singletask-v0 --no_wandb
  python recap_phase0_ogbench.py --env cube-single-play-singletask-v0 \\
    --wandb_project recap --wandb_run phase0-seed42 --seed 42
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pickle
import time
from typing import Sequence, Tuple
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

import ogbench
import wandb
import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

# =====  Data Loading (OGBench) =====
def load_ogbench_data(env_name: str):
    """Load environment and dataset from OGBench.

    """
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)

    obs = np.asarray(train_dataset["observations"], dtype=np.float32)
    actions = np.asarray(train_dataset["actions"], dtype=np.float32)
    raw_rews = np.asarray(train_dataset["rewards"], dtype=np.float32)
    terminals = np.asarray(train_dataset["terminals"], dtype=bool)
    masks = np.asarray(train_dataset["masks"], dtype=np.float32)

    print(f"  raw rewards unique: {np.unique(raw_rews)}")
    print(f"  raw rewards == 0 count: {(raw_rews == 0).sum()}")
    print(f"  raw rewards == -1 count: {(np.abs(raw_rews + 1) < 1e-6).sum()}")
    print(f"  terminals sum: {terminals.sum()}")
    print(f"  masks == 0 (task complete): {(masks == 0).sum()}")
    print(f"  rewards at terminals: {np.unique(raw_rews[terminals])}")
    print(f"  dataset keys: {list(train_dataset.keys())}")
    rewards = np.asarray(train_dataset["rewards"], dtype=np.float32)

    return env, obs, actions, rewards, terminals

def reset_env(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def step_env(env, action):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, r, terminated, truncated, info = out
        return obs, float(r), bool(terminated) or bool(truncated), info
    obs, r, done, info = out
    return obs, float(r), bool(done), info


class SinusoidalTimeEmbed(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, eta: jnp.ndarray) -> jnp.ndarray:
        eta = jnp.squeeze(eta, axis=-1) if eta.ndim > 1 else eta
        half = self.dim // 2
        freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / half)
        args = eta[:, None] * freqs[None, :]
        emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
        if self.dim % 2 == 1:
            emb = jnp.concatenate([emb, jnp.zeros_like(emb[:, :1])], axis=-1)
        return emb


class ValueNet(nn.Module):
    hidden_sizes: Sequence[int]
    num_bins: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs
        for hs in self.hidden_sizes:
            x = nn.relu(nn.Dense(hs)(x))
        return nn.Dense(self.num_bins)(x)


class FlowActor(nn.Module):
    hidden_sizes: Sequence[int]
    act_dim: int
    cond_dim: int = 3
    time_embed_dim: int = 64

    @nn.compact
    def __call__(self, obs, cond_id, x_t, eta) -> jnp.ndarray:
        t_emb = SinusoidalTimeEmbed(self.time_embed_dim)(eta)
        cond_oh = jax.nn.one_hot(cond_id, self.cond_dim)
        x = jnp.concatenate([obs, cond_oh, x_t, t_emb], axis=-1)
        for hs in self.hidden_sizes:
            x = nn.Dense(hs)(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
        return nn.Dense(self.act_dim)(x)


def compute_rtg(rewards: np.ndarray, dones: np.ndarray, gamma: float) -> np.ndarray:
    rtg = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            running = 0.0
        running = float(rewards[t]) + gamma * running
        rtg[t] = running
    return rtg


def make_bins(rtg, num_bins, vmin, vmax):
    if vmin is None:
        vmin = float(np.percentile(rtg, 1.0))
    if vmax is None:
        vmax = float(np.percentile(rtg, 99.0))
    if vmax <= vmin + 1e-6:
        vmax = vmin + 1.0
    centers = np.linspace(vmin, vmax, num_bins, dtype=np.float32)
    return centers, vmin, vmax


def discretize_to_bins(rtg, vmin, vmax, num_bins):
    x = (rtg - vmin) / (vmax - vmin)
    idx = np.rint(x * (num_bins - 1)).astype(np.int32)
    return np.clip(idx, 0, num_bins - 1)


def batched_value_expectation(value_net, value_params, obs, bin_centers, batch_size=16384):
    centers = jnp.asarray(bin_centers, dtype=jnp.float32)

    @jax.jit
    def _v_batch(o_batch):
        logits = value_net.apply({"params": value_params}, o_batch)
        probs = jax.nn.softmax(logits, axis=-1)
        return jnp.sum(probs * centers[None, :], axis=-1)

    out = np.zeros((obs.shape[0],), dtype=np.float32)
    for i in range(0, obs.shape[0], batch_size):
        o = jnp.asarray(obs[i : i + batch_size], dtype=jnp.float32)
        out[i : i + batch_size] = np.array(_v_batch(o), dtype=np.float32)
    return out


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation without scipy."""
    n = len(x)
    if n < 3:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    d = rx - ry
    return float(1.0 - 6.0 * np.sum(d ** 2) / (n * (n ** 2 - 1)))


# Training Steps (return grad_norm for logging)

def make_value_train_step(value_net: ValueNet):
    """Value step that also returns gradient norm."""
    def step(state, obs_b, bin_b):
        def loss_fn(params):
            logits = value_net.apply({"params": params}, obs_b)
            return optax.softmax_cross_entropy_with_integer_labels(logits, bin_b).mean()
            # don't do integer bins
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        grad_norm = optax.global_norm(grads)
        state = state.apply_gradients(grads=grads)
        return state, loss, grad_norm

    return jax.jit(step)


def make_flow_actor_train_step(flow_actor, action_low, action_high, cond_dropout_rate=0.3):
    """Flow actor step that also returns gradient norm."""
    low = jnp.asarray(action_low, dtype=jnp.float32)
    high = jnp.asarray(action_high, dtype=jnp.float32)
    scale = (high - low) / 2.0
    bias = (high + low) / 2.0

    def normalize_actions(a):
        return (a - bias) / (scale + 1e-8)

    def step(state, obs_b, act_b, cond_b, rng):
        rng_eta, rng_noise, rng_dropout = jax.random.split(rng, 3)
        B = obs_b.shape[0]

        dropout_mask = jax.random.uniform(rng_dropout, shape=(B,)) < cond_dropout_rate
        cond_b_dropped = jnp.where(dropout_mask, 0, cond_b)

        a_norm = normalize_actions(act_b)
        eta = jax.random.uniform(rng_eta, shape=(B,))
        omega = jax.random.normal(rng_noise, shape=a_norm.shape)
        eta_bc = eta[:, None]
        x_t = eta_bc * a_norm + (1.0 - eta_bc) * omega
        v_target = a_norm - omega

        def loss_fn(params):
            v_pred = flow_actor.apply(
                {"params": params}, obs_b, cond_b_dropped, x_t, eta
            )
            return jnp.mean(jnp.sum((v_pred - v_target) ** 2, axis=-1))

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        grad_norm = optax.global_norm(grads)
        state = state.apply_gradients(grads=grads)
        return state, loss, grad_norm

    return jax.jit(step)


# ═══════════════════════════════════════════════════════════════
# Diagnostic Functions (separate JIT — called at intervals only)
# ═══════════════════════════════════════════════════════════════

def make_value_diagnostics(value_net, bin_centers_j):
    """Creates JIT-compiled function for value function validation metrics.

    Returns (val_loss, v_pred, mae, mean_bias, entropy) where v_pred is
    the full vector of predictions (pulled to host for rank correlation).
    """
    @jax.jit
    def compute(params, obs_val, bins_val, rtg_val):
        logits = value_net.apply({"params": params}, obs_val)

        # Val cross-entropy loss
        val_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, bins_val
        ).mean()

        # Expected value
        probs = jax.nn.softmax(logits, axis=-1)
        v_pred = jnp.sum(probs * bin_centers_j[None, :], axis=-1)

        # MAE
        mae = jnp.mean(jnp.abs(v_pred - rtg_val))

        # Mean bias: positive means V overestimates
        mean_bias = jnp.mean(v_pred) - jnp.mean(rtg_val)

        # Distribution entropy: high = uncertain, low = collapsed to point estimate
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1).mean()

        return val_loss, v_pred, mae, mean_bias, entropy

    return compute


def make_actor_diagnostics(flow_actor, action_low, action_high):
    """Creates JIT-compiled function for flow actor conditioning diagnostics.

    guidance_gap: v_pos - v_uncon (if this is near zero, CFG inference will produce 
    identical behavior regardless of Beta, meaning RECAP's conditioning mechanism has failed)
    """
    low = jnp.asarray(action_low, dtype=jnp.float32)
    high = jnp.asarray(action_high, dtype=jnp.float32)
    scale = (high - low) / 2.0
    bias = (high + low) / 2.0

    @jax.jit
    def compute(params, obs_b, act_b, cond_b, rng):
        B = obs_b.shape[0]
        rng_eta, rng_noise = jax.random.split(rng)

        a_norm = (act_b - bias) / (scale + 1e-8)
        eta = jax.random.uniform(rng_eta, (B,))
        omega = jax.random.normal(rng_noise, a_norm.shape)
        eta_bc = eta[:, None]
        x_t = eta_bc * a_norm + (1.0 - eta_bc) * omega
        v_target = a_norm - omega

        # Three forward passes: one per condition
        uncond_id = jnp.zeros(B, dtype=jnp.int32)
        neg_id = jnp.ones(B, dtype=jnp.int32)
        pos_id = jnp.full(B, 2, dtype=jnp.int32)

        v_uncond = flow_actor.apply({"params": params}, obs_b, uncond_id, x_t, eta)
        v_neg = flow_actor.apply({"params": params}, obs_b, neg_id, x_t, eta)
        v_pos = flow_actor.apply({"params": params}, obs_b, pos_id, x_t, eta)

        # Per-sample MSE for each condition
        mse_uncond = jnp.sum((v_uncond - v_target) ** 2, axis=-1)
        mse_neg = jnp.sum((v_neg - v_target) ** 2, axis=-1)
        mse_pos = jnp.sum((v_pos - v_target) ** 2, axis=-1)

        # Conditional losses: loss on matching data
        # "Does cond=2 predict well on data that IS positive?"
        is_pos = cond_b == 2
        is_neg = cond_b == 1
        n_pos = jnp.maximum(is_pos.sum(), 1)
        n_neg = jnp.maximum(is_neg.sum(), 1)

        loss_pos = jnp.where(is_pos, mse_pos, 0.0).sum() / n_pos
        loss_neg = jnp.where(is_neg, mse_neg, 0.0).sum() / n_neg
        loss_uncond = mse_uncond.mean()

        # Guidance gap
        # How different are positive vs unconditioned velocity predictions?
        guidance_gap = jnp.mean(
            jnp.sqrt(jnp.sum((v_pos - v_uncond) ** 2, axis=-1) + 1e-8)
        )

        # Velocity R^2: explained variance
        # How much of the velocity variance is the model capturing?
        var_target = jnp.mean(jnp.sum(v_target ** 2, axis=-1))
        r2 = 1.0 - mse_pos.mean() / (var_target + 1e-8)

        # Velocity magnitude: sanity check
        v_mag = jnp.mean(jnp.sqrt(jnp.sum(v_pos ** 2, axis=-1) + 1e-8))

        return (loss_pos, loss_neg, loss_uncond, guidance_gap, r2, v_mag)

    return compute


# CFG Inference
@partial(jax.jit, static_argnums=(0, 5))
def euler_sample_cfg(flow_actor, params, obs, rng, beta: float, num_steps: int = 20):
    B = obs.shape[0]
    act_dim = flow_actor.act_dim
    dt = 1.0 / num_steps
    uncond_id = jnp.zeros((B,), dtype=jnp.int32)
    pos_id = jnp.full((B,), 2, dtype=jnp.int32)
    x0 = jax.random.normal(rng, shape=(B, act_dim))

    def body_fn(k, x):
        eta_k = jnp.full((B,), k / num_steps)
        v_u = flow_actor.apply({"params": params}, obs, uncond_id, x, eta_k)
        v_p = flow_actor.apply({"params": params}, obs, pos_id, x, eta_k)
        v_g = v_u + beta * (v_p - v_u)
        return x + dt * v_g

    def scan_fn(x, k):
        return body_fn(k, x), None

    ks = jnp.arange(num_steps)
    x_final, _ = jax.lax.scan(scan_fn, x0, ks)
    return x_final


def sample_action_cfg(flow_actor, params, obs, beta, rng, action_low, action_high,
                      num_steps=20, deterministic=False):
    obs_j = jnp.asarray(obs[None, :], dtype=jnp.float32)
    if deterministic:
        sample_rng = jax.random.PRNGKey(0)
    else:
        rng, sample_rng = jax.random.split(rng)

    a_norm = euler_sample_cfg(flow_actor, params, obs_j, sample_rng, beta, num_steps)
    low = jnp.asarray(action_low, dtype=jnp.float32)
    high = jnp.asarray(action_high, dtype=jnp.float32)
    a_env = a_norm[0] * ((high - low) / 2.0) + ((high + low) / 2.0)
    a_env = jnp.clip(a_env, low, high)
    return np.array(a_env, dtype=np.float32), rng


def evaluate(env, flow_actor, actor_params, episodes, seed, beta,
             action_low, action_high, num_steps=20):
    rng = jax.random.PRNGKey(seed)
    returns, successes = [], []
    for _ in range(episodes):
        obs = reset_env(env)
        done, ep_ret, success = False, 0.0, 0.0
        while not done:
            action, rng = sample_action_cfg(
                flow_actor, actor_params, obs, beta, rng,
                action_low, action_high, num_steps, deterministic=True,
            )
            obs, r, done, info = step_env(env, action)
            ep_ret += r
            if info.get("success", 0.0):
                success = 1.0
        returns.append(ep_ret)
        successes.append(success)
    return float(np.mean(returns)), float(np.mean(successes))

@dataclasses.dataclass
class Checkpoint:
    env_name: str
    seed: int
    gamma: float
    num_bins: int
    vmin: float
    vmax: float
    bin_centers: np.ndarray
    epsilon: float
    pos_frac: float
    cond_dropout_rate: float
    value_hidden: Tuple[int, ...]
    actor_hidden: Tuple[int, ...]
    actor_params: dict
    value_params: dict
    action_low: np.ndarray
    action_high: np.ndarray
    act_dim: int
    num_euler_steps: int


def save_checkpoint(path: str, ckpt: Checkpoint) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    actor_params_np = jax.tree_util.tree_map(lambda x: np.array(x), ckpt.actor_params)
    value_params_np = jax.tree_util.tree_map(lambda x: np.array(x), ckpt.value_params)
    payload = dataclasses.asdict(ckpt)
    payload["actor_params"] = actor_params_np
    payload["value_params"] = value_params_np
    with open(path, "wb") as f:
        pickle.dump(payload, f)

def main():
    p = argparse.ArgumentParser(
        description="RECAP Phase 0: Offline training with flow matching + CFG"
    )

    p.add_argument("--env", type=str, default="cube-single-play-singletask-v0")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--gamma", type=float, default=0.99)

    p.add_argument("--num_bins", type=int, default=201)
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--value_hidden", type=str, default="256,256")
    p.add_argument("--value_lr", type=float, default=3e-4)
    p.add_argument("--value_steps", type=int, default=29000000)

    p.add_argument("--actor_hidden", type=str, default="256,256,256")
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--actor_steps", type=int, default=29000000)
    p.add_argument("--num_euler_steps", type=int, default=20)

    p.add_argument("--pos_frac", type=float, default=0.30)
    p.add_argument("--cond_dropout_rate", type=float, default=0.3)

    p.add_argument("--batch", type=int, default=1024)

    p.add_argument("--beta_eval", type=float, default=1.0)

    p.add_argument("--eval_episodes", type=int, default=20)

    p.add_argument("--out", type=str, default="./checkpoints/phase0.pkl")

    p.add_argument("--wandb_project", type=str, default="recap")
    p.add_argument("--wandb_run", type=str, default=None,
                   help="Run name (auto-generated if not set)")
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable wandb entirely (mode='disabled')")
    p.add_argument("--log_interval", type=int, default=100,
                   help="Log loss + grad_norm every N training steps")
    p.add_argument("--diag_interval", type=int, default=5000,
                   help="Run heavier diagnostics every N steps")
    p.add_argument("--eval_interval", type=int, default=100000,
                   help="Run evaluation rollouts every N actor steps (expensive)")
    p.add_argument("--val_frac", type=float, default=0.1,
                   help="Fraction of data held out for validation diagnostics")
    p.add_argument("--diag_batch", type=int, default=2048,
                   help="Batch size for diagnostic forward passes")
    p.add_argument("--use_lambda_returns", action="store_true",
                   help="Tag run name as lambda vs MC returns")

    args = p.parse_args()

    method_tag = "lambda" if args.use_lambda_returns else "mc"
    run_name = args.wandb_run or f"phase0-{args.env}-s{args.seed}-{method_tag}"
    if args.no_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    print(f"{'='*60}")
    print(f"RECAP Phase 0: {args.env}")
    print(f"{'='*60}")

    print("\n[1/7] Loading OGBench dataset...")
    env, obs, acts, rews, dones = load_ogbench_data(args.env)

    act_dim = acts.shape[-1]
    obs_dim = obs.shape[-1]
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)

    print(f"  obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"  dataset size: {len(obs)} transitions")
    print(f"  action range: [{low.min():.2f}, {high.max():.2f}]")
    print(f"  reward range: [{rews.min():.2f}, {rews.max():.2f}]")

    wandb.log({
        "data/n_transitions": len(obs),
        "data/obs_dim": obs_dim,
        "data/act_dim": act_dim,
        "data/reward_min": float(rews.min()),
        "data/reward_max": float(rews.max()),
        "data/n_episodes": int(dones.sum()),
    }, step=0)

    # ── Compute episode returns for dataset quality check ──
    ep_returns = []
    running = 0.0
    for t in range(len(rews)):
        running += rews[t]
        if dones[t]:
            ep_returns.append(running)
            running = 0.0
    ep_returns = np.array(ep_returns)
    if len(ep_returns) > 0:
        wandb.log({"data/episode_returns": wandb.Histogram(ep_returns)}, step=0)
        print(f"  episode returns: mean={ep_returns.mean():.2f}, "
              f"std={ep_returns.std():.2f}, "
              f"min={ep_returns.min():.2f}, max={ep_returns.max():.2f}")

    # Compute RTG
    print("\n[2/7] Computing return-to-go...")
    rtg = compute_rtg(rews, dones, gamma=args.gamma)
    print(f"  RTG range: [{rtg.min():.2f}, {rtg.max():.2f}]")

    bin_centers, vmin, vmax = make_bins(rtg, args.num_bins, args.vmin, args.vmax)
    rtg_bins = discretize_to_bins(rtg, vmin, vmax, args.num_bins)
    print(f"  bins: {args.num_bins} bins in [{vmin:.2f}, {vmax:.2f}]")

    # Train / val split
    # Random split at the transition level. RTG is pre-computed so this is safe
    n_data = obs.shape[0]
    rng_np = np.random.RandomState(args.seed + 7777)
    perm = rng_np.permutation(n_data)
    n_val = max(1, int(n_data * args.val_frac))
    n_train = n_data - n_val
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    print(f"  train: {n_train}, val: {n_val}")

    obs_val_j = jnp.asarray(obs[val_idx], dtype=jnp.float32)
    bins_val_j = jnp.asarray(rtg_bins[val_idx], dtype=jnp.int32)
    rtg_val_j = jnp.asarray(rtg[val_idx], dtype=jnp.float32)
    rtg_val_np = rtg[val_idx]

    # Full arrays for training
    obs_j = jnp.asarray(obs, dtype=jnp.float32)
    bins_j = jnp.asarray(rtg_bins, dtype=jnp.int32)
    train_idx_j = jnp.asarray(train_idx, dtype=jnp.int32)

    # Parse hidden sizes ──
    value_hidden = tuple(int(x) for x in args.value_hidden.split(",") if x.strip())
    actor_hidden = tuple(int(x) for x in args.actor_hidden.split(",") if x.strip())

    # models
    print("\n[3/7] Initializing models...")
    key = jax.random.PRNGKey(args.seed)
    key, k_val, k_act = jax.random.split(key, 3)

    value_net = ValueNet(hidden_sizes=value_hidden, num_bins=args.num_bins)
    value_params = value_net.init(
        k_val, jnp.zeros((1, obs_dim), dtype=jnp.float32)
    )["params"]

    flow_actor = FlowActor(hidden_sizes=actor_hidden, act_dim=act_dim)
    actor_params = flow_actor.init(
        k_act,
        jnp.zeros((1, obs_dim)),
        jnp.zeros((1,), jnp.int32),
        jnp.zeros((1, act_dim)),
        jnp.zeros((1,)),
    )["params"]

    val_param_count = sum(x.size for x in jax.tree_util.tree_leaves(value_params))
    act_param_count = sum(x.size for x in jax.tree_util.tree_leaves(actor_params))
    print(f"  Value net: {val_param_count:,} params")
    print(f"  Flow actor: {act_param_count:,} params")
    wandb.log({"model/value_params": val_param_count,
               "model/actor_params": act_param_count}, step=0)

    # [4/7] Train Value Function
    print(f"\n[4/7] Training value function ({args.value_steps} steps)...")

    value_state = TrainState.create(
        apply_fn=value_net.apply,
        params=value_params,
        tx=optax.adam(args.value_lr),
    )
    value_step = make_value_train_step(value_net)
    value_diag = make_value_diagnostics(
        value_net, jnp.asarray(bin_centers, dtype=jnp.float32)
    )

    @jax.jit
    def sample_train_batch(rng_key):
        # Sample from train indices only
        raw_idx = jax.random.randint(rng_key, (args.batch,), 0, n_train)
        return train_idx_j[raw_idx]

    rng = jax.random.PRNGKey(args.seed + 123)
    t0 = time.time()

    for t in range(args.value_steps):
        rng, sub = jax.random.split(rng)
        idx = sample_train_batch(sub)
        value_state, loss, grad_norm = value_step(value_state, obs_j[idx], bins_j[idx])

        if (t + 1) % args.log_interval == 0:
            wandb.log({
                "value/train_loss": float(loss),
                "value/grad_norm": float(grad_norm),
                "value/step": t + 1,
            }, step=t + 1)

        if (t + 1) % args.diag_interval == 0:
            val_loss, v_pred_j, mae, mean_bias, entropy = value_diag(
                value_state.params, obs_val_j, bins_val_j, rtg_val_j
            )
            v_pred_np = np.array(v_pred_j, dtype=np.float32)
            rho = rank_correlation(v_pred_np, rtg_val_np)

            elapsed = time.time() - t0
            sps = (t + 1) / elapsed

            wandb.log({
                "value/val_loss": float(val_loss),
                "value/val_mae": float(mae),
                "value/val_rank_corr": rho,
                "value/val_mean_bias": float(mean_bias),
                "value/val_entropy": float(entropy),
                "value/step": t + 1,
                "perf/value_sps": sps,
            }, step=t + 1)

            print(f"  step {t+1:>6d}/{args.value_steps}  "
                  f"train_loss={float(loss):.4f}  "
                  f"val_mae={float(mae):.3f}  "
                  f"rank_ρ={rho:.3f}  "
                  f"bias={float(mean_bias):.3f}  "
                  f"entropy={float(entropy):.2f}  "
                  f"[{sps:.0f} sps]")

    # [5/7] Compute Advantages
    print("\n[5/7] Computing advantages...")
    v_pred = batched_value_expectation(value_net, value_state.params, obs, bin_centers)
    adv = rtg - v_pred # RTG is an unbiased estimate of Q.

    epsilon = float(np.quantile(adv, 1.0 - args.pos_frac))
    is_pos = adv > epsilon
    cond_id = np.where(is_pos, 2, 1).astype(np.int32)

    n_pos = is_pos.sum()
    actual_pos_frac = float(n_pos / len(adv))

    print(f"  advantage range: [{adv.min():.2f}, {adv.max():.2f}]")
    print(f"  advantage std: {adv.std():.4f}")
    print(f"  epsilon (threshold): {epsilon:.4f}")
    print(f"  positive: {n_pos} / {len(adv)} ({100*actual_pos_frac:.1f}%)")

    # Log adv distribution 
    adv_step = args.value_steps + 1
    wandb.log({
        "advantage/epsilon": epsilon,
        "advantage/std": float(adv.std()),
        "advantage/pos_frac": actual_pos_frac,
        "advantage/mean": float(adv.mean()),
        "advantage/min": float(adv.min()),
        "advantage/max": float(adv.max()),
        "advantage/histogram": wandb.Histogram(adv, num_bins=100),
        "advantage/mean_pos": float(adv[is_pos].mean()) if n_pos > 0 else 0.0,
        "advantage/mean_neg": float(adv[~is_pos].mean()) if (~is_pos).sum() > 0 else 0.0,
    }, step=adv_step)

    # [6/7] Train Flow Actor
    print(f"\n[6/7] Training flow actor ({args.actor_steps} steps, "
          f"dropout={args.cond_dropout_rate})...")

    actor_state = TrainState.create(
        apply_fn=flow_actor.apply,
        params=actor_params,
        tx=optax.adam(args.actor_lr),
    )
    actor_step = make_flow_actor_train_step(
        flow_actor, action_low=low, action_high=high,
        cond_dropout_rate=args.cond_dropout_rate,
    )
    actor_diag = make_actor_diagnostics(flow_actor, low, high)

    acts_j = jnp.asarray(acts, dtype=jnp.float32)
    cond_j = jnp.asarray(cond_id, dtype=jnp.int32)

    # ── Fixed diagnostic batch ──
    # Same samples every time -> metrics are comparable across training steps.
    # Uses val data so we're measuring generalization, not memorization.
    diag_rng_fixed = jax.random.PRNGKey(args.seed + 5555)
    n_diag = min(args.diag_batch, n_val)
    diag_idx = val_idx[:n_diag]
    diag_obs_j = jnp.asarray(obs[diag_idx], dtype=jnp.float32)
    diag_acts_j = jnp.asarray(acts[diag_idx], dtype=jnp.float32)
    diag_cond_j = jnp.asarray(cond_id[diag_idx], dtype=jnp.int32)

    @jax.jit
    def sample_full_batch(rng_key):
        return jax.random.randint(rng_key, (args.batch,), 0, n_data)

    # Use a separate step counter for actor (offset from value steps)
    actor_step_offset = args.value_steps + 2
    t0 = time.time()

    for t in range(args.actor_steps):
        rng, sub_batch, sub_flow = jax.random.split(rng, 3)
        idx = sample_full_batch(sub_batch)
        actor_state, loss, grad_norm = actor_step(
            actor_state, obs_j[idx], acts_j[idx], cond_j[idx], sub_flow
        )
        global_step = actor_step_offset + t + 1

        # ── Light logging ──
        if (t + 1) % args.log_interval == 0:
            wandb.log({
                "actor/train_loss": float(loss),
                "actor/grad_norm": float(grad_norm),
                "actor/step": t + 1,
            }, step=global_step)

        # ── Heavy diagnostics: conditional losses + guidance gap ──
        if (t + 1) % args.diag_interval == 0:
            (loss_pos, loss_neg, loss_uncond,
             guidance_gap, velocity_r2, v_mag) = actor_diag(
                actor_state.params,
                diag_obs_j, diag_acts_j, diag_cond_j,
                diag_rng_fixed,  # same random draw every time
            )

            elapsed = time.time() - t0
            sps = (t + 1) / elapsed

            wandb.log({
                "actor/loss_pos": float(loss_pos),
                "actor/loss_neg": float(loss_neg),
                "actor/loss_uncond": float(loss_uncond),
                "actor/guidance_gap": float(guidance_gap),
                "actor/velocity_r2": float(velocity_r2),
                "actor/v_magnitude": float(v_mag),
                "actor/step": t + 1,
                "perf/actor_sps": sps,
            }, step=global_step)

            print(f"  step {t+1:>6d}/{args.actor_steps}  "
                  f"loss={float(loss):.4f}  "
                  f"gap={float(guidance_gap):.4f}  "
                  f"R²={float(velocity_r2):.3f}  "
                  f"L_pos={float(loss_pos):.3f}  "
                  f"L_neg={float(loss_neg):.3f}  "
                  f"L_unc={float(loss_uncond):.3f}  "
                  f"[{sps:.0f} sps]")

        # ── Periodic evaluation rollouts ──
        if (t + 1) % args.eval_interval == 0 and (t + 1) < args.actor_steps:
            print(f"  [eval @ step {t+1}]")
            ret0, succ0 = evaluate(
                env, flow_actor, actor_state.params,
                args.eval_episodes, args.seed + 999,
                beta=0.0, action_low=low, action_high=high,
                num_steps=args.num_euler_steps,
            )
            ret1, succ1 = evaluate(
                env, flow_actor, actor_state.params,
                args.eval_episodes, args.seed + 999,
                beta=1.0, action_low=low, action_high=high,
                num_steps=args.num_euler_steps,
            )
            guidance_lift = ret1 - ret0
            wandb.log({
                "eval/return_beta0": ret0,
                "eval/success_beta0": succ0,
                "eval/return_beta1": ret1,
                "eval/success_beta1": succ1,
                "eval/guidance_lift": guidance_lift,
                "actor/step": t + 1,
            }, step=global_step)
            print(f"    Beta=0: ret={ret0:.2f} succ={succ0:.2%}  "
                  f"Beta=1: ret={ret1:.2f} succ={succ1:.2%}  "
                  f"lift={guidance_lift:.2f}")

    # ═══════════════════════════════════════════════════════════
    # [7/7] Final Evaluation
    # ═══════════════════════════════════════════════════════════
    print(f"\n[7/7] Final evaluation ({args.eval_episodes} episodes)...")
    final_step = actor_step_offset + args.actor_steps + 1

    betas_to_test = [0.0, 1.0]
    if args.beta_eval not in betas_to_test:
        betas_to_test.append(args.beta_eval)

    eval_results = {}
    for beta in betas_to_test:
        label = {0.0: "uncond", 1.0: "pos"}.get(beta, f"cfg={beta:g}")
        avg_ret, avg_succ = evaluate(
            env, flow_actor, actor_state.params,
            args.eval_episodes, args.seed + 999,
            beta=beta, action_low=low, action_high=high,
            num_steps=args.num_euler_steps,
        )
        eval_results[beta] = (avg_ret, avg_succ)
        print(f"  Beta={beta:4.1f} ({label:>8s}): return={avg_ret:8.2f}  success={avg_succ:.2%}")

    # Log final eval
    final_log = {}
    if 0.0 in eval_results and 1.0 in eval_results:
        ret0, succ0 = eval_results[0.0]
        ret1, succ1 = eval_results[1.0]
        final_log.update({
            "eval/return_beta0": ret0,
            "eval/success_beta0": succ0,
            "eval/return_beta1": ret1,
            "eval/success_beta1": succ1,
            "eval/guidance_lift": ret1 - ret0,
        })
    for beta, (ret, succ) in eval_results.items():
        if beta not in (0.0, 1.0):
            final_log[f"eval/return_beta{beta:g}"] = ret
            final_log[f"eval/success_beta{beta:g}"] = succ
    wandb.log(final_log, step=final_step)

    # ── Summary metrics (shown in wandb run overview) ──
    if 0.0 in eval_results and 1.0 in eval_results:
        wandb.run.summary["final/return_beta0"] = eval_results[0.0][0]
        wandb.run.summary["final/return_beta1"] = eval_results[1.0][0]
        wandb.run.summary["final/success_beta0"] = eval_results[0.0][1]
        wandb.run.summary["final/success_beta1"] = eval_results[1.0][1]
        wandb.run.summary["final/guidance_lift"] = eval_results[1.0][0] - eval_results[0.0][0]
        wandb.run.summary["final/advantage_epsilon"] = epsilon
        wandb.run.summary["final/advantage_std"] = float(adv.std())

    # ── Save checkpoint ──
    ckpt = Checkpoint(
        env_name=args.env,
        seed=args.seed,
        gamma=args.gamma,
        num_bins=args.num_bins,
        vmin=vmin,
        vmax=vmax,
        bin_centers=bin_centers,
        epsilon=epsilon,
        pos_frac=args.pos_frac,
        cond_dropout_rate=args.cond_dropout_rate,
        value_hidden=value_hidden,
        actor_hidden=actor_hidden,
        actor_params=actor_state.params,
        value_params=value_state.params,
        action_low=low,
        action_high=high,
        act_dim=act_dim,
        num_euler_steps=args.num_euler_steps,
    )
    save_checkpoint(args.out, ckpt)
    print(f"\nCheckpoint saved to: {args.out}")

    wandb.finish()


if __name__ == "__main__":
    main()