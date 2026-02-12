#!/usr/bin/env python3
"""
Baseline: Behavioral Cloning via Flow Matching (no RECAP conditioning).

Trains a flow matching policy to clone the dataset: obs -> action.
No condition IDs, no dropout, no CFG — just unconditional flow matching BC.

This answers the question: "Can the flow matching architecture learn useful
actions from this offline dataset at all?" 

Usage:
  python bc_flow_baseline.py --env cube-single-play-singletask-v0
  python bc_flow_baseline.py --env cube-single-play-singletask-v0 --steps 200000
"""

from __future__ import annotations

import argparse
import time
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from typing import Sequence

import ogbench


class SinusoidalTimeEmbed(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, eta):
        eta = jnp.squeeze(eta, axis=-1) if eta.ndim > 1 else eta
        half = self.dim // 2
        freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / half)
        args = eta[:, None] * freqs[None, :]
        emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
        if self.dim % 2 == 1:
            emb = jnp.concatenate([emb, jnp.zeros_like(emb[:, :1])], axis=-1)
        return emb


class BCFlowActor(nn.Module):
    """Unconditional flow matching velocity network.

    Same architecture as the RECAP FlowActor but without cond_id input.
    Maps (obs, x_t, η) -> velocity.
    """
    hidden_sizes: Sequence[int]
    act_dim: int
    time_embed_dim: int = 64

    @nn.compact
    def __call__(self, obs, x_t, eta):
        t_emb = SinusoidalTimeEmbed(self.time_embed_dim)(eta)
        x = jnp.concatenate([obs, x_t, t_emb], axis=-1)
        for hs in self.hidden_sizes:
            x = nn.Dense(hs)(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
        return nn.Dense(self.act_dim)(x)

def make_train_step(flow_actor, action_low, action_high):
    low = jnp.asarray(action_low, dtype=jnp.float32)
    high = jnp.asarray(action_high, dtype=jnp.float32)
    scale = (high - low) / 2.0
    bias = (high + low) / 2.0

    def normalize(a):
        return (a - bias) / (scale + 1e-8)

    def step(state, obs_b, act_b, rng):
        rng_eta, rng_noise = jax.random.split(rng)
        B = obs_b.shape[0]

        a_norm = normalize(act_b)
        eta = jax.random.uniform(rng_eta, (B,))
        omega = jax.random.normal(rng_noise, a_norm.shape)
        eta_bc = eta[:, None]
        x_t = eta_bc * a_norm + (1.0 - eta_bc) * omega
        v_target = a_norm - omega

        def loss_fn(params):
            v_pred = flow_actor.apply({"params": params}, obs_b, x_t, eta)
            return jnp.mean(jnp.sum((v_pred - v_target) ** 2, axis=-1))

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    return jax.jit(step)


@partial(jax.jit, static_argnums=(0, 4))
def euler_sample(flow_actor, params, obs, rng, num_steps: int = 20):
    B = obs.shape[0]
    act_dim = flow_actor.act_dim
    dt = 1.0 / num_steps
    x = jax.random.normal(rng, (B, act_dim))

    def body_fn(k, x):
        eta_k = jnp.full((B,), k / num_steps)
        v = flow_actor.apply({"params": params}, obs, x, eta_k)
        return x + dt * v

    return jax.lax.fori_loop(0, num_steps, body_fn, x)


def sample_action(flow_actor, params, obs, rng, action_low, action_high,
                  num_steps=20, deterministic=False):
    obs_j = jnp.asarray(obs[None, :], dtype=jnp.float32)
    if deterministic:
        sample_rng = jax.random.PRNGKey(0)
    else:
        rng, sample_rng = jax.random.split(rng)

    a_norm = euler_sample(flow_actor, params, obs_j, sample_rng, num_steps)
    low = jnp.asarray(action_low, dtype=jnp.float32)
    high = jnp.asarray(action_high, dtype=jnp.float32)
    a_env = a_norm[0] * ((high - low) / 2.0) + ((high + low) / 2.0)
    a_env = jnp.clip(a_env, low, high)
    return np.array(a_env, dtype=np.float32), rng


def evaluate(env, flow_actor, params, n_episodes, seed, action_low, action_high,
             num_steps=20):
    rng = jax.random.PRNGKey(seed)
    returns, successes, ep_lengths = [], [], []
    for _ in range(n_episodes):
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        done, ep_ret, success, steps = False, 0.0, 0.0, 0
        while not done:
            action, rng = sample_action(
                flow_actor, params, obs, rng,
                action_low, action_high, num_steps, deterministic=True,
            )
            out = env.step(action)
            if len(out) == 5:
                obs, r, terminated, truncated, info = out
                done = bool(terminated) or bool(truncated)
            else:
                obs, r, done, info = out
            ep_ret += float(r)
            steps += 1
            if info.get("success", 0.0):
                success = 1.0
        returns.append(ep_ret)
        successes.append(success)
        ep_lengths.append(steps)
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "success_mean": float(np.mean(successes)),
        "ep_length_mean": float(np.mean(ep_lengths)),
    }


def main():
    p = argparse.ArgumentParser(description="BC Flow Matching Baseline")
    p.add_argument("--env", type=str, default="cube-single-play-singletask-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=str, default="256,256,256")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=10000000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--num_euler_steps", type=int, default=20)
    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=10000)
    p.add_argument("--log_interval", type=int, default=1000)
    args = p.parse_args()

    print(f"{'='*60}")
    print(f"BC Flow Matching Baseline: {args.env}")
    print(f"{'='*60}")

    print("\nLoading dataset...")
    env, train_dataset, _ = ogbench.make_env_and_datasets(args.env)

    obs = np.asarray(train_dataset["observations"], dtype=np.float32)
    acts = np.asarray(train_dataset["actions"], dtype=np.float32)
    rews = np.asarray(train_dataset["rewards"], dtype=np.float32)
    terminals = np.asarray(train_dataset["terminals"], dtype=bool)

    act_dim = acts.shape[-1]
    obs_dim = obs.shape[-1]
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    n_data = obs.shape[0]

    n_success = (rews > -0.5).sum()  # 0 = success in this dataset
    n_episodes = terminals.sum()
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"  {n_data} transitions, {n_episodes} episodes")
    print(f"  {n_success} success transitions ({100*n_success/n_data:.1f}%)")
    print(f"  rewards: {np.unique(rews)}")

    hidden = tuple(int(x) for x in args.hidden.split(","))
    flow_actor = BCFlowActor(hidden_sizes=hidden, act_dim=act_dim)

    key = jax.random.PRNGKey(args.seed)
    key, k_init = jax.random.split(key)
    params = flow_actor.init(
        k_init,
        jnp.zeros((1, obs_dim)),
        jnp.zeros((1, act_dim)),
        jnp.zeros((1,)),
    )["params"]

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  {param_count:,} parameters")

    state = TrainState.create(
        apply_fn=flow_actor.apply,
        params=params,
        tx=optax.adam(args.lr),
    )
    train_step = make_train_step(flow_actor, low, high)

    obs_j = jnp.asarray(obs, dtype=jnp.float32)
    acts_j = jnp.asarray(acts, dtype=jnp.float32)

    @jax.jit
    def sample_idx(rng_key):
        return jax.random.randint(rng_key, (args.batch,), 0, n_data)

    # ── Train ──
    print(f"\nTraining for {args.steps} steps...")
    rng = jax.random.PRNGKey(args.seed + 123)
    t0 = time.time()

    for t in range(args.steps):
        rng, sub_batch, sub_flow = jax.random.split(rng, 3)
        idx = sample_idx(sub_batch)
        state, loss = train_step(state, obs_j[idx], acts_j[idx], sub_flow)

        if (t + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            sps = (t + 1) / elapsed
            print(f"  step {t+1:>7d}/{args.steps}  loss={float(loss):.4f}  [{sps:.0f} sps]")

        if (t + 1) % args.eval_interval == 0:
            print(f"  [eval @ step {t+1}]")
            results = evaluate(
                env, flow_actor, state.params, args.eval_episodes,
                args.seed + 999, low, high, args.num_euler_steps,
            )
            print(f"    return: {results['return_mean']:.2f} ± {results['return_std']:.2f}")
            print(f"    success: {results['success_mean']:.2%}")
            print(f"    ep_length: {results['ep_length_mean']:.0f}")

    print(f"\nFinal evaluation ({args.eval_episodes} episodes)...")
    results = evaluate(
        env, flow_actor, state.params, args.eval_episodes,
        args.seed + 999, low, high, args.num_euler_steps,
    )
    print(f"  return: {results['return_mean']:.2f} ± {results['return_std']:.2f}")
    print(f"  success: {results['success_mean']:.2%}")
    print(f"  ep_length: {results['ep_length_mean']:.0f}")

    ep_returns = []
    running = 0.0
    for i in range(len(rews)):
        running += rews[i]
        if terminals[i]:
            ep_returns.append(running)
            running = 0.0
    ep_returns = np.array(ep_returns)
    ds_success = (ep_returns > -999.5).sum()  # return > -1000 means at least 1 success
    print(f"\n  Dataset baseline: {n_episodes} episodes, "
          f"return={ep_returns.mean():.2f} ± {ep_returns.std():.2f}, "
          f"~{100*ds_success/max(1,len(ep_returns)):.1f}% with any success")
    print(f"  Policy should match or beat these numbers if BC is working.")


if __name__ == "__main__":
    main()