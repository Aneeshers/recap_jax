# RECAP: RL with Experience and Corrections via Advantage-conditioned Policies

JAX/Flax implementation of RECAP ([Physical Intelligence, 2024](https://www.pi.website/download/pistar06.pdf)) on OGBench environments. Implements flow matching policies with classifier-free guidance (CFG) for offline-to-online RL.

## Method Overview

RECAP trains a flow matching policy conditioned on a binary advantage indicator (pos/neg). During training, the indicator is randomly dropped (conditioning dropout), enabling classifier-free guidance at inference. The guided policy steers actions toward high-advantage regions of the action space.

**Phase 0 (offline pre-training):**
1. Train a distributional value function V(s) on Monte Carlo returns.
2. Compute advantages A = RTG - E[V(s)], threshold to assign positive/negative labels.
3. Train a flow matching policy with 30% conditioning dropout.
4. Sample actions via Euler ODE integration with CFG.

**Phase 1 (online improvement) - not implemented, but plan lol:**
1. Collect rollouts with the current policy.
2. Merge online data with the offline dataset.
3. Retrain value function and policy from Phase 0 initialization (not previous iteration).
4. Use N-step advantages (N=50) instead of Monte Carlo.

## Files

| File | Description |
|------|-------------|
| `recap_phase0_ogbench.py` | Offline pre-training with wandb diagnostics |
| `bc_flow_baseline.py` | Unconditional flow matching BC baseline |
| `flow_actor.py` | Standalone flow matching components (network, training, sampling) |

## Installation

Requires Python 3.9+, JAX, Flax, Optax, OGBench, and wandb.

```bash
pip install jax jaxlib flax optax wandb ogbench
```

For headless servers:
```bash
export MUJOCO_GL=egl
```

## Docker (dev)

This mirrors the `unifloral` dev container setup for reproducible GPU runs.

1. Add your wandb API key:
```bash
echo "YOUR_WANDB_KEY" > dev/wandb_key
```
2. Build the image:
```bash
cd dev
./build.sh
```
3. Launch an interactive container on GPU 0:
```bash
cd ..
./launch_container.sh 0
```
4. Or run a single command (example: BC baseline):
```bash
cd ..
./launch_run.sh 0 python BC_policy.py --env cube-single-play-singletask-v0 --steps 100000
```

## Usage

### BC baseline (run this first)

Validates that the flow matching architecture can learn useful actions from the dataset before adding RECAP complexity.

```bash
python bc_flow_baseline.py \
  --env cube-single-play-singletask-v0 \
  --steps 100000
```

Expected: success rate near the dataset's ~2% indicates the architecture works.

### Phase 0: offline pre-training

```bash
python recap_phase0_ogbench.py \
  --env cube-single-play-singletask-v0 \
  --value_steps 100000 \
  --actor_steps 100000 \
  --out checkpoints/phase0.pkl
```
## Paper Reference

Key equations and sections referenced in the code:
- Flow matching objective: Section V-A, Eq. 4
- Conditioning dropout: Section V-B, Appendix F
- CFG inference: Appendix E, Eq. 13
- Reward definition: Section V-C, Eq. 5
- Distributional value function: Section IV-A, Eq. 1
- N-step advantages (Phase 1): Appendix F
- Fine-tune from pre-trained checkpoint: Section V-D
- Correction handling: Section IV-B