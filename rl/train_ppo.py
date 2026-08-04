"""PPO training for the green-cube pick task (rl.env.PickCubeEnv).

    python -m rl.train_ppo --timesteps 3_000_000

Logs to rl/runs/tb (TensorBoard) and saves checkpoints + the best model by eval
success rate to rl/runs/models. Watch a trained policy with:

    python -m rl.rollout --model rl/runs/models/best_model.zip \
        --vecnormalize rl/runs/models/vecnormalize.pkl --viewer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from rl.env import PickCubeEnv


def make_env_kwargs(max_steps: int, noise: float) -> dict:
    return {"max_episode_steps": max_steps, "block_pos_noise": noise}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--block-noise", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="rl/runs")
    args = p.parse_args()

    out = Path(args.outdir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "tb").mkdir(parents=True, exist_ok=True)

    env_kwargs = make_env_kwargs(args.max_steps, args.block_noise)
    venv = make_vec_env(
        PickCubeEnv, n_envs=args.n_envs, seed=args.seed,
        env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv,
    )
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Eval on a separate deterministic-reset venv (no reward norm on eval).
    eval_venv = make_vec_env(
        PickCubeEnv, n_envs=4, seed=args.seed + 100,
        env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv,
    )
    eval_venv = VecNormalize(eval_venv, norm_obs=True, norm_reward=False,
                             clip_obs=10.0, training=False)
    # Share running obs stats from the training venv into the eval venv.
    eval_venv.obs_rms = venv.obs_rms

    eval_cb = EvalCallback(
        eval_venv, best_model_save_path=str(out / "models"),
        log_path=str(out / "tb"), eval_freq=max(20_000 // args.n_envs, 1),
        n_eval_episodes=20, deterministic=True, render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(200_000 // args.n_envs, 1),
        save_path=str(out / "models"), name_prefix="ppo_pickcube",
        save_vecnormalize=True,
    )

    model = PPO(
        "MlpPolicy", venv, seed=args.seed, verbose=1,
        tensorboard_log=str(out / "tb"),
        n_steps=1024, batch_size=4096, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005, learning_rate=3e-4,
        policy_kwargs={"net_arch": [256, 256]},
        device="cuda",
    )
    model.learn(total_timesteps=args.timesteps, callback=[eval_cb, ckpt_cb],
                progress_bar=True)
    model.save(str(out / "models" / "final_model"))
    venv.save(str(out / "models" / "vecnormalize.pkl"))
    print("done. best model in", out / "models")


if __name__ == "__main__":
    main()
