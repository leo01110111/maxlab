"""Roll out a trained PPO policy on the pick task, optionally rendering a video
or opening the live viewer.

    python -m rl.rollout --model rl/runs/models/best_model.zip \
        --vecnormalize rl/runs/models/vecnormalize.pkl --episodes 10 --video out.mp4

    python -m rl.rollout --model rl/runs/models/best_model.zip \
        --vecnormalize rl/runs/models/vecnormalize.pkl --viewer   # live window
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.env import PickCubeEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--vecnormalize", default=None)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--block-noise", type=float, default=0.06)
    p.add_argument("--video", default=None, help="write an mp4 of the rollouts")
    p.add_argument("--viewer", action="store_true",
                   help="open the live MuJoCo viewer and play back in real time")
    args = p.parse_args()

    def factory():
        return PickCubeEnv(max_episode_steps=args.max_steps,
                           block_pos_noise=args.block_noise,
                           render_mode="rgb_array" if args.video else None,
                           show_viewer=args.viewer)

    venv = DummyVecEnv([factory])
    if args.vecnormalize and Path(args.vecnormalize).exists():
        venv = VecNormalize.load(args.vecnormalize, venv)
        venv.training = False
        venv.norm_reward = False

    model = PPO.load(args.model, device="cpu")

    frames = []
    successes = 0
    for ep in range(args.episodes):
        obs = venv.reset()
        done = False
        ep_success = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = venv.step(action)
            done = dones[0]
            ep_success = max(ep_success, infos[0].get("success", 0))
            if args.video:
                frames.append(venv.envs[0].render())
            if args.viewer:
                time.sleep(1.0 / 20.0)  # pace to ~control_hz so it's watchable
        successes += ep_success
        print(f"episode {ep}: success={ep_success}")

    print(f"\nsuccess rate: {successes}/{args.episodes} = {successes/args.episodes:.0%}")

    if args.video and frames:
        import imageio
        imageio.mimsave(args.video, frames, fps=20)
        print("wrote", args.video)


if __name__ == "__main__":
    main()
