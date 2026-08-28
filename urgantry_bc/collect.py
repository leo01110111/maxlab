"""Collect scripted-expert demonstrations for behavior cloning.

Only SUCCESSFUL episodes are kept -- the expert drops the cube on a minority of
starts, and cloning a failed carry teaches the policy to drop it too.

    uv run python -m urgantry_bc.collect --episodes 300 --out data/demos
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from urgantry_bc.expert import ScriptedExpert
from urgantry_bc.task_env import CubeInBoxEnv


def collect(episodes: int, out: Path, image_size: int, seed0: int,
            noise_std: float = 0.002, max_steps: int = 300) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    env = CubeInBoxEnv(image_size=image_size, max_episode_steps=max_steps)
    expert = ScriptedExpert(env)

    rng = np.random.default_rng(seed0)
    images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    ep_starts: list[int] = []
    n_success = 0
    t0 = time.time()

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        expert.reset()
        ep_img, ep_st, ep_ac = [], [], []
        success = 0
        for _ in range(max_steps):
            action = expert.act()
            ep_img.append(obs["image"])
            ep_st.append(obs["state"])
            # Label with the expert's CLEAN action but execute a noisy one (DART).
            # The arm actions are absolute joint targets, so the nominal action is
            # also the correction from wherever the noise pushed the arm -- the
            # policy learns to come back to the path instead of only ever seeing
            # the path. Closure is left clean; jittering it drops the cube.
            ep_ac.append(action.astype(np.float32))
            if noise_std > 0:
                action = action.copy()
                action[:6] += rng.normal(0.0, noise_std, size=6)
                action = np.clip(action, -1.0, 1.0)
            obs, _, term, trunc, info = env.step(action)
            success = max(success, info["success"])
            if term or trunc:
                break
        if success:
            ep_starts.append(len(images))
            images.extend(ep_img)
            states.extend(ep_st)
            actions.extend(ep_ac)
            n_success += 1
        if (ep + 1) % 25 == 0:
            print(f"  {ep+1}/{episodes} episodes, {n_success} kept, "
                  f"{len(images)} frames, {time.time()-t0:.0f}s", flush=True)

    env.close()
    imgs = np.asarray(images, dtype=np.uint8)
    np.save(out / "images.npy", imgs)
    np.savez(out / "traj.npz",
             states=np.asarray(states, dtype=np.float32),
             actions=np.asarray(actions, dtype=np.float32),
             ep_starts=np.asarray(ep_starts, dtype=np.int64))
    meta = {"episodes_run": episodes, "episodes_kept": n_success,
            "frames": int(imgs.shape[0]), "image_size": image_size,
            "noise_std": noise_std, "expert_success_rate": n_success / episodes,
            "seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--out", type=Path, default=Path("urgantry_bc/data/demos"))
    p.add_argument("--image-size", type=int, default=84)
    p.add_argument("--seed0", type=int, default=1000)
    # 1 normalized unit spans the whole ctrlrange -- 6.28 rad on most arm joints.
    # 0.03 therefore injects ~11 deg of joint noise every 50 ms and collapses the
    # expert to 1.75% success; 0.002 is ~0.7 deg, which perturbs without breaking.
    p.add_argument("--noise-std", type=float, default=0.002,
                   help="std of exploration noise on executed arm actions, in "
                        "normalized units (1.0 = full joint range)")
    collect(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
