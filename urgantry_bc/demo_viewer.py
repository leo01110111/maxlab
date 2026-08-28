"""Watch the scripted expert do the cube-into-box task in the interactive viewer.

    uv run python -m urgantry_bc.demo_viewer            # loops episodes forever
    uv run python -m urgantry_bc.demo_viewer --seed 3 --episodes 1

Close the window to stop. Use --policy to watch a trained BC checkpoint instead.
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from urgantry_sim.build_urgantry import BOARD_TOP
from urgantry_bc.expert import ScriptedExpert
from urgantry_bc.task_env import CubeInBoxEnv

VIEW = {"azimuth": 118.0, "elevation": -22.0, "distance": 1.9,
        "lookat": [0.0, -0.06, BOARD_TOP + 0.12]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=0, help="0 = loop until closed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy", type=str, default=None, help="BC checkpoint to run instead")
    p.add_argument("--flow", type=str, default=None, help="flow/DINOv3 checkpoint")
    p.add_argument("--speed", type=float, default=1.0, help="playback rate")
    args = p.parse_args()

    actor = None
    if args.flow:
        from urgantry_bc.load_flow import load_flow_policy
        actor = load_flow_policy(args.flow)

    env = CubeInBoxEnv(image_size=actor.encoder.image_size if args.flow else 96,
                       max_episode_steps=300)
    if args.flow:
        pass
    elif args.policy:
        from urgantry_bc.policy import load_policy
        actor = load_policy(args.policy)
    else:
        actor = ScriptedExpert(env)

    dt = env.n_substeps * env.model.opt.timestep / max(args.speed, 1e-3)
    with mujoco.viewer.launch_passive(env.model, env.data,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        cam = viewer.cam
        cam.azimuth, cam.elevation, cam.distance = (
            VIEW["azimuth"], VIEW["elevation"], VIEW["distance"])
        cam.lookat[:] = VIEW["lookat"]

        ep = 0
        while viewer.is_running() and (args.episodes == 0 or ep < args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            if hasattr(actor, "reset"):
                actor.reset()
            viewer.sync()
            success, steps = 0, 0
            while viewer.is_running():
                t0 = time.time()
                action = (actor.act(env) if isinstance(actor, ScriptedExpert)
                          else actor(obs))
                obs, _, term, trunc, info = env.step(action)
                success = max(success, info["success"])
                steps += 1
                viewer.sync()
                sleep = dt - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)
                if term or trunc:
                    break
            print(f"episode {ep}: steps={steps} success={success} "
                  f"cube={np.round(env.cube_pos(), 3)}", flush=True)
            ep += 1
            time.sleep(0.4)
    env.close()


if __name__ == "__main__":
    main()
