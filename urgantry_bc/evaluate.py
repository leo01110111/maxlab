"""Roll out a trained BC policy (or the scripted expert) and report success rate.

    uv run python -m urgantry_bc.evaluate --policy urgantry_bc/runs/bc/policy.pt -n 50
    uv run python -m urgantry_bc.evaluate --expert -n 50

Evaluation seeds are disjoint from the collection seeds, so the cube starts at
positions the policy never trained on.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from urgantry_bc.task_env import CubeInBoxEnv

EVAL_SEED0 = 90_000          # collection uses seeds from --seed0 (default 1000)


def evaluate(actor, env, n: int, seed0: int = EVAL_SEED0, verbose: bool = True) -> dict:
    successes, lifts, dists = 0, 0, []
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        if hasattr(actor, "reset"):
            actor.reset()
        success, peak = 0, 0.0
        for _ in range(env.max_episode_steps):
            action = actor.act(env) if hasattr(actor, "act") else actor(obs)
            obs, _, term, trunc, info = env.step(action)
            success = max(success, info["success"])
            peak = max(peak, info["cube_height"])
            if term or trunc:
                break
        successes += success
        lifts += int(peak > 0.05)
        dists.append(float(np.linalg.norm((env.cube_pos() - env.box_pos())[:2])))
        if verbose:
            print(f"  ep {i:3d} success={success} peak_lift={peak:.3f} "
                  f"final_xy_to_box={dists[-1]:.3f}", flush=True)
    return {"episodes": n, "success_rate": successes / n, "lift_rate": lifts / n,
            "median_final_xy_to_box": float(np.median(dists))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", default=None)
    p.add_argument("--flow", default=None, help="flow-matching checkpoint (DINOv3)")
    p.add_argument("--expert", action="store_true")
    p.add_argument("-n", "--episodes", type=int, default=50)
    p.add_argument("--image-size", type=int, default=84)
    p.add_argument("--execute", type=int, default=None,
                   help="actions played per predicted chunk")
    p.add_argument("--seed0", type=int, default=EVAL_SEED0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    image_size = args.image_size
    actor = None
    if args.flow:
        # The env must render at whatever resolution the policy was trained on.
        from urgantry_bc.load_flow import load_flow_policy
        actor = load_flow_policy(args.flow, execute=args.execute)
        image_size = actor.encoder.image_size

    env = CubeInBoxEnv(image_size=image_size, max_episode_steps=300)
    if args.expert:
        from urgantry_bc.expert import ScriptedExpert
        actor = ScriptedExpert(env)
    elif args.flow:
        pass
    else:
        from urgantry_bc.policy import load_policy
        actor = load_policy(args.policy, execute=args.execute)

    res = evaluate(actor, env, args.episodes, args.seed0, verbose=not args.quiet)
    res["actor"] = "expert" if args.expert else (args.flow or args.policy)
    print(json.dumps(res, indent=2))
    env.close()


if __name__ == "__main__":
    main()
