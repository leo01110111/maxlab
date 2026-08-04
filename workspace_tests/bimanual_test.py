"""Bimanual waypoint test: can both arms get to a waypoint, cleanly, and stay there.

One verdict per waypoint per layout. No single-arm scoring -- a waypoint only counts
if BOTH arms serve it at once, each on its own side of the point (offset along the
base-to-base axis by `separation`).

A waypoint PASSES when all of the following hold:

  reach     both arms solve IK to the waypoint, tool pointing down
  clean     zero contacts involving either arm at ANY point of the approach, not
            merely at the end -- an arm that swipes the table on the way in fails
  stable    both arms settle (max |q̇| < 0.01 rad/s) and park within 20 mm of goal,
            and stay there for a further second with no contacts

The loose task props (block, cardboard tray) are removed from the scene, so the only
things an arm can hit are the fixed structure, the table, and the other arm.

Physics runs with the implicitfast integrator. With the scenes' default Euler
integrator, the stiff position servos (kp 2000 / kv 400) chatter at the timestep
frequency -- joint angles flip +-0.0008 rad every step with no net motion, which
reads as 0.39 rad/s of velocity and makes a parked arm look unstable. implicitfast
removes it; the poses are unchanged.

    python bimanual_test.py                 # all scenes, table + json
    python bimanual_test.py --scene urtable
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import mujoco

from workspace_benchmark import load_benchmark
from workspace_probe import (
    POS_TOL, ROT_TOL, SCENES, build, reset_pose, solve_ik_restarts, split_targets, tool_quat,
)

HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "bimanual_results.json"

APPROACH_S = 4.0          # time allowed to travel to the waypoint and settle
HOLD_S = 1.0              # must stay parked and contact-free for this long after
SETTLE_VEL = 0.01         # rad/s
HOLD_TOL = 0.02           # meters; how close to the goal counts as holding it


def geom_owner(model, arms) -> np.ndarray:
    """Per-geom arm index (-1 for anything that is not part of an arm). Precomputed
    so the per-step contact check is array work rather than set lookups -- at 450
    waypoints x ~2500 steps the naive version dominates the run."""
    owner = np.full(model.ngeom, -1, np.int8)
    for i, arm in enumerate(arms):
        for g in range(model.ngeom):
            if model.geom_bodyid[g] in arm.bodies:
                owner[g] = i
    return owner


def touching(data, owner) -> bool:
    """True if any arm touches something that is not the same arm. Gripper-internal
    linkage contacts sit inside one arm and are excluded."""
    n = data.ncon
    if n == 0:
        return False
    a1 = owner[data.contact.geom1[:n]]
    a2 = owner[data.contact.geom2[:n]]
    return bool(np.any((a1 != a2) & ((a1 >= 0) | (a2 >= 0))))


def arm_contacts(model, data, arms) -> list[str]:
    """Names of geom pairs where an arm touches something that is not itself. Slow
    path, called only once a contact is known to exist, to label it."""
    def label(geom):
        # arm geoms are mostly unnamed in the menagerie mesh set; fall back to the body
        return model.geom(geom).name or model.body(model.geom_bodyid[geom]).name

    hits = []
    for c in data.contact[: data.ncon]:
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        for arm in arms:
            if (b1 in arm.bodies) != (b2 in arm.bodies):
                pair = sorted((label(c.geom1), label(c.geom2)),
                              key=lambda n: not n.startswith(("left_", "right_")))
                hits.append("|".join(pair))
                break
    return hits


def test_waypoint(mod, model, data, arms, wp, rng, owner) -> dict:
    arm_list = list(arms.values())
    pos = np.array(wp["pos"])
    goals = split_targets(arm_list, data, pos, wp["separation"])

    mujoco.mj_resetData(model, data)
    reset_pose(mod, model, data)
    mujoco.mj_forward(model, data)

    scratch = mujoco.MjData(model)
    for arm in arm_list:
        scratch.qpos[:] = data.qpos
        q, pe, re = solve_ik_restarts(model, scratch, arm, goals[arm.name], tool_quat(0.0),
                                      data.qpos[arm.qadr].copy(), rng)
        if pe > POS_TOL or re > ROT_TOL:
            return {"verdict": "no IK", "arm": arm.name, "ik_mm": pe * 1000}
        data.ctrl[arm.act_ids] = q

    hit = None
    for _ in range(int(APPROACH_S / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if hit is None and touching(data, owner):
            hit = arm_contacts(model, data, arm_list)[0]
    if hit:
        return {"verdict": "collision", "with": hit}

    vel = max(float(np.max(np.abs(data.qvel[a.dofs]))) for a in arm_list)
    err = max(float(np.linalg.norm(data.site_xpos[a.site] - goals[a.name])) for a in arm_list)
    if vel >= SETTLE_VEL:
        return {"verdict": "unstable", "vel": vel, "err_mm": err * 1000}
    if err >= HOLD_TOL:
        return {"verdict": "droop", "vel": vel, "err_mm": err * 1000}

    for _ in range(int(HOLD_S / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if touching(data, owner):
            return {"verdict": "collision", "with": arm_contacts(model, data, arm_list)[0]}
    err = max(float(np.linalg.norm(data.site_xpos[a.site] - goals[a.name])) for a in arm_list)
    vel = max(float(np.max(np.abs(data.qvel[a.dofs]))) for a in arm_list)
    if err >= HOLD_TOL or vel >= SETTLE_VEL:
        return {"verdict": "drifted", "vel": vel, "err_mm": err * 1000}
    return {"verdict": "PASS", "vel": vel, "err_mm": err * 1000}


def run_scene(scene: str) -> dict:
    mod, model, data, arms = build(scene, props=False)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    rng = np.random.default_rng(0)
    owner = geom_owner(model, list(arms.values()))
    return {wp["name"]: test_waypoint(mod, model, data, arms, wp, rng, owner)
            for wp in load_benchmark()}


def print_table(results: dict[str, dict], waypoints: list[dict]) -> None:
    scenes = list(results)
    w = max(14, max(len(s) for s in scenes) + 2)
    header = f"{'waypoint':<22}" + "".join(f"{s:>{w}}" for s in scenes)
    print("\nBimanual waypoint test -- both arms, props removed, implicitfast integrator")
    print("PASS = both arms reach it, touch nothing on the way, and hold it parked\n")
    print(header)
    print("-" * len(header))
    for wp in waypoints:
        line = f"{wp['name']:<22}"
        for s in scenes:
            line += f"{results[s][wp['name']]['verdict']:>{w}}"
        print(line)
    print("-" * len(header))
    print(f"{'PASS':<22}" + "".join(
        f"{sum(1 for r in results[s].values() if r['verdict'] == 'PASS'):>{w}}" for s in scenes))
    print(f"{'   of':<22}" + "".join(f"{len(waypoints):>{w}}" for _ in scenes))
    print()
    for s in scenes:
        why = Counter(r["verdict"] for r in results[s].values() if r["verdict"] != "PASS")
        print(f"  {s:<12} failures: " + (", ".join(f"{k} {v}" for k, v in why.most_common())
                                         or "none"))
    for s in scenes:
        hits = Counter(r["with"] for r in results[s].values() if r["verdict"] == "collision")
        if hits:
            print(f"  {s:<12} hits: " + ", ".join(f"{k} x{v}" for k, v in hits.most_common(4)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", help="one scene; default is every scene")
    args = ap.parse_args()
    scenes = [args.scene] if args.scene else sorted(SCENES)
    results = {s: run_scene(s) for s in scenes}
    print_table(results, load_benchmark())
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nverdicts -> {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
