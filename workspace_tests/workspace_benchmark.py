"""A fixed set of waypoints over the shared work volume in front of the arms, plus
a scorer that runs it against any embodiment.

The waypoints are deliberately scene-independent: the same grid of positions in
table coordinates is scored against build_urtable / build_urtable_45 /
build_urgantry (or any future layout), so the numbers compare directly. What is
scored at each waypoint is not just "can a gripper get there" but "can a gripper
work there":

  reach   - IK to the waypoint with the tool pointing straight down, no collisions
  tilt    - degrees the approach axis can be tilted off straight-down, in *any*
            azimuth, by continuously reorienting about the pinned pinch point
  roll    - degrees of continuous spin about the approach axis at the layer's
            nominal tilt, taken as the worst of four tilt azimuths
  both    - both arms at the waypoint simultaneously (offset by `separation`),
            no arm-vs-arm contact, each still meeting the bimanual tilt floor

A waypoint passes for an arm when reach, tilt and roll all clear the thresholds in
REQUIREMENTS. Edit those to match the task you actually care about -- they are the
specification, and the rest of the file just measures against them.

    python workspace_benchmark.py --write             # (re)generate the json
    python workspace_benchmark.py --scene urtable     # score one embodiment
    python workspace_benchmark.py --all               # score all three, compare
    python workspace_benchmark.py --scene urtable --install   # load into the probe GUI

Positions are absolute in the table frame used by build_urtable.py: origin at the
center of the table footprint on the floor, +x right, +y back, +z up, meters. The
board top is 0.775 in every scene, so absolute z transfers between them.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import mujoco

from workspace_probe import (
    DEFAULT_SEPARATION, POS_TOL, ROT_TOL, SCENES,
    _mat2quat, approach_mats, arm_arm_collisions, break_reason, build, continuous_run,
    continuous_tilt, env_collisions, orientation_sweep, run_sweep, save_points,
    solve_ik_restarts, split_targets, tool_mat, tool_quat,
)

HERE = Path(__file__).resolve().parent
BENCHMARK_PATH = HERE / "workspace_benchmark.json"

BOARD_TOP = 0.775

# --------------------------------------------------------------- the work volume
# A rectangle on the board centered between the arms -- the region both arms are
# expected to share. 70 cm across the table length, 45 cm deep, sampled on a
# uniform 5 cm grid (15 x 10) at three working heights: 450 waypoints.
GRID_X = tuple(round(-0.35 + 0.05 * i, 3) for i in range(15))
GRID_Y = tuple(round(-0.225 + 0.05 * i, 3) for i in range(10))
LAYERS = {
    "low": 0.05,      # near-surface: picking, placing, pushing on the board
    "mid": 0.20,      # in-air manipulation, handovers
    "high": 0.35,     # over-the-top reaches, working above an object
}

# --------------------------------------------------------------- the requirements
# What "dexterous enough" means, per layer. tilt_min is the all-azimuth cone the
# approach axis must sweep continuously; roll_min is the continuous spin about the
# approach axis at roll_tilt. Calibrated against what a UR7e + 2F-85 actually
# manages on these scenes -- measured cones run ~30-50 deg median and top out near
# 80 deg, so a 45 deg bar separates good layouts from bad ones instead of failing
# everything. Raise them if your task needs more; that is the point of the file.
REQUIREMENTS = {
    "low": {"tilt_min": 45.0, "roll_min": 270.0, "roll_tilt": 0.0},
    "mid": {"tilt_min": 45.0, "roll_min": 270.0, "roll_tilt": 45.0},
    "high": {"tilt_min": 45.0, "roll_min": 270.0, "roll_tilt": 45.0},
}
# Tilt cone each arm must keep with the other arm parked in the same volume. Lower
# than the single-arm bar because the second arm eats part of every cone.
BIMANUAL_TILT_MIN = 30.0

ROLL_AZIMUTHS = (0.0, 90.0, 180.0, 270.0)
ROLL_HALF_STEPS = 46            # per direction; roll is swept both ways from the seed
SWEEP_STEPS_FULL, SWEEP_TURNS_FULL = 600, 10.0
SWEEP_STEPS_QUICK, SWEEP_TURNS_QUICK = 240, 6.0
# Which IK branch the start solve lands on caps how far a sweep gets, so each cone
# is measured from a few different start solutions and the best one is kept -- that
# is what a planner free to choose its approach posture would get.
TILT_TRIES = 3


def make_waypoints() -> list[dict]:
    """The grid, named by layer and position so a name identifies a location."""
    out = []
    for layer, dz in LAYERS.items():
        req = REQUIREMENTS[layer]
        for y in GRID_Y:
            for x in GRID_X:
                out.append({
                    "name": f"{layer}_x{int(x * 1000):+04d}_y{int(y * 1000):+04d}",
                    "pos": [round(x, 4), round(y, 4), round(BOARD_TOP + dz, 4)],
                    "layer": layer,
                    "separation": DEFAULT_SEPARATION,
                    **req,
                })
    return out


def write_benchmark() -> list[dict]:
    waypoints = make_waypoints()
    BENCHMARK_PATH.write_text(json.dumps({
        "frame": ("table frame of build_urtable.py: origin at the center of the table "
                  "footprint on the floor, +x right, +y back, +z up, meters; z absolute "
                  f"(board top = {BOARD_TOP})"),
        "grid": {"x": list(GRID_X), "y": list(GRID_Y), "layers": LAYERS},
        "requirements": REQUIREMENTS,
        "bimanual_tilt_min": BIMANUAL_TILT_MIN,
        "waypoints": waypoints,
    }, indent=2) + "\n")
    return waypoints


def load_benchmark() -> list[dict]:
    if not BENCHMARK_PATH.exists():
        return write_benchmark()
    return json.loads(BENCHMARK_PATH.read_text())["waypoints"]


# ------------------------------------------------------------------ measurement

def tilt_cone(model, scratch, arm, pos, rng, q_seed, quick: bool) -> tuple[float, str]:
    steps, turns = ((SWEEP_STEPS_QUICK, SWEEP_TURNS_QUICK) if quick else
                    (SWEEP_STEPS_FULL, SWEEP_TURNS_FULL))
    best, why = 0.0, "start unreachable"
    for i in range(TILT_TRIES):
        seed = q_seed if i == 0 else arm.random_q(rng)
        recs, _ = run_sweep(model, scratch, arm, pos,
                            approach_mats(steps=steps, turns=turns), seed, rng)
        if recs and continuous_tilt(recs) >= best:
            best, why = continuous_tilt(recs), break_reason(recs, continuous_run(recs))
    return best, why


def roll_range(model, scratch, arm, pos, rng, roll_tilt: float, q_seed) -> float:
    """Continuous spin about the approach axis, swept both ways from the start pose
    and summed -- sweeping one way only would just measure how close the IK seed
    happened to leave wrist_3 to its limit. Averaged over four tilt azimuths (an
    azimuth the arm can't even hold the pose in scores 0), so the number degrades
    with the number of directions that work instead of collapsing to 0."""
    totals = []
    for azimuth in ROLL_AZIMUTHS:
        start = tool_mat(azimuth, roll_tilt, 0.0)
        q0, pe, re = solve_ik_restarts(model, scratch, arm, pos, _mat2quat(start),
                                       q_seed, rng, restarts=3)
        if pe > POS_TOL or re > ROT_TOL:
            totals.append(0.0)
            continue
        total = 0.0
        for sign in (1.0, -1.0):
            mats = [tool_mat(azimuth, roll_tilt, sign * a)
                    for a in np.linspace(0.0, 180.0, ROLL_HALF_STEPS)]
            recs = orientation_sweep(model, scratch, arm, pos, mats, q0)
            total += 180.0 * continuous_run(recs) / ROLL_HALF_STEPS
        totals.append(total)
    return float(np.mean(totals))


def measure(model, scratch, arm, pos, rng, wp, quick: bool) -> dict:
    """Reach + continuous tilt + continuous roll for one arm at one waypoint. The
    scene state already in `scratch` is kept, so whatever pose the other arm is in
    counts for collisions."""
    q_seed = scratch.qpos[arm.qadr].copy()
    q, pe, re = solve_ik_restarts(model, scratch, arm, pos, tool_quat(0.0), q_seed, rng)
    scratch.qpos[arm.qadr] = q
    mujoco.mj_forward(model, scratch)
    coll = env_collisions(model, scratch, arm)
    reach = pe < POS_TOL and re < ROT_TOL and coll == 0
    if not reach:
        return {"reach": False, "tilt": 0.0, "roll": 0.0, "why": "no reach", "q": q}

    tilt, why = tilt_cone(model, scratch, arm, pos, rng, q, quick)
    roll = roll_range(model, scratch, arm, pos, rng, wp["roll_tilt"], q)
    scratch.qpos[arm.qadr] = q
    mujoco.mj_forward(model, scratch)
    return {"reach": True, "tilt": tilt, "roll": roll, "why": why, "q": q}


def passes(m: dict, wp: dict) -> bool:
    return m["reach"] and m["tilt"] >= wp["tilt_min"] and m["roll"] >= wp["roll_min"]


def score_scene(scene: str, quick: bool = False, verbose: bool = True) -> dict:
    mod, model, data, arms = build(scene)
    arm_list = list(arms.values())
    waypoints = load_benchmark()
    scratch = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    rows, why_counts = [], Counter()

    for wp in waypoints:
        pos = np.array(wp["pos"])
        row = {"wp": wp, "arms": {}}
        for name, arm in arms.items():
            mujoco.mj_resetData(model, scratch)
            mod.set_initial_pose(model, scratch)
            m = measure(model, scratch, arm, pos, rng, wp, quick)
            m["pass"] = passes(m, wp)
            row["arms"][name] = m
            if not m["pass"]:
                why_counts[m["why"] if m["reach"] else "no reach"] += 1

        # Bimanual: both arms in the volume at once, each on its own side.
        goals = split_targets(arm_list, data, pos, wp["separation"])
        mujoco.mj_resetData(model, scratch)
        mod.set_initial_pose(model, scratch)
        placed = {}
        for name, arm in arms.items():
            q, pe, re = solve_ik_restarts(model, scratch, arm, goals[name], tool_quat(0.0),
                                          scratch.qpos[arm.qadr].copy(), rng)
            scratch.qpos[arm.qadr] = q
            placed[name] = (q, pe, re)
        mujoco.mj_forward(model, scratch)
        cross = arm_arm_collisions(model, scratch, *arm_list)
        both_tilt = {}
        if not all(pe < POS_TOL and re < ROT_TOL for _, pe, re in placed.values()):
            both, both_why = False, "ik"
        elif cross:
            both, both_why = False, "coll"
        else:
            for name, arm in arms.items():
                recs, _ = run_sweep(model, scratch, arm, goals[name],
                                    approach_mats(steps=SWEEP_STEPS_QUICK,
                                                  turns=SWEEP_TURNS_QUICK),
                                    placed[name][0], rng)
                both_tilt[name] = continuous_tilt(recs)
                scratch.qpos[arm.qadr] = placed[name][0]
                mujoco.mj_forward(model, scratch)
            both = all(t >= BIMANUAL_TILT_MIN for t in both_tilt.values())
            both_why = "yes" if both else "tilt"
        row["both"] = both
        row["both_why"] = both_why
        row["both_tilt"] = both_tilt
        row["cross"] = cross
        rows.append(row)

    _print_scene(scene, rows, why_counts, table=verbose)
    return {"scene": scene, "rows": rows, "why": why_counts}


def _print_scene(scene: str, rows: list[dict], why_counts: Counter, table: bool = True) -> None:
    print(f"\n=== {scene} ===")
    if table:
        header = (f"{'waypoint':<22}{'left':>26}{'right':>26}{'both':>7}")
        print(header)
        print(f"{'':<22}{'reach tilt roll pass':>26}{'reach tilt roll pass':>26}")
        print("-" * len(header))
        for row in rows:
            line = f"{row['wp']['name']:<22}"
            for name in ("left", "right"):
                m = row["arms"][name]
                line += (f"{'y' if m['reach'] else 'n':>7}"
                         f"{m['tilt']:>6.0f}{m['roll']:>6.0f}"
                         f"{'  PASS' if m['pass'] else '  ----':>7}")
            line += f"{row['both_why']:>7}"
            print(line)
        print()
    layers = sorted({r["wp"]["layer"] for r in rows}, key=lambda l: list(LAYERS).index(l))
    for layer in layers:
        sub = [r for r in rows if r["wp"]["layer"] == layer]
        req = REQUIREMENTS[layer]
        counts = {name: sum(r["arms"][name]["pass"] for r in sub) for name in ("left", "right")}
        either = sum(any(r["arms"][n]["pass"] for n in ("left", "right")) for r in sub)
        both = sum(r["both"] for r in sub)
        print(f"  {layer:<5} (tilt>={req['tilt_min']:.0f} roll>={req['roll_min']:.0f} "
              f"@tilt {req['roll_tilt']:.0f})  n={len(sub):<3} "
              f"left {counts['left']:>2}  right {counts['right']:>2}  "
              f"either {either:>2}  bimanual {both:>2}")
    n = len(rows)
    either = sum(any(r["arms"][x]["pass"] for x in ("left", "right")) for r in rows)
    print(f"  {'TOTAL':<5} {'':<34} n={n:<3} "
          f"left {sum(r['arms']['left']['pass'] for r in rows):>2}  "
          f"right {sum(r['arms']['right']['pass'] for r in rows):>2}  "
          f"either {either:>2}  bimanual {sum(r['both'] for r in rows):>2}")
    if why_counts:
        print("  limiting factor: " +
              ", ".join(f"{k} {v}" for k, v in why_counts.most_common()))


def compare(results: list[dict]) -> None:
    print("\n=== comparison (waypoints passing, out of "
          f"{len(results[0]['rows'])}) ===")
    print(f"{'scene':<14}{'left':>7}{'right':>7}{'either':>8}{'bimanual':>10}"
          f"{'mean tilt':>11}{'mean roll':>11}  limiting factor")
    for res in results:
        rows = res["rows"]
        tilts = [m["tilt"] for r in rows for m in r["arms"].values() if m["reach"]]
        rolls = [m["roll"] for r in rows for m in r["arms"].values() if m["reach"]]
        either = sum(any(r["arms"][x]["pass"] for x in ("left", "right")) for r in rows)
        top = res["why"].most_common(1)
        print(f"{res['scene']:<14}"
              f"{sum(r['arms']['left']['pass'] for r in rows):>7}"
              f"{sum(r['arms']['right']['pass'] for r in rows):>7}"
              f"{either:>8}{sum(r['both'] for r in rows):>10}"
              f"{np.mean(tilts) if tilts else 0:>10.0f}deg"
              f"{np.mean(rolls) if rolls else 0:>10.0f}deg"
              f"  {top[0][0] + ' x' + str(top[0][1]) if top else '-'}")
    print()


def install(scene: str) -> None:
    """Copy the benchmark waypoints into workspace_points.json for `scene` so the
    probe GUI can step through them (overwrites that scene's saved points)."""
    points = [{"name": wp["name"], "pos": wp["pos"], "yaw": 0.0, "tilt": 0.0, "roll": 0.0,
               "arm": "both", "free_orientation": False, "separation": wp["separation"]}
              for wp in load_benchmark()]
    save_points(scene, points)
    print(f"installed {len(points)} benchmark waypoints as the '{scene}' points "
          f"(workspace_points.json) -- open the probe and press 'play sequence'")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="urtable",
                    help="a key of SCENES, a build_*.py module name, or a path")
    ap.add_argument("--all", action="store_true", help="score every scene and compare")
    ap.add_argument("--write", action="store_true", help="regenerate the waypoint json")
    ap.add_argument("--quick", action="store_true", help="coarser sweeps, ~2.5x faster")
    ap.add_argument("--install", action="store_true",
                    help="load the waypoints into the probe GUI's points file")
    ap.add_argument("--quiet", action="store_true", help="summary only, no per-waypoint table")
    args = ap.parse_args()

    if args.write:
        print(f"wrote {len(write_benchmark())} waypoints to {BENCHMARK_PATH.name}")
        return
    if args.install:
        install(args.scene)
        return
    scenes = sorted(SCENES) if args.all else [args.scene]
    results = [score_scene(s, quick=args.quick, verbose=not args.quiet) for s in scenes]
    if len(results) > 1:
        compare(results)


if __name__ == "__main__":
    main()
