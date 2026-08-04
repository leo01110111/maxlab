"""Holding torque at every benchmark waypoint, for every embodiment.

Drives both arms to each waypoint (same IK and same offset-by-separation rule as
workspace_benchmark.py), waits for them to settle, then records the steady-state
actuator torques and sums the magnitudes over all 12 arm joints. The result is a
table of rows = waypoints, columns = embodiments, cell = summed holding torque of
both arms, with the averages compared at the bottom.

The loose props (block, tray) are removed and physics runs with the implicitfast
integrator, so no reading is contaminated by an arm resting on a prop and no waypoint
is dropped for integrator chatter.

Magnitudes are summed rather than signed values: opposing joint torques are both
real load on the hardware and would otherwise cancel. Gravity is on, so this is
dominated by what each layout asks the arms to hold up.

Torque is a stand-in for power draw, so the comparison at the bottom also reports
sum(torque^2). Holding a pose does no mechanical work (velocity is zero) -- what
the motors actually burn is resistive heating, which goes as current^2 and so as
torque^2. sum|torque| says how loaded the joints are; sum(torque^2) is the closer
proxy for watts, and it punishes one badly-loaded joint much harder than several
lightly-loaded ones.

    python workspace_torque.py                    # table + csv for all scenes
    python workspace_torque.py --scene urtable
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import mujoco

from workspace_benchmark import load_benchmark
from workspace_probe import (
    POS_TOL, ROT_TOL, SCENES, arm_arm_collisions, build, reset_pose, solve_ik_restarts,
    split_targets, tool_quat,
)

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "waypoint_torques.csv"

SETTLE_MAX_S = 8.0        # give up waiting for the servos after this
SETTLE_MIN_S = 3.0        # never sample before this, however quiet it looks
# rad/s. 0.01 is far too loose to call a pose settled: the servo is still creeping,
# and the residual dynamic torque lands in the average -- on urtable it manufactured
# 183 N^2m^2 of shoulder_pan load on a base whose pan axis is vertical and cannot
# see gravity at all. At 1e-3 the run agrees with an independent 6 s flat settle.
SETTLE_VEL = 1e-3
AVG_WINDOW_S = 0.25       # steady-state torques are averaged over this window


def hold_torques(model, data, arm_list, goals, solved) -> tuple[dict, dict, bool, float]:
    """Step until both arms stop moving (or time out), then average |actuator force|
    per arm over a short window. Returns per-arm sums, whether it settled, and the
    worst remaining position error."""
    steps = int(round(SETTLE_MAX_S / model.opt.timestep))
    min_steps = int(round(SETTLE_MIN_S / model.opt.timestep))
    settle_check = max(1, int(round(0.05 / model.opt.timestep)))
    for i in range(steps):
        mujoco.mj_step(model, data)
        if i > min_steps and i % settle_check == 0:
            if all(np.max(np.abs(data.qvel[a.dofs])) < SETTLE_VEL for a in arm_list):
                break
    settled = all(np.max(np.abs(data.qvel[a.dofs])) < SETTLE_VEL for a in arm_list)

    window = max(1, int(round(AVG_WINDOW_S / model.opt.timestep)))
    acc = {a.name: np.zeros(len(a.act_ids)) for a in arm_list}
    acc_sq = {a.name: np.zeros(len(a.act_ids)) for a in arm_list}
    for _ in range(window):
        mujoco.mj_step(model, data)
        for a in arm_list:
            tau = data.actuator_force[a.act_ids]
            acc[a.name] += np.abs(tau)
            acc_sq[a.name] += tau**2
    per_arm = {name: v / window for name, v in acc.items()}
    per_arm_sq = {name: v / window for name, v in acc_sq.items()}
    err = max(float(np.linalg.norm(data.site_xpos[a.site] - goals[a.name]))
              for a in arm_list if solved[a.name]) if any(solved.values()) else float("nan")
    return per_arm, per_arm_sq, settled, err


def measure_scene(scene: str) -> dict:
    mod, model, data, arms = build(scene, props=False)
    # Euler + the stiff position servos (kp 2000 / kv 400) chatter at the timestep
    # frequency: joints flip +-0.0008 rad every step with no net motion, which reads
    # as 0.39 rad/s and fails the settle gate, silently dropping good waypoints.
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    arm_list = list(arms.values())
    waypoints = load_benchmark()
    scratch = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    quat = tool_quat(0.0)
    out = {}

    for wp in waypoints:
        pos = np.array(wp["pos"])
        goals = split_targets(arm_list, data, pos, wp["separation"])
        mujoco.mj_resetData(model, data)
        reset_pose(mod, model, data)
        mujoco.mj_forward(model, data)
        solved = {}
        for arm in arm_list:
            scratch.qpos[:] = data.qpos
            q, pe, re = solve_ik_restarts(model, scratch, arm, goals[arm.name], quat,
                                          data.qpos[arm.qadr].copy(), rng)
            solved[arm.name] = pe < POS_TOL and re < ROT_TOL
            if solved[arm.name]:
                data.ctrl[arm.act_ids] = q

        out.setdefault("_joints", mod.ARM_JOINTS)
        per_arm, per_arm_sq, settled, err = hold_torques(model, data, arm_list, goals, solved)
        ok = all(solved.values()) and settled and err < 0.02
        out[wp["name"]] = {
            "total": float(sum(v.sum() for v in per_arm.values())),
            "total_sq": float(sum(v.sum() for v in per_arm_sq.values())),
            "per_arm": {k: float(v.sum()) for k, v in per_arm.items()},
            "per_joint": {k: v.round(2).tolist() for k, v in per_arm.items()},
            "per_joint_sq": {k: v.round(2).tolist() for k, v in per_arm_sq.items()},
            "ok": ok,
            "err": err,
            "contacts": arm_arm_collisions(model, data, *arm_list),
        }
    return out


def _short(joint: str) -> str:
    return {"shoulder_pan": "pan", "shoulder_lift": "lift", "elbow": "elbow",
            "wrist_1": "w1", "wrist_2": "w2", "wrist_3": "w3"}.get(joint, joint[:6])


def print_table(results: dict[str, dict], waypoints: list[dict]) -> None:
    scenes = list(results)
    width = max(14, max(len(s) for s in scenes) + 2)
    header = f"{'waypoint':<22}" + "".join(f"{s:>{width}}" for s in scenes)
    print("\nSummed holding torque of both arms (N m, |torque| over all 12 arm joints)")
    print("'--' = the waypoint is not reachable by both arms in that layout\n")
    print(header)
    print("-" * len(header))
    for wp in waypoints:
        name = wp["name"]
        line = f"{name:<22}"
        for s in scenes:
            r = results[s].get(name)
            line += f"{r['total']:>{width}.1f}" if r and r["ok"] else f"{'--':>{width}}"
        print(line)

    print("-" * len(header))
    common = [wp["name"] for wp in waypoints
              if all(results[s][wp["name"]]["ok"] for s in scenes)]
    row_own = f"{'mean (own reachable)':<22}"
    row_n = f"{'   n waypoints':<22}"
    row_common = f"{'mean (common set)':<22}"
    for s in scenes:
        vals = [r["total"] for n, r in results[s].items() if n != "_joints" and r["ok"]]
        row_own += f"{np.mean(vals) if vals else float('nan'):>{width}.1f}"
        row_n += f"{len(vals):>{width}}"
        cvals = [results[s][n]["total"] for n in common]
        row_common += f"{np.mean(cvals) if cvals else float('nan'):>{width}.1f}"
    print(row_own)
    print(row_n)
    print(row_common)
    print(f"{'   n common':<22}" + "".join(f"{len(common):>{width}}" for _ in scenes))

    row_sq = f"{'mean sum(tau^2), common':<22}"
    for s in scenes:
        svals = [results[s][n]["total_sq"] for n in common]
        row_sq += f"{np.mean(svals) if svals else float('nan'):>{width}.0f}"
    print(row_sq)

    if common:
        def mean_of(scene, key):
            return float(np.mean([results[scene][n][key] for n in common]))

        for key, label, unit in (("total", "sum|tau| (joint load)", "N m"),
                                 ("total_sq", "sum(tau^2) (power proxy)", "N^2m^2")):
            best = min(scenes, key=lambda s: mean_of(s, key))
            base = mean_of(best, key)
            print(f"\n{label} over the {len(common)} waypoints every layout can hold: "
                  f"{best} lowest at {base:.1f} {unit}")
            print("   " + ",  ".join(f"{s} {mean_of(s, key) / base:.2f}x"
                                     for s in scenes if s != best))
    if not common:
        return
    print("\nper-arm split on the common set (N m):")
    for s in scenes:
        arms = sorted(results[s][common[0]]["per_arm"])
        means = {a: np.mean([results[s][n]["per_arm"][a] for n in common]) for a in arms}
        print(f"  {s:<14}" + "  ".join(f"{a} {means[a]:6.1f}" for a in arms))
    print("\nwhere the load sits -- mean |torque| per joint, left arm, common set (N m):")
    for s in scenes:
        joints = results[s]["_joints"]
        per = np.mean([results[s][n]["per_joint"]["left"] for n in common], axis=0)
        worst = joints[int(np.argmax(per))]
        print(f"  {s:<14}" + " ".join(f"{_short(j):>7}" for j in joints)
              + f"    worst: {worst}")
        print(f"  {'':<14}" + " ".join(f"{v:>7.1f}" for v in per))

    print("\nsame thing squared -- mean torque^2 per joint, left arm, common set (N^2 m^2).")
    print("This is the power-proxy column: it is what sum(tau^2) in the table above is made of.")
    for s_ in scenes:
        joints = results[s_]["_joints"]
        per = np.mean([results[s_][n]["per_joint_sq"]["left"] for n in common], axis=0)
        both = np.mean([results[s_][n]["total_sq"] for n in common])
        print(f"  {s_:<14}" + " ".join(f"{_short(j):>8}" for j in joints)
              + f"{'left sum':>11}{'both arms':>11}")
        print(f"  {'':<14}" + " ".join(f"{v:>8.1f}" for v in per)
              + f"{per.sum():>11.1f}{both:>11.1f}")


def write_csv(results: dict[str, dict], waypoints: list[dict]) -> None:
    scenes = list(results)
    with CSV_PATH.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["waypoint", "x", "y", "z", "layer"] +
                   [f"{s}_{col}" for s in scenes
                    for col in ("sum_abs_tau", "sum_tau_sq", "left", "right",
                                "ok", "max_err_m", "contacts")])
        for wp in waypoints:
            row = [wp["name"], *wp["pos"], wp["layer"]]
            for s in scenes:
                r = results[s][wp["name"]]
                row += [round(r["total"], 2), round(r["total_sq"], 1),
                        round(r["per_arm"].get("left", float("nan")), 2),
                        round(r["per_arm"].get("right", float("nan")), 2),
                        int(r["ok"]), round(r["err"], 4), r["contacts"]]
            w.writerow(row)
    print(f"\nper-joint detail and raw values -> {CSV_PATH.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", help="one scene; default is every scene")
    args = ap.parse_args()
    scenes = [args.scene] if args.scene else sorted(SCENES)
    waypoints = load_benchmark()
    results = {s: measure_scene(s) for s in scenes}
    print_table(results, waypoints)
    write_csv(results, waypoints)


if __name__ == "__main__":
    main()
