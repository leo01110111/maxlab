"""Interactive viewer: both gantry arms trace a rectangular grid of waypoints that
lie exactly in the board plane (z = BOARD_TOP), i.e. the TCP touches the surface
rather than hovering above it like the 5/20/35 cm layers in the other tests.

Default playback is kinematic (qpos set directly, mj_forward) so the trace is the
IK solution and nothing else; --physics runs the position servos instead, where
the fingertips press into the board at z = 0 offset.

    uv run python trace_waypoints.py
    uv run python trace_waypoints.py --physics --dwell 0.6
"""

import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer

import build_urgantry as scene
from workspace_probe import (Arm, arm_arm_collisions, solve_ik_restarts, tool_mat,
                             tool_quat, _mat2quat, POS_TOL, ROT_TOL)

MIRROR = np.diag([-1.0, 1.0, 1.0])


def mirror_quat(mat: np.ndarray) -> np.ndarray:
    """The same tool pose reflected across the column plane. M R M keeps det = +1,
    so it is still a rotation, unlike reflecting the columns individually."""
    return _mat2quat(MIRROR @ mat @ MIRROR)

# Rounded so the center column is exactly 0.0 and lands on the alternating arm
# rule rather than on float dust just below zero.
GRID_X = np.round(np.arange(-0.45, 0.4501, 0.15), 3)
GRID_OUT = np.array([0.15, 0.25, 0.35, 0.45])   # distance out from the mount edge
MOUNT_EDGE_Y = scene.COL_Y + scene.COL_FOOT_HALF

CUP_TILT = 90.0     # approach axis horizontal: jaws wrap a cup standing on the board
CUP_ROLL = 0.0      # the 2F-85 closes along tool y, so tilt alone puts the pads on the sides
CUP_YAW = 180.0     # reach inward from outboard; yaw 0 crosses the arms over the column

TRAVEL_S = 1.2
DWELL_S = 0.35
MARK_R = 0.010
OK_RGBA = [0.15, 0.85, 0.30, 1]
FAIL_RGBA = [0.95, 0.20, 0.20, 1]
TCP_RGBA = [1.0, 0.85, 0.10, 1]

PROPS = ("block", "cardboard_box")


def waypoints(z_offset=0.0, xs=GRID_X):
    """Serpentine over the grid so the arms crawl row to row instead of jumping the
    full width between rows."""
    out = []
    for i, dy in enumerate(GRID_OUT):
        row = xs if i % 2 == 0 else xs[::-1]
        for x in row:
            out.append((float(x), float(MOUNT_EDGE_Y + dy), scene.BOARD_TOP + z_offset))
    return out


def arm_waypoints(z_offset=0.0):
    """Each arm covers its own half of the full-width grid, both halves including
    the center column, so the middle is traced by both arms."""
    return [waypoints(z_offset, GRID_X[GRID_X <= 0]),
            waypoints(z_offset, GRID_X[GRID_X >= 0])]


def build(props: bool):
    spec = scene.build_spec()
    if not props:
        for name in PROPS:
            spec.delete(spec.body(name))
    model = spec.compile()
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)
    return model, data


def home(model, data, arms):
    for arm, pose in zip(arms, (scene.LEFT_HOME_POSE, scene.RIGHT_HOME_POSE)):
        data.qpos[arm.qadr] = pose
        data.ctrl[arm.act_ids] = pose
    for side in ("left_", "right_"):
        data.ctrl[model.actuator(f"{side}grip_fingers_actuator").id] = scene.GRIPPER_OPEN
    mujoco.mj_forward(model, data)


def plan_arm(model, scratch, arm, wps, quat, seed, rng):
    track = []
    for x, y, z in wps:
        pos = np.array([x, y, z])
        q, pe, re = solve_ik_restarts(model, scratch, arm, pos, quat, seed, rng)
        ok = pe < POS_TOL and re < ROT_TOL
        if ok:
            seed = q
        track.append({"pos": pos, "q": q if ok else seed, "ok": ok})
        print(f"{'ok ' if ok else 'MISS'} {arm.name:5s} x={x:+.2f} out={y - MOUNT_EDGE_Y:.2f}")
    return track


def pair(model, data, arms, tracks):
    """Both arms move every step, so their sequences have to be phase-shifted or
    they meet in the shared center column. Pick the shift with no arm-arm contact."""
    n = len(tracks[0])
    best = None
    for shift in range(n):
        hits = 0
        for k in range(n):
            for arm, tr, idx in zip(arms, tracks, (k, (k + shift) % n)):
                data.qpos[arm.qadr] = tr[idx]["q"]
            mujoco.mj_forward(model, data)
            hits += arm_arm_collisions(model, data, arms[0], arms[1])
        if best is None or hits < best[1]:
            best = (shift, hits)
        if hits == 0:
            break
    shift, hits = best
    print(f"phase shift {shift} step(s) between the arms, arm-arm contacts {hits}")
    return [[tracks[0][k], tracks[1][(k + shift) % n]] for k in range(n)]


def add_marks(scn, tracks, tcp=None):
    scn.ngeom = 0
    for track in tracks:
        for st in track:
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([MARK_R] * 3),
                                np.asarray(st["pos"], float), np.eye(3).flatten(),
                                np.array(OK_RGBA if st["ok"] else FAIL_RGBA, float))
            scn.ngeom += 1
    if tcp is not None:
        for pos in tcp:
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([MARK_R * 1.8] * 3),
                                np.asarray(pos, float), np.eye(3).flatten(),
                                np.array(TCP_RGBA, float))
            scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--physics", action="store_true", help="drive the servos instead of setting qpos")
    ap.add_argument("--props", action="store_true", help="keep the block and tray on the board")
    ap.add_argument("--dwell", type=float, default=DWELL_S)
    ap.add_argument("--travel", type=float, default=TRAVEL_S)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tilt", type=float, default=CUP_TILT,
                    help="approach axis from straight down (deg); 90 = cup posture, 0 = top-down")
    ap.add_argument("--roll", type=float, default=CUP_ROLL, help="spin about the approach axis (deg)")
    ap.add_argument("--yaw", type=float, default=CUP_YAW,
                    help="left arm's approach azimuth (deg); 180 reaches inward from outboard")
    ap.add_argument("--z", type=float, default=0.0, dest="z_offset",
                    help="lift the plane off the board (m); 0 is the literal board plane, "
                         "where a horizontal hand penetrates the board")
    args = ap.parse_args()

    model, data = build(args.props)
    arms = [Arm(model, "left_", scene.ARM_JOINTS), Arm(model, "right_", scene.ARM_JOINTS)]
    home(model, data, arms)

    rng = np.random.default_rng(args.seed)
    quats = [tool_quat(args.yaw, args.tilt, args.roll),
             mirror_quat(tool_mat(args.yaw, args.tilt, args.roll))]
    seeds = [np.array(scene.LEFT_HOME_POSE), np.array(scene.RIGHT_HOME_POSE)]
    scratch = mujoco.MjData(model)
    home(model, scratch, arms)
    tracks = [plan_arm(model, scratch, arm, wps, quat, seed, rng)
              for arm, wps, quat, seed in zip(arms, arm_waypoints(args.z_offset), quats, seeds)]
    steps = pair(model, scratch, arms, tracks)

    total = sum(len(t) for t in tracks)
    per_arm = ", ".join(f"{a.name} {sum(s['ok'] for s in t)}/{len(t)}"
                        for a, t in zip(arms, tracks))
    print(f"\n{sum(sum(s['ok'] for s in t) for t in tracks)}/{total} waypoints reached "
          f"({per_arm}) at z = {scene.BOARD_TOP + args.z_offset:.3f}, "
          f"tilt={args.tilt} roll={args.roll} yaw={args.yaw}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        scene.apply_initial_view(viewer)
        add_marks(viewer.user_scn, tracks)
        viewer.sync()

        dt = model.opt.timestep
        q_prev = [data.qpos[a.qadr].copy() for a in arms]
        while viewer.is_running():
            for st in steps:
                goal = [s["q"] for s in st]
                n_travel = max(1, int(args.travel / dt))
                for k in range(n_travel + int(args.dwell / dt)):
                    if not viewer.is_running():
                        return
                    t0 = time.time()
                    s = min(1.0, (k + 1) / n_travel)
                    s = s * s * (3 - 2 * s)
                    for arm, qa, qb in zip(arms, q_prev, goal):
                        q = qa + s * (qb - qa)
                        if args.physics:
                            data.ctrl[arm.act_ids] = q
                        else:
                            data.qpos[arm.qadr] = q
                    if args.physics:
                        mujoco.mj_step(model, data)
                    else:
                        mujoco.mj_forward(model, data)
                    add_marks(viewer.user_scn, tracks,
                              tcp=[data.site_xpos[a.site] for a in arms])
                    viewer.sync()
                    rest = dt - (time.time() - t0)
                    if rest > 0:
                        time.sleep(rest)
                q_prev = [q.copy() for q in goal]


if __name__ == "__main__":
    main()
