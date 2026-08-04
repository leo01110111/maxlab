"""Interactive workspace probe for the bimanual UR7e scenes.

Opens a scene (build_tabletop_full / build_tabletop_half) in the MuJoCo
viewer next to a small Tk panel with X/Y/Z/yaw sliders. The sliders drive a
floating target marker over the table; the selected arm(s) track it with a
damped-least-squares IK solve, so you can see live whether a spot is reachable.
When a spot looks right, "Save point" appends it to workspace_points.json (keyed
by scene). "Play sequence" then drives the arms through every saved point in
turn, and "Report" prints a per-point / per-arm reachability table.

In "both" mode the two arms would fight for the same volume, so their goals are
pulled apart by the "separation" slider along the base-to-base axis -- each arm
takes the side its own base is on (orange markers). The separation is saved with
the point, and both the status line and the report show the arm-vs-arm contact
count so you can tell when it is still too small.

Dexterity: the yaw/tilt/roll sliders aim the gripper's approach axis anywhere on
the full sphere (tilt 0 = straight down, 180 = straight up) about a pinned pinch
point. "Sweep approach" then walks that axis along a spherical spiral covering
every direction, warm-starting each IK solve from the previous one, so it measures
what the arm can reach by *continuously* reorienting rather than by teleporting to
another IK branch. Results draw as a sphere of dots around the point -- green =
clean, yellow/orange = increasingly near-singular, red = no IK or a wrist flip,
blue = collision -- and "replay sweep" drives the arm through the clean arc.

    python workspace_probe.py --scene full
    python workspace_probe.py --scene half --report      # headless, no GUI
    python workspace_probe.py --scene full --sweep       # report + orientation sweeps

Coordinates are the scene frame from gantry_core.py: origin on the floor at the
center of the exposed tabletop, +x right along the mount edge, +y away from the
mount, +z up, meters.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

HERE = Path(__file__).resolve().parent
POINTS_PATH = HERE / "workspace_points.json"

SCENES = {
    "full": "build_tabletop_full.py",       # 117 cm mount edge x 112 cm deep
    "half": "build_tabletop_half.py",       # 112 cm mount edge x 58.5 cm deep
}

POS_TOL = 0.005          # meters; counts as "reached" in the report
ROT_TOL = np.deg2rad(5.0)
IK_RESTARTS = 8          # random seeds tried before declaring a pose unreachable
# Meters between the two arms' targets in "both" mode. Both scenes here use the
# same elevated gantry, which is clean well below 0.35; the value is kept from the
# workspace_tests benchmark so numbers stay comparable with it.
DEFAULT_SEPARATION = 0.35


def scene_path(name: str) -> Path:
    """Accepts a key of SCENES, a module name (`build_urgantry_hands`), or a path,
    so a new embodiment can be scored without editing this file."""
    for candidate in (SCENES.get(name), name, f"{name}.py", f"build_{name}.py"):
        if candidate and (HERE / candidate).exists():
            return HERE / candidate
    if Path(name).exists():
        return Path(name)
    raise SystemExit(f"no scene file for '{name}'; known: {', '.join(sorted(SCENES))}, "
                     f"or pass a path to a build_*.py")


def load_scene(name: str):
    path = scene_path(name)
    spec = importlib.util.spec_from_file_location(f"_scene_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mat2quat(mat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(mat, float).flatten())
    return quat


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix about a unit axis (Rodrigues)."""
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


DOWN_FRAME = np.column_stack([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])   # tool z = -world z


def tool_mat(yaw_deg: float, tilt_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """Target orientation for the gripper's 'pinch' site, as a rotation matrix.
    tilt = 0 points the approach axis (site +z) straight down, tilt = 180 straight
    up; `yaw` swings the tilt direction about world z; `roll` spins the gripper
    about its own approach axis."""
    z, y = np.array([0.0, 0, 1.0]), np.array([0.0, 1.0, 0])
    mat = _rot(z, np.deg2rad(yaw_deg)) @ _rot(y, np.deg2rad(tilt_deg)) @ DOWN_FRAME
    return mat @ _rot(z, np.deg2rad(roll_deg))


def tool_quat(yaw_deg: float, tilt_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    return _mat2quat(tool_mat(yaw_deg, tilt_deg, roll_deg))


def point_quat(p: dict):
    """Target orientation of a saved point, or None if it was saved free."""
    if p.get("free_orientation"):
        return None
    return tool_quat(p.get("yaw", 0.0), p.get("tilt", 0.0), p.get("roll", 0.0))


class Arm:
    def __init__(self, model: mujoco.MjModel, prefix: str, joint_names: list[str]):
        self.prefix = prefix
        self.name = prefix.rstrip("_")
        self.joint_ids = [model.joint(f"{prefix}{j}_joint").id for j in joint_names]
        self.qadr = np.array([model.jnt_qposadr[i] for i in self.joint_ids])
        self.dofs = np.array([model.jnt_dofadr[i] for i in self.joint_ids])
        self.act_ids = np.array([model.actuator(f"{prefix}{j}").id for j in joint_names])
        self.site = self._tcp_site(model, prefix)
        self.range = model.jnt_range[self.joint_ids].copy()
        self.bodies = {b for b in range(model.nbody) if model.body(b).name.startswith(prefix)}
        self.base_body = model.jnt_bodyid[self.joint_ids[0]]

    @staticmethod
    def _tcp_site(model: mujoco.MjModel, prefix: str) -> int:
        """The point everything is measured about. 'pinch' is the Robotiq grasp
        center; other end effectors are picked up by name so new hands work without
        changes here."""
        names = [model.site(i).name for i in range(model.nsite)]
        for suffix in ("grip_pinch", "pinch", "tcp", "grasp_site", "attachment_site"):
            if f"{prefix}{suffix}" in names:
                return model.site(f"{prefix}{suffix}").id
        raise SystemExit(f"no TCP site for arm '{prefix}': expected one of "
                         f"{prefix}pinch / {prefix}tcp / {prefix}grasp_site. "
                         f"Add a site with one of those names to the end effector.")

    def base_xy(self, data: mujoco.MjData) -> np.ndarray:
        return data.xpos[self.base_body][:2].copy()

    def random_q(self, rng: np.random.Generator) -> np.ndarray:
        lo, hi = self.range[:, 0], self.range[:, 1]
        return rng.uniform(np.maximum(lo, -np.pi), np.minimum(hi, np.pi))


def solve_ik(model, data, arm: Arm, pos, quat, q_init,
             iters: int = 200, damping: float = 0.05, max_step: float = 0.3):
    """Damped-least-squares IK on one arm's 6 joints, mutating `data` (use a
    scratch MjData). `quat=None` solves position only. Returns (q, pos_err, rot_err)."""
    data.qpos[arm.qadr] = q_init
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    err = np.zeros(6)
    for _ in range(iters):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        err[:3] = pos - data.site_xpos[arm.site]
        if quat is None:
            err[3:] = 0.0
        else:
            site_quat = np.zeros(4)
            mujoco.mju_mat2Quat(site_quat, data.site_xmat[arm.site])
            neg = np.zeros(4)
            mujoco.mju_negQuat(neg, site_quat)
            dq = np.zeros(4)
            mujoco.mju_mulQuat(dq, quat, neg)
            mujoco.mju_quat2Vel(err[3:], dq, 1.0)
        pos_err = float(np.linalg.norm(err[:3]))
        rot_err = float(np.linalg.norm(err[3:]))
        if pos_err < 1e-4 and rot_err < 1e-3:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, arm.site)
        rows = jacp[:, arm.dofs] if quat is None else np.vstack([jacp[:, arm.dofs], jacr[:, arm.dofs]])
        e = err[:3] if quat is None else err
        n = rows.shape[0]
        step = rows.T @ np.linalg.solve(rows @ rows.T + damping**2 * np.eye(n), e)
        step = np.clip(step, -max_step, max_step)
        q = np.clip(data.qpos[arm.qadr] + step, arm.range[:, 0], arm.range[:, 1])
        data.qpos[arm.qadr] = q
    return data.qpos[arm.qadr].copy(), pos_err, rot_err


def solve_ik_restarts(model, data, arm, pos, quat, q_seed, rng, restarts=IK_RESTARTS):
    """IK from the current pose first, then random seeds until one converges.
    Keeps the best attempt so near-misses are still reported honestly."""
    best = None
    for i in range(restarts + 1):
        q_init = q_seed if i == 0 else arm.random_q(rng)
        q, pe, re = solve_ik(model, data, arm, pos, quat, q_init)
        if best is None or (pe, re) < (best[1], best[2]):
            best = (q, pe, re)
        if pe < POS_TOL and (quat is None or re < ROT_TOL):
            break
    return best


def rotation_cost(model, data, arm: Arm) -> float:
    """Joint speed needed per unit tool angular speed while the pinch point is held
    fixed: 1 / smallest singular value of the rotation Jacobian restricted to the
    joint motions that don't move the site. A cost of 4 means the worst-axis joint
    has to run 4x faster than the tool spins; UR joints top out near pi rad/s, so
    <3 is comfortable, 3-10 sluggish, >10 is a singularity in all but name."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, arm.site)
    jp, jr = jacp[:, arm.dofs], jacr[:, arm.dofs]
    _, s, vt = np.linalg.svd(jp)
    null = vt[np.sum(s > 1e-6):].T          # joint directions that hold the point still
    if null.shape[1] < 3:
        return np.inf
    sigma = np.linalg.svd(jr @ null, compute_uv=False)
    return np.inf if sigma[-1] < 1e-9 else float(1.0 / sigma[-1])


def spiral_dirs(steps: int, turns: float) -> np.ndarray:
    """Unit approach directions along a spherical spiral: straight down at step 0,
    winding through every azimuth, straight up at the last step. Adjacent
    directions are a few degrees apart, so IK can be warm-started along it."""
    t = np.linspace(0.0, 1.0, steps)
    z = -np.cos(np.pi * t)
    r = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    phi = 2 * np.pi * turns * t
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def transport(mat: np.ndarray, old_dir: np.ndarray, new_dir: np.ndarray) -> np.ndarray:
    """Swing `mat` so its z axis follows old_dir -> new_dir by the minimal rotation.
    Parallel transport adds no roll of its own, so a wrist flip along the sweep is
    the arm's doing and not an artifact of how the targets were generated."""
    axis = np.cross(old_dir, new_dir)
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return mat
    return _rot(axis / norm, np.arctan2(norm, float(np.dot(old_dir, new_dir)))) @ mat


SWEEP_STEPS = 600
SWEEP_TURNS = 10.0
JUMP_TOL = 0.30          # rad; a bigger step between adjacent solves is a branch change
COST_OK, COST_MARGINAL = 3.0, 10.0


def orientation_sweep(model, data, arm: Arm, pos, mats, q_start) -> list[dict]:
    """Walk the tool through `mats` with the pinch point pinned at `pos`, warm-starting
    each IK solve from the previous one -- so this measures the orientations reachable
    by *continuously* reorienting, not by jumping to another IK branch. `data` is
    mutated (pass scratch); the other arm keeps whatever pose it already has, so its
    geometry counts for collisions."""
    out = []
    q = np.asarray(q_start, float).copy()
    for mat in mats:
        q_new, pe, re = solve_ik(model, data, arm, pos, _mat2quat(mat), q, iters=60)
        jump = float(np.max(np.abs(q_new - q)))
        data.qpos[arm.qadr] = q_new
        mujoco.mj_forward(model, data)
        coll = env_collisions(model, data, arm)
        cost = rotation_cost(model, data, arm)
        out.append({
            "dir": mat[:, 2].copy(),
            "q": q_new,
            "pos_err": pe,
            "rot_err": re,
            "jump": jump,
            "coll": coll,
            "cost": cost,
            "margin": joint_margin(arm, q_new),
            "ok": pe < POS_TOL and re < ROT_TOL and jump < JUMP_TOL and coll == 0,
        })
        q = q_new
    return out


def approach_mats(yaw_deg: float = 0.0, roll_deg: float = 0.0,
                  steps: int = SWEEP_STEPS, turns: float = SWEEP_TURNS) -> list[np.ndarray]:
    """Target orientations along the spherical spiral, each parallel-transported from
    the last so the roll never jumps."""
    dirs = spiral_dirs(steps, turns)
    mat = tool_mat(yaw_deg, 0.0, roll_deg)
    mats, prev = [], dirs[0]
    for d in dirs:
        mat = transport(mat, prev, d)
        prev = d
        mats.append(mat)
    return mats


def run_sweep(model, data, arm: Arm, pos, mats, q_seed, rng):
    """Solve the first orientation properly (restarts allowed), then sweep from it
    continuously. Returns ([], start_err) if the sweep can't even start."""
    q0, pe, re = solve_ik_restarts(model, data, arm, pos, _mat2quat(mats[0]),
                                   q_seed, rng, restarts=3)
    if pe > POS_TOL or re > ROT_TOL:
        return [], pe
    return orientation_sweep(model, data, arm, pos, mats, q0), pe


def continuous_run(recs: list[dict]) -> int:
    """Number of leading steps reachable without a break -- the part of the sweep the
    arm can actually get to by reorienting from the start pose."""
    n = 0
    while n < len(recs) and recs[n]["ok"]:
        n += 1
    return n


def break_reason(recs: list[dict], run: int) -> str:
    if not recs:
        return "start unreachable"
    if run >= len(recs):
        return "no break"
    r = recs[run]
    return ("no IK" if r["pos_err"] > POS_TOL or r["rot_err"] > ROT_TOL else
            "wrist flip" if r["jump"] >= JUMP_TOL else
            "collision")


def continuous_tilt(recs: list[dict]) -> float:
    """Degrees the approach axis can be tilted off straight-down, in any azimuth,
    before the sweep breaks. The spiral interleaves azimuths as tilt grows, so the
    run ends at the first azimuth that fails -- a conservative, all-azimuth cone."""
    run = continuous_run(recs)
    if run == 0:
        return 0.0
    return float(np.rad2deg(np.arccos(np.clip(-recs[run - 1]["dir"][2], -1, 1))))


def sweep_summary(recs: list[dict], label: str) -> str:
    """One-line reading of a sweep: how far it got before the first break, why it
    broke, and how hard the arm has to work over the part that does work."""
    if not recs:
        return f"{label}: start orientation unreachable"
    n = len(recs)
    run = continuous_run(recs)
    clean = sum(r["ok"] for r in recs)
    costs = [r["cost"] for r in recs[:run] if np.isfinite(r["cost"])]
    worst = max(costs) if costs else float("inf")
    why = break_reason(recs, run) + (f" ({recs[run]['coll']})"
                                     if run < n and break_reason(recs, run) == "collision" else "")
    # The spiral's tilt grows monotonically, so where the continuous run ends is
    # "how far the approach can be tilted off straight-down without a wrist flip".
    tilt = np.rad2deg(np.arccos(np.clip(-recs[max(run - 1, 0)]["dir"][2], -1, 1)))
    moved = run > 0 and abs(recs[-1]["dir"][2] - recs[0]["dir"][2]) > 1e-3
    return (f"{label}: continuous {run}/{n} ({100 * run / n:.0f}%)"
            + (f" up to tilt {tilt:.0f}deg" if moved else "")
            + f"   reachable overall {100 * clean / n:.0f}%   break: {why}   "
            + (f"worst joint-speed cost {worst:.1f}x" if np.isfinite(worst) else "no clean arc"))


def sweep_color(rec: dict) -> list[float]:
    if not rec["ok"]:
        return [0.85, 0.15, 0.15, 0.9] if rec["coll"] == 0 else [0.35, 0.35, 0.95, 0.9]
    if rec["cost"] < COST_OK:
        return [0.15, 0.85, 0.25, 0.9]
    if rec["cost"] < COST_MARGINAL:
        return [0.95, 0.85, 0.1, 0.9]
    return [0.95, 0.5, 0.05, 0.9]


def split_targets(arms: list[Arm], data, pos, separation: float) -> dict[str, np.ndarray]:
    """Per-arm goal positions. With one arm it is just `pos`; with two, the goals
    straddle `pos` by `separation` along the horizontal base-to-base axis, each arm
    taking the side its own base is on, so the two arms approach from opposite
    sides instead of fighting for the same volume."""
    pos = np.asarray(pos, float)
    if len(arms) < 2 or separation <= 0:
        return {a.name: pos.copy() for a in arms}
    near, far = arms[0], arms[1]
    axis = far.base_xy(data) - near.base_xy(data)
    norm = np.linalg.norm(axis)
    axis = axis / norm if norm > 1e-6 else np.array([1.0, 0.0])
    offset = np.array([axis[0], axis[1], 0.0]) * (separation / 2)
    return {near.name: pos - offset, far.name: pos + offset}


def arm_arm_collisions(model, data, a: Arm, b: Arm) -> int:
    n = 0
    for c in data.contact[: data.ncon]:
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if (b1 in a.bodies and b2 in b.bodies) or (b1 in b.bodies and b2 in a.bodies):
            n += 1
    return n


def env_collisions(model, data, arm: Arm) -> int:
    """Contacts between this arm and anything that is not part of it (table,
    objects, walls, the other arm). Self-contacts inside the gripper linkage are
    always present, so they are excluded."""
    n = 0
    for c in data.contact[: data.ncon]:
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if (b1 in arm.bodies) != (b2 in arm.bodies):
            n += 1
    return n


def joint_margin(arm: Arm, q: np.ndarray) -> float:
    """Smallest distance (rad) from any joint to its limit."""
    return float(np.min(np.minimum(q - arm.range[:, 0], arm.range[:, 1] - q)))


# ------------------------------------------------------------------ points file

def load_points(scene: str) -> list[dict]:
    if not POINTS_PATH.exists():
        return []
    return json.loads(POINTS_PATH.read_text()).get(scene, [])


def save_points(scene: str, points: list[dict]) -> None:
    all_points = json.loads(POINTS_PATH.read_text()) if POINTS_PATH.exists() else {}
    all_points[scene] = points
    POINTS_PATH.write_text(json.dumps(all_points, indent=2) + "\n")


# ------------------------------------------------------------------ scene setup

PROP_BODIES = ("block", "cardboard_box")


def reset_pose(mod, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Home pose for both arms, grippers open. Does the arm/gripper part of the
    scene's own set_initial_pose without touching prop joints, which may not exist
    (see build(props=False))."""
    for prefix, home in (("left_", mod.LEFT_HOME_POSE), ("right_", mod.RIGHT_HOME_POSE)):
        for joint, angle in zip(mod.ARM_JOINTS, home):
            data.qpos[model.joint(f"{prefix}{joint}_joint").qposadr[0]] = angle
            data.ctrl[model.actuator(f"{prefix}{joint}").id] = angle
        data.ctrl[model.actuator(f"{prefix}grip_fingers_actuator").id] = mod.GRIPPER_OPEN


def build(scene_name: str, props: bool = True):
    """`props=False` deletes the loose task objects (block, cardboard tray) from the
    spec, so a reachability test measures the arms against the fixed structure only
    and doesn't score a movable prop as an obstacle."""
    mod = load_scene(scene_name)
    spec = mod.build_spec()
    if not props:
        for name in PROP_BODIES:
            body = spec.body(name)
            if body is not None:
                spec.delete(body)
    marker = spec.worldbody.add_body(name="ik_target", mocap=True, pos=[0, 0, mod.BOARD_TOP + 0.2])
    marker.add_geom(name="ik_target", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.012] * 3,
                    rgba=[1.0, 0.2, 0.8, 0.6], contype=0, conaffinity=0, mass=0)
    model = spec.compile()
    data = mujoco.MjData(model)
    (mod.set_initial_pose if props else lambda m, d: reset_pose(mod, m, d))(model, data)
    mujoco.mj_forward(model, data)
    arms = {p.rstrip("_"): Arm(model, p, mod.ARM_JOINTS) for p in ("left_", "right_")}
    return mod, model, data, arms


def report(scene_name: str, with_sweep: bool = False) -> None:
    mod, model, data, arms = build(scene_name)
    points = load_points(scene_name)
    if not points:
        print(f"no saved points for scene '{scene_name}' in {POINTS_PATH}")
        return
    scratch = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    print(f"\nscene: {scene_name}   points: {len(points)}   "
          f"(reach = pos err < {POS_TOL * 1000:.0f} mm, rot err < {np.rad2deg(ROT_TOL):.0f} deg)\n")
    header = (f"{'point':<14}{'x':>7}{'y':>7}{'z':>7}{'arm':>7}"
              f"{'pos mm':>9}{'rot deg':>9}{'lim rad':>9}{'coll':>6}  reach")
    print(header)
    print("-" * len(header))
    reach_count = {name: 0 for name in arms}
    for p in points:
        pos = np.array(p["pos"])
        quat = point_quat(p)
        arm_list = list(arms.values())
        both = p.get("arm") == "both"
        goals = split_targets(arm_list, data, pos,
                              p.get("separation", DEFAULT_SEPARATION) if both else 0.0)
        solved, colls, cross = {}, {}, 0
        if both:
            # Both arms placed at once, so the collision counts include arm-vs-arm.
            mujoco.mj_resetData(model, scratch)
            mod.set_initial_pose(model, scratch)
            for name, arm in arms.items():
                q, pe, re = solve_ik_restarts(model, scratch, arm, goals[name], quat,
                                              scratch.qpos[arm.qadr].copy(), rng)
                scratch.qpos[arm.qadr] = q
                solved[name] = (q, pe, re)
            mujoco.mj_forward(model, scratch)
            colls = {name: env_collisions(model, scratch, arm) for name, arm in arms.items()}
            cross = arm_arm_collisions(model, scratch, *arm_list)
        else:
            for name, arm in arms.items():
                mujoco.mj_resetData(model, scratch)
                mod.set_initial_pose(model, scratch)
                q, pe, re = solve_ik_restarts(model, scratch, arm, goals[name], quat,
                                              scratch.qpos[arm.qadr].copy(), rng)
                scratch.qpos[arm.qadr] = q
                mujoco.mj_forward(model, scratch)
                solved[name] = (q, pe, re)
                colls[name] = env_collisions(model, scratch, arm)
        for name, arm in arms.items():
            q, pe, re = solved[name]
            goal = goals[name]
            coll = colls[name]
            ok = pe < POS_TOL and (quat is None or re < ROT_TOL) and coll == 0
            reach_count[name] += ok
            print(f"{p['name']:<14}{goal[0]:>7.3f}{goal[1]:>7.3f}{goal[2]:>7.3f}{name:>7}"
                  f"{pe * 1000:>9.1f}{np.rad2deg(re):>9.1f}{joint_margin(arm, q):>9.2f}{coll:>6}"
                  f"  {'yes' if ok else 'NO'}")
        if both:
            sep = p.get("separation", DEFAULT_SEPARATION)
            print(f"{'':<14}both arms, separation {sep:.3f} m -> "
                  f"arm-arm contacts: {cross}"
                  f"{'' if cross == 0 else '   <-- increase separation'}")
        if with_sweep:
            for name, arm in arms.items():
                q, pe, _ = solved[name]
                if pe > POS_TOL:
                    continue
                mujoco.mj_resetData(model, scratch)
                mod.set_initial_pose(model, scratch)
                scratch.qpos[arm.qadr] = q
                recs, _ = run_sweep(model, scratch, arm, goals[name],
                                    approach_mats(p.get("yaw", 0.0), p.get("roll", 0.0)), q, rng)
                print(f"{'':<14}{sweep_summary(recs, name + ' approach sweep')}")
    print()
    for name in arms:
        print(f"{name:>7}: {reach_count[name]}/{len(points)} points reachable")
    print()


# ------------------------------------------------------------------ interactive

class Probe:
    def __init__(self, scene_name: str):
        import tkinter as tk

        self.tk = tk
        self.scene_name = scene_name
        self.mod, self.model, self.data, self.arms = build(scene_name)
        self.scratch = mujoco.MjData(self.model)
        self.rng = np.random.default_rng()
        self.points = load_points(scene_name)
        self.mocap = self.model.body("ik_target").mocapid[0]
        self.runner = None
        self.steps_per_tick = max(1, int(0.02 / self.model.opt.timestep))

        board = self.mod.BOARD_TOP
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data,
                                                   show_left_ui=False, show_right_ui=False)
        self.mod.apply_initial_view(self.viewer)

        self.root = tk.Tk()
        self.root.title(f"workspace probe — {scene_name}")
        self.x = tk.DoubleVar(value=0.0)
        self.y = tk.DoubleVar(value=0.0)
        self.z = tk.DoubleVar(value=board + 0.10)
        self.yaw = tk.DoubleVar(value=0.0)
        self.tilt = tk.DoubleVar(value=0.0)
        self.roll = tk.DoubleVar(value=0.0)
        self.sep = tk.DoubleVar(value=DEFAULT_SEPARATION)
        self.sweep_dots: list[tuple[np.ndarray, list[float], float]] = []
        self.sweep_recs: list[dict] = []
        self.arm_choice = tk.StringVar(value="nearest")
        self.live = tk.BooleanVar(value=True)
        self.free_orient = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="")
        self.point_name = tk.StringVar(value="")
        self._build_panel(board)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_panel(self, board):
        tk = self.tk
        hl = self.mod.HALF_LEN
        for label, var, lo, hi, res in (
            ("x  (left/right)", self.x, -hl, hl, 0.005),
            ("y  (out from the mount)", self.y, self.mod.EXPOSED_Y0, self.mod.EXPOSED_Y1, 0.005),
            ("z  (height)", self.z, board, board + 0.60, 0.005),
            ("yaw  (deg)", self.yaw, -180.0, 180.0, 1.0),
            ("tilt  (deg from straight down)", self.tilt, 0.0, 180.0, 1.0),
            ("roll  (deg about approach axis)", self.roll, -180.0, 180.0, 1.0),
            ("both: separation (m)", self.sep, 0.0, 0.60, 0.005),
        ):
            tk.Scale(self.root, variable=var, from_=lo, to=hi, resolution=res,
                     orient=tk.HORIZONTAL, length=380, label=label).pack(fill=tk.X, padx=6)

        row = tk.Frame(self.root)
        row.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(row, text="arm:").pack(side=tk.LEFT)
        for text in ("left", "right", "both", "nearest"):
            tk.Radiobutton(row, text=text, value=text, variable=self.arm_choice).pack(side=tk.LEFT)

        row = tk.Frame(self.root)
        row.pack(fill=tk.X, padx=6)
        tk.Checkbutton(row, text="live IK", variable=self.live).pack(side=tk.LEFT)
        tk.Checkbutton(row, text="free orientation", variable=self.free_orient).pack(side=tk.LEFT)
        tk.Button(row, text="solve once", command=self.solve_now).pack(side=tk.LEFT, padx=4)
        tk.Button(row, text="home", command=self.go_home).pack(side=tk.LEFT)

        row = tk.Frame(self.root)
        row.pack(fill=tk.X, padx=6, pady=4)
        tk.Entry(row, textvariable=self.point_name, width=14).pack(side=tk.LEFT)
        tk.Button(row, text="save point", command=self.save_point).pack(side=tk.LEFT, padx=4)
        tk.Button(row, text="delete", command=self.delete_point).pack(side=tk.LEFT)

        self.listbox = tk.Listbox(self.root, height=8)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=6)
        self.listbox.bind("<<ListboxSelect>>", self.load_selected)
        self.refresh_list()

        row = tk.Frame(self.root)
        row.pack(fill=tk.X, padx=6, pady=4)
        tk.Button(row, text="play sequence", command=self.play).pack(side=tk.LEFT)
        tk.Button(row, text="stop", command=self.stop).pack(side=tk.LEFT, padx=4)
        tk.Button(row, text="report",
                  command=lambda: report(self.scene_name)).pack(side=tk.LEFT)

        row = tk.Frame(self.root)
        row.pack(fill=tk.X, padx=6, pady=4)
        tk.Label(row, text="dexterity:").pack(side=tk.LEFT)
        tk.Button(row, text="sweep approach", command=self.sweep_approach).pack(side=tk.LEFT, padx=4)
        tk.Button(row, text="sweep roll", command=self.sweep_roll).pack(side=tk.LEFT)
        tk.Button(row, text="replay sweep", command=self.replay_sweep).pack(side=tk.LEFT, padx=4)
        tk.Button(row, text="clear dots", command=self.clear_sweep).pack(side=tk.LEFT)

        tk.Label(self.root, textvariable=self.status, anchor="w", justify="left",
                 font=("TkFixedFont", 9)).pack(fill=tk.X, padx=6, pady=4)

    # -- helpers -----------------------------------------------------------
    def target_pos(self) -> np.ndarray:
        return np.array([self.x.get(), self.y.get(), self.z.get()])

    def target_quat(self):
        if self.free_orient.get():
            return None
        return tool_quat(self.yaw.get(), self.tilt.get(), self.roll.get())

    def selected_arms(self, pos) -> list[Arm]:
        choice = self.arm_choice.get()
        if choice == "both":
            return list(self.arms.values())
        if choice == "nearest":
            return [min(self.arms.values(),
                        key=lambda a: np.linalg.norm(a.base_xy(self.data) - pos[:2]))]
        return [self.arms[choice]]

    def goals_for(self, arms, pos, separation=None) -> dict[str, np.ndarray]:
        sep = self.sep.get() if separation is None else separation
        return split_targets(arms, self.data, pos, sep)

    def command(self, arms, goals, quat) -> str:
        """IK in scratch data, then command the position actuators. Returns a status line."""
        bits = []
        for arm in arms:
            goal = goals[arm.name]
            self.scratch.qpos[:] = self.data.qpos
            q, pe, re = solve_ik(self.model, self.scratch, arm, goal, quat,
                                 self.data.qpos[arm.qadr].copy())
            if pe > POS_TOL:
                q, pe, re = solve_ik_restarts(self.model, self.scratch, arm, goal, quat,
                                              q, self.rng, restarts=3)
            self.data.ctrl[arm.act_ids] = q
            bits.append(f"{arm.name}: {pe * 1000:5.1f}mm {np.rad2deg(re):5.1f}deg "
                        f"lim {joint_margin(arm, q):.2f}")
        if len(arms) > 1:
            bits.append(f"arm-arm contacts {arm_arm_collisions(self.model, self.data, *arms)}")
        return "   ".join(bits)

    def tracking_error(self, arms, goals) -> str:
        return "   ".join(
            f"{a.name} actual "
            f"{np.linalg.norm(self.data.site_xpos[a.site] - goals[a.name]) * 1000:5.1f}mm"
            for a in arms)

    def refresh_list(self):
        self.listbox.delete(0, self.tk.END)
        for p in self.points:
            x, y, z = p["pos"]
            self.listbox.insert(self.tk.END,
                                f"{p['name']:<12} ({x:+.3f}, {y:+.3f}, {z:.3f})  "
                                f"yaw {p.get('yaw', 0):.0f}  {p.get('arm', 'nearest')}")

    # -- buttons -----------------------------------------------------------
    def save_point(self):
        name = self.point_name.get().strip() or f"p{len(self.points) + 1}"
        pos = self.target_pos()
        self.points.append({
            "name": name,
            "pos": [round(float(v), 4) for v in pos],
            "yaw": round(float(self.yaw.get()), 1),
            "tilt": round(float(self.tilt.get()), 1),
            "roll": round(float(self.roll.get()), 1),
            "arm": self.arm_choice.get(),
            "free_orientation": bool(self.free_orient.get()),
            "separation": round(float(self.sep.get()), 3),
        })
        save_points(self.scene_name, self.points)
        self.point_name.set("")
        self.refresh_list()
        self.status.set(f"saved '{name}' -> {POINTS_PATH.name}")

    def delete_point(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        removed = self.points.pop(sel[0])
        save_points(self.scene_name, self.points)
        self.refresh_list()
        self.status.set(f"deleted '{removed['name']}'")

    def load_selected(self, _event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        p = self.points[sel[0]]
        self.x.set(p["pos"][0])
        self.y.set(p["pos"][1])
        self.z.set(p["pos"][2])
        self.yaw.set(p.get("yaw", 0.0))
        self.tilt.set(p.get("tilt", 0.0))
        self.roll.set(p.get("roll", 0.0))
        self.arm_choice.set(p.get("arm", "nearest"))
        self.free_orient.set(bool(p.get("free_orientation")))
        self.sep.set(p.get("separation", DEFAULT_SEPARATION))

    def solve_now(self):
        pos = self.target_pos()
        arms = self.selected_arms(pos)
        self.status.set(self.command(arms, self.goals_for(arms, pos), self.target_quat()))

    def go_home(self):
        self.runner = None
        self.mod.set_initial_pose(self.model, self.data)
        self.data.qvel[:] = 0.0

    def stop(self):
        self.runner = None
        self.status.set("stopped")

    def play(self):
        if not self.points:
            self.status.set("no saved points")
            return
        self.runner = self._sequence()

    # -- dexterity sweeps --------------------------------------------------
    def sweep_arm(self) -> Arm:
        pos = self.target_pos()
        arms = self.selected_arms(pos)
        return arms[0] if len(arms) == 1 else min(
            arms, key=lambda a: np.linalg.norm(a.base_xy(self.data) - pos[:2]))

    def _run_sweep(self, mats, label: str):
        """Kinematic continuous sweep on the currently selected arm, drawn as dots on
        a sphere around the point and kept for `replay sweep`."""
        arm = self.sweep_arm()
        raw = self.target_pos()
        pos = self.goals_for(self.selected_arms(raw), raw)[arm.name]
        self.scratch.qpos[:] = self.data.qpos
        recs, pe = run_sweep(self.model, self.scratch, arm, pos, mats,
                             self.data.qpos[arm.qadr].copy(), self.rng)
        if not recs:
            self.status.set(f"{arm.name}: start orientation not reachable ({pe * 1000:.0f}mm)")
            return
        self.sweep_dots = [(pos + 0.09 * r["dir"], sweep_color(r), 0.007) for r in recs]
        run = continuous_run(recs)
        for i in range(run):
            self.sweep_dots[i] = (self.sweep_dots[i][0], self.sweep_dots[i][1], 0.012)
        self.sweep_recs = recs[:run]
        line = sweep_summary(recs, f"{arm.name} {label}")
        print(line)
        self.status.set(line)

    def sweep_approach(self):
        """Full sphere of approach directions, straight down -> straight up."""
        self._run_sweep(approach_mats(self.yaw.get(), self.roll.get()), "approach sweep")

    def sweep_roll(self):
        """360 deg spin about the current approach axis, tool position pinned."""
        mats = [tool_mat(self.yaw.get(), self.tilt.get(), self.roll.get() + a)
                for a in np.linspace(0.0, 360.0, 181)]
        self._run_sweep(mats, "roll sweep")

    def replay_sweep(self):
        if not getattr(self, "sweep_recs", None):
            self.status.set("no sweep to replay -- run a sweep first")
            return
        self.runner = self._replay()

    def _replay(self):
        arm = self.sweep_arm()
        for i, rec in enumerate(self.sweep_recs):
            self.data.ctrl[arm.act_ids] = rec["q"]
            self.status.set(f"replay {i + 1}/{len(self.sweep_recs)}   "
                            f"tilt {np.rad2deg(np.arccos(-rec['dir'][2])):5.1f}deg   "
                            f"cost {rec['cost']:5.1f}x   margin {rec['margin']:.2f}")
            yield
        self.runner = None

    def clear_sweep(self):
        self.sweep_dots = []
        self.sweep_recs = []

    def _sequence(self):
        hold = int(2.0 / (self.steps_per_tick * self.model.opt.timestep))
        for p in self.points:
            pos = np.array(p["pos"])
            quat = point_quat(p)
            self.x.set(pos[0]), self.y.set(pos[1]), self.z.set(pos[2])
            self.yaw.set(p.get("yaw", 0.0))
            self.tilt.set(p.get("tilt", 0.0))
            self.roll.set(p.get("roll", 0.0))
            self.sep.set(p.get("separation", DEFAULT_SEPARATION))
            self.arm_choice.set(p.get("arm", "nearest"))
            arms = self.selected_arms(pos)
            goals = self.goals_for(arms, pos)
            ik = self.command(arms, goals, quat)
            for _ in range(hold):
                self.status.set(f"[{p['name']}] {ik}")
                yield
            print(f"{p['name']:<14} {ik}   {self.tracking_error(arms, goals)}")
            self.status.set(f"[{p['name']}] {ik}   {self.tracking_error(arms, goals)}")
        self.runner = None

    # -- loop --------------------------------------------------------------
    def _add_marker(self, pos, rgba, size=0.01):
        scn = self.viewer.user_scn
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([size, 0, 0]), np.asarray(pos, float),
                            np.eye(3).flatten(), np.array(rgba, np.float32))
        scn.ngeom += 1

    def draw_markers(self, goals):
        self.viewer.user_scn.ngeom = 0
        for pos, rgba, size in self.sweep_dots:
            self._add_marker(pos, rgba, size)
        for p in self.points:
            self._add_marker(p["pos"], [0.1, 0.9, 0.4, 0.8])
        if len(goals) > 1:
            for goal in goals.values():
                self._add_marker(goal, [1.0, 0.6, 0.1, 0.9], size=0.012)

    def tick(self):
        if not self.viewer.is_running():
            self.close()
            return
        start = time.perf_counter()
        pos = self.target_pos()
        self.data.mocap_pos[self.mocap] = pos
        goals = self.goals_for(self.selected_arms(pos), pos)
        if self.runner is not None:
            try:
                next(self.runner)
            except StopIteration:
                self.runner = None
        elif self.live.get():
            arms = self.selected_arms(pos)
            goals = self.goals_for(arms, pos)
            self.status.set(f"{self.command(arms, goals, self.target_quat())}   "
                            f"{self.tracking_error(arms, goals)}")
        for _ in range(self.steps_per_tick):
            mujoco.mj_step(self.model, self.data)
        self.draw_markers(goals)
        self.viewer.sync()
        delay = self.steps_per_tick * self.model.opt.timestep - (time.perf_counter() - start)
        self.root.after(max(1, int(delay * 1000)), self.tick)

    def run(self):
        self.tick()
        self.root.mainloop()
        # Tearing down the GLFW viewer and Tk in the same process segfaults on
        # interpreter exit; both windows are already gone by here, so bail out
        # before the GL context is finalized.
        import os
        os._exit(0)

    def close(self):
        self.viewer.close()
        self.root.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="full",
                    help="a key of SCENES, a build_*.py module name, or a path")
    ap.add_argument("--report", action="store_true", help="print the reachability table and exit")
    ap.add_argument("--sweep", action="store_true",
                    help="with --report, also run the continuous orientation sweep per point")
    args = ap.parse_args()
    if args.report or args.sweep:
        report(args.scene, with_sweep=args.sweep)
    else:
        Probe(args.scene).run()


if __name__ == "__main__":
    main()
