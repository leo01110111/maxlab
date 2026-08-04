"""Render each embodiment driving the benchmark waypoints, in two modes.

Writes one mp4 per scene plus a side-by-side of both, so the tabletop sizes can be
compared on the same motion (each scene walks its own grid, so the full table's tour
simply keeps going where the half table's has run out of board). Scene setup matches
bimanual_test.py -- props removed, implicitfast integrator.

  shared  both arms serve the SAME waypoint, offset from it by `separation` along
          the base-to-base axis. This is what bimanual_test.py scores, so each frame
          carries that test's verdict.
  split   each arm follows its OWN sampled list of waypoints, independently and
          simultaneously. Nothing coordinates them, so this shows what the two arms
          do when they are not sharing a target -- including running into each other.
          The lists are drawn from each scene's own grid.

The arms run continuously, chasing one waypoint after the next from wherever they
ended up, rather than restarting from the home pose the way the test does. So the
*approach path* differs from the tested one: the live contact counter is what this
run touched, while `verdict:` is the test's from-home result.

    python waypoint_video.py                        # shared mode, all scenes
    python waypoint_video.py --mode split
    python waypoint_video.py --mode split --n 60 --seed 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from workspace_benchmark import load_benchmark
from bimanual_test import RESULTS_PATH, arm_contacts, geom_owner, touching
from workspace_probe import (
    POS_TOL, ROT_TOL, SCENES, build, reset_pose, solve_ik_restarts, split_targets,
    tool_quat,
)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "waypoint_videos"

FPS = 25
DWELL_S = 1.0                    # seconds per waypoint; most of it is travel
WIDTH, HEIGHT = 640, 480
# Both scenes mount their arms on the near (-y) edge, so the camera sits on the far
# (+y) edge looking back across the table at them -- from the near side the arms and
# the gantry column stand between the lens and the work area. The framing is anchored
# to the mount edge and identical for both scenes, so the panels differ only by how
# much tabletop there is.
CAM = {"azimuth": 270.0, "elevation": -26.0, "distance": 2.9, "lookat_ahead": 0.45}
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# Stride between visited waypoints in shared mode, drawn uniformly from this range.
# 1,1 walks the whole grid in serpentine order -- the full table's 330 waypoints at
# 1 s each is a 5.5 minute clip. Widen the range to sample the grid instead, which
# also mixes short nudges with long reaches.
STRIDE_MIN, STRIDE_MAX = 1, 1
SPLIT_N = 40


def _font(size: int):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _overlay(frame: np.ndarray, lines: list[str], font) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((10, 8 + 20 * i), line, font=font,
                  fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(img)


def travel_order(waypoints: list[dict]) -> list[dict]:
    """Serpentine the grid so consecutive waypoints are adjacent: visiting them in
    file order jumps the full width of the table between rows, and the arms spend
    the whole clip in transit instead of at the waypoint."""
    out, row = [], []
    for wp in waypoints:
        if row and (wp["pos"][1] != row[-1]["pos"][1] or wp["layer"] != row[-1]["layer"]):
            out.extend(row if len(out) // max(len(row), 1) % 2 == 0 else row[::-1])
            row = []
        row.append(wp)
    if row:
        out.extend(row if len(out) // max(len(row), 1) % 2 == 0 else row[::-1])
    return out


def random_stride(waypoints: list[dict], lo: int, hi: int, seed: int) -> list[dict]:
    """Walk the serpentine order in random increments. Seeded, so both layouts
    are driven through the identical sequence and the triptych stays frame-aligned."""
    rng = np.random.default_rng(seed)
    out, i = [], 0
    while i < len(waypoints):
        out.append(waypoints[i])
        i += int(rng.integers(lo, hi + 1))
    return out


def sample_lists(waypoints: list[dict], n: int, seed: int) -> dict[str, list[dict]]:
    """One independent waypoint list per arm, drawn from the same grid. Seeded so
    both layouts are driven through identical lists."""
    rng = np.random.default_rng(seed)
    return {name: [waypoints[i] for i in rng.choice(len(waypoints), n, replace=False)]
            for name in ("left", "right")}


def load_verdicts(scene: str) -> dict:
    if not RESULTS_PATH.exists():
        return {}
    return {k: v["verdict"] for k, v in json.loads(RESULTS_PATH.read_text()).get(scene, {}).items()}


def _setup(scene: str):
    mod, model, data, arms = build(scene, props=False)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = CAM["azimuth"], CAM["elevation"]
    camera.distance = CAM["distance"]
    camera.lookat[:] = [0.0, mod.EXPOSED_Y0 + CAM["lookat_ahead"], mod.BOARD_TOP + 0.05]
    return mod, model, data, arms, camera


def _aim(model, scratch, data, arm, goal, quat, rng) -> bool:
    """IK to `goal` and command it. A failed solve returns the best attempt, which can
    be an arbitrary posture; commanding it would look like the arm lunging somewhere
    random, so leave the arm where it is and let the caller label it."""
    scratch.qpos[:] = data.qpos
    q, pe, re = solve_ik_restarts(model, scratch, arm, goal, quat,
                                  data.qpos[arm.qadr].copy(), rng)
    if pe < POS_TOL and re < ROT_TOL:
        data.ctrl[arm.act_ids] = q
        return True
    return False


def render_scene(scene: str, out_path: Path, mode: str, dwell_s: float,
                 stride: tuple[int, int], split_n: int, seed: int,
                 reset: bool = False, limit: int | None = None) -> int:
    mod, model, data, arms, camera = _setup(scene)
    arm_list = list(arms.values())
    owner = geom_owner(model, arm_list)
    verdicts = load_verdicts(scene)
    scratch = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    mocap = model.body("ik_target").mocapid[0]
    quat = tool_quat(0.0)

    grid = load_benchmark(scene)
    if mode == "shared":
        sequence = random_stride(travel_order(grid), stride[0], stride[1], seed)
    else:
        lists = sample_lists(grid, split_n, seed)
        sequence = list(zip(lists["left"], lists["right"]))
    if limit:
        sequence = sequence[:limit]

    steps_per_frame = max(1, int(round((1.0 / FPS) / model.opt.timestep)))
    frames_per_wp = max(1, int(round(dwell_s * FPS)))
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    writer = imageio.get_writer(out_path, fps=FPS, macro_block_size=1)
    font = _font(15)
    n = 0

    for idx, item in enumerate(sequence):
        if reset:
            # Matches bimanual_test.py / randomized_reaching.py, which score every point
            # from the home pose. Without it the video shows an approach the test never
            # ran, and a collision verdict can disagree with what is on screen.
            mujoco.mj_resetData(model, data)
            reset_pose(mod, model, data)
            mujoco.mj_forward(model, data)
        if mode == "shared":
            wp = item
            goals = split_targets(arm_list, data, np.array(wp["pos"]), wp["separation"])
            data.mocap_pos[mocap] = wp["pos"]
            marks = [np.array(wp["pos"])]
            label = f"[{idx + 1}/{len(sequence)}] {wp['name']}"
            extra = f"verdict: {verdicts.get(wp['name'], '?')}"
        else:
            per_arm = dict(zip(("left", "right"), item))
            goals = {name: np.array(per_arm[name]["pos"]) for name in per_arm}
            data.mocap_pos[mocap] = goals["left"]
            marks = [goals["left"], goals["right"]]
            label = f"[{idx + 1}/{len(sequence)}]"
            extra = (f"left -> {per_arm['left']['name']}   "
                     f"right -> {per_arm['right']['name']}")
        solved = {a.name: _aim(model, scratch, data, a, goals[a.name], quat, rng)
                  for a in arm_list}

        for _ in range(frames_per_wp):
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            renderer.update_scene(data, camera)
            _draw_marks(renderer.scene, grid, marks, mode)
            errs = "  ".join(
                f"{a.name} " + (
                    f"{np.linalg.norm(data.site_xpos[a.site] - goals[a.name]) * 1000:5.0f}mm"
                    if solved[a.name] else "NO IK")
                for a in arm_list)
            hits = len(arm_contacts(model, data, arm_list)) if touching(data, owner) else 0
            writer.append_data(_overlay(renderer.render(), [
                f"{scene}  {mode}  {label}",
                f"{errs}   contacts {hits}",
                extra,
            ], font))
            n += 1
    writer.close()
    renderer.close()
    return n


def _sphere(scene_, pos, rgba, size: float) -> None:
    if scene_.ngeom >= scene_.maxgeom:
        return
    mujoco.mjv_initGeom(scene_.geoms[scene_.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([size, 0, 0]), np.asarray(pos, float),
                        np.eye(3).flatten(), np.array(rgba, np.float32))
    scene_.ngeom += 1


def _draw_marks(scene_, grid, marks, mode: str) -> None:
    """The whole waypoint grid as faint green dots for context, plus the target(s)
    currently being chased. In split mode the two arms have different targets, so
    they are coloured per arm (left magenta, right cyan)."""
    for wp in grid:
        _sphere(scene_, wp["pos"], [0.15, 0.8, 0.35, 0.35], 0.005)
    colors = ([[1.0, 0.3, 0.8, 0.95]] if mode == "shared"
              else [[1.0, 0.3, 0.8, 0.95], [0.2, 0.85, 0.95, 0.95]])
    for pos, rgba in zip(marks, colors):
        _sphere(scene_, pos, rgba, 0.016)


def combine(paths: list[Path], out_path: Path) -> None:
    """Side-by-side panels, padded to the longest run so a shorter scene's last frame
    holds while the others finish."""
    clips = []
    for path in paths:
        reader = imageio.get_reader(path)
        clips.append([f for f in reader])
        reader.close()
    n = max(len(c) for c in clips)
    writer = imageio.get_writer(out_path, fps=FPS, macro_block_size=1)
    try:
        for i in range(n):
            writer.append_data(np.hstack([c[min(i, len(c) - 1)] for c in clips]))
    finally:
        writer.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="shared", choices=("shared", "split"))
    ap.add_argument("--scene", help="one scene; default is every scene plus the combined video")
    ap.add_argument("--dwell", type=float, default=DWELL_S, help="seconds per waypoint")
    ap.add_argument("--stride-min", type=int, default=STRIDE_MIN,
                    help="shared mode: smallest step between visited waypoints")
    ap.add_argument("--stride-max", type=int, default=STRIDE_MAX,
                    help="shared mode: largest step between visited waypoints")
    ap.add_argument("--n", type=int, default=SPLIT_N, help="split mode: waypoints per arm")
    ap.add_argument("--seed", type=int, default=0, help="split mode: sampling seed")
    ap.add_argument("--reset", action="store_true",
                    help="return to the home pose before each target, as the tests do")
    ap.add_argument("--limit", type=int, help="render only the first N targets (preview)")
    ap.add_argument("--suffix", default="", help="output filename suffix")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    suffix = ("" if args.mode == "shared" else "_split") + args.suffix
    scenes = [args.scene] if args.scene else sorted(SCENES)
    paths = []
    for scene in scenes:
        out = OUT_DIR / f"{scene}{suffix}.mp4"
        n = render_scene(scene, out, args.mode, args.dwell,
                         (args.stride_min, args.stride_max), args.n, args.seed,
                         args.reset, args.limit)
        print(f"{scene}: {n} frames -> {out.relative_to(HERE)}")
        paths.append(out)
    if len(paths) > 1:
        combined = OUT_DIR / f"both_scenes{suffix}.mp4"
        combine(paths, combined)
        print(f"combined -> {combined.relative_to(HERE)}")


if __name__ == "__main__":
    main()
