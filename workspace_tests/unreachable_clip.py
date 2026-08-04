"""Clip of the one benchmark waypoint the urgantry scene cannot reach.

Both arms solve IK to 0.1 mm at `low_x+175_y+000`; the right gripper pad just lands
inside the green task cube. Segment 1 shows the collision, segment 2 repeats the move
with the cube shoved aside so the same goal is reached cleanly.

    python unreachable_clip.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco
import imageio.v2 as imageio

from waypoint_video import CAM, FPS, HEIGHT, WIDTH, _draw_waypoints, _font, _overlay
from workspace_benchmark import load_benchmark
from workspace_probe import POS_TOL, ROT_TOL, build, solve_ik_restarts, split_targets, tool_quat

HERE = Path(__file__).resolve().parent
OUT = HERE / "waypoint_videos" / "urgantry_unreachable.mp4"

SCENE = "urgantry"
WP_NAME = "low_x+175_y+000"
SETTLE_S = 4.0
HOLD_S = 3.0
BLOCK_ASIDE = np.array([0.0, 0.5, 0.9, 1.0, 0.0, 0.0, 0.0])


def block_contacts(model, data, arm, block_gid: int) -> int:
    n = 0
    for c in data.contact[:data.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 != block_gid and g2 != block_gid:
            continue
        other = g2 if g1 == block_gid else g1
        if model.geom_bodyid[other] in arm.bodies:
            n += 1
    return n


def render_segment(writer, renderer, camera, font, mod, model, data, arms, waypoints,
                   wp_idx, block_gid, label, move_block: bool) -> int:
    wp = waypoints[wp_idx]
    arm_list = list(arms.values())
    right = arms["right"]

    mod.set_initial_pose(model, data)
    data.ctrl[:] = 0.0
    for a in arm_list:
        data.ctrl[a.act_ids] = data.qpos[a.qadr]
    if move_block:
        adr = model.joint("block_joint").qposadr[0]
        data.qpos[adr:adr + 7] = BLOCK_ASIDE
        data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    pos = np.array(wp["pos"])
    goals = split_targets(arm_list, data, pos, wp["separation"])
    data.mocap_pos[model.body("ik_target").mocapid[0]] = pos

    scratch = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    quat = tool_quat(0.0)
    solved = {}
    for a in arm_list:
        scratch.qpos[:] = data.qpos
        q, pe, re = solve_ik_restarts(model, scratch, a, goals[a.name], quat,
                                      data.qpos[a.qadr].copy(), rng)
        solved[a.name] = pe < POS_TOL and re < ROT_TOL
        data.ctrl[a.act_ids] = q

    steps_per_frame = max(1, int(round((1.0 / FPS) / model.opt.timestep)))
    n_frames = int(round((SETTLE_S + HOLD_S) * FPS))
    peak = 0
    for _ in range(n_frames):
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        nc = block_contacts(model, data, right, block_gid)
        peak = max(peak, nc)
        renderer.update_scene(data, camera)
        _draw_waypoints(renderer.scene, waypoints, wp_idx)
        errs = "   ".join(
            f"{a.name} {np.linalg.norm(data.site_xpos[a.site] - goals[a.name]) * 1000:5.1f}mm"
            for a in arm_list)
        writer.append_data(_overlay(renderer.render(), [
            f"{SCENE}   {wp['name']}   [{label}]",
            f"TCP error:  {errs}",
            "",
            f"right gripper vs block: {nc} contacts",
        ], font))
    return peak


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    mod, model, data, arms = build(SCENE)
    waypoints = load_benchmark()
    wp_idx = next(i for i, w in enumerate(waypoints) if w["name"] == WP_NAME)
    block_gid = model.geom("block").id

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = CAM["azimuth"], CAM["elevation"]
    camera.distance = 1.15
    camera.lookat[:] = [0.28, -0.08, mod.BOARD_TOP + 0.05]

    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    writer = imageio.get_writer(OUT, fps=FPS, macro_block_size=1)
    font = _font(15)
    try:
        p1 = render_segment(writer, renderer, camera, font, mod, model, data, arms,
                            waypoints, wp_idx, block_gid, "block in place", False)
        p2 = render_segment(writer, renderer, camera, font, mod, model, data, arms,
                            waypoints, wp_idx, block_gid, "block moved aside", True)
    finally:
        writer.close()
        renderer.close()
    print(f"peak block contacts: segment1={p1} segment2={p2}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
