"""Draw a predicted grasp into the sim, to check it landed where HUG meant it to.

Two views of the same claim. The 2D overlay reprojects the prediction into the
image it came from, which catches nothing about the world transform. Rendering
the same landmarks as world-space geometry from a DIFFERENT viewpoint does: if
the camera-to-world conversion were wrong, the hand would still overlay correctly
in 2D but drift off the object as soon as the camera moves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

from .frames import project, transform_points
from .interface import MANO_LANDMARKS, MANO_WRIST_LANDMARK

BONES = [(MANO_WRIST_LANDMARK, chain[0]) for chain in MANO_LANDMARKS.values()]
BONES += [(chain[i], chain[i + 1])
          for chain in MANO_LANDMARKS.values() for i in range(3)]

FINGER_COLORS = {
    "thumb": (1.0, 0.30, 0.30, 1.0),
    "index": (1.0, 0.75, 0.20, 1.0),
    "middle": (0.35, 0.85, 0.35, 1.0),
    "ring": (0.35, 0.65, 1.0, 1.0),
    "pinky": (0.80, 0.45, 1.0, 1.0),
}
WRIST_COLOR = (1.0, 1.0, 1.0, 1.0)
CLICK_COLOR = (1.0, 0.15, 0.60, 1.0)


def landmark_color(index: int):
    for finger, chain in MANO_LANDMARKS.items():
        if index in chain:
            return FINGER_COLORS[finger]
    return WRIST_COLOR


def _add_geom(scene, gtype, size, pos, mat, rgba) -> bool:
    if scene.ngeom >= scene.maxgeom:
        return False
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, np.float64),
                        np.asarray(pos, np.float64),
                        np.asarray(mat, np.float64).flatten(),
                        np.asarray(rgba, np.float32))
    scene.ngeom += 1
    return True


def draw_grasp(scene, grasp, joint_radius: float = 0.006,
               bone_radius: float = 0.003) -> None:
    """Add the predicted hand skeleton to an mjvScene as world-space geometry."""
    eye = np.eye(3)
    for i, p in enumerate(grasp.landmarks):
        r = joint_radius * (1.6 if i == MANO_WRIST_LANDMARK else 1.0)
        _add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [r, 0, 0], p, eye,
                  landmark_color(i))

    for a, b in BONES:
        pa, pb = grasp.landmarks[a], grasp.landmarks[b]
        mid = 0.5 * (pa + pb)
        d = pb - pa
        length = np.linalg.norm(d)
        if length < 1e-6:
            continue
        z = d / length
        helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(helper, z)
        x /= np.linalg.norm(x)
        mat = np.column_stack([x, np.cross(z, x), z])
        _add_geom(scene, mujoco.mjtGeom.mjGEOM_CAPSULE,
                  [bone_radius, bone_radius, 0.5 * length], mid, mat,
                  landmark_color(b))

    if grasp.click_world is not None:
        _add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.008, 0, 0],
                  grasp.click_world, eye, CLICK_COLOR)


def render_with_grasp(model, data, grasp, camera="top1", width: int = 640,
                      height: int = 480) -> np.ndarray:
    """Render `camera` with the predicted hand drawn in. `camera` may be a name
    or an mjvCamera for a free viewpoint."""
    renderer = mujoco.Renderer(model, height=height, width=width, max_geom=20000)
    try:
        renderer.update_scene(data, camera=camera)
        draw_grasp(renderer.scene, grasp)
        return renderer.render().copy()
    finally:
        renderer.close()


def free_camera(model, lookat, distance: float, azimuth: float, elevation: float):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def overlay_2d(capture, grasp) -> np.ndarray:
    """Reproject the prediction onto the image it was predicted from."""
    import cv2

    img = capture.rgb.copy()
    cam_pts = transform_points(np.linalg.inv(capture.T_world_cam), grasp.landmarks)
    uv = project(cam_pts, capture.K)

    for a, b in BONES:
        c = tuple(int(255 * v) for v in landmark_color(b)[:3][::-1])
        cv2.line(img, tuple(np.int32(uv[a])), tuple(np.int32(uv[b])), c, 2,
                 cv2.LINE_AA)
    for i, p in enumerate(uv):
        c = tuple(int(255 * v) for v in landmark_color(i)[:3][::-1])
        cv2.circle(img, tuple(np.int32(p)), 4 if i else 6, c, -1, cv2.LINE_AA)
    if grasp.click_uv is not None:
        cam_click = transform_points(np.linalg.inv(capture.T_world_cam),
                                     grasp.click_world[None])
        cu = project(cam_click, capture.K)[0]
        cv2.drawMarker(img, tuple(np.int32(cu)),
                       tuple(int(255 * v) for v in CLICK_COLOR[:3][::-1]),
                       cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    return img


def main() -> None:
    import argparse
    import cv2

    from urgantry_sim.build_urgantry import build_scene, set_initial_pose
    from .capture import RGBDRenderer
    from .predict import DEFAULT_CHECKPOINT, HugPredictor

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("hug_vis"))
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model, data = build_scene()
    set_initial_pose(model, data)
    mujoco.mj_forward(model, data)

    renderer = RGBDRenderer(model, width=640, height=480)
    capture = renderer(data, camera="top1")

    cam_pt = transform_points(np.linalg.inv(capture.T_world_cam),
                              data.body("block").xpos[None])
    uv = project(cam_pt, capture.K)[0]

    predictor = HugPredictor(args.checkpoint)
    grasp = predictor.predict(capture, uv, seed=args.seed)
    block = data.body("block").xpos

    cv2.imwrite(str(args.out / "overlay_2d.png"),
                cv2.cvtColor(overlay_2d(capture, grasp), cv2.COLOR_RGB2BGR))

    views = {
        "world_top1": "top1",
        "world_side": free_camera(model, block, 0.55, azimuth=90, elevation=-10),
        "world_front": free_camera(model, block, 0.55, azimuth=180, elevation=-20),
        "world_above": free_camera(model, block, 0.5, azimuth=135, elevation=-70),
    }
    for name, cam in views.items():
        img = render_with_grasp(model, data, grasp, camera=cam, width=640, height=480)
        cv2.imwrite(str(args.out / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    tips = {f: grasp.landmarks[c[3]] for f, c in MANO_LANDMARKS.items()}
    print(f"block center {np.round(block, 4)}  (5 cm cube, top at {block[2] + 0.025:.3f})")
    print(f"click        {np.round(grasp.click_world, 4)}")
    print(f"wrist        {np.round(grasp.T_world_wrist[:3, 3], 4)}")
    for f, p in tips.items():
        print(f"  {f:<7} tip {np.round(p, 4)}  d(block) "
              f"{np.linalg.norm(p - block):.4f}")

    below = int((grasp.landmarks[:, 2] < 0.775).sum())
    print(f"\nlandmarks below table: {below}/21")
    print(f"wrote {len(views) + 1} images to {args.out}/")

    renderer.close()
    predictor.close()


if __name__ == "__main__":
    main()
