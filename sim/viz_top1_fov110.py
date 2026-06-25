"""Visualize the UR7e table scene with the **top1** camera widened to a 110°
*horizontal* field of view, and show a live render of both cameras side by side.

This script is self-contained: it builds its own MjSpec via build_urtable.build_spec(),
overrides only the top1 camera's fovy, and compiles its own model. It never mutates
build_urtable's globals or the shared model, so it cannot change the behavior of the
simulation for env.py, test_viz.py, or any other script.

MuJoCo cameras are parameterized by the *vertical* FOV (`fovy`). The horizontal
FOV depends on the render aspect ratio, so we convert:
    tan(hfov/2) = aspect * tan(vfov/2)   ->   vfov = 2*atan(tan(hfov/2) / aspect)

Run:  uv run python viz_top1_fov110.py
"""

import math
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from build_urtable import (
    CAM1_POS,
    _lookat_quat,
    apply_initial_view,
    build_spec,
    capture_state,
    set_initial_pose,
)

CAMERAS = ["top1", "top2"]
RENDER_W, RENDER_H = 450, 300          # per-camera render size; aspect = W/H
TOP1_HFOV_DEG = 70.0                   # desired horizontal FOV for top1


def hfov_to_fovy(hfov_deg: float, aspect: float) -> float:
    """Vertical FOV (deg) giving `hfov_deg` horizontal FOV at the given aspect (W/H)."""
    hfov = math.radians(hfov_deg)
    fovy = 2.0 * math.atan(math.tan(hfov / 2.0) / aspect)
    return math.degrees(fovy)


def build_scene_top1_wide() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build the standard scene but with top1's fovy set for a 110° horizontal FOV.

    Works on a fresh spec so the shared build_urtable model is untouched."""
    spec = build_spec()
    fovy = hfov_to_fovy(TOP1_HFOV_DEG, RENDER_W / RENDER_H)
    spec.camera("top1").fovy = fovy
    print(f"top1: horizontal FOV {TOP1_HFOV_DEG:.1f}deg "
          f"(aspect {RENDER_W}/{RENDER_H}) -> fovy {fovy:.2f}deg")

    # Reorient top1: move it to the midpoint of the table length (x=0, keeping its
    # front-edge y and height), and aim it straight forward (horizontal, along +y,
    # the "into the table" direction) instead of angled down at the work center.
    new_pos = (0.0, CAM1_POS[1], CAM1_POS[2])
    forward_target = (new_pos[0], new_pos[1] + 1.0, new_pos[2])   # +y, same height
    quat = _lookat_quat(new_pos, forward_target)
    top1_body = spec.body("top1_body")
    top1_body.pos = list(new_pos)
    top1_body.quat = list(quat)
    print(f"top1: moved to {tuple(round(p, 4) for p in new_pos)}, facing +y (straight forward)")

    model = spec.compile()
    data = mujoco.MjData(model)
    set_initial_pose(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def main():
    model, data = build_scene_top1_wide()
    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        apply_initial_view(viewer)   # open at the hard-coded INITIAL_VIEW
        viewer.sync()
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)
            viewer.sync()

            frames = []
            for cam in CAMERAS:
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render())     # (H, W, 3) uint8 RGB

            combined = np.hstack(frames)             # side by side
            cv2.imshow(f"cameras (left: top1 @{TOP1_HFOV_DEG:.0f}deg HFOV, right: top2)",
                       combined[..., ::-1])          # RGB -> BGR for OpenCV
            # Controls (focus the OpenCV window): 's' = print pose+view, 'q' = quit.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                capture_state(data, viewer)
            elif key == ord("q"):
                break

            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)

    renderer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
