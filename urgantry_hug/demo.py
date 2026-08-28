"""End-to-end: render the gantry scene, predict a HUG grasp for a clicked pixel,
retarget it onto the Wuji hand, and execute the pick.

    python -m urgantry_hug.demo                     # aim at the block
    python -m urgantry_hug.demo --uv 442 263        # aim at a pixel
    python -m urgantry_hug.demo --viewer            # watch it run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import mujoco

from urgantry_sim.build_urgantry import apply_initial_view, build_scene, set_initial_pose

from .capture import RGBDRenderer
from .execute import GraspExecutor
from .frames import project, transform_points
from .predict import DEFAULT_CHECKPOINT, HugPredictor


def load_retargeter():
    try:
        from .retarget import retarget
    except ImportError as e:
        raise SystemExit(
            "urgantry_hug.retarget is not available yet -- it is owned by the "
            "retargeting work. It must expose retarget(grasp, model, side) -> "
            "WujiGrasp; see RETARGETING_SPEC.md.\n"
            f"({e})"
        )
    return retarget


def block_pixel(model, data, capture) -> np.ndarray:
    cam = transform_points(np.linalg.inv(capture.T_world_cam),
                           data.body("block").xpos[None])
    return project(cam, capture.K)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uv", type=float, nargs=2, default=None,
                    help="pixel to grasp at, in full-resolution image coords")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--side", default="right", choices=["left", "right"],
                    help="HUG predicts right hands; left needs a mirror")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    retarget = load_retargeter()

    model, data = build_scene()
    set_initial_pose(model, data)
    mujoco.mj_forward(model, data)

    viewer = None
    if args.viewer:
        viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                              show_right_ui=False)
        apply_initial_view(viewer)

    renderer = RGBDRenderer(model, width=args.width, height=args.height)
    capture = renderer(data, camera="top1")

    uv = np.array(args.uv) if args.uv else block_pixel(model, data, capture)
    print(f"grasping at pixel {np.round(uv, 1)}")

    predictor = HugPredictor(args.checkpoint, sampling_steps=50)
    grasp = predictor.predict(capture, uv, seed=args.seed)
    print(f"MANO wrist at {np.round(grasp.T_world_wrist[:3, 3], 4)}")
    print(f"clicked surface point {np.round(grasp.click_world, 4)}")

    wuji = retarget(grasp, model, side=args.side)
    print(f"palm target {np.round(wuji.T_world_palm[:3, 3], 4)}")

    executor = GraspExecutor(model, data,
                             on_step=(lambda: viewer.sync()) if viewer else None)
    result = executor.run(wuji)
    print(f"\n{'SUCCESS' if result.success else 'FAILED'}: {result.reason}")
    print(f"block height {result.block_height:.4f} "
          f"(rest 0.80, success above 0.85)")

    renderer.close()
    predictor.close()
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
