"""Exercise the executor with a hand-built top-down grasp, no HUG and no
retargeting involved.

Separates "the arm/hand pipeline works" from "the prediction was any good", so a
failed demo can be attributed to one side or the other.
"""

import numpy as np
import mujoco

from urgantry_bc.hand import FLEX_FULL, THUMB_FLEX_FULL
from urgantry_bc.ik import GRASP_CENTER, palm_target_mat
from urgantry_sim.build_urgantry import build_scene, set_initial_pose

from .execute import GraspExecutor
from .interface import HandTarget, WujiGrasp


def closed_hand() -> HandTarget:
    angles = np.zeros((5, 4))
    for f in range(5):
        table = THUMB_FLEX_FULL if f == 0 else FLEX_FULL
        for j, val in table.items():
            angles[f, j - 1] = val
    return HandTarget(angles)


def block_grasp(model, data, side: str = "right") -> WujiGrasp:
    """A grasp of the block that this hand can actually close on.

    Fingers extend horizontally, not downward: open fingertips reach ~4.5 cm past
    GRASP_CENTER, so a top-down approach drives them through the block and the
    table before anything closes, and the block just gets swept away.
    """
    target = data.body("block").xpos.copy()
    mat = palm_target_mat(finger_dir=(0.0, -1.0, 0.0), grasp_dir=(0.0, 0.0, -1.0))
    T = np.eye(4)
    T[:3, :3] = mat
    T[:3, 3] = target - mat @ GRASP_CENTER
    return WujiGrasp(T_world_palm=T, hand=closed_hand(), side=side)


def main() -> None:
    model, data = build_scene()
    set_initial_pose(model, data)
    mujoco.mj_forward(model, data)

    grasp = block_grasp(model, data)
    print(f"palm target {np.round(grasp.T_world_palm[:3, 3], 4)}")
    print(f"block at    {np.round(data.body('block').xpos, 4)}")

    result = GraspExecutor(model, data).run(grasp)
    print(f"\n{'SUCCESS' if result.success else 'FAILED'}: {result.reason}")
    print(f"block height {result.block_height:.4f} (rest 0.80)")
    print("\nPASS" if result.success else "\nFAIL")


if __name__ == "__main__":
    main()
