"""Round-trip check on the camera model: does a pixel + its depth land back on
the object it came from, in world coordinates?

This validates the intrinsics, the depth units, and the MuJoCo->OpenCV camera
flip together. If retargeting ever produces a hand that is mirrored or offset by
a constant, run this first.
"""

import numpy as np
import mujoco

from urgantry_sim.build_urgantry import build_scene
from urgantry_hug.capture import capture_rgbd
from urgantry_hug.frames import pixel_depth, project, transform_points, unproject


def main() -> None:
    model, data = build_scene()
    cap = capture_rgbd(model, data, camera="top1", width=640, height=480)

    block_world = data.body("block").xpos.copy()

    T_cam_world = np.linalg.inv(cap.T_world_cam)
    block_cam = transform_points(T_cam_world, block_world[None])[0]
    uv = project(block_cam[None], cap.K)[0]
    print(f"block world      {block_world}")
    print(f"block cam        {block_cam}")
    print(f"projects to px   {uv}")

    d = pixel_depth(cap, uv[0], uv[1])
    print(f"depth at px      {d:.4f} m  (block center is {block_cam[2]:.4f} m out)")

    back_cam = unproject(uv[0], uv[1], d, cap.K)
    back_world = transform_points(cap.T_world_cam, back_cam[None])[0]
    print(f"unprojects to    {back_world}")

    # The hit point sits on the block's top face, not at its center: the camera
    # views the block obliquely, so the ray through the center pixel stops short.
    # What must hold is that the hit lies on the camera ray toward the center and
    # on the top face.
    eye = cap.T_world_cam[:3, 3]
    ray = block_world - eye
    ray /= np.linalg.norm(ray)
    off_ray = np.linalg.norm(np.cross(back_world - eye, ray))
    top_z = block_world[2] + 0.025
    on_face = np.abs(back_world[:2] - block_world[:2]).max()

    print(f"\noff camera ray:   {off_ray * 1000:.2f} mm")
    print(f"z vs block top:   {(back_world[2] - top_z) * 1000:+.2f} mm")
    print(f"xy within face:   {on_face * 1000:.1f} mm (half-extent 25 mm)")

    ok = off_ray < 2e-3 and abs(back_world[2] - top_z) < 5e-3 and on_face < 0.025
    print("\nPASS" if ok else "\nFAIL")


if __name__ == "__main__":
    main()
