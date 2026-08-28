"""RGB-D capture from a MuJoCo camera, in the form HUG's prepare_inputs wants."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

from .frames import camera_extrinsic


@dataclass
class Capture:
    rgb: np.ndarray          # (H, W, 3) uint8
    depth_mm: np.ndarray     # (H, W) uint16, 0 = invalid
    K: np.ndarray            # (3, 3) at the rgb resolution
    T_world_cam: np.ndarray  # (4, 4), OpenCV camera axes
    camera: str

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.rgb.shape[:2]
        return w, h


def camera_intrinsics(model: mujoco.MjModel, camera: str, width: int, height: int
                      ) -> np.ndarray:
    """Pinhole K for a MuJoCo camera rendered at width x height.

    MuJoCo's fovy is the VERTICAL field of view and its pixels are square, so fx
    and fy are both set from the image height.
    """
    fovy = np.deg2rad(model.cam_fovy[model.camera(camera).id])
    f = 0.5 * height / np.tan(0.5 * fovy)
    return np.array([[f, 0.0, 0.5 * width],
                     [0.0, f, 0.5 * height],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


class RGBDRenderer:
    """Paired color + depth renderers for one model, reused across captures.

    Kept as an object because mujoco.Renderer allocates a GL context per
    instance and toggling depth mode per frame is far cheaper than rebuilding.
    """

    def __init__(self, model: mujoco.MjModel, width: int = 640, height: int = 480):
        self.model = model
        self.width = width
        self.height = height
        self._rgb = mujoco.Renderer(model, height=height, width=width)
        self._depth = mujoco.Renderer(model, height=height, width=width)
        self._depth.enable_depth_rendering()

    def __call__(self, data: mujoco.MjData, camera: str = "top1",
                 max_depth: float = 5.0) -> Capture:
        self._rgb.update_scene(data, camera=camera)
        rgb = self._rgb.render().copy()

        self._depth.update_scene(data, camera=camera)
        depth_m = self._depth.render().copy()

        return Capture(
            rgb=rgb,
            depth_mm=to_depth_mm(depth_m, max_depth),
            K=camera_intrinsics(self.model, camera, self.width, self.height),
            T_world_cam=camera_extrinsic(self.model, data, camera),
            camera=camera,
        )

    def close(self) -> None:
        self._rgb.close()
        self._depth.close()


def to_depth_mm(depth_m: np.ndarray, max_depth: float = 5.0) -> np.ndarray:
    """Metric float depth -> uint16 millimetres, background zeroed.

    Rays that hit nothing come back at the far clipping plane, which would read
    as a real surface several metres out and drag the point cloud with it.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0.0) & (depth < max_depth)
    mm = np.where(valid, depth * 1000.0, 0.0)
    return np.clip(mm, 0, 65535).astype(np.uint16)


def capture_rgbd(model: mujoco.MjModel, data: mujoco.MjData, camera: str = "top1",
                 width: int = 640, height: int = 480,
                 max_depth: float = 5.0) -> Capture:
    """One-shot capture. Use RGBDRenderer directly in a loop."""
    renderer = RGBDRenderer(model, width=width, height=height)
    try:
        return renderer(data, camera=camera, max_depth=max_depth)
    finally:
        renderer.close()


def write_capture(capture: Capture, out_dir, stem: str = "sim") -> None:
    """Dump a capture as the rgb/depth/intrinsics trio `hug.prepare_inputs` reads."""
    import cv2
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "rgb.png"), cv2.cvtColor(capture.rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "depth.png"), capture.depth_mm)
    np.savetxt(out / "intrinsics.txt", capture.K)
    np.savetxt(out / f"{stem}_T_world_cam.txt", capture.T_world_cam)
