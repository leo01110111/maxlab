"""Coordinate conversions between MuJoCo world, MuJoCo camera, and HUG camera.

HUG works in the OpenCV camera convention (+x right, +y down, +z into the
scene). MuJoCo cameras look down their own -z with +y up, so the two differ by a
180 degree roll about x -- get this wrong and predictions land mirrored through
the camera plane, which still looks plausible in a single view.
"""

from __future__ import annotations

import numpy as np
import mujoco

MUJOCO_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


def camera_extrinsic(model: mujoco.MjModel, data: mujoco.MjData,
                     camera: str = "top1") -> np.ndarray:
    """(4, 4) pose of the OpenCV-convention camera frame in world coordinates."""
    cid = model.camera(camera).id
    T = np.eye(4)
    T[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ MUJOCO_TO_OPENCV
    T[:3, 3] = data.cam_xpos[cid]
    return T


def transform_pose(T: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Compose two (4, 4) homogeneous transforms."""
    return np.asarray(T, float) @ np.asarray(pose, float)


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a (4, 4) transform to (N, 3) points."""
    pts = np.asarray(points, float)
    return pts @ np.asarray(T, float)[:3, :3].T + np.asarray(T, float)[:3, 3]


def unproject(u: float, v: float, depth_m: float, K: np.ndarray) -> np.ndarray:
    """Pixel + metric depth -> (3,) point in the OpenCV camera frame."""
    K = np.asarray(K, float)
    return np.array([(u - K[0, 2]) * depth_m / K[0, 0],
                     (v - K[1, 2]) * depth_m / K[1, 1],
                     depth_m])


def project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(N, 3) camera-frame points -> (N, 2) pixels."""
    proj = np.asarray(points, float) @ np.asarray(K, float).T
    return proj[:, :2] / (proj[:, 2:3] + 1e-8)


def pixel_depth(capture, u: float, v: float) -> float:
    """Metric depth at a pixel of a Capture, 0.0 where invalid."""
    h, w = capture.depth_mm.shape[:2]
    uu = int(np.clip(round(u), 0, w - 1))
    vv = int(np.clip(round(v), 0, h - 1))
    return float(capture.depth_mm[vv, uu]) / 1000.0
