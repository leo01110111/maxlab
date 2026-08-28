"""HUG grasp prediction driving the gantry sim's Wuji hand.

Pipeline: render an RGB-D frame from a sim camera (capture.py), predict a MANO
grasp for a clicked pixel (predict.py), lift it into world coordinates
(frames.py), retarget it onto the Wuji hand + UR7e arm (retarget.py, owned
separately -- see RETARGETING_SPEC.md), and execute the reach/close/lift
(execute.py). demo.py runs the whole thing.
"""

from .capture import Capture, capture_rgbd, camera_intrinsics
from .frames import (
    camera_extrinsic,
    transform_points,
    transform_pose,
    unproject,
)
from .interface import HandTarget, WujiGrasp

__all__ = [
    "Capture",
    "capture_rgbd",
    "camera_intrinsics",
    "camera_extrinsic",
    "transform_points",
    "transform_pose",
    "unproject",
    "HandTarget",
    "WujiGrasp",
]
