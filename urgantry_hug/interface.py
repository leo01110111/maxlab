"""The data contract between HUG prediction and Wuji execution.

`predict.py` produces a ManoGrasp, `execute.py` consumes a WujiGrasp, and
retargeting is the function between them (see RETARGETING_SPEC.md). Nothing here
depends on either implementation, so both sides can be built and tested apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np
import mujoco

FINGERS = (1, 2, 3, 4, 5)
FINGER_JOINTS = (1, 2, 3, 4)
FINGER_NAMES = {1: "thumb", 2: "index", 3: "middle", 4: "ring", 5: "pinky"}

# MANO's 15 joint rotations run index, middle, pinky, ring, thumb; its 21
# landmarks run thumb, index, middle, ring, pinky. Ring and pinky swap between
# the two, so neither ordering can be assumed from the other.
MANO_POSE_SLICES = {
    "index": slice(0, 3),
    "middle": slice(3, 6),
    "pinky": slice(6, 9),
    "ring": slice(9, 12),
    "thumb": slice(12, 15),
}
MANO_LANDMARKS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
MANO_WRIST_LANDMARK = 0


@dataclass
class ManoGrasp:
    """One HUG prediction, expressed in MuJoCo world coordinates."""

    T_world_wrist: np.ndarray       # (4, 4) MANO wrist frame in world
    pose: np.ndarray                # (15, 3) axis-angle, MANO_POSE_SLICES order
    landmarks: np.ndarray           # (21, 3) world, MANO_LANDMARKS order
    vertices: np.ndarray            # (778, 3) world
    faces: Optional[np.ndarray] = None   # (F, 3)
    T_world_cam: Optional[np.ndarray] = None
    click_uv: Optional[np.ndarray] = None    # (2,) pixel in the 224 crop
    click_world: Optional[np.ndarray] = None  # (3,) clicked surface point

    def landmark(self, finger: str, index: int) -> np.ndarray:
        return self.landmarks[MANO_LANDMARKS[finger][index]]


@dataclass
class HandTarget:
    """Joint targets for one Wuji hand.

    `angles[f - 1, j - 1]` is fingerF_jointJ in radians, finger 1 = thumb
    through finger 5 = pinky. joint1/3/4 are flexion (positive curls toward palm
    +x), joint2 is abduction.
    """

    angles: np.ndarray = field(
        default_factory=lambda: np.zeros((5, 4), dtype=np.float64))

    def __post_init__(self) -> None:
        self.angles = np.asarray(self.angles, dtype=np.float64).reshape(5, 4)

    def ctrl(self, model: mujoco.MjModel, side: str) -> dict[int, float]:
        """{actuator id: clipped target} for one hand of the compiled model."""
        out = {}
        for f in FINGERS:
            for j in FINGER_JOINTS:
                act = model.actuator(f"{side}_hand_finger{f}_joint{j}")
                lo, hi = act.ctrlrange
                out[act.id] = float(np.clip(self.angles[f - 1, j - 1], lo, hi))
        return out

    @classmethod
    def open(cls) -> "HandTarget":
        return cls(np.zeros((5, 4)))


@dataclass
class WujiGrasp:
    """A grasp the sim can execute: where to put the palm, and how to close."""

    T_world_palm: np.ndarray      # (4, 4) target pose of <side>_hand_palm_link
    hand: HandTarget              # joint targets at full close
    side: str = "right"
    pregrasp: Optional[HandTarget] = None  # posture during approach; open if None
    approach_m: float = 0.08      # standoff along palm -z at the pre-grasp pose
    quality: Optional[float] = None

    def __post_init__(self) -> None:
        self.T_world_palm = np.asarray(self.T_world_palm, float).reshape(4, 4)
        if self.pregrasp is None:
            self.pregrasp = HandTarget.open()


class Retargeter(Protocol):
    """MANO grasp -> Wuji grasp. See RETARGETING_SPEC.md."""

    def __call__(self, grasp: ManoGrasp, model: mujoco.MjModel,
                 side: str = "right") -> WujiGrasp: ...
