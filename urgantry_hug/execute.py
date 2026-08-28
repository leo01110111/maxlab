"""Execute a retargeted grasp: approach, close, lift.

Consumes a WujiGrasp (see interface.py) and drives the compiled gantry scene
through it with the existing damped-least-squares arm IK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import mujoco

from urgantry_bc.ik import GRASP_CENTER, solve_ik
from urgantry_sim.build_urgantry import ARM_JOINTS, pick_success

from .interface import FINGERS, HandTarget, WujiGrasp

# Retargeting reproduces the hand SHAPE, which only touches the object. Position
# servos commanded to exactly that configuration apply no grip force, so the
# object slips the moment the arm lifts. Curl the flexion joints past the fitted
# pose so the servos push into the object.
SQUEEZE_RAD = 0.25
FLEXION_JOINTS = (1, 3, 4)

# The arm's position servos lag: right after a move the grasp center is ~15 mm
# off target, decaying to ~4 mm. Closing the fingers before it settles grasps a
# pose the hand is not actually at, and the first finger to touch shoves the
# object away.
SETTLE_TICKS = 150


@dataclass
class GraspResult:
    success: bool
    reason: str
    block_height: float
    pos_err: float
    rot_err: float


def palm_pose_to_ik_target(T_world_palm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Palm pose -> (position, rotation) in solve_ik's convention.

    solve_ik drives GRASP_CENTER, a point offset inside the palm body, not the
    palm origin -- so the offset has to be added in the palm frame first.
    """
    T = np.asarray(T_world_palm, float)
    R = T[:3, :3]
    return T[:3, 3] + R @ GRASP_CENTER, R


def standoff_pose(T_world_palm: np.ndarray, distance: float) -> np.ndarray:
    """Back the palm off along its own -z (fingers extend along palm +z)."""
    T = np.array(T_world_palm, float)
    T[:3, 3] = T[:3, 3] - T[:3, 2] * distance
    return T


class GraspExecutor:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 control_hz: float = 50.0, squeeze_rad: float = SQUEEZE_RAD,
                 settle_ticks: int = SETTLE_TICKS,
                 on_step: Optional[Callable[[], None]] = None):
        self.model = model
        self.data = data
        self.n_substeps = max(1, round((1.0 / control_hz) / model.opt.timestep))
        self.squeeze_rad = squeeze_rad
        self.settle_ticks = settle_ticks
        self.on_step = on_step

    def _settle(self, ticks: int) -> None:
        for _ in range(ticks):
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.model, self.data)
            if self.on_step is not None:
                self.on_step()

    def set_hand(self, target: HandTarget, side: str) -> None:
        for aid, val in target.ctrl(self.model, side).items():
            self.data.ctrl[aid] = val

    def move_arm(self, q_target: np.ndarray, side: str, ticks: int = 60) -> None:
        """Ramp the arm actuators from their current command to q_target."""
        act_ids = [self.model.actuator(f"{side}_{j}").id for j in ARM_JOINTS]
        q_start = self.data.ctrl[act_ids].copy()
        for i in range(1, ticks + 1):
            alpha = i / ticks
            self.data.ctrl[act_ids] = (1 - alpha) * q_start + alpha * q_target
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.model, self.data)
            if self.on_step is not None:
                self.on_step()

    def close_hand(self, target: HandTarget, side: str, ticks: int = 50) -> None:
        start = HandTarget(np.array([
            [self.data.ctrl[self.model.actuator(f"{side}_hand_finger{f}_joint{j}").id]
             for j in (1, 2, 3, 4)] for f in (1, 2, 3, 4, 5)]))
        for i in range(1, ticks + 1):
            alpha = i / ticks
            blend = HandTarget((1 - alpha) * start.angles + alpha * target.angles)
            self.set_hand(blend, side)
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.model, self.data)
            if self.on_step is not None:
                self.on_step()

    def run(self, grasp: WujiGrasp, lift_m: float = 0.15) -> GraspResult:
        side = grasp.side

        pre = standoff_pose(grasp.T_world_palm, grasp.approach_m)
        pos, mat = palm_pose_to_ik_target(pre)
        q_pre, pe, re, ok = solve_ik(self.model, self.data, side, pos, mat)
        if not ok:
            return GraspResult(False, f"pre-grasp IK failed (pos {pe:.4f} rot {re:.3f})",
                               _block_z(self.model, self.data), pe, re)
        self.set_hand(grasp.pregrasp, side)
        self.move_arm(q_pre, side)
        self._settle(self.settle_ticks)

        pos, mat = palm_pose_to_ik_target(grasp.T_world_palm)
        q_grasp, pe, re, ok = solve_ik(self.model, self.data, side, pos, mat,
                                       q_init=q_pre)
        if not ok:
            return GraspResult(False, f"grasp IK failed (pos {pe:.4f} rot {re:.3f})",
                               _block_z(self.model, self.data), pe, re)
        self.move_arm(q_grasp, side, ticks=40)
        self._settle(self.settle_ticks)

        self.close_hand(grasp.hand, side)
        self._settle(10)
        if self.squeeze_rad > 0.0:
            self.close_hand(squeezed(grasp.hand, self.squeeze_rad), side, ticks=25)
        self._settle(20)

        lift = np.array(grasp.T_world_palm, float)
        lift[2, 3] += lift_m
        pos, mat = palm_pose_to_ik_target(lift)
        q_lift, lpe, lre, lok = solve_ik(self.model, self.data, side, pos, mat,
                                        q_init=q_grasp)
        if lok:
            self.move_arm(q_lift, side, ticks=60)
        self._settle(30)

        success = pick_success(self.model, self.data)
        reason = "lifted" if success else (
            "lift IK failed" if not lok else "grasp slipped")
        return GraspResult(bool(success), reason,
                           _block_z(self.model, self.data), pe, re)


def squeezed(target: HandTarget, amount: float) -> HandTarget:
    """Curl the flexion joints further, leaving abduction where it was."""
    angles = target.angles.copy()
    for f in FINGERS:
        for j in FLEXION_JOINTS:
            angles[f - 1, j - 1] += amount
    return HandTarget(angles)


def _block_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    return float(data.body("block").xpos[2])
