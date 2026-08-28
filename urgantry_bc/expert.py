"""Scripted IK expert for the cube-into-box task.

Uses privileged state (the cube's true pose) to solve a fixed waypoint sequence
once per episode, then emits the interpolated joint targets as normalized env
actions. This is the demonstrator for behavior cloning -- the BC policy sees only
the camera image and proprioception, never these waypoints.

Grasp geometry was found empirically, not assumed (see hand.py): the Wuji fingers
curl toward palm +x, so the palm approaches with +x pointing down and the fingers
pitched 30 deg below horizontal, with the grasp center 1 cm BELOW the cube center
-- aiming at the cube center closes the fingers around its upper half and it
slips out on the lift.
"""

from __future__ import annotations

import numpy as np
import mujoco

from urgantry_sim.build_urgantry import ARM_JOINTS, BOARD_TOP
from urgantry_bc import ik as IK
from urgantry_bc.task_env import SIDE

PITCH = np.deg2rad(30.0)          # fingers below horizontal
AZIMUTH = np.deg2rad(180.0)       # fingers point back toward the far side of the board
GRASP_Z_OFFSET = -0.010           # grasp center below the cube center
PREGRASP_H = 0.14
LIFT_H = 0.22
CARRY_Z = BOARD_TOP + 0.30
RELEASE_Z = BOARD_TOP + 0.17

# The carry is split into CARRY_LEGS Cartesian sub-waypoints, each IK-solved with
# the same palm orientation. Interpolating joints straight from the lift pose to
# the over-tray pose lets the palm tilt through the middle of the swing and the
# cube falls out; holding orientation along the path is what keeps it in the hand.
CARRY_LEGS = 5
CARRY_LEG_STEPS = 14

SCHEDULE = [
    ("pregrasp", 26),
    ("descend1", 12),
    ("descend2", 12),
    ("grasp", 20),
    ("close", 26),
    ("lift", 30),
    *[(f"carry{i}", CARRY_LEG_STEPS) for i in range(CARRY_LEGS)],
    ("lower", 22),
    ("release", 20),
    ("retreat", 16),
]
CLOSE_PHASES = {"close", "lift", "lower"} | {f"carry{i}" for i in range(CARRY_LEGS)}


def _first_carry_index() -> int:
    return next(i for i, (label, _) in enumerate(SCHEDULE) if label.startswith("carry"))


def approach_mat() -> np.ndarray:
    finger = np.array([np.cos(PITCH) * np.cos(AZIMUTH),
                       np.cos(PITCH) * np.sin(AZIMUTH),
                       -np.sin(PITCH)])
    return IK.palm_target_mat(finger)


class ScriptedExpert:
    """Replans at reset; call `act(env)` once per env step."""

    def __init__(self, env):
        self.env = env
        self.R = approach_mat()
        self.plan: list[tuple[str, np.ndarray, np.ndarray, float]] = []
        self.t = 0
        self.failed = False

    def reset(self):
        env = self.env
        m, d = env.model, env.data
        cube = env.cube_pos().copy()
        box = env.box_pos().copy()

        lift_pt = cube + [0, 0, LIFT_H]
        over_box = np.array([box[0], box[1], CARRY_Z])
        targets = [
            ("pregrasp", cube + [0, 0, PREGRASP_H]),
            ("descend1", cube + [0, 0, 0.07]),
            ("descend2", cube + [0, 0, 0.025]),
            ("grasp", cube + [0, 0, GRASP_Z_OFFSET]),
            ("close", cube + [0, 0, GRASP_Z_OFFSET]),
            ("lift", lift_pt),
            *[(f"carry{i}", lift_pt + (over_box - lift_pt) * ((i + 1) / CARRY_LEGS))
              for i in range(CARRY_LEGS)],
            ("lower", np.array([box[0], box[1], RELEASE_Z])),
            ("release", np.array([box[0], box[1], RELEASE_Z])),
            ("retreat", over_box),
        ]

        IK.GRASP_CENTER = IK.GRASP_CENTER  # module default; kept explicit for clarity
        q = env.arm_qpos()
        self.plan = []
        self.failed = False
        for (label, steps), (_, pos) in zip(SCHEDULE, targets):
            q_next, pos_err, rot_err, ok = IK.solve_ik(m, d, SIDE, pos, self.R, q_init=q)
            if not ok:
                self.failed = True
            self.plan.append((label, q.copy(), q_next.copy(), steps))
            q = q_next
        self.t = 0
        self._replanned = False
        self._carry_start = sum(steps for label, steps in
                                SCHEDULE[:_first_carry_index()])
        return not self.failed

    def _replan_delivery(self):
        """Re-aim the delivery at the cube, not at the hand.

        The cube seats wherever the fingers caught it, which is offset from the
        grasp center by up to ~12 cm; delivering the grasp center over the tray
        center therefore drops the cube just outside the rim. Once it is off the
        board we can measure that offset and subtract it from the remaining
        targets."""
        env = self.env
        env = self.env
        here, _ = IK.grasp_frame(env.model, env.data, SIDE)
        offset = env.cube_pos() - here          # where the cube sits in the grip
        box = env.box_pos()
        over_box = np.array([box[0] - offset[0], box[1] - offset[1], CARRY_Z])
        release = np.array([box[0] - offset[0], box[1] - offset[1],
                            RELEASE_Z - offset[2]])

        carry_i = _first_carry_index()
        targets = [here + (over_box - here) * ((i + 1) / CARRY_LEGS)
                   for i in range(CARRY_LEGS)]
        targets += [release, release, over_box]     # lower, release, retreat

        q = self.plan[carry_i][1]
        for k, pos in enumerate(targets):
            idx = carry_i + k
            label, _, _, steps = self.plan[idx]
            q_next, _, _, _ = IK.solve_ik(env.model, env.data, SIDE, pos, self.R, q_init=q)
            self.plan[idx] = (label, q.copy(), q_next.copy(), steps)
            q = q_next

    def act(self, env=None) -> np.ndarray:
        """Normalized (7,) action for the current step."""
        env = env or self.env
        if not self._replanned and self.t == self._carry_start:
            self._replan_delivery()
            self._replanned = True
        t = self.t
        for label, q0, q1, steps in self.plan:
            if t < steps:
                frac = (t + 1) / steps
                q = q0 + (q1 - q0) * frac
                closure = self._closure(label, frac)
                self.t += 1
                return np.concatenate([env.norm_arm(q), [closure * 2.0 - 1.0]])
            t -= steps
        self.t += 1
        q = self.plan[-1][2]
        return np.concatenate([env.norm_arm(q), [-1.0]])

    @staticmethod
    def _closure(label: str, frac: float) -> float:
        if label == "close":
            return float(np.clip(frac * 1.4, 0.0, 1.0))    # ramp shut, then hold
        if label in CLOSE_PHASES:
            return 1.0
        if label == "release":
            return float(max(0.0, 1.0 - frac * 1.6))       # ramp open over the tray
        return 0.0

    @property
    def horizon(self) -> int:
        return sum(steps for _, steps in SCHEDULE)
