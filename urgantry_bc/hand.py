"""Wuji hand posture helpers.

The scene's `set_hand` drives every joint of a hand to the same angle, which
saturates the abduction joints long before the flexion joints have closed, and
its "closed" value of 1.2 rad is applied to joints whose useful travel differs
per axis. Grasping needs them treated separately.

Read off the compiled model (the XML's joint ranges are asymmetric, so signs are
not guessable): flexion is joint1/joint3/joint4, POSITIVE angles, curling the
fingertips toward palm +x -- that is the grasping side. joint2 is abduction and
stays at 0. At full close the fingertips travel from (x=-0.015, z=0.16) to
(x=+0.06, z=0.09) and the thumb crosses from y=+0.096 to y=0.
"""

from __future__ import annotations

import numpy as np
import mujoco

from urgantry_sim.build_urgantry import HAND_FINGERS

# Flexion angle at full close, per joint, in radians. Inside each joint's own
# range (joint1 up to 1.64, joint3/joint4 up to 1.63) so a full close never fights
# a joint limit.
# The distal joints (3, 4) doing most of the closing is what actually secures a
# 5 cm cube: at 1.0 rad the fingers cage it but it shakes loose while the arm
# traverses; at 1.5 they wrap under it.
FLEX_FULL = {1: 1.30, 3: 1.50, 4: 1.50}
THUMB_FLEX_FULL = {1: 1.30, 3: 1.50, 4: 1.50}
ABDUCT_JOINT = 2


def hand_ctrl(model: mujoco.MjModel, side: str, amount: float) -> dict[int, float]:
    """{actuator id: ctrl} closing one hand by `amount` in [0, 1] (0 = flat open,
    1 = fully curled)."""
    amount = float(np.clip(amount, 0.0, 1.0))
    out = {}
    for f in HAND_FINGERS:
        table = THUMB_FLEX_FULL if f == 1 else FLEX_FULL
        for j, full in table.items():
            act = model.actuator(f"{side}_hand_finger{f}_joint{j}")
            lo, hi = act.ctrlrange
            out[act.id] = float(np.clip(full * amount, lo, hi))
        act = model.actuator(f"{side}_hand_finger{f}_joint{ABDUCT_JOINT}")
        out[act.id] = 0.0
    return out


def set_hand_closure(model: mujoco.MjModel, data: mujoco.MjData, side: str,
                     amount: float) -> None:
    """Command one hand to a closure in [0, 1] (ctrl only; does not step)."""
    for aid, val in hand_ctrl(model, side, amount).items():
        data.ctrl[aid] = val


def hand_actuator_ids(model: mujoco.MjModel, side: str) -> list[int]:
    return sorted(hand_ctrl(model, side, 0.0))
