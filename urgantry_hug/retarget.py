"""MANO grasp -> Wuji grasp. See RETARGETING_SPEC.md.

The palm pose comes from a rigid fit of the Wuji finger bases onto MANO's wrist
and MCP landmarks; the finger angles come from damped least-squares position
fitting of the Wuji joints onto MANO's landmarks, in the palm frame.

Fitting positions rather than mapping angles avoids `grasp.pose` entirely, and
with it the index-ordering trap and the 3-vs-4 joint mismatch.
"""

from __future__ import annotations

import numpy as np
import mujoco

from .interface import (FINGERS, FINGER_JOINTS, FINGER_NAMES, MANO_LANDMARKS,
                        MANO_WRIST_LANDMARK, HandTarget, ManoGrasp, WujiGrasp)

# MANO landmark index within a finger, for the Wuji link it corresponds to.
# link1/link2 are 4.6 mm apart -- both sit at the MCP -- so the fitted points
# start at link3.
_LINK_TO_MANO = {3: 1, 4: 2}      # link3 -> PIP, link4 -> DIP
_TIP_MANO = 3

# The fingertip is mesh geometry ~29 mm beyond the link4 origin, so link4 is not
# a like-for-like match against a MANO tip landmark.
# Tip-dominant because the tips are what touch the object, but the PIP/DIP terms
# carry real weight: dropping them entirely leaves the redundant joints
# unconstrained and makes the fit worse (1.6 cm tip error, against 0.2 cm here).
_FIT_WEIGHTS = {"link3": 0.2, "link4": 0.5, "tip": 3.0}

PREGRASP_FRACTION = 0.3


def retarget(grasp: ManoGrasp, model: mujoco.MjModel,
             side: str = "right") -> WujiGrasp:
    if side != "right":
        raise NotImplementedError(
            "HUG predicts right hands, and the scene's right hand is rolled 180 "
            "deg about the flange relative to the left (HAND_ROLL), so a left "
            "target needs a mirrored grasp, not a renamed prefix.")

    scratch = mujoco.MjData(model)
    _zero_fingers(model, scratch, side)
    mujoco.mj_kinematics(model, scratch)

    T_world_palm = _fit_palm_pose(model, scratch, side, grasp)
    angles = _fit_finger_angles(model, scratch, side, grasp, T_world_palm)

    return WujiGrasp(
        T_world_palm=T_world_palm,
        hand=HandTarget(angles),
        side=side,
        pregrasp=HandTarget(_pregrasp_angles(model, side, angles)),
    )


def _joint_ids(model: mujoco.MjModel, side: str):
    qadr, dofs, lo, hi = [], [], [], []
    for f in FINGERS:
        for j in FINGER_JOINTS:
            jt = model.joint(f"{side}_hand_finger{f}_joint{j}")
            qadr.append(jt.qposadr[0])
            dofs.append(jt.dofadr[0])
            lo.append(jt.range[0])
            hi.append(jt.range[1])
    return np.array(qadr), np.array(dofs), np.array(lo), np.array(hi)


def _zero_fingers(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> None:
    qadr, _, _, _ = _joint_ids(model, side)
    data.qpos[qadr] = 0.0


def _fingertip_offsets(model: mujoco.MjModel, side: str) -> dict[int, np.ndarray]:
    """Fingertip in each link4 body frame: the farthest mesh vertex from its origin."""
    out = {}
    for f in FINGERS:
        bid = model.body(f"{side}_hand_finger{f}_link4").id
        best = np.zeros(3)
        for gi in range(model.ngeom):
            if model.geom_bodyid[gi] != bid or model.geom_dataid[gi] < 0:
                continue
            did = model.geom_dataid[gi]
            verts = model.mesh_vert[
                model.mesh_vertadr[did]:model.mesh_vertadr[did] + model.mesh_vertnum[did]
            ].reshape(-1, 3)
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.geom_quat[gi])
            local = verts @ rot.reshape(3, 3).T + model.geom_pos[gi]
            far = local[np.argmax(np.linalg.norm(local, axis=1))]
            if np.linalg.norm(far) > np.linalg.norm(best):
                best = far
        out[f] = best
    return out


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform taking `src` onto `dst`, both (N, 3)."""
    sc, dc = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - sc).T @ (dst - dc))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return R, dc - R @ sc


def _fit_palm_pose(model, scratch, side: str, grasp: ManoGrasp) -> np.ndarray:
    """Place the palm by aligning the Wuji finger bases with MANO's wrist and MCPs."""
    palm = scratch.body(f"{side}_hand_palm_link")
    R_palm = palm.xmat.reshape(3, 3)

    src = [np.zeros(3)]
    dst = [grasp.landmarks[MANO_WRIST_LANDMARK]]
    for f in FINGERS:
        base = scratch.body(f"{side}_hand_finger{f}_link1")
        src.append(R_palm.T @ (base.xpos - palm.xpos))
        dst.append(grasp.landmarks[MANO_LANDMARKS[FINGER_NAMES[f]][0]])

    R, t = _kabsch(np.array(src), np.array(dst))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _fit_finger_angles(model, scratch, side: str, grasp: ManoGrasp,
                       T_world_palm: np.ndarray, iters: int = 200,
                       damping: float = 0.05, reg: float = 0.001,
                       max_step: float = 0.2) -> np.ndarray:
    """Damped least-squares fit of the Wuji joints onto MANO's landmarks.

    Runs in the palm frame: finger joints never move the palm body, so the
    palm rotation is constant and world Jacobians rotate into it directly.
    """
    qadr, dofs, lo, hi = _joint_ids(model, side)
    tips = _fingertip_offsets(model, side)

    R_world_palm = T_world_palm[:3, :3]
    to_palm = lambda p: R_world_palm.T @ (p - T_world_palm[:3, 3])

    targets, weights = [], []
    for f in FINGERS:
        marks = MANO_LANDMARKS[FINGER_NAMES[f]]
        for link, mano_i in _LINK_TO_MANO.items():
            targets.append(to_palm(grasp.landmarks[marks[mano_i]]))
            weights.append(_FIT_WEIGHTS[f"link{link}"])
        targets.append(to_palm(grasp.landmarks[marks[_TIP_MANO]]))
        weights.append(_FIT_WEIGHTS["tip"])
    targets = np.array(targets)
    weights = np.repeat(weights, 3)

    palm = scratch.body(f"{side}_hand_palm_link")
    q = np.zeros(len(qadr))
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(iters):
        scratch.qpos[qadr] = q
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        R_palm = palm.xmat.reshape(3, 3)
        palm_pos = palm.xpos

        current, rows = [], []
        for f in FINGERS:
            for link in _LINK_TO_MANO:
                body = scratch.body(f"{side}_hand_finger{f}_link{link}")
                bid = model.body(f"{side}_hand_finger{f}_link{link}").id
                current.append(R_palm.T @ (body.xpos - palm_pos))
                mujoco.mj_jac(model, scratch, jacp, jacr, body.xpos, bid)
                rows.append(R_palm.T @ jacp[:, dofs])
            body = scratch.body(f"{side}_hand_finger{f}_link4")
            bid = model.body(f"{side}_hand_finger{f}_link4").id
            tip_world = body.xpos + body.xmat.reshape(3, 3) @ tips[f]
            current.append(R_palm.T @ (tip_world - palm_pos))
            mujoco.mj_jac(model, scratch, jacp, jacr, tip_world, bid)
            rows.append(R_palm.T @ jacp[:, dofs])

        err = ((targets - np.array(current)).reshape(-1)) * weights
        J = np.vstack(rows) * weights[:, None]

        # Regularizing toward the open hand keeps the redundant joints out of the
        # flat basins where DLS otherwise flips between equal-cost postures.
        JtJ = J.T @ J + (damping ** 2 + reg) * np.eye(len(q))
        step = np.linalg.solve(JtJ, J.T @ err - reg * q)
        q = np.clip(q + np.clip(step, -max_step, max_step), lo, hi)

    return q.reshape(5, 4)


def _pregrasp_angles(model: mujoco.MjModel, side: str,
                     angles: np.ndarray) -> np.ndarray:
    """Partly-curled approach posture.

    A flat-open hand reaches ~4.5 cm past the grasp center and sweeps the object
    off the table before anything closes; keeping abduction while backing off the
    flexion shrinks the swept volume without pre-closing on the object.
    """
    pre = angles * PREGRASP_FRACTION
    pre[:, 1] = angles[:, 1]
    return pre
