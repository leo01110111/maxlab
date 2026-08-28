"""Damped least-squares IK for one UR7e arm of the gantry scene.

Solves for the 6 arm joints that put a hand-frame point (the grasp center between
the fingers) at a target position with a target palm orientation. Works on a
scratch MjData so the live sim is never disturbed, and warm-starts from the
current pose -- so it descends to the local branch, not the globally shortest one.
"""

from __future__ import annotations

import numpy as np
import mujoco

from urgantry_sim.build_urgantry import ARM_JOINTS

# Grasp center in the palm frame: the middle of the volume the fingers sweep as
# they close. Measured on the compiled model -- fingertips travel from
# (x=-0.015, z=0.16) open to (x=+0.06, z=0.09) closed, and the thumb crosses from
# y=+0.096 to y=0. The fingers curl toward palm +x, so that is the grasping side
# (the hand MJCF's joint ranges are asymmetric: flexion is POSITIVE).
GRASP_CENTER = np.array([0.035, 0.010, 0.115])


def arm_indices(model: mujoco.MjModel, side: str) -> tuple[np.ndarray, np.ndarray]:
    """(qpos addresses, dof indices) of one arm's 6 joints, in ARM_JOINTS order."""
    qadr, dofs = [], []
    for name in ARM_JOINTS:
        j = model.joint(f"{side}_{name}_joint")
        qadr.append(j.qposadr[0])
        dofs.append(j.dofadr[0])
    return np.array(qadr), np.array(dofs)


def arm_limits(model: mujoco.MjModel, side: str) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = [], []
    for name in ARM_JOINTS:
        j = model.joint(f"{side}_{name}_joint")
        if j.limited:
            lo.append(j.range[0])
            hi.append(j.range[1])
        else:
            lo.append(-2 * np.pi)
            hi.append(2 * np.pi)
    return np.array(lo), np.array(hi)


def grasp_frame(model: mujoco.MjModel, data: mujoco.MjData, side: str
                ) -> tuple[np.ndarray, np.ndarray]:
    """World (position, rotation) of the grasp center / palm frame."""
    palm = data.body(f"{side}_hand_palm_link")
    R = palm.xmat.reshape(3, 3)
    return palm.xpos + R @ GRASP_CENTER, R


def palm_target_mat(finger_dir, grasp_dir=(0.0, 0.0, -1.0)) -> np.ndarray:
    """Palm rotation whose +z (the direction the fingers extend) is `finger_dir`
    and whose +x (the grasping side the fingers curl toward) is as close to
    `grasp_dir` as the first axis allows. Default: fingers close downward, for
    picking off the table."""
    z = np.asarray(finger_dir, float)
    z /= np.linalg.norm(z)
    want = np.asarray(grasp_dir, float)
    x = want - np.dot(want, z) * z
    if np.linalg.norm(x) < 1e-8:                   # fingers point along grasp_dir
        alt = np.array([1.0, 0.0, 0.0])
        x = alt - np.dot(alt, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def solve_ik(model: mujoco.MjModel, data: mujoco.MjData, side: str,
             target_pos, target_mat=None, *, q_init=None, iters: int = 200,
             damping: float = 0.12, pos_tol: float = 1.5e-3, rot_tol: float = 0.02,
             rot_weight: float = 0.5, max_step: float = 0.25):
    """Solve for the arm joints putting the grasp center at `target_pos` (and the
    palm at `target_mat`, if given). Returns (q, pos_err, rot_err, converged).

    Runs on a scratch copy of `data`; the caller decides whether to apply q.
    """
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    qadr, dofs = arm_indices(model, side)
    lo, hi = arm_limits(model, side)
    q = np.array(data.qpos[qadr] if q_init is None else q_init, float)

    target_pos = np.asarray(target_pos, float)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    bid = model.body(f"{side}_hand_palm_link").id

    pos_err = rot_err = np.inf
    for _ in range(iters):
        scratch.qpos[qadr] = q
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)

        R = scratch.xmat[bid].reshape(3, 3)
        cur = scratch.xpos[bid] + R @ GRASP_CENTER
        err_p = target_pos - cur
        pos_err = float(np.linalg.norm(err_p))

        if target_mat is None:
            err = err_p
            rot_err = 0.0
        else:
            q_cur, q_tgt, q_del, vel = (np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3))
            mujoco.mju_mat2Quat(q_cur, R.flatten())
            mujoco.mju_mat2Quat(q_tgt, np.asarray(target_mat, float).flatten())
            mujoco.mju_negQuat(q_cur, q_cur)
            mujoco.mju_mulQuat(q_del, q_tgt, q_cur)
            mujoco.mju_quat2Vel(vel, q_del, 1.0)
            rot_err = float(np.linalg.norm(vel))
            err = np.concatenate([err_p, rot_weight * vel])

        if pos_err < pos_tol and rot_err < rot_tol:
            return q, pos_err, rot_err, True

        # Jacobian of the grasp center (a point offset in the palm body).
        mujoco.mj_jac(model, scratch, jacp, jacr, cur, bid)
        J = jacp[:, dofs] if target_mat is None else np.vstack(
            [jacp[:, dofs], rot_weight * jacr[:, dofs]])
        JJt = J @ J.T + damping ** 2 * np.eye(J.shape[0])
        step = J.T @ np.linalg.solve(JJt, err)
        q = np.clip(q + np.clip(step, -max_step, max_step), lo, hi)

    return q, pos_err, rot_err, bool(pos_err < pos_tol and rot_err < rot_tol)
