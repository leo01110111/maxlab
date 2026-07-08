"""Interactive tuner for the wrist-camera bracket + camera poses.

Opens an orbitable 3D view of the scene plus a window of sliders that move the
`wrist_mount` (bracket, relative to the Robotiq `base` frame) and the `wrist_cam`
(camera, relative to the bracket frame).  A readout window shows the live values;
press `p` to dump paste-ready XML to the terminal, `r` to reset to the values the
model loaded with, `q`/Esc to quit.

Run from the sim/ directory:  python tune_wrist_cam.py

The xyz/quat shown are exactly what go into <body name="..." pos="..." quat="...">
in gripper/robotiq-2f85.xml — body_pos / body_quat are stored relative to the
parent body, which is precisely how MuJoCo reads those XML attributes back.
"""
import os
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

import build_urtable as b

os.chdir(Path(__file__).parent)        # so the gripper meshdir resolves

ARM = "left_"                          # tune on this arm (both arms share the XML)
RENDER_W, RENDER_H = 640, 400          # wrist-camera preview size

# Slider encoding: trackbars are integer + start at 0, so we offset/scale.
POS_RANGE_MM = 300                     # +/- 0.3 m, 1 mm steps
RPY_RANGE_DEG = 180                    # +/- 180 deg, 1 deg steps

# The two bodies we tune and the frame each is expressed in (for the readout).
TARGETS = {
    "mount": (f"{ARM}grip_wrist_mount", "base frame"),
    "cam":   (f"{ARM}grip_wrist_cam",   "mount frame"),
}


def rpy_to_quat(rpy):
    """Standard roll-pitch-yaw (R = Rz(yaw) @ Ry(pitch) @ Rx(roll)) -> quat [w,x,y,z]."""
    r, p, y = rpy
    cr, cp, cy = np.cos([r, p, y])
    sr, sp, sy = np.sin([r, p, y])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def quat_to_rpy(q):
    """Inverse of rpy_to_quat -> (roll, pitch, yaw) in radians."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    R = R.reshape(3, 3)
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def make_sliders(win, model):
    """Create 6 sliders (x y z roll pitch yaw) per target, seeded from the model."""
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 460, 560)
    for key, (bname, _) in TARGETS.items():
        bid = model.body(bname).id
        xyz = model.body_pos[bid]
        rpy = np.rad2deg(quat_to_rpy(model.body_quat[bid]))
        for axis, v in zip("xyz", xyz):
            s = int(round(v * 1000)) + POS_RANGE_MM
            cv2.createTrackbar(f"{key}.{axis}[mm]", win, int(np.clip(s, 0, 2 * POS_RANGE_MM)),
                               2 * POS_RANGE_MM, lambda _v: None)
        for axis, v in zip(("roll", "pitch", "yaw"), rpy):
            s = int(round(v)) + RPY_RANGE_DEG
            cv2.createTrackbar(f"{key}.{axis}[deg]", win, int(np.clip(s, 0, 2 * RPY_RANGE_DEG)),
                               2 * RPY_RANGE_DEG, lambda _v: None)


def read_sliders(win):
    """Return {key: (xyz_m (3,), rpy_deg (3,))} from the current slider positions."""
    out = {}
    for key in TARGETS:
        xyz = np.array([cv2.getTrackbarPos(f"{key}.{a}[mm]", win) - POS_RANGE_MM
                        for a in "xyz"]) / 1000.0
        rpy = np.array([cv2.getTrackbarPos(f"{key}.{a}[deg]", win) - RPY_RANGE_DEG
                        for a in ("roll", "pitch", "yaw")], float)
        out[key] = (xyz, rpy)
    return out


def apply(model, vals):
    """Write slider values onto BOTH arms' bodies so the viewer stays symmetric."""
    for key, (bname, _) in TARGETS.items():
        xyz, rpy = vals[key]
        quat = rpy_to_quat(np.deg2rad(rpy))
        for arm in ("left_", "right_"):
            bid = model.body(bname.replace(ARM, arm, 1)).id
            model.body_pos[bid] = xyz
            model.body_quat[bid] = quat


def fmt_xml(vals):
    lines = []
    for key, (bname, frame) in TARGETS.items():
        xyz, rpy = vals[key]
        quat = rpy_to_quat(np.deg2rad(rpy))
        name = bname.replace(ARM, "", 1)       # strip the per-arm prefix for the XML
        p = " ".join(f"{v:.5g}" for v in xyz)
        q = " ".join(f"{v:.5f}" for v in quat)
        lines.append(f'  <body name="{name}" pos="{p}" quat="{q}">   '
                     f'<!-- {frame}; rpy(deg)={rpy[0]:.1f} {rpy[1]:.1f} {rpy[2]:.1f} -->')
    return "\n".join(lines)


def readout_canvas(vals):
    canvas = np.zeros((220, 760, 3), np.uint8)
    y = 26
    cv2.putText(canvas, "wrist-cam tuner   [p]=print XML  [r]=reset  [q]=quit",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    for key, (_, frame) in TARGETS.items():
        xyz, rpy = vals[key]
        quat = rpy_to_quat(np.deg2rad(rpy))
        y += 34
        cv2.putText(canvas, f"{key} ({frame})", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 220, 120), 1, cv2.LINE_AA)
        y += 26
        cv2.putText(canvas, f"  pos  {xyz[0]:+.4f} {xyz[1]:+.4f} {xyz[2]:+.4f}   "
                    f"rpy  {rpy[0]:+.1f} {rpy[1]:+.1f} {rpy[2]:+.1f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        y += 24
        cv2.putText(canvas, "  quat " + " ".join(f"{v:+.4f}" for v in quat),
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def main():
    model = b.build_spec().compile()
    data = mujoco.MjData(model)
    b.set_initial_pose(model, data)

    win = "controls"
    make_sliders(win, model)
    initial = read_sliders(win)            # remember load-time values for reset

    renderer = mujoco.Renderer(model, RENDER_H, RENDER_W)
    cam = f"{ARM}grip_wrist"

    print("Tuning", ARM, "wrist camera. Orbit the 3D view; drag sliders; press 'p' to print XML.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            vals = read_sliders(win)
            apply(model, vals)
            mujoco.mj_forward(model, data)
            viewer.sync()

            renderer.update_scene(data, camera=cam)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.imshow("wrist view", frame)
            cv2.imshow("readout", readout_canvas(vals))

            k = cv2.waitKey(16) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("p"):
                print("\n--- paste into gripper/robotiq-2f85.xml ---")
                print(fmt_xml(vals))
                print("-------------------------------------------\n", flush=True)
            if k == ord("r"):
                for key in TARGETS:
                    for a, v in zip("xyz", initial[key][0]):
                        cv2.setTrackbarPos(f"{key}.{a}[mm]", win,
                                           int(round(v * 1000)) + POS_RANGE_MM)
                    for a, v in zip(("roll", "pitch", "yaw"), initial[key][1]):
                        cv2.setTrackbarPos(f"{key}.{a}[deg]", win,
                                           int(round(v)) + RPY_RANGE_DEG)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
