"""Interactive Tk tool to tune the wrist-camera mount and camera poses.

Opens three things at once:
  * a Tk window of sliders for the pose (pos + euler) of the wrist-camera MOUNT
    (local to the gripper base_mount) and the wrist CAMERA (local to the mount);
  * the MuJoCo passive viewer, so you can orbit the arm and watch the bracket move;
  * an OpenCV window showing the live left/right wrist-camera renders.

Poses are applied live to BOTH arms (they share the same local mount/cam pose, as
in build_urtable.py). Hit "Save YAML" to dump the current poses to a yaml file;
each entry has pos + quat (wxyz, ready to paste into an XML <body>) plus euler_deg
for readability.

Run:  uv run python tune_wrist_cam.py
"""

from pathlib import Path

import cv2
import numpy as np
import mujoco
import mujoco.viewer
import tkinter as tk

from build_urtable import build_scene

# Bodies we tune. Each is (label, [left_name, right_name]); left/right share pose.
MOUNT_BODIES = ["left_grip_wrist_mount", "right_grip_wrist_mount"]
CAM_BODIES = ["left_grip_wrist_cam", "right_grip_wrist_cam"]
WRIST_CAMERAS = ["left_grip_wrist", "right_grip_wrist"]

OUT_YAML = Path(__file__).with_name("wrist_cam_pose.yaml")
PREVIEW_W, PREVIEW_H = 420, 340


# ------------------------------------------------------------ math helpers
def _R_from_euler(rx, ry, rz):
    """Rotation matrix from intrinsic X-Y-Z euler angles (degrees): R = Rz@Ry@Rx."""
    rx, ry, rz = np.radians([rx, ry, rz])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _euler_from_R(R):
    """Intrinsic X-Y-Z euler angles (degrees) from a rotation matrix (inverse of above)."""
    sy = -R[2, 0]
    ry = np.arcsin(np.clip(sy, -1.0, 1.0))
    if abs(sy) < 0.99999:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:                                    # gimbal lock
        rx = np.arctan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return np.degrees([rx, ry, rz])


def _quat_from_euler(rx, ry, rz):
    """MuJoCo quat (wxyz) from intrinsic X-Y-Z euler angles (degrees)."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, _R_from_euler(rx, ry, rz).flatten())
    return quat


def _euler_from_quat(quat):
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, float))
    return _euler_from_R(mat.reshape(3, 3))


# ----------------------------------------------------------------- the app
class WristCamTuner:
    def __init__(self):
        self.model, self.data = build_scene()
        mujoco.mj_forward(self.model, self.data)

        # Read current poses (from the first arm) as slider start values.
        mid = self.model.body(MOUNT_BODIES[0]).id
        cid = self.model.body(CAM_BODIES[0]).id
        self.mount_id = [self.model.body(n).id for n in MOUNT_BODIES]
        self.cam_id = [self.model.body(n).id for n in CAM_BODIES]
        mount_pos = self.model.body_pos[mid].copy()
        mount_eul = _euler_from_quat(self.model.body_quat[mid])
        cam_pos = self.model.body_pos[cid].copy()
        cam_eul = _euler_from_quat(self.model.body_quat[cid])

        self.renderer = mujoco.Renderer(self.model, height=PREVIEW_H, width=PREVIEW_W)
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False)

        self._build_ui(mount_pos, mount_eul, cam_pos, cam_eul)
        self._apply()                        # push initial values
        self._tick()                         # start the live loop

    # --- UI ---------------------------------------------------------------
    def _build_ui(self, mount_pos, mount_eul, cam_pos, cam_eul):
        self.root = tk.Tk()
        self.root.title("Wrist camera pose tuner")
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.vars = {}

        def group(title, keys_specs):
            frame = tk.LabelFrame(self.root, text=title, padx=8, pady=4)
            frame.pack(fill="x", padx=8, pady=6)
            for key, label, lo, hi, res, init in keys_specs:
                row = tk.Frame(frame)
                row.pack(fill="x")
                tk.Label(row, text=label, width=8, anchor="w").pack(side="left")
                var = tk.DoubleVar(value=float(init))
                self.vars[key] = var
                tk.Scale(row, variable=var, from_=lo, to=hi, resolution=res,
                         orient="horizontal", length=320,
                         command=lambda _v: self._apply()).pack(side="left", fill="x")

        group("Mount position (m, in gripper base_mount frame)", [
            ("mx", "x", -0.15, 0.15, 0.001, mount_pos[0]),
            ("my", "y", -0.15, 0.15, 0.001, mount_pos[1]),
            ("mz", "z", -0.15, 0.15, 0.001, mount_pos[2]),
        ])
        group("Mount rotation (deg, intrinsic XYZ)", [
            ("mrx", "roll x", -180, 180, 1, mount_eul[0]),
            ("mry", "pitch y", -180, 180, 1, mount_eul[1]),
            ("mrz", "yaw z", -180, 180, 1, mount_eul[2]),
        ])
        group("Camera position (m, in mount frame)", [
            ("cx", "x", -0.05, 0.05, 0.0005, cam_pos[0]),
            ("cy", "y", -0.05, 0.05, 0.0005, cam_pos[1]),
            ("cz", "z", -0.05, 0.05, 0.0005, cam_pos[2]),
        ])
        group("Camera rotation (deg, intrinsic XYZ)", [
            ("crx", "roll x", -180, 180, 1, cam_eul[0]),
            ("cry", "pitch y", -180, 180, 1, cam_eul[1]),
            ("crz", "yaw z", -180, 180, 1, cam_eul[2]),
        ])

        btns = tk.Frame(self.root)
        btns.pack(fill="x", padx=8, pady=8)
        tk.Button(btns, text="Save YAML", command=self._save).pack(side="left")
        self.status = tk.Label(btns, text="", anchor="w")
        self.status.pack(side="left", padx=10)

    # --- live update ------------------------------------------------------
    def _current(self):
        v = {k: var.get() for k, var in self.vars.items()}
        mount_pos = np.array([v["mx"], v["my"], v["mz"]])
        mount_quat = _quat_from_euler(v["mrx"], v["mry"], v["mrz"])
        cam_pos = np.array([v["cx"], v["cy"], v["cz"]])
        cam_quat = _quat_from_euler(v["crx"], v["cry"], v["crz"])
        return mount_pos, mount_quat, cam_pos, cam_quat

    def _apply(self):
        mount_pos, mount_quat, cam_pos, cam_quat = self._current()
        for bid in self.mount_id:
            self.model.body_pos[bid] = mount_pos
            self.model.body_quat[bid] = mount_quat
        for bid in self.cam_id:
            self.model.body_pos[bid] = cam_pos
            self.model.body_quat[bid] = cam_quat
        mujoco.mj_forward(self.model, self.data)

    def _tick(self):
        if self.viewer.is_running():
            self.viewer.sync()
            frames = []
            for cam in WRIST_CAMERAS:
                self.renderer.update_scene(self.data, camera=cam)
                frames.append(self.renderer.render())
            combined = np.hstack(frames)[..., ::-1]      # RGB -> BGR
            cv2.imshow("wrist cameras (left | right)", combined)
            cv2.waitKey(1)
            self.root.after(33, self._tick)
        else:
            self._close()

    # --- save -------------------------------------------------------------
    def _save(self):
        mount_pos, mount_quat, cam_pos, cam_quat = self._current()
        mount_eul = [self.vars[k].get() for k in ("mrx", "mry", "mrz")]
        cam_eul = [self.vars[k].get() for k in ("crx", "cry", "crz")]

        def block(name, pos, quat, eul):
            f = lambda a: "[" + ", ".join(f"{x:.6g}" for x in a) + "]"
            return (f"{name}:\n"
                    f"  pos: {f(pos)}\n"
                    f"  quat: {f(quat)}   # wxyz, paste into <body quat=...>\n"
                    f"  euler_deg: {f(eul)}   # intrinsic XYZ\n")

        text = ("# Wrist-camera poses tuned with tune_wrist_cam.py\n"
                "# Local frames: mount is in the gripper base_mount frame,\n"
                "# camera is in the mount frame. Applies to both arms.\n"
                + block("wrist_mount", mount_pos, mount_quat, mount_eul)
                + block("wrist_cam", cam_pos, cam_quat, cam_eul))
        OUT_YAML.write_text(text)
        msg = f"saved {OUT_YAML.name}"
        self.status.config(text=msg)
        print("\n" + text)

    def _close(self):
        try:
            self.viewer.close()
        except Exception:
            pass
        self.renderer.close()
        cv2.destroyAllWindows()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    WristCamTuner().run()
