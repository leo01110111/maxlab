import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from build_urgantry_hands import (
    HAND_CURL_CLOSED,
    HAND_OPEN,
    apply_initial_view,
    build_scene,
    capture_state,
    set_hand,
)

CAMERAS = ["top1", "left_hand_wrist", "right_hand_wrist"]


def main():
    model, data = build_scene()
    renderer = mujoco.Renderer(model, height=300, width=450)

    # Each camera's user[0] (set in its MJCF, e.g. gripper/arducam_ov9782.xml)
    # is its target capture fps; cameras without one update every sim step.
    # Track a per-camera next-due wall-clock time and reuse the last frame
    # in between, so e.g. the 15fps wrist cams don't render at the ~500Hz
    # sim step rate.
    cam_fps = {}
    for cam in CAMERAS:
        cid = model.camera(cam).id
        cam_fps[cam] = float(model.cam_user[cid][0]) if model.nuser_cam else 0.0
    next_due = {cam: 0.0 for cam in CAMERAS}
    last_frame = {cam: None for cam in CAMERAS}

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        apply_initial_view(viewer)   # open at the hard-coded INITIAL_VIEW
        viewer.sync()
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)
            viewer.sync()

            frames = []
            for cam in CAMERAS:
                fps = cam_fps[cam]
                if fps > 0 and step_start < next_due[cam] and last_frame[cam] is not None:
                    frames.append(last_frame[cam])
                    continue
                renderer.update_scene(data, camera=cam)
                frame = renderer.render()             # (300, 450, 3) uint8 RGB
                last_frame[cam] = frame
                if fps > 0:
                    next_due[cam] = step_start + 1.0 / fps
                frames.append(frame)

            combined = np.hstack(frames)             # side by side
            cv2.imshow("cameras", combined[..., ::-1])  # RGB -> BGR for OpenCV
            # Controls (focus the OpenCV window): 'o'/'c' = open / close both
            # hands, '[' / ']' = same for the left / right hand alone,
            # 's' = print current pose+view as paste-ready constants, 'q' = quit.
            key = cv2.waitKey(1) & 0xFF              # 1 ms wait
            if key == ord("o"):
                for side in ("left", "right"):
                    set_hand(model, data, side, HAND_OPEN)
            elif key == ord("c"):
                for side in ("left", "right"):
                    set_hand(model, data, side, HAND_CURL_CLOSED)
            elif key == ord("["):
                set_hand(model, data, "left", _toggled(model, data, "left"))
            elif key == ord("]"):
                set_hand(model, data, "right", _toggled(model, data, "right"))
            elif key == ord("s"):
                capture_state(data, viewer)
            elif key == ord("q"):
                break

            # Keep the loop roughly real-time.
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)

    renderer.close()
    cv2.destroyAllWindows()


def _toggled(model, data, side):
    """Whichever of HAND_OPEN / HAND_CURL_CLOSED this hand is not currently
    commanded to, judged from its first finger's curl command."""
    curl = data.ctrl[model.actuator(f"{side}_hand_finger1_joint2").id]
    midpoint = (HAND_OPEN + HAND_CURL_CLOSED) / 2
    return HAND_OPEN if curl > midpoint else HAND_CURL_CLOSED


if __name__ == "__main__":
    main()
