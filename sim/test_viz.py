import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from build_urtable import apply_initial_view, build_scene, capture_state

CAMERAS = ["top1", "top2"]


def main():
    model, data = build_scene()
    renderer = mujoco.Renderer(model, height=300, width=450)
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
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render())     # (300, 450, 3) uint8 RGB

            combined = np.hstack(frames)             # side by side
            cv2.imshow("cameras", combined[..., ::-1])  # RGB -> BGR for OpenCV
            # Controls (focus the OpenCV window): 's' = print current pose+view
            # as paste-ready constants; 'q' = quit.
            key = cv2.waitKey(1) & 0xFF              # 1 ms wait
            if key == ord("s"):
                capture_state(data, viewer)
            elif key == ord("q"):
                break

            # Keep the loop roughly real-time.
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)

    renderer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
