"""Build the bimanual UR7e table scene with the MuJoCo spec (mjSpec) API,
matching the real lab setup measured in setup_specs.md. Defines reward functions
used in env.py

NOTE FOR FUTURE AGENTS: the arm is a **UR7e**. The MJCF still lives at
`universal_robots_ur5e/ur5e.xml` (and the menagerie dir keeps its UR5e name)
because it started as the menagerie UR5e and is being re-tuned in place to UR7e
physics. Treat the robot as a UR7e everywhere; only the on-disk file paths keep
the legacy "ur5e" name. See models/universal_robots_ur5e/README.md.

Pure-XML <attach> of an <include>d robot doesn't work (include merges the robot's
bodies into the worldbody; attach needs the robot kept as a separate child model),
so we load the arm MJCF once per arm and attach a prefixed copy.

Coordinate frame: origin at the center of the table footprint on the floor.
  +x = right, +y = back (away from cameras), +z = up.   Units: meters.

Run directly to open the interactive viewer:  uv run python build_urtable.py
"""

from pathlib import Path

import numpy as np
import mujoco

# Resolved relative to this file (not the process cwd) so the package works
# whether it's run in-place (`cd ur_sim && python test_viz.py`) or installed and
# imported from anywhere (`import ur_sim`).
ASSET_DIR = Path(__file__).resolve().parent

# UR7e arm MJCF. Path keeps the legacy "ur5e" name (see module docstring); the
# physics in that file is being changed to a UR7e.
UR7E_PATH = str(ASSET_DIR / "universal_robots_ur5e" / "ur5e.xml")
ROBOTIQ_PATH = str(ASSET_DIR / "gripper" / "robotiq-2f85.xml")
ARDUCAM_PATH = str(ASSET_DIR / "gripper" / "arducam_ov9782.xml")

# Wrist camera (one per arm): an ArduCam OV9782 board on a printed bracket that
# saddles the Robotiq base, looking over the fingers at the grasp. The bracket
# ("wrist_mount") is defined directly in the Robotiq XML; the camera module is
# a separate MjSpec attached onto the bracket's "cam_attach" site (see
# _arm_with_gripper below). Tune the bracket pose with tune_wrist_cam.py. See
# gripper/robotiq-2f85.xml and gripper/arducam_ov9782.xml.

# ---------------------------------------------------------------- measurements
TABLE_H = 0.76          # aluminum frame top height
BOARD_T = 0.015         # black board thickness (on top of the frame)
PLATE_T = 0.006         # blue Vention mounting plate thickness (under each arm)
TABLE_LEN = 1.725       # along x (left-right); cameras lie on this long edge
TABLE_W = 1.14          # along y (front-back)

BOARD_TOP = TABLE_H + BOARD_T          # 0.775
# Blue plates bolt to the aluminum frame (not on top of the board); the board is
# cut around them. So the plate sits on the aluminum at 0.76 and the arm base on it.
ARM_Z = TABLE_H + PLATE_T              # 0.766  (UR7e base height)

HALF_LEN = TABLE_LEN / 2               # 0.8625
HALF_W = TABLE_W / 2                   # 0.57

# Arm base placement: both arms centered in the width (56.5 cm from the front
# edge ~= width/2), separated along the length, 12 cm in from each end edge,
# facing each other across the length.
ARM_FRONT_DIST = 0.565   # from front edge -> centered in the 114 cm width
ARM_END_DIST = 0.12      # in from the left/right end edge
ARM_Y = -HALF_W + ARM_FRONT_DIST                        # ~= -0.005 (centered)
ARM_RIGHT = (HALF_LEN - ARM_END_DIST, ARM_Y)            # (+0.7425, -0.005)
ARM_LEFT = (-(HALF_LEN - ARM_END_DIST), ARM_Y)          # (-0.7425, -0.005)

# Single overhead camera (Intel RealSense D435): centered along the table length
# on the front edge, elevated, tilted slightly down toward the work surface.
CAM_POS = (0.0, -HALF_W + 0.05, BOARD_TOP + 0.675)  # centered in x, front edge, 67.5cm above table top
CAM_DOWN_TILT = np.deg2rad(30.0)                    # 30 deg downward tilt from the table top (horizontal)
CAM_TARGET = (CAM_POS[0],                           # look forward (+y, into the table)
              CAM_POS[1] + np.cos(CAM_DOWN_TILT),   # and slightly down
              CAM_POS[2] - np.sin(CAM_DOWN_TILT))
CAM_FOVY = 42.0                        # D435 color vertical FOV (deg)

# UR7e initial pose (radians): arms reach out over the table (not folded up), flange
# pointing down — matching the photos.  [pan, lift, elbow, w1, w2, w3]
# The two arms face each other: each has its own full home pose so they reach
# toward the table center (+x for the left arm, -x for the right).
LEFT_HOME_POSE = [ 1.5700, -1.5700,  1.5700, -1.5700, -1.5700, -1.5700]
RIGHT_HOME_POSE = [-1.5700, -1.5700, -1.5700, -1.5700,  1.5700,  1.5700]
# Left base yaw: -90deg about z (clockwise from above) so the left arm faces the
# table at LEFT_HOME_POSE. Quat [w, x, y, z] for a rotation of theta about +z.
LEFT_BASE_YAW = np.pi / 2
LEFT_BASE_QUAT = [np.cos(LEFT_BASE_YAW / 2), 0.0, 0.0, np.sin(LEFT_BASE_YAW / 2)]
RIGHT_BASE_YAW = -np.pi / 2
RIGHT_BASE_QUAT = [np.cos(RIGHT_BASE_YAW / 2), 0.0, 0.0, np.sin(RIGHT_BASE_YAW / 2)]
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

# Robotiq 2F-85 gripper: one actuator per arm, ctrl 0 = open, 255 = closed.
GRIPPER_OPEN, GRIPPER_CLOSED = 0.0, 255.0

# --------------------------------------------------------------- pick task
# A graspable block sits on the board; the task is to lift it. Placed within the
# right arm's reach. Success = block lifted LIFT_SUCCESS_H above its rest height.
BLOCK_HALF = 0.025                                   # 5 cm cube
BLOCK_REST_Z = BOARD_TOP + BLOCK_HALF
BLOCK_INIT_POS = (0.45, 0.0, BLOCK_REST_Z)
BLOCK_RGBA = [0.15, 0.75, 0.20, 1]
LIFT_SUCCESS_H = 0.05                                # meters above rest to count as a pick

# --------------------------------------------------------------- cardboard tray
# Open-top box (corrugated cardboard tray) sitting on the board, clear of the
# block's pick site and both arm bases.
BOX_OUTER = 0.16                                     # outer footprint, square
BOX_WALL_T = 0.006
BOX_WALL_H = 0.035
BOX_FLOOR_T = 0.004
BOX_INIT_POS = (-0.35, 0.25, BOARD_TOP + BOX_FLOOR_T / 2)
BOX_RGBA = [0.72, 0.53, 0.34, 1]

# colors
COL_ALU = [0.62, 0.64, 0.66, 1]
COL_PLATE = [0.20, 0.42, 0.78, 1]
COL_LEG = [0.12, 0.13, 0.15, 1]


def _lookat_quat(cam_pos, target):
    """Quaternion orienting a MuJoCo camera (looks down -z) from cam_pos at target."""
    cam_pos, target = np.asarray(cam_pos, float), np.asarray(target, float)
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    # camera axes as columns: x=right, y=up, z=-forward  (row-major 3x3)
    mat = np.array([[right[0], up[0], -forward[0]],
                    [right[1], up[1], -forward[1]],
                    [right[2], up[2], -forward[2]]]).flatten()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)
    return quat


def _arm_with_gripper() -> mujoco.MjSpec:
    """Load a UR7e and bolt a Robotiq 2F-85 onto its wrist attachment site. The
    gripper's coupling (equality constraints + tendon), its single actuator, and
    its wrist camera all come along with the attach, prefixed 'grip_' (final camera
    names: 'left_grip_wrist' / 'right_grip_wrist')."""
    arm = mujoco.MjSpec.from_file(UR7E_PATH)
    gripper = mujoco.MjSpec.from_file(ROBOTIQ_PATH)
    cam = mujoco.MjSpec.from_file(ARDUCAM_PATH)
    # Empty prefix so the camera's names stay "wrist_cam"/"wrist" through this
    # attach, matching what the grip_/left_/right_ prefixes below expect.
    gripper.site("cam_attach").attach_body(cam.body("wrist_cam"), "", "")
    arm.site("attachment_site").attach_body(gripper.body("base_mount"), "grip_", "")
    return arm


def build_spec() -> mujoco.MjSpec:
    """Construct the floor, table, board, plates, two UR7e arms, and cameras."""
    spec = mujoco.MjSpec()
    spec.compiler.autolimits = True
    # Offscreen buffer sized for the OV9782 wrist cameras (1280x800); covers the
    # smaller top-camera / policy renders too.
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 800
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    # Pin the model extent to the working area (table + arms), not the room's
    # bounding box. znear is znear_ratio * extent (default ratio 0.01); left
    # unpinned, the 6x6m floor/walls inflate extent to ~12m, pushing znear past
    # 12cm and clipping the wrist-cam mount/arm links, which sit only a few cm
    # from the lens.
    spec.stat.extent = 1.5

    spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     rgb1=[0.8, 0.8, 0.78], rgb2=[0.8, 0.8, 0.78], width=512, height=3072)

    spec.add_texture(name="carpet", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     file=str(ASSET_DIR / "universal_robots_ur5e" / "assets" / "carpet.png"))
    grid_mat = spec.add_material(name="grid", rgba=[1, 1, 1, 1], reflectance=0.0,
                                  texrepeat=[10, 10], texuniform=True)
    grid_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "carpet"

    spec.add_material(name="wallpaper", rgba=[0.96, 0.96, 0.94, 1], reflectance=0.05)

    spec.add_texture(name="board_plastic", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, mark=mujoco.mjtMark.mjMARK_RANDOM,
                     rgb1=[0, 0, 0], rgb2=[0, 0, 0], markrgb=[0.03, 0.03, 0.03], random=0.02,
                     width=512, height=512)
    board_mat = spec.add_material(name="board_plastic", texrepeat=[4, 4], texuniform=True,
                                   specular=0.9, shininess=0.9, reflectance=0.2)
    board_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "board_plastic"

    spec.add_texture(name="cardboard", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, mark=mujoco.mjtMark.mjMARK_RANDOM,
                     rgb1=BOX_RGBA[:3], rgb2=BOX_RGBA[:3], markrgb=[0.55, 0.38, 0.22],
                     random=0.08, width=256, height=256)
    cardboard_mat = spec.add_material(name="cardboard", texrepeat=[2, 2], texuniform=True,
                                       specular=0.05, shininess=0.05, reflectance=0.0)
    cardboard_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "cardboard"

    wb = spec.worldbody
    wb.add_light(pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 diffuse=[0.35, 0.35, 0.35])
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], material="grid")

    # --- room walls: plain white wallpaper enclosing the floor --------------
    ROOM_HALF = 3.0     # matches floor plane half-extent
    WALL_H = 2.4         # typical room ceiling height
    WALL_T = 0.05
    for name, pos, size in (
        ("wall_neg_x", [-ROOM_HALF, 0, WALL_H / 2], [WALL_T / 2, ROOM_HALF, WALL_H / 2]),
        ("wall_pos_x", [ROOM_HALF, 0, WALL_H / 2], [WALL_T / 2, ROOM_HALF, WALL_H / 2]),
        ("wall_neg_y", [0, -ROOM_HALF, WALL_H / 2], [ROOM_HALF, WALL_T / 2, WALL_H / 2]),
        ("wall_pos_y", [0, ROOM_HALF, WALL_H / 2], [ROOM_HALF, WALL_T / 2, WALL_H / 2]),
    ):
        wb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, material="wallpaper")

    # --- table: aluminum frame top + 4 legs ---------------------------------
    wb.add_geom(name="alu_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[HALF_LEN, HALF_W, 0.02], pos=[0, 0, TABLE_H - 0.02], rgba=COL_ALU)
    leg_h = TABLE_H - 0.04
    leg_inset = 0.05
    for sx in (-1, 1):
        for sy in (-1, 1):
            wb.add_geom(name=f"leg_{sx}_{sy}", type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[0.02, 0.02, leg_h / 2],
                        pos=[sx * (HALF_LEN - leg_inset), sy * (HALF_W - leg_inset), leg_h / 2],
                        rgba=COL_LEG)

    # --- blue Vention mounting plates: bolted to the aluminum frame ---------
    plate_half = 0.09
    plate_z = TABLE_H + PLATE_T / 2
    for name, (ax, ay) in (("left", ARM_LEFT), ("right", ARM_RIGHT)):
        wb.add_geom(name=f"plate_{name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[plate_half, plate_half, PLATE_T / 2], pos=[ax, ay, plate_z],
                    rgba=COL_PLATE)

    # --- black board on top, cut around each plate --------------------------
    board_z = TABLE_H + BOARD_T / 2
    hx = abs(ARM_LEFT[0]) - plate_half        # inner x edge of the plate holes (0.6525)
    hy_lo, hy_hi = ARM_Y - plate_half, ARM_Y + plate_half
    pieces = [
        ("mid", (-hx, hx), (-HALF_W, HALF_W)),               # full-width center span
        ("Lfront", (-HALF_LEN, -hx), (-HALF_W, hy_lo)),      # left end, front of plate
        ("Lback", (-HALF_LEN, -hx), (hy_hi, HALF_W)),        # left end, behind plate
        ("Rfront", (hx, HALF_LEN), (-HALF_W, hy_lo)),        # right end, front of plate
        ("Rback", (hx, HALF_LEN), (hy_hi, HALF_W)),          # right end, behind plate
    ]
    for name, (x0, x1), (y0, y1) in pieces:
        wb.add_geom(name=f"board_{name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[(x1 - x0) / 2, (y1 - y0) / 2, BOARD_T / 2],
                    pos=[(x0 + x1) / 2, (y0 + y1) / 2, board_z], material="board_plastic")

    # --- arms ----------------------------------------------------------------
    # The left base is yawed -90deg about z (clockwise viewed from above) so the
    # left arm faces into the table (reaches +x toward center) at LEFT_HOME_POSE. The
    # right base keeps the default orientation. (Unlike a frame's euler, a body
    # quat DOES propagate through attach, so we orient the base via the mount.)
    left_mount = wb.add_body(name="left_robot_mount", pos=[*ARM_LEFT, ARM_Z],
                             quat=LEFT_BASE_QUAT)
    right_mount = wb.add_body(name="right_robot_mount", pos=[*ARM_RIGHT, ARM_Z],
                              quat=RIGHT_BASE_QUAT)
    left_mount.add_frame().attach_body(_arm_with_gripper().body("base"), "left_", "")
    right_mount.add_frame().attach_body(_arm_with_gripper().body("base"), "right_", "")

    # --- graspable block (free joint) for the pick task ---------------------
    block = wb.add_body(name="block", pos=list(BLOCK_INIT_POS))
    block.add_freejoint(name="block_joint")
    block.add_geom(name="block", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[BLOCK_HALF] * 3, rgba=BLOCK_RGBA,
                   mass=0.05, friction=[1.0, 0.01, 0.001])

    # --- open-top cardboard tray (free joint) -------------------------------
    box_half = BOX_OUTER / 2
    box_in_half = box_half - BOX_WALL_T
    box = wb.add_body(name="cardboard_box", pos=list(BOX_INIT_POS))
    box.add_freejoint(name="cardboard_box_joint")
    box.add_geom(name="cardboard_box_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[box_half, box_half, BOX_FLOOR_T / 2], material="cardboard", mass=0.03)
    wall_z = BOX_FLOOR_T / 2 + BOX_WALL_H / 2
    for name, size, pos in (
        ("neg_x", [BOX_WALL_T / 2, box_half, BOX_WALL_H / 2], [-box_in_half, 0, wall_z]),
        ("pos_x", [BOX_WALL_T / 2, box_half, BOX_WALL_H / 2], [box_in_half, 0, wall_z]),
        ("neg_y", [box_in_half, BOX_WALL_T / 2, BOX_WALL_H / 2], [0, -box_in_half, wall_z]),
        ("pos_y", [box_in_half, BOX_WALL_T / 2, BOX_WALL_H / 2], [0, box_in_half, wall_z]),
    ):
        box.add_geom(name=f"cardboard_box_wall_{name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=size, pos=pos, material="cardboard", mass=0.01)

    # --- camera (Intel RealSense D435) --------------------------------------
    quat = _lookat_quat(CAM_POS, CAM_TARGET)
    cam_body = wb.add_body(name="top1_body", pos=list(CAM_POS), quat=list(quat))
    # D435 housing: 90 x 25 x 25 mm; camera looks down its local -z.
    cam_body.add_geom(name="top1_housing", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[0.045, 0.0125, 0.0125], rgba=[1, 1, 1, 1])
    cam_body.add_camera(name="top1", fovy=CAM_FOVY)

    return spec


def build_model() -> mujoco.MjModel:
    return build_spec().compile()


# ======================================================================== #
#  HARD-CODED INITIAL STATE — edit these two blocks to change what the      #
#  viewer opens at. To recapture them after rearranging the scene, run the  #
#  viewer and press 's' (see capture_state); paste its output back here.    #
# ======================================================================== #

# Robot pose is set per-joint by name (see set_initial_pose) from LEFT_HOME_POSE /
# RIGHT_HOME_POSE, so it survives layout changes (added gripper/object joints). Edit
# those / BLOCK_INIT_POS above to change the starting state.

# Viewport: the free camera's orbit angle / zoom / look-at target.
INITIAL_VIEW = {
    "azimuth": 90.0,
    "elevation": -20.0,
    "distance": 2.5,
    "lookat": [0.0, 0.0, BOARD_TOP],   # center of the work surface
}


def set_initial_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put both arms at the home pose with grippers open, command the actuators to
    hold it (so nothing sags under gravity), and place the block at its rest pose.
    Set by joint/body name so it survives layout changes."""
    for prefix, home_pose in (("left_", LEFT_HOME_POSE), ("right_", RIGHT_HOME_POSE)):
        for joint, angle in zip(ARM_JOINTS, home_pose):
            data.qpos[model.joint(f"{prefix}{joint}_joint").qposadr[0]] = angle
            data.ctrl[model.actuator(f"{prefix}{joint}").id] = angle
        # gripper: finger linkage rests open at qpos 0; just command it open.
        data.ctrl[model.actuator(f"{prefix}grip_fingers_actuator").id] = GRIPPER_OPEN

    # block: free joint qpos is [x, y, z, qw, qx, qy, qz]
    adr = model.joint("block_joint").qposadr[0]
    data.qpos[adr:adr + 7] = [*BLOCK_INIT_POS, 1, 0, 0, 0]


def block_height(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Current height (z, meters) of the block's center."""
    return float(data.body("block").xpos[2])


def pick_success(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """True once the block has been lifted LIFT_SUCCESS_H above its rest height."""
    return block_height(model, data) > BLOCK_REST_Z + LIFT_SUCCESS_H


def apply_initial_view(viewer) -> None:
    """Point the viewer's free camera at INITIAL_VIEW. Call right after
    launch_passive(), then viewer.sync(). Edit INITIAL_VIEW above to change it."""
    cam = viewer.cam
    cam.azimuth = INITIAL_VIEW["azimuth"]
    cam.elevation = INITIAL_VIEW["elevation"]
    cam.distance = INITIAL_VIEW["distance"]
    cam.lookat[:] = INITIAL_VIEW["lookat"]


def capture_state(data: mujoco.MjData, viewer) -> None:
    """Print the current viewport as a paste-ready INITIAL_VIEW block, plus the
    current arm joint angles (for LEFT_HOME_POSE) and block position, so you can
    hard-code the state you've navigated to."""
    cam = viewer.cam
    print("\n# --- paste INITIAL_VIEW into build_urtable.py ---")
    print("INITIAL_VIEW = {")
    print(f'    "azimuth": {cam.azimuth:.3f},')
    print(f'    "elevation": {cam.elevation:.3f},')
    print(f'    "distance": {cam.distance:.4f},')
    print(f'    "lookat": [{cam.lookat[0]:.4f}, {cam.lookat[1]:.4f}, {cam.lookat[2]:.4f}],')
    print("}")
    print("# current left-arm joints:",
          [round(float(data.qpos[i]), 4) for i in range(6)], "\n")


def build_scene() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build the scene and return model + data initialized to INITIAL_QPOS, with
    the position actuators commanded to hold it and forward kinematics evaluated."""
    model = build_model()
    data = mujoco.MjData(model)
    set_initial_pose(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def main() -> None:
    import time
    import mujoco.viewer

    model, data = build_scene()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        viewer.opt.sitegroup[5] = True  # sites use group 5 (see robotiq-2f85.xml); off by default
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
