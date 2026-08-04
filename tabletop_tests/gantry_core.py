"""Shared builder for the UR7e gantry scenes used to compare *tabletop sizes*.

Same hardware in every variant -- one Vention column at the middle of the near
edge of the table, two UR7e arms hanging upside down off its head, splayed 45 deg
outward, each with a Robotiq 2F-85 and a wrist camera. The only thing that changes
between scenes is the rectangle of table exposed in front of the mount, which is
what `Dims` parameterises. See build_tabletop_*.py for the variants.

NOTE FOR FUTURE AGENTS: the arm is a **UR7e**. The MJCF still lives at
`universal_robots_ur5e/ur5e.xml` because it started as the menagerie UR5e and is
being re-tuned in place to UR7e physics; only the file path keeps the legacy name.

Pure-XML <attach> of an <include>d robot doesn't work (include merges the robot's
bodies into the worldbody; attach needs the robot kept as a separate child model),
so we load the arm MJCF once per arm and attach a prefixed copy.

Coordinate frame: origin on the floor at the **center of the exposed tabletop**.
  +x = right along the mount edge, +y = away from the mount, +z = up. Meters.
  Exposed tabletop = x in [-L/2, L/2], y in [-D/2, D/2] at z = BOARD_TOP.
  The column stands on its own strip of table behind y = -D/2, so it never eats
  into the exposed rectangle.
"""

from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
import mujoco

ASSET_DIR = Path(__file__).resolve().parent

UR7E_PATH = str(ASSET_DIR / "universal_robots_ur5e" / "ur5e.xml")
ROBOTIQ_PATH = str(ASSET_DIR / "gripper" / "robotiq-2f85.xml")
ARDUCAM_PATH = str(ASSET_DIR / "gripper" / "arducam_ov9782.xml")

# ---------------------------------------------------------------- measurements
TABLE_H = 0.76          # aluminum frame top height
BOARD_T = 0.015         # black board thickness (on top of the frame)
PLATE_T = 0.006         # blue Vention mounting plate thickness (column footplate)
BOARD_TOP = TABLE_H + BOARD_T          # 0.775

# --------------------------------------------------------------- gantry column
COL_HALF = 0.045                       # 9x9 cm extrusion
COL_FOOT_HALF = 0.09                   # blue footplate under the column
COL_STRIP = 2 * COL_FOOT_HALF          # table depth taken up behind the exposed area
COL_TOP_Z = BOARD_TOP + 0.78           # top of the column above the board

HEAD_HALF_X = 0.24
HEAD_HALF_Y = 0.10
HEAD_T = 0.03
HEAD_Z = COL_TOP_Z + HEAD_T / 2

# --------------------------------------------------------------- arm mounting
# Both arms hang upside down (base flange up, arm pointing down) and are rolled
# 45 deg outward about +y, so the left arm leans toward -x and the right toward +x.
ARM_TILT = np.deg2rad(45.0)
ARM_MOUNT_X = 0.155                    # mount center offset from the column axis
ARM_Z = COL_TOP_Z - 0.06               # base flange height

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
LEFT_HOME_POSE = [ 1.3969, -1.6581,  2.3117, -2.3623,  0.7424, 0.0]
RIGHT_HOME_POSE = [-1.3969, -1.4835, -2.3117, -0.7793, -0.7424, 0.0]

GRIPPER_OPEN, GRIPPER_CLOSED = 0.0, 255.0

CAM_FOVY = 42.0                        # D435 color vertical FOV (deg)

# --------------------------------------------------------------- props
BLOCK_HALF = 0.025                                   # 5 cm cube
BLOCK_REST_Z = BOARD_TOP + BLOCK_HALF
BLOCK_RGBA = [0.15, 0.75, 0.20, 1]
LIFT_SUCCESS_H = 0.05                                # meters above rest to count as a pick

BOX_OUTER = 0.16
BOX_WALL_T = 0.006
BOX_WALL_H = 0.035
BOX_FLOOR_T = 0.004
BOX_RGBA = [0.72, 0.53, 0.34, 1]

COL_ALU = [0.62, 0.64, 0.66, 1]
COL_PLATE = [0.20, 0.42, 0.78, 1]
COL_LEG = [0.12, 0.13, 0.15, 1]


class Dims(NamedTuple):
    """The exposed tabletop in front of the mount: `length` along the mount edge
    (x, with the mount centered on it), `depth` away from the mount (y)."""

    length: float
    depth: float

    @property
    def half_len(self) -> float:
        return self.length / 2

    @property
    def y0(self) -> float:
        """Near edge of the exposed area (the mount edge)."""
        return -self.depth / 2

    @property
    def y1(self) -> float:
        return self.depth / 2

    @property
    def table_y0(self) -> float:
        return self.y0 - COL_STRIP

    @property
    def table_cy(self) -> float:
        return (self.table_y0 + self.y1) / 2

    @property
    def half_w(self) -> float:
        """Half depth of the *whole* table, exposed area plus the column strip."""
        return (self.depth + COL_STRIP) / 2

    @property
    def col_y(self) -> float:
        return self.y0 - COL_FOOT_HALF

    def block_pos(self) -> tuple[float, float, float]:
        return (min(0.32, self.half_len - 0.12), self.y0 + 0.22, BLOCK_REST_Z)

    def box_pos(self) -> tuple[float, float, float]:
        return (-min(0.32, self.half_len - 0.12), self.y0 + 0.22,
                BOARD_TOP + BOX_FLOOR_T / 2)

    def cam_pos(self) -> tuple[float, float, float]:
        """Overhead D435 on the near (mount) side, standing off the table: every
        spot on the edge itself is inside the arms' swept volume."""
        return (0.0, self.table_y0 - 0.70, BOARD_TOP + 0.90)

    def cam_target(self) -> tuple[float, float, float]:
        return (0.0, self.y0 + 0.35 * self.depth, BOARD_TOP)


def _upside_down_quat(tilt: float) -> list[float]:
    """Quat [w,x,y,z] for Ry(tilt) * Rx(180deg): flip the base over (its +z, which
    the arm grows along, now points down) then lean it by `tilt` about +y, so the
    arm hangs and splays toward -x for tilt>0 / +x for tilt<0."""
    flip = np.array([np.cos(np.pi / 2), np.sin(np.pi / 2), 0.0, 0.0])   # Rx(pi)
    lean = np.array([np.cos(tilt / 2), 0.0, np.sin(tilt / 2), 0.0])     # Ry(tilt)
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, lean, flip)
    return quat.tolist()


LEFT_BASE_QUAT = _upside_down_quat(ARM_TILT)
RIGHT_BASE_QUAT = _upside_down_quat(-ARM_TILT)


def _lookat_quat(cam_pos, target):
    """Quaternion orienting a MuJoCo camera (looks down -z) from cam_pos at target."""
    cam_pos, target = np.asarray(cam_pos, float), np.asarray(target, float)
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    mat = np.array([[right[0], up[0], -forward[0]],
                    [right[1], up[1], -forward[1]],
                    [right[2], up[2], -forward[2]]]).flatten()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)
    return quat


def _arm_with_gripper() -> mujoco.MjSpec:
    """Load a UR7e and bolt a Robotiq 2F-85 onto its wrist attachment site. The
    gripper's coupling, its actuator and its wrist camera come along with the
    attach, prefixed 'grip_'."""
    arm = mujoco.MjSpec.from_file(UR7E_PATH)
    gripper = mujoco.MjSpec.from_file(ROBOTIQ_PATH)
    cam = mujoco.MjSpec.from_file(ARDUCAM_PATH)
    # Empty prefix so the camera's names stay "wrist_cam"/"wrist" through this
    # attach, matching what the grip_/left_/right_ prefixes below expect.
    gripper.site("cam_attach").attach_body(cam.body("wrist_cam"), "", "")
    arm.site("attachment_site").attach_body(gripper.body("base_mount"), "grip_", "")
    return arm


def _materials(spec: mujoco.MjSpec) -> None:
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


def build_spec(dims: Dims) -> mujoco.MjSpec:
    """Floor, room, table sized to `dims`, gantry column, two hanging UR7e arms,
    props and the overhead camera."""
    spec = mujoco.MjSpec()
    spec.compiler.autolimits = True
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 800
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    # Pin the extent to the working area, not the room: znear is znear_ratio *
    # extent, and a 6 m room pushes it past 12 cm, clipping the wrist cameras.
    spec.stat.extent = 1.5

    _materials(spec)

    hl, hw, cy = dims.half_len, dims.half_w, dims.table_cy

    wb = spec.worldbody
    wb.add_light(pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 diffuse=[0.35, 0.35, 0.35])
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], material="grid")

    ROOM_HALF = 3.0
    WALL_H = 2.4
    WALL_T = 0.05
    for name, pos, size in (
        ("wall_neg_x", [-ROOM_HALF, 0, WALL_H / 2], [WALL_T / 2, ROOM_HALF, WALL_H / 2]),
        ("wall_pos_x", [ROOM_HALF, 0, WALL_H / 2], [WALL_T / 2, ROOM_HALF, WALL_H / 2]),
        ("wall_neg_y", [0, -ROOM_HALF, WALL_H / 2], [ROOM_HALF, WALL_T / 2, WALL_H / 2]),
        ("wall_pos_y", [0, ROOM_HALF, WALL_H / 2], [ROOM_HALF, WALL_T / 2, WALL_H / 2]),
    ):
        wb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size,
                    material="wallpaper")

    # --- table: aluminum frame top + 4 legs ---------------------------------
    wb.add_geom(name="alu_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[hl, hw, 0.02], pos=[0, cy, TABLE_H - 0.02], rgba=COL_ALU)
    leg_h = TABLE_H - 0.04
    leg_inset = 0.05
    for sx in (-1, 1):
        for sy in (-1, 1):
            wb.add_geom(name=f"leg_{sx}_{sy}", type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[0.02, 0.02, leg_h / 2],
                        pos=[sx * (hl - leg_inset), cy + sy * (hw - leg_inset), leg_h / 2],
                        rgba=COL_LEG)

    # --- black board on top, cut around the column footplate ----------------
    board_z = TABLE_H + BOARD_T / 2
    nx0, nx1 = -COL_FOOT_HALF, COL_FOOT_HALF
    ny0, ny1 = dims.col_y - COL_FOOT_HALF, dims.col_y + COL_FOOT_HALF
    for name, (x0, x1), (y0, y1) in (
        ("neg_x", (-hl, nx0), (dims.table_y0, dims.y1)),
        ("pos_x", (nx1, hl), (dims.table_y0, dims.y1)),
        ("mid_front", (nx0, nx1), (dims.table_y0, ny0)),
        ("mid_back", (nx0, nx1), (ny1, dims.y1)),
    ):
        if x1 - x0 < 1e-9 or y1 - y0 < 1e-9:
            continue
        wb.add_geom(name=f"board_{name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[(x1 - x0) / 2, (y1 - y0) / 2, BOARD_T / 2],
                    pos=[(x0 + x1) / 2, (y0 + y1) / 2, board_z], material="board_plastic")

    # --- gantry: blue footplate + vertical extrusion + head plate -----------
    wb.add_geom(name="column_foot", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[COL_FOOT_HALF, COL_FOOT_HALF, PLATE_T / 2],
                pos=[0, dims.col_y, TABLE_H + PLATE_T / 2], rgba=COL_PLATE)
    col_z0 = TABLE_H + PLATE_T
    wb.add_geom(name="column", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[COL_HALF, COL_HALF, (COL_TOP_Z - col_z0) / 2],
                pos=[0, dims.col_y, (col_z0 + COL_TOP_Z) / 2], rgba=COL_PLATE)
    wb.add_geom(name="gantry_head", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[HEAD_HALF_X, HEAD_HALF_Y, HEAD_T / 2],
                pos=[0, dims.col_y, HEAD_Z], rgba=COL_PLATE)

    # --- arms: hung upside down off the head, splayed 45 deg outward --------
    # (Unlike a frame's euler, a body quat DOES propagate through attach, so we
    # orient each base via its mount body.)
    for side, sx, quat in (("left", -1, LEFT_BASE_QUAT), ("right", 1, RIGHT_BASE_QUAT)):
        mount = wb.add_body(name=f"{side}_robot_mount",
                            pos=[sx * ARM_MOUNT_X, dims.col_y, ARM_Z], quat=quat)
        mount.add_geom(name=f"plate_{side}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[0.075, 0.075, PLATE_T / 2], pos=[0, 0, -PLATE_T / 2],
                       rgba=COL_PLATE, contype=0, conaffinity=0)
        mount.add_frame().attach_body(_arm_with_gripper().body("base"), f"{side}_", "")

    # Wrist F/T: reads nonzero at rest (tool weight + Robotiq linkage preload), so
    # consumers should tare against a no-contact reference pose.
    for side in ("left", "right"):
        spec.body(f"{side}_grip_base_mount").add_site(name=f"{side}_ft_site")
        spec.add_sensor(name=f"{side}_ft_force", type=mujoco.mjtSensor.mjSENS_FORCE,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"{side}_ft_site")
        spec.add_sensor(name=f"{side}_ft_torque", type=mujoco.mjtSensor.mjSENS_TORQUE,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"{side}_ft_site")

    # --- graspable block (free joint) ---------------------------------------
    block = wb.add_body(name="block", pos=list(dims.block_pos()))
    block.add_freejoint(name="block_joint")
    block.add_geom(name="block", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[BLOCK_HALF] * 3, rgba=BLOCK_RGBA,
                   mass=0.05, friction=[1.0, 0.01, 0.001])

    # --- open-top cardboard tray (free joint) -------------------------------
    box_half = BOX_OUTER / 2
    box_in_half = box_half - BOX_WALL_T
    box = wb.add_body(name="cardboard_box", pos=list(dims.box_pos()))
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
    cam_pos, cam_target = dims.cam_pos(), dims.cam_target()
    quat = _lookat_quat(cam_pos, cam_target)
    cam_body = wb.add_body(name="top1_body", pos=list(cam_pos), quat=list(quat))
    # D435 housing: 90 x 25 x 25 mm; camera looks down its local -z.
    cam_body.add_geom(name="top1_housing", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[0.045, 0.0125, 0.0125], rgba=[1, 1, 1, 1])
    cam_body.add_camera(name="top1", fovy=CAM_FOVY)

    return spec


def set_initial_pose(dims: Dims, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Both arms at the home pose with grippers open, actuators commanded to hold
    it, props at their rest poses. Set by name so it survives layout changes."""
    for prefix, home_pose in (("left_", LEFT_HOME_POSE), ("right_", RIGHT_HOME_POSE)):
        for joint, angle in zip(ARM_JOINTS, home_pose):
            data.qpos[model.joint(f"{prefix}{joint}_joint").qposadr[0]] = angle
            data.ctrl[model.actuator(f"{prefix}{joint}").id] = angle
        data.ctrl[model.actuator(f"{prefix}grip_fingers_actuator").id] = GRIPPER_OPEN

    adr = model.joint("block_joint").qposadr[0]
    data.qpos[adr:adr + 7] = [*dims.block_pos(), 1, 0, 0, 0]
    adr = model.joint("cardboard_box_joint").qposadr[0]
    data.qpos[adr:adr + 7] = [*dims.box_pos(), 1, 0, 0, 0]


def initial_view(dims: Dims) -> dict:
    return {
        "azimuth": 90.0,
        "elevation": -12.0,
        "distance": 2.6 + dims.depth,
        "lookat": [0.0, dims.y0 + 0.3 * dims.depth, BOARD_TOP + 0.30],
    }


def apply_initial_view(dims: Dims, viewer) -> None:
    view = initial_view(dims)
    cam = viewer.cam
    cam.azimuth = view["azimuth"]
    cam.elevation = view["elevation"]
    cam.distance = view["distance"]
    cam.lookat[:] = view["lookat"]


def build_scene(dims: Dims) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = build_spec(dims).compile()
    data = mujoco.MjData(model)
    set_initial_pose(dims, model, data)
    mujoco.mj_forward(model, data)
    return model, data


def block_height(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    return float(data.body("block").xpos[2])


def pick_success(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    return block_height(model, data) > BLOCK_REST_Z + LIFT_SUCCESS_H


def viewer_main(dims: Dims) -> None:
    import time
    import mujoco.viewer

    model, data = build_scene(dims)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        apply_initial_view(dims, viewer)
        viewer.sync()
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


def scene_api(dims: Dims) -> dict:
    """The module-level API every tool here expects (build_spec, build_model,
    build_scene, set_initial_pose, apply_initial_view, main, plus the layout
    constants). A scene module is this dict poured into its globals."""
    return {
        "DIMS": dims,
        "EXPOSED_LEN": dims.length,
        "EXPOSED_DEPTH": dims.depth,
        "EXPOSED_X0": -dims.half_len,
        "EXPOSED_X1": dims.half_len,
        "EXPOSED_Y0": dims.y0,
        "EXPOSED_Y1": dims.y1,
        "HALF_LEN": dims.half_len,
        "HALF_W": dims.half_w,
        "TABLE_CY": dims.table_cy,
        "COL_Y": dims.col_y,
        "BOARD_TOP": BOARD_TOP,
        "ARM_JOINTS": ARM_JOINTS,
        "LEFT_HOME_POSE": LEFT_HOME_POSE,
        "RIGHT_HOME_POSE": RIGHT_HOME_POSE,
        "GRIPPER_OPEN": GRIPPER_OPEN,
        "GRIPPER_CLOSED": GRIPPER_CLOSED,
        "BLOCK_INIT_POS": dims.block_pos(),
        "BOX_INIT_POS": dims.box_pos(),
        "INITIAL_VIEW": initial_view(dims),
        "build_spec": partial(build_spec, dims),
        "build_model": lambda: build_spec(dims).compile(),
        "build_scene": partial(build_scene, dims),
        "set_initial_pose": partial(set_initial_pose, dims),
        "apply_initial_view": partial(apply_initial_view, dims),
        "block_height": block_height,
        "pick_success": pick_success,
        "main": partial(viewer_main, dims),
    }
