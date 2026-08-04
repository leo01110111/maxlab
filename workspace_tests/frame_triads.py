"""Render the gantry with triads at the stand base point and both arm bases.

Two passes: the full scene, and a stripped one (arms/props deleted, column made
translucent) so no triad axis is buried inside a link.
"""

import numpy as np
import mujoco
import imageio.v2 as imageio

import build_urgantry as scene

AXIS_LEN = 0.26
AXIS_W = 0.010
AXIS_RGBA = ([1, 0.15, 0.15, 1], [0.15, 0.9, 0.2, 1], [0.25, 0.45, 1, 1])
DOT_R = 0.020

ORIGIN = np.array([0.0, scene.COL_Y, scene.BOARD_TOP])
FRAMES = [
    ("STAND BASE (0, 0, 0)", ORIGIN, np.eye(3)),
    ("LEFT BASE (-0.155, 0, +0.720)", ORIGIN + [-scene.ARM_MOUNT_X, 0, scene.ARM_Z - scene.BOARD_TOP], None),
    ("RIGHT BASE (+0.155, 0, +0.720)", ORIGIN + [scene.ARM_MOUNT_X, 0, scene.ARM_Z - scene.BOARD_TOP], None),
]

VIEWS = {"front": (90, -10), "iso": (130, -22), "side": (180, -10)}
DIST = {"scene": 2.3, "bare": 1.9}


def _compile(spec):
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1400, 900
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


def full_scene():
    model, data = _compile(scene.build_spec())
    scene.set_initial_pose(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def stripped_scene():
    spec = scene.build_spec()
    for name in ("left_robot_mount", "right_robot_mount", "block", "cardboard_box"):
        spec.delete(spec.body(name))
    for g in spec.worldbody.geoms:
        if g.name.startswith(("column", "gantry_head")):
            g.rgba = [0.20, 0.42, 0.78, 0.30]
    model, data = _compile(spec)
    mujoco.mj_forward(model, data)
    return model, data


def base_orientations():
    model, data = full_scene()
    return [data.body(f"{s}_base").xmat.reshape(3, 3).copy() for s in ("left", "right")]


def add_triads(sc, frames_, axis_labels):
    for name, pos, R in frames_:
        g = sc.geoms[sc.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([DOT_R] * 3),
                            pos, np.eye(3).flatten(), np.array([1.0, 0.85, 0.1, 1.0]))
        g.label = name
        sc.ngeom += 1
        for i, letter in enumerate("xyz"):
            g = sc.geoms[sc.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                                np.eye(3).flatten(), np.array(AXIS_RGBA[i], float))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, AXIS_W,
                                 pos, pos + AXIS_LEN * R[:, i])
            if axis_labels:
                g.label = letter
            sc.ngeom += 1


def render(model, data, frames_, tag, axis_labels):
    renderer = mujoco.Renderer(model, 900, 1400)
    cam = mujoco.MjvCamera()
    for view, (az, el) in VIEWS.items():
        cam.lookat[:] = [0.0, scene.COL_Y, scene.BOARD_TOP + 0.42]
        cam.azimuth, cam.elevation, cam.distance = az, el, DIST[tag]
        renderer.update_scene(data, cam)
        add_triads(renderer.scene, frames_, axis_labels)
        path = f"frame_triads_{tag}_{view}.png"
        imageio.imwrite(path, renderer.render())
        print("wrote", path)
    renderer.close()


def main():
    lR, rR = base_orientations()
    frames_ = [(FRAMES[0][0], FRAMES[0][1], FRAMES[0][2]),
               (FRAMES[1][0], FRAMES[1][1], lR),
               (FRAMES[2][0], FRAMES[2][1], rR)]
    for name, pos, R in frames_:
        print(f"{name:36s} rel={np.round(pos - ORIGIN, 4)}  z_axis={np.round(R[:, 2], 4)}")

    m, d = full_scene()
    render(m, d, frames_, "scene", axis_labels=False)
    m, d = stripped_scene()
    render(m, d, frames_, "bare", axis_labels=False)


if __name__ == "__main__":
    main()
