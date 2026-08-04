"""Gantry scene A: the full tabletop.

117 cm along the mount edge (the gantry column, and so both arm bases, sit centered
on it) x 112 cm deep of exposed board in front of the mount. Everything else --
column, head, arms, grippers, cameras -- is gantry_core's standard hardware.

Run directly to open the interactive viewer:  uv run python build_tabletop_full.py
"""

from gantry_core import Dims, scene_api

DIMS = Dims(length=1.17, depth=1.12)

globals().update(scene_api(DIMS))

if __name__ == "__main__":
    main()   # noqa: F821  (from scene_api)
