"""Gantry scene B: the table turned 90 deg, half depth.

112 cm along the mount edge (the gantry column, and so both arm bases, sit centered
on it) x 58.5 cm (= 117/2) deep of exposed board in front of the mount. Same
hardware as build_tabletop_full.py; only the exposed rectangle differs.

Run directly to open the interactive viewer:  uv run python build_tabletop_half.py
"""

from gantry_core import Dims, scene_api

DIMS = Dims(length=1.12, depth=0.585)

globals().update(scene_api(DIMS))

if __name__ == "__main__":
    main()   # noqa: F821  (from scene_api)
