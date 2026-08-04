# Tabletop size tests

Two UR7e gantry scenes that differ only in how much table is exposed in front of the
mount, plus the workspace tests from `workspace_tests/` re-pointed at them.

| scene  | file                      | exposed tabletop                       |
|--------|---------------------------|----------------------------------------|
| `full` | `build_tabletop_full.py`  | 117 cm mount edge x 112 cm deep        |
| `half` | `build_tabletop_half.py`  | 112 cm mount edge x 58.5 cm (117/2) deep |

The hardware is identical in both (`gantry_core.py`): one Vention column centered on
the mount edge, two UR7e arms hanging upside down off its head splayed 45 deg
outward, Robotiq 2F-85 grippers, wrist cameras, one overhead D435. The column stands
on its own 18 cm strip of table *behind* the exposed rectangle, so the quoted
tabletop is all usable board.

Frame: origin on the floor at the center of the exposed tabletop, +x right along the
mount edge, +y away from the mount, +z up, meters. Board top = 0.775.

```
uv run python build_tabletop_full.py          # interactive viewer
uv run python workspace_benchmark.py --all    # reach + tilt + roll over each tabletop
uv run python bimanual_test.py                # both-arms verdict per waypoint
uv run python waypoint_video.py               # mp4 per scene + side by side
```

## The grid

Both scenes are scored over *their own* exposed rectangle, since the size of that
rectangle is what is being compared. The two grids share one lattice anchored on the
mount edge (10 cm pitch, first row 15 cm out, x = 0 on the column axis), so the half
table's 165 waypoints are exactly the near 58.5 cm of the full table's 330, and the
runs can also be compared on identical positions. Waypoint names carry the distance
out from the mount edge (`low_x+200_d0350` = low layer, x = +20 cm, 35 cm out).

Three layers, 5 / 20 / 35 cm above the board. The first 10 cm of board is skipped:
it is directly under the stand base, where the arms foul each other and the
extrusion whatever the tabletop size, so it says nothing about the size.

## Result: the far half of the big table is dead space

`bimanual_test.py` (both arms at the waypoint at once, offset 35 cm, no contact on
the way in, parked and stable) passes **116 waypoints on each table** -- the same
116 positions, out of 330 for `full` and 165 for `half`.

Pass rate by distance out from the mount edge (33 waypoints per row, all layers):

| distance | 15 cm | 25 | 35 | 45 | 55 | 65 | 75 | 85 | 95 | 105 |
|----------|-------|----|----|----|----|----|----|----|----|-----|
| `full`   | 26    | 31 | 29 | 21 | 9  | 0  | 0  | 0  | 0  | 0   |
| `half`   | 26    | 31 | 29 | 21 | 9  | -- | -- | -- | -- | --  |

Nothing past 55 cm from the mount edge is reachable bimanually (`no IK` from 65 cm
out), and the last row before that already only manages 9/33. The two scenes score
identically row for row; the whole difference between them is board no arm can use.

So the 112 cm deep table costs 83 % more footprint than the 58.5 cm one and buys no
extra working area. If more reach is wanted, it has to come from the mount (a taller
or forward-cantilevered head, or a longer arm), not from more board.
