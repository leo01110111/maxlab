# urgantry_hug

Drives the gantry sim's Wuji hand from [HUG](https://grasping.io) grasp
predictions: render an RGB-D frame, click an object, get a human grasp, execute
it.

```
capture.py    MuJoCo camera -> RGB + metric depth + intrinsics
predict.py    HUG inference -> ManoGrasp, in world coordinates
frames.py     MuJoCo <-> OpenCV camera conventions
interface.py  the shared data contract (no HUG or sim imports)
retarget.py   MANO -> Wuji joint angles          [owned separately]
execute.py    approach / close / lift
visualize.py  draw a prediction into the scene
demo.py       all of the above, end to end
```

## Setup

Both stacks live in the `hug` conda env — HUG pins Python 3.10, so `mujoco` and
`gymnasium` were installed there rather than the other way around. MuJoCo needs
an EGL context for offscreen rendering:

```bash
cd ~/maxlab
MUJOCO_GL=egl conda run -n hug python -m urgantry_hug.demo
```

HUG weights are expected at `~/hug/checkpoints/hug_full.safetensors`
(`--checkpoint` to override).

## Tests

Both are runnable and currently pass. Run them before debugging anything
downstream.

| command | checks | last result |
|---|---|---|
| `python -m urgantry_hug.test_frames` | intrinsics, depth units, camera flip | 0.00 mm off the camera ray |
| `python -m urgantry_hug.test_execute` | IK, approach, close, lift | block lifted 0.80 → 0.928 |

`python -m urgantry_hug.visualize` writes five images: the prediction reprojected
onto its own capture, plus the same landmarks as world geometry from four
viewpoints. The extra viewpoints are the point — a wrong camera-to-world
transform still overlays perfectly in 2D and only shows up when the camera moves.

## Status

Every stage runs. `demo.py` goes from render to prediction to retargeting to
execution without error. It does not yet lift the block — see below.

Verified:

- The camera model round-trips: a pixel unprojects back onto the object it came
  from, 0.00 mm off the camera ray and 1.3 mm in depth.
- HUG works on synthetic MuJoCo renders — the domain gap was the main risk and it
  did not materialise. Clicking the block gives a wrist above and behind it with
  the thumb on the near face and four fingers wrapping the far one.
- Retargeting is accurate: posed Wuji fingertips land within **3.0 mm mean** of
  their MANO targets (2.1 / 3.1 / 2.9 / 3.1 / 3.8 mm, thumb through pinky).
- The executor reaches, closes, squeezes, and lifts — `test_execute.py` lifts the
  block from 0.80 to 0.936.

### Open issue: predicted grasps interpenetrate the object

`demo.py` reports `grasp slipped`. It is not retargeting (3 mm) and not the
executor (it lifts a non-penetrating grasp fine). **HUG's predicted fingertips
are inside the block**, measured against the block's AABB:

| seed | tips inside | max penetration |
|---|---|---|
| 0 | 4/5 | 20.1 mm |
| 1 | 2/5 | 12.5 mm |
| 2 | 5/5 | 18.3 mm |
| 3 | 3/5 | 6.1 mm |

The model has no collision constraint against the scene, so it places fingers
where the object is. Executing that faithfully pushes the block away — during the
close it travels from y=−0.0725 to −0.109 and contact is lost. This is consistent
with everything else observed: retargeting reproduces an impossible target
*accurately*, and no executor parameter rescues it.

Things already ruled out, so nobody repeats them:

- 12 combinations of pre-shape fraction (0.3/0.7/0.9), approach distance
  (0.08/0.04 m) and squeeze (0.25/0.45 rad) — all slip.
- 12 combinations of palm back-off along −z (0–0.05 m) and squeeze
  (0.25/0.5/0.8 rad) — all slip. A rigid translation along the approach axis does
  not resolve the penetration.

The likely fix is a penetration-resolution stage between prediction and
retargeting: push the fingertip targets out to the observed surface (the captured
depth point cloud is right there in `Capture`) before fitting joint angles,
rather than translating the whole hand. `sample.py` scoring could also weight
against penetration to prefer the shallower samples.

## Gotchas

- **HUG predicts right hands only.** `side="right"`.
- **MANO's `pose` and `landmarks` arrays order fingers differently** — ring and
  pinky are transposed. Use `MANO_POSE_SLICES` / `MANO_LANDMARKS` from
  `interface.py`, never literal indices. See `RETARGETING_SPEC.md` §2.1.
- **Open fingertips reach ~4.5 cm past `GRASP_CENTER`**, so a top-down approach
  drives the open hand through the object before it closes. `test_execute.py`
  documents the measured consequences.
