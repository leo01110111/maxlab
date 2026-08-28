# Retargeting spec: MANO grasp → Wuji hand + UR7e

Everything around retargeting is built and tested. This document is the contract
for the missing middle: turning a HUG grasp prediction into something the gantry
sim can execute.

All the numbers below were measured on the compiled model or verified
empirically, not read off documentation. Where something is a starting estimate
rather than a measurement, it says so.

---

## 1. What you implement

Create `urgantry_hug/retarget.py` exposing exactly one entry point:

```python
def retarget(grasp: ManoGrasp, model: mujoco.MjModel, side: str = "right") -> WujiGrasp:
    ...
```

Both types live in `urgantry_hug/interface.py` and neither imports HUG or the
executor, so you can develop against them standalone. `demo.py` imports this
function by name; nothing else in the package needs to change.

Run `python -m urgantry_hug.demo` to exercise the whole path once it exists.

---

## 2. Input: `ManoGrasp`

Already lifted into **MuJoCo world coordinates, metres**. You do not need to
touch camera frames — `predict.py` has done that and it is validated (see §6).

| Field | Shape | Meaning |
|---|---|---|
| `T_world_wrist` | (4,4) | MANO wrist frame in world |
| `pose` | (15,3) | Per-joint axis-angle, **relative to the parent joint** |
| `landmarks` | (21,3) | Joint positions in world |
| `vertices` | (778,3) | Hand mesh in world |
| `click_world` | (3,) | The surface point the grasp was requested at |

### 2.1 The index ordering trap

`pose` and `landmarks` use **different finger orders**. This is verified by
perturbing each pose entry and observing which landmarks move:

| `pose` rows | finger | `landmarks` indices (MCP→tip) |
|---|---|---|
| 0,1,2 | index | 5, 6, 7, 8 |
| 3,4,5 | middle | 9, 10, 11, 12 |
| 6,7,8 | **pinky** | 17, 18, 19, 20 |
| 9,10,11 | **ring** | 13, 14, 15, 16 |
| 12,13,14 | thumb | 1, 2, 3, 4 |

`pose` follows the raw MANO kinematic tree (index, middle, **pinky, ring**,
thumb); `landmarks` follows OpenPose order (thumb, index, middle, **ring,
pinky**). Ring and pinky are transposed between them. Landmark 0 is the wrist.

`interface.py` exports `MANO_POSE_SLICES` and `MANO_LANDMARKS` so you never have
to hardcode these.

Within a finger, `pose[k]` is the rotation *at* joint k, so it moves all joints
distal to it: `pose[0]` (index MCP) moves landmarks 6, 7, 8 but not 5.

---

## 3. Output: `WujiGrasp`

```python
WujiGrasp(
    T_world_palm,   # (4,4) target pose of the <side>_hand_palm_link body
    hand,           # HandTarget: joint angles at full close
    side="right",
    pregrasp=None,  # HandTarget during approach; defaults to flat open
    approach_m=0.08,
)
```

### 3.1 `T_world_palm`

The pose of the **`{side}_hand_palm_link` body**, not the tool flange and not the
grasp center. The executor converts to whatever the IK wants. Palm frame axes,
measured on the compiled model:

- **palm +z** — the direction the fingers extend
- **palm +x** — the grasping side, the direction fingers curl toward
- **palm +y** — the direction the thumb sticks out

### 3.2 `HandTarget.angles`

`(5, 4)` float array, radians. `angles[f-1, j-1]` is `finger{f}_joint{j}`.

- **finger 1 = thumb**, 2 = index, 3 = middle, 4 = ring, 5 = pinky
- **joint1, joint3, joint4 = flexion**, positive curls toward palm +x
- **joint2 = abduction** (spread). Note joint2's axis differs from the flexion
  joints, and on the thumb joint1 is a rotation rather than a clean flex.

Joint limits from the MJCF (the executor clips to `ctrlrange` anyway, but
targets outside the range silently saturate and cost you fidelity):

| joint | range (rad) | notes |
|---|---|---|
| finger1_joint1 | −0.045 … 1.651 | thumb rotation |
| finger1_joint2 | −0.166 … 0.934 | |
| finger2-5_joint1 | −0.327 … 1.636 | |
| finger2-5_joint2 | −0.495 … 0.495 | abduction, symmetric |
| all joint3, joint4 | −0.493 … 1.627 | |

**The ranges are asymmetric — flexion is positive.** Signs are not guessable
from the axis attributes alone because the link frames carry quaternions;
`urgantry_bc/hand.py` documents the conventions read off the compiled model.

---

## 4. Measurements you'll want

### 4.1 The Wuji hand is close to MANO scale

Distance from palm origin to the distal link (`link4`) origin, hand flat open,
against MANO's rest-pose wrist-to-tip distances:

| finger | Wuji palm→link4 | MANO wrist→tip | ratio |
|---|---|---|---|
| thumb | 0.1115 | 0.1285 | 0.87 |
| index | 0.1732 | 0.1684 | 1.03 |
| middle | 0.1675 | 0.1756 | 0.95 |
| ring | 0.1629 | 0.1632 | 1.00 |
| pinky | 0.1579 | 0.1404 | 1.12 |

**A uniform rescale is probably not worth applying** — the hands are within ~±12%
per finger. Note the shape difference though: MANO's *middle* finger is longest,
Wuji's *index* is longest. Per-finger normalization will behave differently from
a single global scale.

(`link4` is the distal link origin, not the fingertip — there is tip geometry
beyond it. Compare like with like if you compute your own ratios.)

### 4.2 Wuji finger bases in the palm frame, hand open

```
finger1 (thumb)  base [ 0.0085  0.0206  0.0287]   link4 [ 0.0116  0.0957  0.0562]
finger2 (index)  base [-0.0059  0.0294  0.0919]   link4 [-0.0118  0.0395  0.1682]
finger3 (middle) base [-0.0105  0.0075  0.0895]   link4 [-0.0197  0.0075  0.1661]
finger4 (ring)   base [-0.0077 -0.0137  0.0844]   link4 [-0.0163 -0.0211  0.1608]
finger5 (pinky)  base [-0.0023 -0.0340  0.0743]   link4 [-0.0094 -0.0501  0.1494]
```

Every non-thumb finger has an identical 0.0772 m base→link4 length.

### 4.3 A starting estimate for the wrist→palm rotation

MANO's rest pose extends its fingers along **−x** with the thumb splayed toward
**+z**. The Wuji palm extends fingers along **+z** with the thumb toward **+y**.
Matching those two axis pairs gives

```python
R_palm_manowrist = np.array([[0, -1, 0],
                             [0,  0, 1],
                             [-1, 0, 0]])   # v_palm = R @ v_mano
```

**This is a starting estimate, not a measurement.** MANO's flat-open pose and the
Wuji zero pose are not the same posture, so expect to refine it — most likely by
fitting rather than by reasoning. Treat it as an initialization for §5, not an
answer.

---

## 5. Suggested approach

Two sub-problems, and they are separable.

**Fingers.** Start with direct angle mapping: decompose each MANO joint's
axis-angle into flexion and abduction components and map MCP→joint1 (+joint2 for
spread), PIP→joint3, DIP→joint4, then clip. It is cheap and gives something to
look at.

If that isn't faithful enough, do position fitting: express MANO landmarks in the
palm frame and least-squares solve the 20 joint angles so the Wuji joint
positions match, using MuJoCo Jacobians on a scratch `MjData`. `urgantry_bc/ik.py`
has the damped-least-squares machinery to copy — same structure, different
Jacobian targets. This handles the 3-vs-4 joint mismatch gracefully, which the
direct mapping fundamentally cannot.

**Wrist.** Apply your calibrated `T_palm_manowrist` to `T_world_wrist`. Validate
by posing the hand and eyeballing it against `grasp.vertices` before trusting it.

---

## 6. What's already verified, so you can trust it

- **Camera model** — `python -m urgantry_hug.test_frames` unprojects a pixel back
  onto the object it came from: 0.00 mm off the camera ray, 1.3 mm in depth.
  Intrinsics, depth units, and the MuJoCo→OpenCV flip are all confirmed.
- **HUG on synthetic renders** — the domain gap does not break it. Clicking the
  block yields a wrist above/behind it with all five fingertips at z ≈ 0.805–0.813
  spanning x 0.289→0.338, against a block spanning x 0.295→0.345. The predicted
  hand wraps the block.
- **Executor** — `python -m urgantry_hug.test_execute` reaches, closes, and lifts
  the block to z = 0.928 from a rest height of 0.80.

---

## 7. Two constraints that will cost you time if you don't know them

**Open fingertips reach ~4.5 cm past the grasp center.** `GRASP_CENTER` is
`[0.035, 0.010, 0.115]` in the palm frame, but an open hand's fingertips sit at
z ≈ 0.16. So a top-down approach that puts the grasp center at an object's height
drives the *open* fingers straight through the object and into the table before
anything closes.

This is not hypothetical. Sweeping 18 top-down variants (three heights × three
closure amounts × two curl directions) failed every time — the block was swept
from x=0.32 to as far as x=0.70, and in one case knocked off the table entirely.
The same grasp with the fingers pointing **horizontally** lifted it on the first
try. Whatever posture you emit, check that the *pre-grasp* posture clears the
scene, not just the final one. `WujiGrasp.pregrasp` exists for exactly this.

**HUG predicts right hands only.** Use `side="right"`. The scene's right hand is
rolled 180° about the flange relative to the left (`HAND_ROLL` in
`build_urgantry.py`), so a left-hand version needs a mirror somewhere, not just a
renamed prefix.

---

## 8. Acceptance

Minimum bar:

1. `python -m urgantry_hug.demo` runs start to finish without an exception.
2. The posed Wuji hand visually matches the MANO target. Use
   `visualize.render_with_grasp(model, data, grasp, camera=...)` — it renders the
   **current sim state** with the MANO skeleton drawn over it, so once you have
   commanded the Wuji joints, one image shows both the target and what you
   achieved. Do this before caring about pick success.

   ```python
   from urgantry_hug.visualize import free_camera, render_with_grasp
   cam = free_camera(model, data.body("block").xpos, 0.55, azimuth=90, elevation=-10)
   img = render_with_grasp(model, data, mano_grasp, camera=cam)
   ```

   Render from at least two viewpoints. A single view hides depth error
   completely — that is exactly how a wrong frame convention survives review.
3. `python -m urgantry_hug.demo --viewer` shows a reach that doesn't scatter the
   block across the table.

Pick success is the goal but it is a poor first-order signal: it conflates
retargeting quality with grasp physics, and my own hand-built grasp needed the
right approach direction before it lifted anything. Judge the pose first.
