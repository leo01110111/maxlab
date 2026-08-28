# urgantry_bc — cube-into-box policy for the gantry scene

Trains a vision + proprioception policy that picks the green cube off the board
and drops it into the cardboard tray, on the `urgantry_sim` scene (two UR7e arms
hanging off a gantry column, Wuji hands).

The policy sees **only** the `top1` camera image and proprioception — never the
cube's pose. Demonstrations come from a scripted IK expert that does use the
cube pose; behavior cloning distills it into a pixel policy.

## Pipeline

The current policy is a **frozen DINOv3 ViT-S/16 backbone + flow-matching action
head**, ported to torch from `~/wuji-hands/student/wuji_bc` (`encoder.py`,
`nets.py`, `flow.py`). A small-CNN + MSE variant is kept for comparison
(`policy.py` / `train_bc.py`).

```bash
# 1. watch the scripted expert
uv run python -m urgantry_bc.demo_viewer

# 2. collect demonstrations at 224px (successful episodes only)
MUJOCO_GL=egl uv run python -m urgantry_bc.collect \
    --episodes 300 --image-size 224 --noise-std 0.002 --out urgantry_bc/data/demos224

# 3. cache the frozen backbone's features once
HF_HUB_OFFLINE=1 uv run --with transformers python -m urgantry_bc.cache_features \
    --data urgantry_bc/data/demos224 --preview 4

# 4. flow-matching BC
uv run python -m urgantry_bc.train_flow --data urgantry_bc/data/demos224 --epochs 60

# 5. evaluate on unseen cube positions
MUJOCO_GL=egl HF_HUB_OFFLINE=1 uv run --with transformers python -m urgantry_bc.evaluate \
    --flow urgantry_bc/runs/flow/policy.pt -n 50
MUJOCO_GL=egl uv run python -m urgantry_bc.evaluate --expert -n 50      # baseline

# 6. watch the trained policy
HF_HUB_OFFLINE=1 uv run --with transformers python -m urgantry_bc.demo_viewer \
    --flow urgantry_bc/runs/flow/policy.pt
```

`transformers` is pulled in per-command with `uv run --with` rather than added to
the project dependencies. The DINOv3 repo is license-gated but its weights are
already in the local HF cache, hence `HF_HUB_OFFLINE=1`.

## Interfaces

| | |
|---|---|
| Action | `(7,)` in `[-1, 1]`: 6 right-arm joint targets + hand closure |
| Action chunk | 16 predicted per observation, 8 executed before replanning |
| Observation | `image` `(224, 224, 3)` uint8 from `top1`; `state` `(13,)` = 6 arm qpos, 6 arm qvel, closure |
| Success | cube resting inside the tray footprint, below the rim, in the tray's own frame |
| Control | 20 Hz, `implicitfast` integrator |

Only the right arm is actuated — it is the one arm that reaches both the cube
(+x side) and the tray (−x side). The left arm holds its home pose.

## What the hand geometry actually is

Three things had to be measured on the compiled model rather than assumed, and
each one silently produced a non-grasping policy until it was fixed:

1. **Flexion is positive and closes toward palm +x.** The scene docstring says
   the grasping side is palm −x, and the hand XML's joint ranges *look*
   symmetric, but the compiled ranges are asymmetric (`joint1` is
   `[-0.33, +1.64]`). Fingertips travel from `(x=-0.015, z=0.16)` open to
   `(x=+0.06, z=0.09)` closed.
2. **Abduction must stay at 0.** The scene's `set_hand` drives every joint to
   one angle, which saturates abduction long before the hand has closed.
3. **The distal joints do the work.** Closing `joint3`/`joint4` to 1.0 rad cages
   the cube but it shakes loose during the carry; 1.5 rad wraps under it.

Grasp pose: palm +x pointing down, fingers pitched 30° below horizontal, grasp
center 1 cm **below** the cube center — aiming at the center closes the fingers
around the cube's upper half and it slips out on the lift.

## Two fixes that took the expert from 50% to ~90%

* **Carry in Cartesian legs.** Interpolating joints straight from the lift pose
  to the tray lets the palm tilt through the middle of the swing and the cube
  falls out. The carry is split into 5 IK-solved sub-waypoints at constant palm
  orientation.
* **Re-aim the release at the cube, not the hand.** The cube seats wherever the
  fingers caught it — up to ~12 cm from the grasp center. Delivering the grasp
  center over the tray center therefore drops the cube just outside the rim, a
  systematic miss that looked like random failure. After the lift the expert
  measures the cube's offset in the grip and subtracts it from the remaining
  targets.

## Files

| file | what |
|---|---|
| `ik.py` | damped least-squares IK on one arm, grasp-center frame |
| `hand.py` | Wuji closure mapping (flexion only, measured signs) |
| `task_env.py` | `CubeInBoxEnv` — the Gym task |
| `expert.py` | scripted IK demonstrator |
| `collect.py` | demo collection (keeps successes only, DART noise) |
| `encoder.py` | frozen DINOv3 ViT backbone (torch port) |
| `cache_features.py` | encode frames once into a float16 feature memmap |
| `flow_policy.py` | TokenPool + adaLN-Zero velocity field, flow loss and sampling |
| `train_flow.py` | flow-matching BC on cached features |
| `load_flow.py` | rebuild backbone + head for rollout |
| `policy.py`, `train_bc.py` | earlier CNN + MSE baseline |
| `evaluate.py` | success-rate rollouts on held-out seeds |
| `demo_viewer.py` | watch expert or policy in the interactive viewer |

## Why flow matching, and why chunks

The first attempt — single-step MSE regression on a small CNN — scored **0%**
despite a near-zero training loss, and the trace showed why: the hand closure
sat at ~0.6 forever, the *mean* of open (0) and closed (1). Where the image and
proprioception barely differ between "still descending" and "now close", an MSE
fit returns the average of the two branches. Predicting a 16-step chunk
conditions the whole burst on one observation, and a flow-matching head can
represent a branching action distribution instead of averaging it.

Two measurement mistakes worth not repeating:

* **Frame-level validation split.** It left frames of the same trajectory on both
  sides and reported val MSE 1e-5 while the policy scored 0%. Split by episode.
* **Noise in the wrong units.** DART noise of σ=0.03 in *normalized* action units
  is ~11° of joint noise every 50 ms (1 unit = 6.28 rad), which collapsed the
  expert from 89% to 1.75%. σ=0.002 is the right scale.
