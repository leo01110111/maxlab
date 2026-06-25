# Policy evaluation interface

The MuJoCo sim is used **for evaluation only** (not training). It exposes one
interface — **camera images + joint positions in, joint targets out** — through
a single Gymnasium env, `env.py:SimBimanualUR7eEnv`. Two policy ecosystems consume
it:

| | OGPO | OpenPi (VLA / BC) |
|---|---|---|
| Where the policy runs | in-process (JAX) | remote **websocket server** |
| Sim's role | the `gym.Env` it rolls out | a **client** of the server |
| Action space | `Box(-1, 1)` (`normalized_actions=True`) | raw joint targets (`normalized_actions=False`) |
| Observation | `{'state', 'image'}` dict | `{observation/image, observation/wrist_image, observation/state, prompt}` |
| Entry point | use `SimBimanualUR7eEnv` directly | `openpi_eval.py` |

## Observation / action contract

- **Observation** `SimBimanualUR7eEnv._get_obs()`:
  - `state`: `(14,)` float32 — 12 arm joint positions + 2 gripper drivers.
  - `image`: `(H, W, 3*n_cam)` uint8 — cameras (`top1`, `top2`) concatenated on channels.
- **Action**: 14 actuator position targets (12 arm + 2 gripper). Normalized
  `[-1,1]` actions map onto each actuator's `ctrlrange`; raw actions are arm
  radians + gripper `0=open..255=closed`.

## OpenPi (remote VLA)

The policy is served separately and holds the model/GPU; the sim is the client.

```bash
# 1) Start a policy server in the openpi repo (separate process/host):
uv run scripts/serve_policy.py --env=DROID          # or your own config

# 2) Install the lightweight client into THIS venv (one time):
uv pip install -e /path/to/openpi/packages/openpi-client

# 3) Run the eval client against the sim:
uv run python openpi_eval.py --host 127.0.0.1 --port 8000 \
    --prompt "pick up the block" --episodes 5 --replan-steps 5

# Add --view to open a live MuJoCo window and watch the rollout:
uv run python openpi_eval.py --host 127.0.0.1 --port 8000 \
    --prompt "pick up the block" --view
```

`openpi_eval.py` builds the obs with `env.get_openpi_observation()`, resizes
images to 224×224 (via `openpi_client.image_tools` when available), calls
`client.infer(obs)["actions"]`, and replays the action chunk open-loop,
replanning every `--replan-steps` steps.

> Camera → key mapping (`top1 → observation/image`, `top2 → observation/wrist_image`)
> and the action dimension/order must match what the checkpoint was trained on.
> Adjust `get_openpi_observation()` for your checkpoint.

## OGPO (in-process)

OGPO's eval (`ogpo/utils/evaluation.py`) wants a Gymnasium env returning a
`{'state', 'image'}` dict, `Box(-1,1)` actions, and `info['success']` — which
`SimBimanualUR7eEnv(normalized_actions=True)` provides. To plug in, add a branch in
OGPO's `envs/env_utils.py:make_env` that returns it, e.g.:

```python
# in OGPO/envs/env_utils.py
if env_name.startswith("maxlab"):
    import sys; sys.path.append("/home/leo/Documents/maxlab/models")
    from env import SimBimanualUR7eEnv
    # show_viewer=True opens a live MuJoCo window during eval rollouts.
    return SimBimanualUR7eEnv(normalized_actions=True, image_size=96,
                           show_viewer=("render" in env_name))
```

The `show_viewer` flag is the viewer option for both ecosystems: pass it to
`SimBimanualUR7eEnv(...)` from OGPO (e.g. gate it on the env name as above, so
`maxlab_render` opens a window and `maxlab` stays headless), or use `--view`
on `openpi_eval.py`. It launches a passive viewer alongside the offscreen
camera renderer and `sync()`s it every `reset()`/`step()`.

## Task

Each arm has a **Robotiq 2F-85** gripper (mounted on the UR7e `attachment_site`),
and a red **block** sits on the board. The task is to **lift the block**:

- `step()` returns `reward = 1.0` on success (else a small block-height shaping
  term), `terminated = True` on success, and `info = {success, block_height}`.
- Success criterion: `build_urtable.pick_success` — block lifted `LIFT_SUCCESS_H`
  (5 cm) above its rest height. Tune `BLOCK_INIT_POS` / `LIFT_SUCCESS_H` there.

Action dim is now **14** (12 arm joints + 2 gripper drivers, ctrl `0=open..255=closed`).
The policy/checkpoint must emit 14-D actions in this order (see env actuator list).
