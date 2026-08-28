"""Cube-into-box task on the gantry scene, as a Gymnasium env.

Observation is exactly what was asked for: the top camera image plus
proprioception. Action drives the RIGHT arm only -- it is the arm that reaches
both the cube (+x side) and the tray (-x side); the left arm is held at its home
pose every step, so the policy never has to learn to keep it still.

  Action (7,)  : 6 right-arm joint targets + 1 hand closure, all in [-1, 1].
                 Joint targets map onto each actuator's ctrlrange; closure maps
                 onto [0, 1] and is expanded to the 20 hand joints by
                 hand.set_hand_closure (flexion only, abduction pinned at 0).
  Obs          : {'image': (H, W, 3) uint8 from 'top1',
                  'state': (13,) 6 arm qpos + 6 arm qvel + hand closure}
  Success      : the cube is resting inside the tray's footprint, below the rim.

The cube's start position is randomized per episode inside the right arm's
comfortable reach; the tray stays where the scene puts it.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import mujoco

from urgantry_sim.build_urgantry import (
    ARM_JOINTS, BLOCK_HALF, BLOCK_REST_Z, BOARD_TOP, BOX_FLOOR_T, BOX_INIT_POS,
    BOX_OUTER, BOX_WALL_T, build_model, set_initial_pose,
)
from urgantry_bc.hand import hand_ctrl, set_hand_closure

SIDE = "right"

# Cube start region: inside the right arm's reach, clear of the tray and the
# column strip. x is the arm's own side of the board.
CUBE_X_RANGE = (0.24, 0.40)
CUBE_Y_RANGE = (-0.16, 0.04)

BOX_IN_HALF = BOX_OUTER / 2 - BOX_WALL_T          # inner half-width, 0.074
IN_BOX_XY = BOX_IN_HALF - BLOCK_HALF + 0.012      # cube center tolerance, ~0.061
IN_BOX_Z = BOARD_TOP + BOX_FLOOR_T + 2 * BLOCK_HALF + 0.02
REST_SPEED = 0.05                                 # m/s below which the cube counts as settled


class CubeInBoxEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, image_size: int = 96, control_hz: float = 20.0,
                 max_episode_steps: int = 260, randomize_cube: bool = True,
                 seed: int | None = None):
        super().__init__()
        self.model = build_model()
        # Stiff position servos + the default Euler integrator chatter at the
        # timestep frequency; implicitfast keeps the parked arm actually parked.
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.data = mujoco.MjData(self.model)
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self.randomize_cube = randomize_cube
        self.n_substeps = max(1, round((1.0 / control_hz) / self.model.opt.timestep))
        self._renderer = mujoco.Renderer(self.model, height=image_size, width=image_size)
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

        m = self.model
        self._arm_act = np.array([m.actuator(f"{SIDE}_{j}").id for j in ARM_JOINTS])
        self._arm_qadr = np.array([m.joint(f"{SIDE}_{j}_joint").qposadr[0] for j in ARM_JOINTS])
        self._arm_dofadr = np.array([m.joint(f"{SIDE}_{j}_joint").dofadr[0] for j in ARM_JOINTS])
        self._hand_act = np.array(sorted(hand_ctrl(m, SIDE, 0.0)))
        self._arm_lo = m.actuator_ctrlrange[self._arm_act, 0].copy()
        self._arm_hi = m.actuator_ctrlrange[self._arm_act, 1].copy()
        self._block_qadr = m.joint("block_joint").qposadr[0]
        self._box_qadr = m.joint("cardboard_box_joint").qposadr[0]
        self._closure = 0.0

        self.action_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = Dict({
            "image": Box(low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8),
            "state": Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32),
        })

    # ------------------------------------------------------------------ helpers
    def arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self._arm_qadr].copy()

    def cube_pos(self) -> np.ndarray:
        return self.data.body("block").xpos.copy()

    def box_pos(self) -> np.ndarray:
        return self.data.body("cardboard_box").xpos.copy()

    def denorm_arm(self, a: np.ndarray) -> np.ndarray:
        a = np.clip(a, -1.0, 1.0)
        return self._arm_lo + (a + 1.0) * 0.5 * (self._arm_hi - self._arm_lo)

    def norm_arm(self, q: np.ndarray) -> np.ndarray:
        return np.clip(2.0 * (q - self._arm_lo) / (self._arm_hi - self._arm_lo) - 1.0, -1, 1)

    def _state(self) -> np.ndarray:
        return np.concatenate([
            self.data.qpos[self._arm_qadr],
            self.data.qvel[self._arm_dofadr],
            [self._closure],
        ]).astype(np.float32)

    def render_top(self) -> np.ndarray:
        self._renderer.update_scene(self.data, camera="top1")
        return self._renderer.render()

    def _obs(self) -> dict:
        return {"image": self.render_top(), "state": self._state()}

    # ------------------------------------------------------------------ task
    def cube_in_box(self) -> bool:
        """Cube resting inside the tray: within the inner footprint (in the tray's
        own current frame -- the tray is a free body and can be nudged), below the
        rim, and no longer moving."""
        cube, box = self.cube_pos(), self.box_pos()
        rel = cube - box
        if abs(rel[0]) > IN_BOX_XY or abs(rel[1]) > IN_BOX_XY:
            return False
        if cube[2] > IN_BOX_Z:
            return False
        speed = float(np.linalg.norm(self.data.qvel[
            self.model.joint("block_joint").dofadr[0]:
            self.model.joint("block_joint").dofadr[0] + 3]))
        return speed < REST_SPEED

    def _reward(self) -> tuple[float, bool]:
        """Sparse success plus shaping: approach the cube, lift it, carry it over
        the tray. Enough signal to log progress; the BC policy does not use it."""
        cube, box = self.cube_pos(), self.box_pos()
        palm = self.data.body(f"{SIDE}_hand_palm_link").xpos
        reach = float(np.linalg.norm(palm - cube))
        lift = max(0.0, cube[2] - BLOCK_REST_Z)
        over = float(np.linalg.norm(cube[:2] - box[:2]))
        success = self.cube_in_box()
        r = (1.0 - np.tanh(3.0 * reach)) * 0.1 + lift * 2.0 + (1.0 - np.tanh(2.0 * over)) * 0.5
        return (10.0 if success else r), success

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        set_initial_pose(self.model, self.data)
        self._closure = 0.0
        set_hand_closure(self.model, self.data, SIDE, 0.0)

        if self.randomize_cube:
            x = self._rng.uniform(*CUBE_X_RANGE)
            y = self._rng.uniform(*CUBE_Y_RANGE)
            self.data.qpos[self._block_qadr:self._block_qadr + 3] = [x, y, BLOCK_REST_Z]
        self.data.qpos[self._box_qadr:self._box_qadr + 7] = [*BOX_INIT_POS, 1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)
        # let the scene settle so the cube starts at rest on the board
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        self._step_count = 0
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(7)
        self.data.ctrl[self._arm_act] = self.denorm_arm(action[:6])
        self._closure = float(np.clip((action[6] + 1.0) * 0.5, 0.0, 1.0))
        set_hand_closure(self.model, self.data, SIDE, self._closure)

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        reward, success = self._reward()
        terminated = bool(success)
        truncated = self._step_count >= self.max_episode_steps
        info = {"success": int(success), "cube_pos": self.cube_pos(),
                "cube_height": float(self.cube_pos()[2] - BLOCK_REST_Z)}
        return self._obs(), reward, terminated, truncated, info

    def close(self):
        self._renderer.close()
