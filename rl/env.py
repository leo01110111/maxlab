"""State-based single-arm pick environment for PPO training.

The eval env (env.py:SimBimanualUR7eEnv) exposes a camera+proprio observation
meant for VLA/BC policies and only a sparse lift reward. That is the wrong
interface for training a controller from scratch: pixels make PPO enormously
sample-hungry, the full 14-actuator bimanual action space is needless (the
green block sits in the LEFT arm's reach), and the flat reward gives almost no
gradient before the first accidental lift.

This env fixes all three for RL:

  * Action  : 7 = left arm (6 joints) + left gripper. The right arm is held at
              its home pose every step. Normalized to Box(-1, 1).
  * Obs     : low-dim state vector (joints + gripper->block vector + block pose
              + block height), no rendering. Fast and Markov.
  * Reward  : dense shaping — reach the block, close the gripper on it, lift it —
              plus a sparse success bonus once it clears LIFT_SUCCESS_H.

Same physics model as the eval env (build_urtable.build_model), so a policy
trained here can be rolled out through the eval env for rendering/inspection.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import mujoco
import mujoco.viewer

from urtable_sim.build_urtable import (
    build_model, set_initial_pose, block_height, pick_success,
    BLOCK_REST_Z, BLOCK_INIT_POS, LIFT_SUCCESS_H, GRIPPER_OPEN, GRIPPER_CLOSED,
    apply_initial_view,
)

LEFT_ACTUATORS = 7          # actuators 0..6: left 6 arm joints + left gripper
PINCH_SITE = "left_grip_pinch"


class PickCubeEnv(gym.Env):
    """Left-arm pick of the green cube, state observation, shaped reward."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        control_hz: float = 20.0,
        max_episode_steps: int = 200,
        block_pos_noise: float = 0.06,
        render_mode: str | None = None,
        image_size: int = 224,
        show_viewer: bool = False,
    ):
        super().__init__()
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.max_episode_steps = max_episode_steps
        self.block_pos_noise = block_pos_noise
        self.render_mode = render_mode
        self.image_size = image_size
        self._renderer = None

        # Optional live on-screen viewer (for watching an eval); separate from the
        # offscreen rgb_array renderer used for video.
        self.show_viewer = show_viewer
        self._viewer = None
        if show_viewer:
            self._viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False)
            apply_initial_view(self._viewer)
            self._viewer.sync()

        self.n_substeps = max(1, round((1.0 / control_hz) / self.model.opt.timestep))
        self._step_count = 0

        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self._pinch_id = self.model.site(PINCH_SITE).id
        self._block_qadr = self.model.joint("block_joint").qposadr[0]
        self._left_qadr = np.array(
            [self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]
             for i in range(LEFT_ACTUATORS)])

        self.action_space = Box(-1.0, 1.0, shape=(LEFT_ACTUATORS,), dtype=np.float32)
        obs_dim = self._get_obs().shape[0]
        self.observation_space = Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._prev_dist = None

    # ------------------------------------------------------------- helpers
    def _pinch_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._pinch_id].copy()

    def _block_pos(self) -> np.ndarray:
        return self.data.xpos[self.model.body("block").id].copy()

    def _get_obs(self) -> np.ndarray:
        joints = self.data.qpos[self._left_qadr].astype(np.float32)
        pinch = self._pinch_pos()
        block = self._block_pos()
        to_block = block - pinch
        grip_ctrl = np.array([self.data.ctrl[6] / GRIPPER_CLOSED], dtype=np.float32)
        return np.concatenate([
            joints,                              # 7
            pinch.astype(np.float32),            # 3
            block.astype(np.float32),            # 3
            to_block.astype(np.float32),         # 3
            [np.linalg.norm(to_block)],          # 1
            [block[2] - BLOCK_REST_Z],           # 1 lift height
            grip_ctrl,                           # 1
        ]).astype(np.float32)

    # ----------------------------------------------------------------- gym
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        set_initial_pose(self.model, self.data)
        if self.block_pos_noise > 0:
            dx, dy = self.np_random.uniform(-self.block_pos_noise, self.block_pos_noise, size=2)
            self.data.qpos[self._block_qadr + 0] = BLOCK_INIT_POS[0] + dx
            self.data.qpos[self._block_qadr + 1] = BLOCK_INIT_POS[1] + dy
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._prev_dist = float(np.linalg.norm(self._block_pos() - self._pinch_pos()))
        self._sync_viewer()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # Map the 7 normalized left actions onto their ctrlrange; hold the right
        # arm + right gripper at their commanded home ctrl (set in reset).
        lo, hi = self._ctrl_low[:LEFT_ACTUATORS], self._ctrl_high[:LEFT_ACTUATORS]
        self.data.ctrl[:LEFT_ACTUATORS] = lo + (action + 1.0) * 0.5 * (hi - lo)

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        self._sync_viewer()

        pinch = self._pinch_pos()
        block = self._block_pos()
        dist = float(np.linalg.norm(block - pinch))
        lift = block_height(self.model, self.data) - BLOCK_REST_Z
        success = pick_success(self.model, self.data)

        # Dense shaping:
        #   reach  : reward getting the pinch site closer to the block
        #   close  : once near, reward closing the gripper (encourages a grasp)
        #   lift   : reward height gained off the table
        #   bonus  : large sparse reward for a completed pick
        reach = self._prev_dist - dist                       # potential-based
        near = dist < 0.06
        grip_ctrl = self.data.ctrl[6] / GRIPPER_CLOSED       # 0 open .. 1 closed
        close = 0.05 * grip_ctrl if near else 0.0
        reward = 10.0 * reach + close + 20.0 * max(0.0, lift)
        if success:
            reward += 100.0
        self._prev_dist = dist

        terminated = bool(success)
        truncated = self._step_count >= self.max_episode_steps
        info = {
            "success": int(success),
            "block_height": float(block[2]),
            "dist": dist,
            "lift": float(lift),
            "is_success": bool(success),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.image_size, self.image_size)
        self._renderer.update_scene(self.data, camera="top1")
        return self._renderer.render()

    def _sync_viewer(self) -> None:
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


ENV_ID = "PickCube-v0"
if ENV_ID not in gym.envs.registry:
    gym.register(id=ENV_ID, entry_point="rl.env:PickCubeEnv")
