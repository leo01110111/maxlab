"""Flow-matching action head over frozen DINOv3 tokens.

Torch port of ~/wuji-hands/student/wuji_bc/{nets,flow}.py:

  * TokenPool          -- patch tokens to one conditioning vector, via three
                          readouts: spatial-softmax keypoints (where), attention
                          pooling (what) and CLS (global).
  * AdaLNVelocityField -- v_theta(obs, x_t, t) with adaLN-Zero time conditioning;
                          modulation and output projections are zero-init so the
                          net starts as the identity and outputs zero velocity.
  * flow_targets/loss  -- x_t = (1-t) x_0 + t x_1, regress v = x_1 - x_0.
  * sample_actions     -- Euler-integrate the velocity field from noise.

The action being sampled is the whole chunk flattened: action_dim = chunk * 7.
Flow matching replaces the MSE head that collapsed to the mean of open and
closed at the grasp -- a velocity field can represent that branch instead of
averaging it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def time_features(t: torch.Tensor, num_freqs: int = 32, max_freq: float = 100.0):
    """(B, 1) time -> (B, 2*num_freqs) sin/cos features, log-spaced in [1, max_freq]."""
    freqs = torch.exp(torch.linspace(0.0, float(np.log(max_freq)), num_freqs,
                                     device=t.device, dtype=t.dtype))
    ang = t * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class TokenPool(nn.Module):
    """Frozen DINOv3 tokens -> one conditioning vector."""

    def __init__(self, embed_dim: int, num_patches: int, out_dim: int = 128,
                 num_queries: int = 4, num_keypoints: int = 16):
        super().__init__()
        self.g = int(round(num_patches ** 0.5))
        self.num_queries = num_queries
        self.norm_in = nn.LayerNorm(embed_dim)
        self.heat = nn.Linear(embed_dim, num_keypoints)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.query = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        in_dim = num_keypoints * 2 + num_queries * embed_dim + embed_dim
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm_out = nn.LayerNorm(out_dim)

        coord = torch.linspace(-1.0, 1.0, self.g)
        gy, gx = torch.meshgrid(coord, coord, indexing="ij")
        self.register_buffer("gx", gx.reshape(-1))
        self.register_buffer("gy", gy.reshape(-1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, 1 + P, D), index 0 is CLS -> (B, out_dim)."""
        tokens = self.norm_in(tokens)
        cls, patches = tokens[:, 0], tokens[:, 1:]
        b, p, d = patches.shape

        # Expected patch coordinate per channel. Attention pooling alone averages
        # patch *content*, which carries no position; without this the grid's
        # spatial layout -- where the cube is -- is thrown away.
        heat = F.softmax(self.heat(patches), dim=1)              # (B, P, K)
        kx = torch.einsum("bpk,p->bk", heat, self.gx)
        ky = torch.einsum("bpk,p->bk", heat, self.gy)
        keypoints = torch.stack([kx, ky], dim=-1).reshape(b, -1)

        x = patches + self.pos_embed
        attn = F.softmax(torch.einsum("qd,bnd->bqn", self.query, x) / d ** 0.5, dim=-1)
        pooled = torch.einsum("bqn,bnd->bqd", attn, x).reshape(b, -1)

        h = torch.cat([keypoints, pooled, cls], dim=-1)
        return self.norm_out(self.proj(h))


class AdaLNBlock(nn.Module):
    def __init__(self, width: int, time_dim: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.mod = nn.Linear(time_dim, 3 * width)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)
        self.fc1 = nn.Linear(width, mlp_ratio * width)
        self.fc2 = nn.Linear(mlp_ratio * width, width)

    def forward(self, h, c):
        scale, shift, gate = self.mod(F.silu(c)).chunk(3, dim=-1)
        y = self.norm(h) * (1 + scale) + shift
        y = self.fc2(F.silu(self.fc1(y)))
        return h + gate * y


class AdaLNVelocityField(nn.Module):
    """v_theta(obs, x_t, t): obs and x_t go through the trunk, t modulates it."""

    def __init__(self, action_dim: int, obs_dim: int, width: int = 512,
                 num_blocks: int = 4, mlp_ratio: int = 2, time_dim: int = 128,
                 num_freqs: int = 32, max_freq: float = 100.0):
        super().__init__()
        self.num_freqs, self.max_freq = num_freqs, max_freq
        self.t_fc1 = nn.Linear(2 * num_freqs, time_dim)
        self.t_fc2 = nn.Linear(time_dim, time_dim)
        self.inp = nn.Linear(obs_dim + action_dim, width)
        self.blocks = nn.ModuleList(
            [AdaLNBlock(width, time_dim, mlp_ratio) for _ in range(num_blocks)])
        self.norm_out = nn.LayerNorm(width, elementwise_affine=False)
        self.mod_out = nn.Linear(time_dim, 2 * width)
        nn.init.zeros_(self.mod_out.weight)
        nn.init.zeros_(self.mod_out.bias)
        self.out = nn.Linear(width, action_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, obs, x_t, t):
        c = self.t_fc2(F.silu(self.t_fc1(time_features(t, self.num_freqs, self.max_freq))))
        h = self.inp(torch.cat([obs, x_t], dim=-1))
        for blk in self.blocks:
            h = blk(h, c)
        scale, shift = self.mod_out(F.silu(c)).chunk(2, dim=-1)
        h = self.norm_out(h) * (1 + scale) + shift
        return self.out(h)


class FlowPolicy(nn.Module):
    """TokenPool over DINOv3 tokens + proprioception -> velocity field over a
    flattened action chunk."""

    def __init__(self, embed_dim: int = 384, num_patches: int = 196,
                 state_dim: int = 13, action_dim: int = 7, chunk: int = 16,
                 vision_dim: int = 128, width: int = 512, num_blocks: int = 4):
        super().__init__()
        self.chunk = chunk
        self.action_dim = action_dim
        self.flat_dim = action_dim * chunk
        self.pool = TokenPool(embed_dim, num_patches, out_dim=vision_dim)
        self.field = AdaLNVelocityField(self.flat_dim, vision_dim + state_dim,
                                        width=width, num_blocks=num_blocks)

    def obs_vector(self, tokens, state):
        return torch.cat([state, self.pool(tokens)], dim=-1)

    def forward(self, tokens, state, x_t, t):
        return self.field(self.obs_vector(tokens, state), x_t, t)


def flow_targets(actions: torch.Tensor, generator=None):
    """x_0 ~ N(0, I), t ~ U(0, 1), x_t = (1-t) x_0 + t x_1, v = x_1 - x_0."""
    x_0 = torch.randn(actions.shape, device=actions.device, generator=generator)
    t = torch.rand((actions.shape[0], 1), device=actions.device, generator=generator)
    x_t = (1 - t) * x_0 + t * actions
    return x_t, actions - x_0, t


def flow_loss(model: FlowPolicy, tokens, state, actions, generator=None):
    x_t, v_target, t = flow_targets(actions, generator)
    return F.mse_loss(model(tokens, state, x_t, t), v_target)


@torch.no_grad()
def sample_actions(model: FlowPolicy, tokens, state, flow_steps: int = 8):
    """Euler-integrate the velocity field from noise to an action chunk."""
    b = state.shape[0]
    x = torch.randn(b, model.flat_dim, device=state.device)
    obs = model.obs_vector(tokens, state)
    for i in range(flow_steps):
        t = torch.full((b, 1), i / flow_steps, device=state.device)
        x = x + model.field(obs, x, t) * (1.0 / flow_steps)
    return x.clamp(-1, 1).view(b, model.chunk, model.action_dim)


class FlowRunner:
    """Callable actor: renders -> DINOv3 tokens -> sampled action chunk."""

    def __init__(self, model: FlowPolicy, encoder, state_mean, state_std,
                 device="cuda", execute: int | None = None, flow_steps: int = 8):
        self.model = model.to(device).eval()
        self.encoder = encoder
        self.device = device
        self.flow_steps = flow_steps
        self.execute = execute or max(1, model.chunk // 2)
        self.state_mean = torch.as_tensor(state_mean, dtype=torch.float32, device=device)
        self.state_std = torch.as_tensor(state_std, dtype=torch.float32, device=device)
        self._queue: list[np.ndarray] = []

    def __call__(self, obs: dict) -> np.ndarray:
        if not self._queue:
            img = torch.from_numpy(np.ascontiguousarray(obs["image"]))[None].to(self.device)
            tokens = self.encoder(img).float()
            st = torch.as_tensor(obs["state"], dtype=torch.float32,
                                 device=self.device)[None]
            st = (st - self.state_mean) / self.state_std
            chunk = sample_actions(self.model, tokens, st, self.flow_steps)[0].cpu().numpy()
            self._queue = list(chunk[:self.execute])
        return np.clip(self._queue.pop(0), -1.0, 1.0)

    def reset(self):
        self._queue = []
