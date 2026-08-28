"""Vision + proprioception policy for the cube-into-box task.

A small conv encoder over the top-camera image, concatenated with the
proprioceptive vector, into an MLP that regresses the 7-dim action. Deliberately
small: the scene is fixed and only the cube's start position varies, so capacity
is not the bottleneck -- data is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    """Predicts a CHUNK of future actions, not just the next one.

    Single-step regression is ambiguous here: while the hand hovers at the grasp
    pose, the image and proprioception barely change between "still descending"
    and "now close", so an MSE fit returns the average of the two -- a closure
    that sits at 0.6 forever instead of going 0 -> 1. Predicting `chunk` steps at
    once conditions the whole burst on one observation, which resolves the phase.
    """

    def __init__(self, image_size: int = 84, state_dim: int = 13, action_dim: int = 7,
                 chunk: int = 16, width: int = 32, feat_dim: int = 256):
        super().__init__()
        self.image_size = image_size
        self.chunk = chunk
        self.action_dim = action_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, width, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.encoder(torch.zeros(1, 3, image_size, image_size)).shape[1]
        self.img_proj = nn.Sequential(nn.Linear(n_flat, feat_dim), nn.LayerNorm(feat_dim),
                                      nn.Tanh())
        self.head = nn.Sequential(
            nn.Linear(feat_dim + state_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, action_dim * chunk),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """image: (B, 3, H, W) normalized; state: (B, state_dim).
        Returns (B, chunk, action_dim)."""
        z = self.img_proj(self.encoder(image))
        out = self.head(torch.cat([z, state], dim=1))
        return out.view(-1, self.chunk, self.action_dim)


def normalize_images(x: torch.Tensor) -> torch.Tensor:
    return x.float().div_(255.0).sub_(0.5).div_(0.5)


class PolicyRunner:
    """Callable wrapper turning an env observation dict into an action.

    Replans every `execute` steps and plays the predicted chunk open-loop in
    between: fewer replans means less chance of stepping off-distribution
    mid-motion, at the cost of reacting more slowly.
    """

    def __init__(self, model: BCPolicy, state_mean, state_std, device="cuda",
                 execute: int | None = None):
        self.model = model.to(device).eval()
        self.device = device
        self.state_mean = torch.as_tensor(state_mean, dtype=torch.float32, device=device)
        self.state_std = torch.as_tensor(state_std, dtype=torch.float32, device=device)
        self.execute = execute or max(1, model.chunk // 2)
        self._queue: list[np.ndarray] = []

    @torch.no_grad()
    def _predict_chunk(self, obs: dict) -> np.ndarray:
        img = torch.as_tensor(np.ascontiguousarray(obs["image"]),
                              device=self.device).permute(2, 0, 1)[None]
        img = normalize_images(img.clone())
        st = torch.as_tensor(obs["state"], dtype=torch.float32, device=self.device)[None]
        st = (st - self.state_mean) / self.state_std
        return self.model(img, st)[0].cpu().numpy()

    def __call__(self, obs: dict) -> np.ndarray:
        if not self._queue:
            chunk = self._predict_chunk(obs)
            self._queue = list(chunk[:self.execute])
        return np.clip(self._queue.pop(0), -1.0, 1.0)

    def reset(self):
        self._queue = []


def load_policy(path: str | Path, device: str = "cuda",
                execute: int | None = None) -> PolicyRunner:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = BCPolicy(image_size=ckpt["image_size"], state_dim=ckpt["state_dim"],
                     action_dim=ckpt["action_dim"], chunk=ckpt.get("chunk", 1))
    model.load_state_dict(ckpt["model"])
    return PolicyRunner(model, ckpt["state_mean"], ckpt["state_std"], device=device,
                        execute=execute)
