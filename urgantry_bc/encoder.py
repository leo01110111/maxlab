"""Frozen DINOv3 vision backbone -- pixels to patch tokens.

Torch port of ~/wuji-hands/student/wuji_bc/encoder.py (that one is JAX-side and
returns numpy; this one stays in torch so the policy can call it inline during
rollout as well as offline for caching).

The HF repo is license-gated; the weights are already in the local HF cache, so
runs set HF_HUB_OFFLINE=1 rather than re-authenticating.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

IMAGE_SIZE = 224
PATCH = 16

# DINOv3 inherits DINOv2's ImageNet preprocessing.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

REPOS = {
    "s": "facebook/dinov3-vits16-pretrain-lvd1689m",   # 384-dim, 21M
    "b": "facebook/dinov3-vitb16-pretrain-lvd1689m",   # 768-dim, 86M
    "l": "facebook/dinov3-vitl16-pretrain-lvd1689m",   # 1024-dim, 300M
}


class DINOv3Encoder(nn.Module):
    """Frozen DINOv3 ViT. Returns [CLS] + patch tokens, registers dropped.

    Output is (B, 1 + num_patches, embed_dim); index 0 is CLS and the rest is the
    patch grid, row-major, reshapeable to (g, g, embed_dim).
    """

    def __init__(self, size: str = "s", device: str = "cuda", dtype=torch.float16,
                 image_size: int = IMAGE_SIZE):
        super().__init__()
        from transformers import AutoModel

        if size not in REPOS:
            raise ValueError(f"unknown size {size!r}, expected one of {sorted(REPOS)}")
        if image_size % PATCH:
            raise ValueError(f"image_size must be a multiple of {PATCH}, got {image_size}")
        self.model = AutoModel.from_pretrained(REPOS[size]).eval().to(device=device,
                                                                     dtype=dtype)
        for p in self.model.parameters():
            p.requires_grad = False
        cfg = self.model.config
        self.embed_dim = cfg.hidden_size
        # ViT position embeddings interpolate, so any multiple of the patch size
        # works. Resolution is what decides whether a 5 cm cube is even one token:
        # at 224 it spans 11 px (under one 16 px patch), at 448 about 22.
        self.image_size = image_size
        self.num_patches = (image_size // PATCH) ** 2
        self.n_registers = getattr(cfg, "num_register_tokens", 0)
        self.device = device
        self.dtype = dtype
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, H, W, 3) uint8 or (B, 3, H, W) float already in [0, 1]."""
        if images.dtype == torch.uint8:
            images = images.permute(0, 3, 1, 2).float().div_(255.0)
        x = (images.to(self.device) - self.mean.to(self.device)) / self.std.to(self.device)
        tokens = self.model(pixel_values=x.to(self.dtype)).last_hidden_state
        cls, patches = tokens[:, :1], tokens[:, 1 + self.n_registers:]
        return torch.cat([cls, patches], dim=1)

    @torch.inference_mode()
    def encode_numpy(self, images: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """(N, S, S, 3) uint8 -> (N, 1 + num_patches, embed_dim) float16."""
        s = self.image_size
        if images.shape[1:] != (s, s, 3):
            raise ValueError(f"expected (N, {s}, {s}, 3) uint8, got {images.shape}")
        out = []
        for i in range(0, len(images), batch_size):
            chunk = torch.from_numpy(np.ascontiguousarray(images[i:i + batch_size]))
            out.append(self(chunk.to(self.device)).to(torch.float16).cpu().numpy())
        return np.concatenate(out)
