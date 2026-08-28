"""Rebuild a trained flow policy (backbone + head) for rollout."""

from __future__ import annotations

from pathlib import Path

import torch

from urgantry_bc.encoder import DINOv3Encoder
from urgantry_bc.flow_policy import FlowPolicy, FlowRunner


def load_flow_policy(path: str | Path, device: str = "cuda",
                     execute: int | None = None, flow_steps: int | None = None,
                     size: str = "s") -> FlowRunner:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = FlowPolicy(embed_dim=ckpt["embed_dim"], num_patches=ckpt["num_patches"],
                       state_dim=ckpt["state_dim"], action_dim=ckpt["action_dim"],
                       chunk=ckpt["chunk"], width=ckpt["width"],
                       num_blocks=ckpt["blocks"])
    model.load_state_dict(ckpt["model"])
    encoder = DINOv3Encoder(size=size, device=device,
                            image_size=ckpt.get("image_size", 224))
    return FlowRunner(model, encoder, ckpt["state_mean"], ckpt["state_std"],
                      device=device, execute=execute,
                      flow_steps=flow_steps or ckpt.get("flow_steps", 8))
