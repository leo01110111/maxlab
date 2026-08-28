"""Run HUG on a sim capture and return the grasp in world coordinates.

Mirrors the inference path in hug's src/app.py:handle_click, but headless and
driven by a pixel we choose rather than a mouse click.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from hug.dataloader.grasp_dataset import GraspDataset
from hug.inference import load_model, resolve_checkpoint_path
from hug.models.mano import mano_params_to_grasp_dict
from hug.prepare_inputs import TARGET_SIZE, prepare_pkl
from hug.utils.pcl_utils import depth_to_pcl_tensors, pixel_to_xyz

from .capture import Capture
from .frames import transform_points, transform_pose, unproject
from .interface import ManoGrasp

DEFAULT_CHECKPOINT = Path.home() / "hug" / "checkpoints" / "hug_full.safetensors"


def crop_to_224(uv, width: int, height: int) -> np.ndarray:
    """Full-resolution pixel -> its coordinate in the 224 center crop.

    prepare_pkl center-crops to the shorter side then resizes, so a click has to
    travel the same path or it points at a different part of the scene.
    """
    size = min(width, height)
    x_off = (width - size) // 2
    y_off = (height - size) // 2
    scale = TARGET_SIZE / size
    u = (float(uv[0]) - x_off) * scale
    v = (float(uv[1]) - y_off) * scale
    return np.array([np.clip(u, 0, TARGET_SIZE - 1), np.clip(v, 0, TARGET_SIZE - 1)])


class HugPredictor:
    """Loads the HUG checkpoint once; call it per grasp."""

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, device: Optional[str] = None,
                 use_ema: bool = True, sampling_steps: int = 50):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_steps = sampling_steps
        self.model = load_model(resolve_checkpoint_path(Path(checkpoint_path)),
                                use_ema, self.device)
        self.use_rgb = getattr(self.model, "use_rgb", True)
        self.use_depth = getattr(self.model, "use_depth", False)
        self.pcl_use_rgb = getattr(self.model, "pcl_use_rgb", False)
        self._tmp = tempfile.TemporaryDirectory(prefix="urgantry_hug_")
        self._dir = Path(self._tmp.name)

    def _sample(self, capture: Capture, stem: str = "frame"):
        prepare_pkl(capture.rgb, capture.depth_mm, capture.K, stem, self._dir)
        dataset = GraspDataset(str(self._dir), split="val",
                               use_rgb=self.use_rgb, use_depth=self.use_depth)
        return dataset.get_inference_data(stem)

    def predict(self, capture: Capture, uv, seed: Optional[int] = None) -> ManoGrasp:
        """Predict a grasp at full-resolution pixel `uv` of `capture`."""
        if seed is not None:
            torch.manual_seed(seed)

        width, height = capture.size
        uv224 = crop_to_224(uv, width, height)
        sample = self._sample(capture)
        K224 = np.asarray(sample["camera_K"], dtype=np.float64)

        depth_image = sample.get("depth_image")
        depth_m = 0.0
        if depth_image is not None:
            dh, dw = depth_image.shape[:2]
            du = int(np.clip(uv224[0] * dw / TARGET_SIZE, 0, dw - 1))
            dv = int(np.clip(uv224[1] * dh / TARGET_SIZE, 0, dh - 1))
            depth_m = float(depth_image[dv, du]) / 1000.0
        if depth_m <= 0.0:
            raise ValueError(f"no valid depth at pixel {tuple(np.round(uv, 1))}")

        point_uv = torch.tensor([[uv224[0], uv224[1], depth_m]],
                                dtype=torch.float32, device=self.device)
        camera_K = torch.from_numpy(K224).float().unsqueeze(0).to(self.device)
        rgb = sample["rgb"].unsqueeze(0).to(self.device) if self.use_rgb else None

        pcl_xyz = pcl_rgb = None
        if self.use_depth:
            crop_r = getattr(self.model, "pcl_crop_radius", None)
            if crop_r is not None and depth_image is not None:
                center = pixel_to_xyz(uv224[0], uv224[1], depth_m, K224)
                xyz, rgb_pcl = depth_to_pcl_tensors(
                    depth_image.astype(np.float32) / 1000.0,
                    sample["rgb_original"], K224,
                    center=center, crop_radius=crop_r)
                pcl_xyz = xyz.unsqueeze(0).to(self.device)
                pcl_rgb = rgb_pcl.unsqueeze(0).to(self.device) if self.pcl_use_rgb else None
            else:
                pcl_xyz = sample["pcl_xyz"].unsqueeze(0).to(self.device)
                pcl_rgb = (sample["pcl_rgb"].unsqueeze(0).to(self.device)
                           if self.pcl_use_rgb else None)

        with torch.no_grad():
            preds = self.model.sample(point_uv, camera_K, steps=self.sampling_steps,
                                      rgb=rgb, pcl_xyz=pcl_xyz, pcl_rgb=pcl_rgb)

        pred = mano_params_to_grasp_dict(preds[0], self.model.fixed_betas.squeeze(0),
                                         self.model.mano, K224, self.model.mesh_faces)

        T = capture.T_world_cam
        return ManoGrasp(
            T_world_wrist=transform_pose(T, pred["T_camera_wrist"]),
            pose=np.asarray(pred["pose"]).reshape(15, 3),
            landmarks=transform_points(T, pred["landmarks_3d"]),
            vertices=transform_points(T, pred["mesh_vertices"]),
            faces=pred["mesh_faces"],
            T_world_cam=T,
            click_uv=uv224,
            click_world=transform_points(
                T, unproject(uv224[0], uv224[1], depth_m, K224)[None])[0],
        )

    def close(self) -> None:
        self._tmp.cleanup()
