"""Encode the collected frames once with the frozen DINOv3 backbone.

The backbone never trains, so its output is a constant function of the dataset:
computing it once turns each training epoch from a ViT forward pass over 60k
images into a memmap read.

    HF_HUB_OFFLINE=1 uv run --with transformers python -m urgantry_bc.cache_features \
        --data urgantry_bc/data/demos224 --preview 4

Writes features.npy as (N, 1 + num_patches, embed_dim) float16 next to the data.
`--preview` PCAs a few patch grids to RGB first: working features look like a
crude segmentation of the scene, noise means the preprocessing or weights are wrong.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def preview(images: np.ndarray, enc, idx, out: Path) -> None:
    from PIL import Image

    imgs = np.asarray(images[idx])
    patches = enc.encode_numpy(imgs)[:, 1:].astype(np.float32)
    mean = patches.reshape(-1, enc.embed_dim).mean(0)
    comps = np.linalg.svd(patches.reshape(-1, enc.embed_dim) - mean,
                          full_matrices=False)[2][:3]
    proj = (patches - mean) @ comps.T
    lo, hi = proj.min((0, 1)), proj.max((0, 1))
    proj = ((proj - lo) / (hi - lo) * 255).astype(np.uint8)
    g = int(round(enc.num_patches ** 0.5))
    scale = imgs.shape[1] // g
    pca = proj.reshape(len(idx), g, g, 3).repeat(scale, 1).repeat(scale, 2)
    grid = np.concatenate([np.concatenate(list(imgs), axis=1),
                           np.concatenate(list(pca), axis=1)], axis=0)
    Image.fromarray(grid).save(out)
    print(f"wrote {out} (top row: frames, bottom row: PCA'd patch grids)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("urgantry_bc/data/demos224"))
    p.add_argument("--size", default="s", choices=("s", "b", "l"))
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--preview", type=int, default=0)
    args = p.parse_args()

    from urgantry_bc.encoder import DINOv3Encoder

    images = np.load(args.data / "images.npy", mmap_mode="r")
    enc = DINOv3Encoder(size=args.size, image_size=images.shape[1])
    print(f"{len(images)} frames, embed_dim={enc.embed_dim}, "
          f"num_patches={enc.num_patches}")

    if args.preview:
        idx = np.sort(np.random.default_rng(0).choice(len(images), args.preview,
                                                      replace=False))
        preview(images, enc, idx, args.data / "feature_preview.png")

    n = len(images)
    out_path = args.data / "features.npy"
    feats = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float16,
        shape=(n, 1 + enc.num_patches, enc.embed_dim))
    t0 = time.time()
    for i in range(0, n, args.batch):
        feats[i:i + args.batch] = enc.encode_numpy(
            np.asarray(images[i:i + args.batch]), batch_size=args.batch)
        if (i // args.batch) % 20 == 0:
            done = min(i + args.batch, n)
            print(f"  {done}/{n} ({time.time()-t0:.0f}s)", flush=True)
    feats.flush()
    print(f"wrote {out_path} {feats.shape} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
