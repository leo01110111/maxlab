"""Behavior cloning on the scripted-expert demonstrations.

    uv run python -m urgantry_bc.train_bc --epochs 40

Regresses the expert action from (top-camera image, proprioception). Random-shift
augmentation on the image is the one non-obvious ingredient: without it a small
conv net memorises the fixed scene and the policy stops tracking the cube.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from urgantry_bc.policy import BCPolicy, normalize_images


def random_shift(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """DrQ-style random translation: pad by replication, crop back at random."""
    n, c, h, w = x.shape
    x = F.pad(x, (pad,) * 4, mode="replicate")
    dx = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    dy = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    rows = (torch.arange(h, device=x.device)[None, :] + dy[:, None])   # (n, h)
    cols = (torch.arange(w, device=x.device)[None, :] + dx[:, None])   # (n, w)
    idx_r = rows[:, None, :, None].expand(n, c, h, w + 2 * pad)
    x = torch.gather(x, 2, idx_r)
    idx_c = cols[:, None, None, :].expand(n, c, h, w)
    return torch.gather(x, 3, idx_c)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("urgantry_bc/data/demos"))
    p.add_argument("--out", type=Path, default=Path("urgantry_bc/runs/bc"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.10, help="fraction of EPISODES")
    p.add_argument("--chunk", type=int, default=16, help="actions predicted per obs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.data / "meta.json").read_text())
    images = np.load(args.data / "images.npy", mmap_mode="r")
    traj = np.load(args.data / "traj.npz")
    states, actions = traj["states"], traj["actions"]
    n = len(images)
    print(f"{n} frames from {meta['episodes_kept']} demos "
          f"(expert success {meta['expert_success_rate']:.2f})")

    # Whole frames fit comfortably in GPU memory at 84x84; keeping them there is
    # what makes 40 epochs take a couple of minutes instead of an hour.
    dev = args.device
    imgs_t = torch.as_tensor(np.ascontiguousarray(images)).permute(0, 3, 1, 2).to(dev)
    states_t = torch.as_tensor(states).to(dev)
    actions_t = torch.as_tensor(actions).to(dev)

    state_mean = states_t.mean(0)
    state_std = states_t.std(0).clamp_min(1e-3)
    states_t = (states_t - state_mean) / state_std

    # Split by EPISODE, not by frame. A random frame split leaves frames from the
    # same trajectory on both sides, so val loss measures memorization of these
    # demos rather than generalization to a new cube position -- it read 1e-5
    # while the policy scored 0% in rollout.
    ep_starts = traj["ep_starts"]
    ep_bounds = list(zip(ep_starts, list(ep_starts[1:]) + [n]))
    g = torch.Generator(device="cpu").manual_seed(0)
    ep_perm = torch.randperm(len(ep_bounds), generator=g).tolist()
    n_val_ep = max(1, int(len(ep_bounds) * args.val_frac))
    val_eps = set(ep_perm[:n_val_ep])

    # A chunk target must stay inside its own episode, so drop the last
    # (chunk - 1) frames of each episode as chunk start points.
    chunk = args.chunk
    train_starts, val_starts = [], []
    targets = np.zeros((n, chunk, actions.shape[1]), dtype=np.float32)
    for e, (lo, hi) in enumerate(ep_bounds):
        for t in range(lo, hi):
            idxs = np.minimum(np.arange(t, t + chunk), hi - 1)   # clamp at episode end
            targets[t] = actions[idxs]
        (val_starts if e in val_eps else train_starts).extend(range(lo, hi))
    targets_t = torch.as_tensor(targets).to(dev)
    train_idx = torch.as_tensor(train_starts, device=dev)
    val_idx = torch.as_tensor(val_starts, device=dev)
    print(f"split: {len(ep_bounds) - n_val_ep} train eps / {n_val_ep} val eps, "
          f"chunk={chunk}")

    model = BCPolicy(image_size=meta["image_size"], state_dim=states.shape[1],
                     action_dim=actions.shape[1], chunk=chunk).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    hist = []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=dev)]
        tot, nb = 0.0, 0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            img = normalize_images(imgs_t[idx].clone())
            img = random_shift(img)
            pred = model(img, states_t[idx])
            loss = F.mse_loss(pred, targets_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            vl, vb = 0.0, 0
            for i in range(0, len(val_idx), args.batch):
                idx = val_idx[i:i + args.batch]
                img = normalize_images(imgs_t[idx].clone())
                vl += float(F.mse_loss(model(img, states_t[idx]), targets_t[idx]))
                vb += 1
        hist.append({"epoch": epoch, "train": tot / max(nb, 1), "val": vl / max(vb, 1)})
        print(f"epoch {epoch:3d}  train {hist[-1]['train']:.5f}  val {hist[-1]['val']:.5f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    ckpt = {"model": model.state_dict(), "image_size": meta["image_size"],
            "state_dim": states.shape[1], "action_dim": actions.shape[1],
            "chunk": chunk,
            "state_mean": state_mean.cpu().numpy(), "state_std": state_std.cpu().numpy(),
            "history": hist, "data_meta": meta}
    torch.save(ckpt, args.out / "policy.pt")
    (args.out / "history.json").write_text(json.dumps(hist, indent=2))
    print("saved", args.out / "policy.pt")


if __name__ == "__main__":
    main()
