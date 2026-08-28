"""Flow-matching behavior cloning on cached DINOv3 features.

    uv run python -m urgantry_bc.train_flow --data urgantry_bc/data/demos224 --epochs 60

Same data and chunking as train_bc.py, but the head is a conditional velocity
field trained with the flow-matching objective instead of an MSE regressor. The
MSE version collapsed to the mean of open and closed at the grasp; a velocity
field can represent that branch.

Validation splits by EPISODE -- a frame-level split leaves frames of the same
trajectory on both sides and reports memorization.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from urgantry_bc.flow_policy import FlowPolicy, flow_loss, sample_actions


def build_chunk_targets(actions: np.ndarray, ep_bounds, chunk: int) -> np.ndarray:
    """(N, chunk, A) targets, clamped at each episode's last frame."""
    targets = np.zeros((len(actions), chunk, actions.shape[1]), dtype=np.float32)
    for lo, hi in ep_bounds:
        for t in range(lo, hi):
            idxs = np.minimum(np.arange(t, t + chunk), hi - 1)
            targets[t] = actions[idxs]
    return targets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("urgantry_bc/data/demos224"))
    p.add_argument("--out", type=Path, default=Path("urgantry_bc/runs/flow"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--val-frac", type=float, default=0.10, help="fraction of EPISODES")
    p.add_argument("--flow-steps", type=int, default=8)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = args.device

    meta = json.loads((args.data / "meta.json").read_text())
    feats = np.load(args.data / "features.npy", mmap_mode="r")
    traj = np.load(args.data / "traj.npz")
    states, actions, ep_starts = traj["states"], traj["actions"], traj["ep_starts"]
    n = len(feats)
    ep_bounds = list(zip(ep_starts, list(ep_starts[1:]) + [n]))
    print(f"{n} frames, {len(ep_bounds)} demos, tokens {feats.shape[1:]}")

    targets = build_chunk_targets(actions, ep_bounds, args.chunk)
    flat_targets = torch.as_tensor(targets.reshape(n, -1)).to(dev)
    states_t = torch.as_tensor(states).to(dev)
    state_mean, state_std = states_t.mean(0), states_t.std(0).clamp_min(1e-3)
    states_t = (states_t - state_mean) / state_std
    # Keep the token cache on the GPU when it fits (an epoch is then seconds);
    # at 448 px the cache is ~22 GB, so fall back to pinned host memory and move
    # each batch across.
    feats_cpu = torch.as_tensor(np.ascontiguousarray(feats))
    # The head is tiny (5.7M params) and batches are small, so nearly all of the
    # card can go to the cache. Falling back to host memory costs ~100x: the
    # per-batch CPU gather of 150 MB of tokens leaves the GPU idle.
    free, _ = torch.cuda.mem_get_info() if dev == "cuda" else (0, 0)
    on_gpu = dev == "cuda" and feats_cpu.numel() * 2 < free - 4e9
    feats_t = feats_cpu.to(dev) if on_gpu else feats_cpu.pin_memory()
    print(f"token cache {feats_cpu.numel()*2/1e9:.1f} GB on "
          f"{'gpu' if on_gpu else 'cpu (pinned)'}")

    def get_feats(idx):
        return (feats_t[idx].float() if on_gpu
                else feats_t[idx.cpu()].to(dev, non_blocking=True).float())

    g = torch.Generator().manual_seed(0)
    ep_perm = torch.randperm(len(ep_bounds), generator=g).tolist()
    n_val_ep = max(1, int(len(ep_bounds) * args.val_frac))
    val_eps = set(ep_perm[:n_val_ep])
    train_idx, val_idx = [], []
    for e, (lo, hi) in enumerate(ep_bounds):
        (val_idx if e in val_eps else train_idx).extend(range(lo, hi))
    train_idx = torch.as_tensor(train_idx, device=dev)
    val_idx = torch.as_tensor(val_idx, device=dev)
    print(f"split: {len(ep_bounds)-n_val_ep} train eps / {n_val_ep} val eps, "
          f"chunk={args.chunk}")

    model = FlowPolicy(embed_dim=feats.shape[2], num_patches=feats.shape[1] - 1,
                       state_dim=states.shape[1], action_dim=actions.shape[1],
                       chunk=args.chunk, width=args.width,
                       num_blocks=args.blocks).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    hist, t0 = [], time.time()
    for epoch in range(args.epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=dev)]
        tot, nb = 0.0, 0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            loss = flow_loss(model, get_feats(idx), states_t[idx], flat_targets[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()

        # Validation reports the quantity we actually care about -- how far a
        # SAMPLED chunk is from the expert's -- not just the velocity residual.
        model.eval()
        with torch.no_grad():
            vloss, verr, vb = 0.0, 0.0, 0
            for i in range(0, len(val_idx), args.batch):
                idx = val_idx[i:i + args.batch]
                f, s, y = get_feats(idx), states_t[idx], flat_targets[idx]
                vloss += float(flow_loss(model, f, s, y))
                pred = sample_actions(model, f, s, args.flow_steps).reshape(len(idx), -1)
                verr += float((pred - y).abs().mean())
                vb += 1
        hist.append({"epoch": epoch, "train_flow": tot / max(nb, 1),
                     "val_flow": vloss / max(vb, 1), "val_action_mae": verr / max(vb, 1)})
        print(f"epoch {epoch:3d}  flow {hist[-1]['train_flow']:.4f}  "
              f"val_flow {hist[-1]['val_flow']:.4f}  "
              f"val_action_mae {hist[-1]['val_action_mae']:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    torch.save({"model": model.state_dict(), "chunk": args.chunk,
                "state_dim": states.shape[1], "action_dim": actions.shape[1],
                "embed_dim": feats.shape[2], "num_patches": feats.shape[1] - 1,
                "image_size": meta["image_size"],
                "width": args.width, "blocks": args.blocks,
                "flow_steps": args.flow_steps,
                "state_mean": state_mean.cpu().numpy(),
                "state_std": state_std.cpu().numpy(),
                "history": hist, "data_meta": meta}, args.out / "policy.pt")
    (args.out / "history.json").write_text(json.dumps(hist, indent=2))
    print("saved", args.out / "policy.pt")


if __name__ == "__main__":
    main()
