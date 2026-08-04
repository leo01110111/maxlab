"""Every ordered pair of waypoints, both arms independently: 450 x 449 = 202,050
configurations per layout.

Unlike bimanual_test.py, the two arms do NOT share a target. Each is sent to its own
waypoint, so this measures what the pair of arms can do when the two halves of the task
are unrelated -- which is where interference shows up.

Same verdict rules as bimanual_test.py: both arms must solve IK, touch nothing during
the approach, and settle parked within 20 mm. Collisions are split by what was hit,
since arm-vs-arm and arm-vs-structure mean different things about a layout.

Runs across processes; each worker builds its own copy of the scene.

    python randomized_reaching.py                 # all scenes, all pairs
    python randomized_reaching.py --workers 32 --limit 5000
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from pathlib import Path

import numpy as np
import mujoco

from workspace_benchmark import load_benchmark
from workspace_probe import (
    POS_TOL, ROT_TOL, SCENES, build, reset_pose, solve_ik_restarts, tool_quat,
)

HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "randomized_reaching.npz"

APPROACH_S = 4.0
HOLD_S = 1.0
SETTLE_VEL = 0.01
HOLD_TOL = 0.02

PASS, COLL_ARM, COLL_ENV, DROOP, UNSTABLE, NO_IK, DRIFTED, ERROR = range(8)
LABELS = ["PASS", "collision (arm-arm)", "collision (structure)", "droop",
          "unstable", "no IK", "drifted", "error"]

_W: dict = {}


def _init(scene: str) -> None:
    # One BLAS thread per worker: 64 processes each spawning a thread pool sized to
    # the machine's 128 cores oversubscribes badly and can take a worker down.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    mod, model, data, arms = build(scene, props=False)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    arm_list = list(arms.values())
    owner = np.full(model.ngeom, -1, np.int8)
    for i, arm in enumerate(arm_list):
        for g in range(model.ngeom):
            if model.geom_bodyid[g] in arm.bodies:
                owner[g] = i
    _W.update(mod=mod, model=model, data=data, arms=arm_list, owner=owner,
              scratch=mujoco.MjData(model), rng=np.random.default_rng(0),
              points=np.array([wp["pos"] for wp in load_benchmark()]),
              quat=tool_quat(0.0))


def _contact(data, owner):
    """(touching, arm_vs_arm). Gripper-internal contacts sit inside one arm and are
    excluded; a contact between the two arms has two distinct non-negative owners."""
    n = data.ncon
    if n == 0:
        return False, False
    a1, a2 = owner[data.contact.geom1[:n]], owner[data.contact.geom2[:n]]
    real = (a1 != a2) & ((a1 >= 0) | (a2 >= 0))
    if not np.any(real):
        return False, False
    return True, bool(np.any(real & (a1 >= 0) & (a2 >= 0)))


def _run_chunk(args) -> tuple[int, np.ndarray]:
    start, pairs = args
    mod, model, data = _W["mod"], _W["model"], _W["data"]
    arms, owner, scratch, rng = _W["arms"], _W["owner"], _W["scratch"], _W["rng"]
    points, quat = _W["points"], _W["quat"]
    approach = int(APPROACH_S / model.opt.timestep)
    hold = int(HOLD_S / model.opt.timestep)
    out = np.empty(len(pairs), np.uint8)

    for k, (i, j) in enumerate(pairs):
        try:
            out[k] = _one(mod, model, data, arms, owner, scratch, rng, quat,
                          [points[i], points[j]], approach, hold)
        except Exception:
            # A diverged config must not take the worker (and the pool) down with it.
            out[k] = ERROR
    return start, out


def _one(mod, model, data, arms, owner, scratch, rng, quat, goals, approach, hold) -> int:
    mujoco.mj_resetData(model, data)
    reset_pose(mod, model, data)
    mujoco.mj_forward(model, data)

    for arm, goal in zip(arms, goals):
        scratch.qpos[:] = data.qpos
        q, pe, re = solve_ik_restarts(model, scratch, arm, goal, quat,
                                      data.qpos[arm.qadr].copy(), rng)
        if pe > POS_TOL or re > ROT_TOL:
            return NO_IK
        data.ctrl[arm.act_ids] = q

    hit = arm_arm = False
    for _ in range(approach):
        mujoco.mj_step(model, data)
        if not hit:
            hit, arm_arm = _contact(data, owner)
    if not np.isfinite(data.qpos).all():
        return ERROR
    if hit:
        return COLL_ARM if arm_arm else COLL_ENV

    vel = max(np.max(np.abs(data.qvel[a.dofs])) for a in arms)
    err = max(np.linalg.norm(data.site_xpos[a.site] - g) for a, g in zip(arms, goals))
    if vel >= SETTLE_VEL:
        return UNSTABLE
    if err >= HOLD_TOL:
        return DROOP

    for _ in range(hold):
        mujoco.mj_step(model, data)
        h, aa = _contact(data, owner)
        if h:
            return COLL_ARM if aa else COLL_ENV
    err = max(np.linalg.norm(data.site_xpos[a.site] - g) for a, g in zip(arms, goals))
    vel = max(np.max(np.abs(data.qvel[a.dofs])) for a in arms)
    return DRIFTED if (err >= HOLD_TOL or vel >= SETTLE_VEL) else PASS


def run_scene(scene: str, pairs: np.ndarray, workers: int, chunk: int) -> np.ndarray:
    """Chunks are retried with a fresh, smaller pool if a worker dies. MuJoCo takes a
    process down hard on some configs, and losing six minutes of finished work to one
    of them is not acceptable."""
    codes = np.empty(len(pairs), np.uint8)
    pending = [(s, pairs[s:s + chunk]) for s in range(0, len(pairs), chunk)]
    t0 = time.perf_counter()
    done = 0
    for attempt in range(6):
        finished = set()
        try:
            with ProcessPoolExecutor(workers, initializer=_init, initargs=(scene,),
                                     max_tasks_per_child=100) as pool:
                for start, out in pool.map(_run_chunk, pending):
                    codes[start:start + len(out)] = out
                    finished.add(start)
                    done += len(out)
                    if done % (chunk * 50) < chunk:
                        rate = done / (time.perf_counter() - t0)
                        print(f"  {scene}: {done:,}/{len(pairs):,}  {rate:,.0f} config/s  "
                              f"eta {(len(pairs) - done) / rate / 60:.1f} min", flush=True)
        except BrokenExecutor:
            workers = max(8, workers // 2)
            print(f"  {scene}: worker died, retrying {len(pending) - len(finished)} "
                  f"chunks with {workers} workers", flush=True)
        pending = [t for t in pending if t[0] not in finished]
        if not pending:
            break
    if pending:
        left = sum(len(t[1]) for t in pending)
        print(f"  {scene}: giving up with {left:,} configs unfinished", flush=True)
        for start, chunk_pairs in pending:
            codes[start:start + len(chunk_pairs)] = ERROR
    print(f"  {scene}: done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return codes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--limit", type=int, help="test only the first N pairs (smoke test)")
    ap.add_argument("--scene", help="one scene; default is all")
    args = ap.parse_args()

    n = len(load_benchmark())
    pairs = np.array([(i, j) for i in range(n) for j in range(n) if i != j], np.int32)
    if args.limit:
        pairs = pairs[:args.limit]
    scenes = [args.scene] if args.scene else sorted(SCENES)
    print(f"{len(pairs):,} ordered pairs x {len(scenes)} scenes "
          f"= {len(pairs) * len(scenes):,} configs, {args.workers} workers\n")

    # Merge with anything already on disk so a single-scene rerun keeps the others.
    results = {}
    if RESULTS_PATH.exists():
        old = np.load(RESULTS_PATH, allow_pickle=True)
        if len(old["pairs"]) == len(pairs):
            results = {k: old[k] for k in old.files if k not in ("pairs", "labels")}
    for s in scenes:
        results[s] = run_scene(s, pairs, args.workers, args.chunk)
        # Checkpoint per scene: a crash in a later scene must not discard finished ones.
        np.savez_compressed(RESULTS_PATH, pairs=pairs, labels=np.array(LABELS), **results)

    scenes = [s for s in sorted(results)]
    width = max(22, max(len(s) for s in scenes) + 2)
    print(f"\n{'verdict':<24}" + "".join(f"{s:>{width}}" for s in scenes))
    print("-" * (24 + width * len(scenes)))
    for code, label in enumerate(LABELS):
        row = f"{label:<24}"
        for s in scenes:
            c = int((results[s] == code).sum())
            row += f"{c:>{width - 8},}{100 * c / len(pairs):>7.1f}%"
        print(row)
    print(f"\nverdict codes -> {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
