"""Draw several HUG grasps for one click and rank them by geometry.

HUG is a generative flow model: the same pixel with a different seed gives a
different grasp, and the spread in quality is large. Sampling N and keeping the
best costs one model load and N cheap forward passes.

    python -m urgantry_hug.sample            # rank 8 grasps on the block
    python -m urgantry_hug.sample -n 16
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import mujoco

from urgantry_sim.build_urgantry import build_scene, set_initial_pose

from .capture import Capture, RGBDRenderer
from .frames import project, transform_points
from .interface import MANO_LANDMARKS, MANO_WRIST_LANDMARK, ManoGrasp
from .predict import DEFAULT_CHECKPOINT, HugPredictor

FINGERTIPS = tuple(MANO_LANDMARKS[f][-1] for f in
                   ("thumb", "index", "middle", "ring", "pinky"))

TABLE_Z = 0.775
SPREAD_TARGET = 0.07
SPREAD_SIGMA = 0.04
REACH_TARGET = 0.06
REACH_SIGMA = 0.05
PENETRATION_TOLERANCE = 0.02
WRIST_CLEARANCE = 0.05

WEIGHTS = {
    "enclosure": 0.30,
    "table": 0.25,
    "reach": 0.20,
    "wrist": 0.15,
    "spread": 0.10,
}


def sample_grasps(predictor: HugPredictor, capture: Capture, uv,
                  n: int = 8, seeds: Optional[Sequence[int]] = None
                  ) -> list[ManoGrasp]:
    """Predict `n` grasps at the same pixel, one per seed."""
    if seeds is None:
        seeds = range(n)
    return [predictor.predict(capture, uv, seed=int(s)) for s in seeds]


def score_terms(grasp: ManoGrasp, table_z: float = TABLE_Z) -> dict[str, float]:
    """The individual terms behind `score_grasp`, each in [0, 1].

    enclosure  Bearings from the clicked point to the five fingertips, in the
               world xy plane. A hand that wraps the object has fingertips all
               around it, so the largest angular gap between consecutive
               bearings is small; a hand that only touches one side leaves a gap
               approaching a full turn. Scores 1 for any gap under 180 degrees
               (the point is inside the fingertip hull) and falls linearly to 0
               at 360.

    table      No part of a real hand can be inside the table. Measures the
               deepest landmark below `table_z` and falls linearly to 0 at 2 cm
               of penetration. HUG has no notion of the support surface, so this
               is the term that rejects the physically impossible samples.

    reach      Mean fingertip distance to the clicked point. Punishes grasps
               that are correctly shaped but placed off the object, which the
               enclosure term alone cannot see.

    wrist      Wrist height above the clicked point. The gantry approaches from
               above, so a wrist below the click is unreachable regardless of
               how good the finger geometry is. Saturates at 5 cm.

    spread     Largest pairwise fingertip distance, scored by a Gaussian around
               a target a little wider than the object. Both a fist and a fully
               splayed hand fail to close on a 5 cm object.
    """
    lm = np.asarray(grasp.landmarks, float)
    tips = lm[list(FINGERTIPS)]
    wrist = lm[MANO_WRIST_LANDMARK]
    click = np.asarray(grasp.click_world, float)

    d = tips[:, :2] - click[:2]
    bearings = np.sort(np.arctan2(d[:, 1], d[:, 0]))
    gaps = np.diff(np.concatenate([bearings, bearings[:1] + 2 * np.pi]))
    max_gap = float(gaps.max())
    enclosure = float(np.clip((2 * np.pi - max_gap) / np.pi, 0.0, 1.0))

    penetration = float(max(0.0, table_z - lm[:, 2].min()))
    table = float(np.clip(1.0 - penetration / PENETRATION_TOLERANCE, 0.0, 1.0))

    reach_err = float(np.linalg.norm(tips - click, axis=1).mean()) - REACH_TARGET
    reach = float(np.exp(-0.5 * (max(0.0, reach_err) / REACH_SIGMA) ** 2))

    wrist_score = float(np.clip((wrist[2] - click[2]) / WRIST_CLEARANCE, 0.0, 1.0))

    spread = float(np.linalg.norm(tips[:, None] - tips[None], axis=-1).max())
    spread_score = float(np.exp(-0.5 * ((spread - SPREAD_TARGET) / SPREAD_SIGMA) ** 2))

    return {
        "enclosure": enclosure,
        "table": table,
        "reach": reach,
        "wrist": wrist_score,
        "spread": spread_score,
    }


def score_grasp(grasp: ManoGrasp, capture: Optional[Capture] = None,
                table_z: float = TABLE_Z) -> float:
    """Weighted mean of `score_terms`, in [0, 1]; higher is better.

    See `score_terms` for what each term measures and why. `capture` is accepted
    so callers can score against the frame the grasp came from once image-space
    terms exist; the geometry below is all in world coordinates and needs none.
    """
    terms = score_terms(grasp, table_z)
    return float(sum(WEIGHTS[k] * terms[k] for k in WEIGHTS))


def rank_grasps(grasps: Sequence[ManoGrasp], capture: Optional[Capture] = None,
                table_z: float = TABLE_Z) -> list[tuple[ManoGrasp, float]]:
    """`(grasp, score)` pairs, best first."""
    scored = [(g, score_grasp(g, capture, table_z)) for g in grasps]
    return sorted(scored, key=lambda gs: gs[1], reverse=True)


def best_grasp(predictor: HugPredictor, capture: Capture, uv, n: int = 8,
               seeds: Optional[Sequence[int]] = None,
               table_z: float = TABLE_Z) -> tuple[ManoGrasp, float]:
    grasps = sample_grasps(predictor, capture, uv, n=n, seeds=seeds)
    return rank_grasps(grasps, capture, table_z)[0]


def block_pixel(data: mujoco.MjData, capture: Capture) -> np.ndarray:
    cam = transform_points(np.linalg.inv(capture.T_world_cam),
                           data.body("block").xpos[None])
    return project(cam, capture.K)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("--uv", type=float, nargs=2, default=None)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--table-z", type=float, default=TABLE_Z)
    args = ap.parse_args()

    model, data = build_scene()
    set_initial_pose(model, data)
    mujoco.mj_forward(model, data)

    renderer = RGBDRenderer(model, width=640, height=480)
    capture = renderer(data, camera="top1")

    uv = np.array(args.uv) if args.uv else block_pixel(data, capture)
    print(f"block at {np.round(data.body('block').xpos, 4)}, pixel {np.round(uv, 1)}")

    predictor = HugPredictor(args.checkpoint, sampling_steps=50)
    seeds = list(range(args.n))
    grasps = sample_grasps(predictor, capture, uv, seeds=seeds)
    order = {id(g): s for g, s in zip(grasps, seeds)}

    print(f"\nclicked surface point {np.round(grasps[0].click_world, 4)}")
    header = ("rank seed  score | " +
              " ".join(f"{k:>9}" for k in WEIGHTS) +
              " |   min_z  wrist_z   spread")
    print("\n" + header)
    print("-" * len(header))

    for rank, (grasp, score) in enumerate(rank_grasps(grasps, capture, args.table_z)):
        terms = score_terms(grasp, args.table_z)
        lm = np.asarray(grasp.landmarks, float)
        tips = lm[list(FINGERTIPS)]
        spread = np.linalg.norm(tips[:, None] - tips[None], axis=-1).max()
        print(f"{rank:>4} {order[id(grasp)]:>4} {score:6.3f} | " +
              " ".join(f"{terms[k]:9.3f}" for k in WEIGHTS) +
              f" | {lm[:, 2].min():7.4f} {lm[0, 2]:8.4f} {spread:8.4f}")

    best, best_score = rank_grasps(grasps, capture, args.table_z)[0]
    worst, worst_score = rank_grasps(grasps, capture, args.table_z)[-1]
    for name, grasp, score in (("best", best, best_score), ("worst", worst, worst_score)):
        tips = np.asarray(grasp.landmarks, float)[list(FINGERTIPS)]
        print(f"\n{name} ({score:.3f}) seed {order[id(grasp)]} fingertips:")
        for finger, tip in zip(("thumb", "index", "middle", "ring", "pinky"), tips):
            print(f"  {finger:>6} {np.round(tip, 4)}")

    renderer.close()
    predictor.close()


if __name__ == "__main__":
    main()
