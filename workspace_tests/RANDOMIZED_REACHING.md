# Randomized reaching: every waypoint pair, both arms independent

**Question:** when the two arms are given *unrelated* targets, how often can the pair
serve them at once?
**Test:** all 450 × 449 = **202,050 ordered waypoint pairs** per layout — left arm to one
point, right arm to another — **606,150 configurations** in total.

Produced by `randomized_reaching.py` (56 workers, ~7 min per layout); verdict codes in
`randomized_reaching.npz`. Same pass rules as `bimanual_test.py`: both arms solve IK,
touch nothing during the approach, and settle parked within 20 mm.

This differs from `REACHABILITY_REPORT.md`, where both arms serve the *same* waypoint
offset by a fixed separation. Here nothing coordinates them, which is what exposes
interference.

---

## Headline

| verdict | `urgantry` | `urtable_45` | `urtable` |
|---|---|---|---|
| **PASS** | **87,039 (43.1 %)** | 75,202 (37.2 %) | 63,294 (31.3 %) |
| collision — arm vs arm | 83,394 (41.3 %) | 93,625 (46.3 %) | **131,663 (65.2 %)** |
| collision — arm vs structure | 2,616 (1.3 %) | 14,113 (7.0 %) | 0 (0.0 %) |
| droop | 26,257 (13.0 %) | 17,876 (8.8 %) | 6,196 (3.1 %) |
| unstable | 29 (0.0 %) | 1,098 (0.5 %) | 0 (0.0 %) |
| no IK | 2,715 (1.3 %) | 136 (0.1 %) | 897 (0.4 %) |

**Most configurations fail in every layout**, and the dominant cause is always the arms
hitting each other. That is a property of random pairing, not a defect: a large share of
pairs asks the two arms to swap sides of the table.

The ordering matches the shared-target test (`urgantry` ≥ `urtable_45` > `urtable`), but
the margins are far wider — 43.1 % vs 31.3 % here, against 84 % vs 76 % there. Independent
targets separate the layouts much more sharply than a shared one.

## The result that matters: crossing

Splitting by where each arm is sent explains nearly everything:

| | `urgantry` | `urtable_45` | `urtable` | n |
|---|---|---|---|---|
| **uncrossed** (left arm → −x, right arm → +x) | **91.6 %** | 79.9 % | 86.9 % | 44,100 |
| **same side** (both targets same half) | 37.4 % | 30.4 % | 18.8 % | 88,650 |
| **crossed** (left arm → +x, right arm → −x) | **3.0 %** | 3.1 % | **0.7 %** | 44,100 |

Arm-vs-arm collision rate over the same split:

| | `urgantry` | `urtable_45` | `urtable` |
|---|---|---|---|
| uncrossed | 0.2 % | 2.0 % | 7.5 % |
| same side | 38.4 % | 48.3 % | 77.0 % |
| crossed | 90.0 % | 89.5 % | **98.3 %** |

**Crossing is fatal in every layout.** When the arms are asked to swap sides they collide
in ~90 % of configurations, and `urtable` — with 20 cm between its bases — fails 98.3 % of
them. No layout tested here tolerates crossed assignment.

**Uncrossed assignment is the opposite story.** Give each arm the side it is on and
`urgantry` serves 91.6 % of pairs and `urtable` 86.9 %. The headline 31–43 % figures are
dominated by the 44,100 crossed pairs and the 88,650 same-side pairs that a real task
allocator would never generate.

The practical reading: **the layout matters much less than the assignment policy.** A
scheduler that assigns each target to the nearer arm converts a ~40 % success rate into a
~90 % one on the same hardware — a bigger gain than any layout change on offer.

## Where each layout actually differs

Once crossing is set aside, the layouts differ in *how* they fail:

- **`urgantry`** — best overall (43.1 %) and best uncrossed (91.6 %), with almost no
  structural collisions (1.3 %). Its distinctive failure is **droop**: 13.0 %, four times
  `urtable`'s rate, from arms hanging at long reach.
- **`urtable_45`** — the only layout with meaningful **structural** collisions (7.0 %,
  14,113 configs): arms striking their own angled stand. Also the only one with `unstable`
  verdicts (1,098). Both are fixture problems, not kinematic ones.
- **`urtable`** — **zero** structural collisions and the lowest droop (3.1 %), because its
  upright bases and short reaches are mechanically easy. It pays for that with the highest
  arm-vs-arm rate in every geometry class, including 7.5 % even when *uncrossed*, where
  the other two are at 0.2 % and 2.0 %.

That last number is the sharpest single indictment of the 20 cm base spacing: `urtable` is
the only layout that collides appreciably even when each arm stays on its own side.

## Target separation

PASS rate against the distance between the two arms' targets:

| separation | `urgantry` | `urtable_45` | `urtable` | n |
|---|---|---|---|---|
| 0.00–0.15 m | 17.3 % | 18.4 % | **0.0 %** | 8,778 |
| 0.15–0.30 m | 39.9 % | 33.5 % | 14.4 % | 56,296 |
| 0.30–0.45 m | 47.6 % | 41.6 % | 37.0 % | 73,674 |
| 0.45–0.60 m | 44.0 % | 39.2 % | 44.7 % | 44,778 |
| 0.60–0.80 m | 45.1 % | 35.5 % | 42.9 % | 18,216 |

`urtable` cannot serve **a single one** of the 8,778 pairs whose targets are within 15 cm
of each other — its two grippers cannot occupy adjacent points at all. The other two
manage ~18 %. This is the quantitative version of the 0.35 m `separation` constant used
in the shared-target test: below that spacing `urtable` simply does not function
bimanually.

Beyond ~0.45 m all three plateau, and `urtable` catches up — at wide separation the arms
are far enough apart that only reach matters.

## Layer pairs

PASS rate by the height layer of each arm's target:

| left / right | `urgantry` | `urtable_45` | `urtable` |
|---|---|---|---|
| low / low | 37.0 % | 26.7 % | 23.3 % |
| mid / mid | 44.0 % | 35.8 % | 20.9 % |
| high / high | 41.6 % | **46.9 %** | **20.7 %** |
| low / high (either order) | ~45 % | ~38 % | ~42 % |

`urtable_45` is the only layout that improves with height (26.7 % → 46.9 %), consistent
with its angled stand lifting the arms clear of the board. `urtable` degrades at height
(23.3 % → 20.7 %) because reaching up brings both arms into the same volume above their
closely-spaced bases. Mixed-height pairs are easiest for every layout — one arm high and
one low keeps them out of each other's way.

## Recommendation

1. **Fix the assignment policy before the layout.** Nearest-arm assignment moves success
   from ~40 % to ~90 % on unchanged hardware. Crossing is unrecoverable in all three
   layouts.
2. **`urgantry` is the most robust to uncoordinated targets** — best overall, best
   uncrossed, and its failures are droop, which a stiffer controller or a looser tolerance
   addresses.
3. **`urtable` should not be used for genuinely independent bimanual work.** 65 % arm-vs-arm
   collisions, 7.5 % even uncrossed, and total failure below 15 cm target separation.
4. **`urtable_45`'s 7 % structural collisions are the cheapest fix in this report** —
   reshape the pedestal and it likely overtakes `urgantry` on uncrossed work.

---

## Method

- All ordered pairs `(i, j)`, `i ≠ j`, of the 450-waypoint grid. Ordered, not unordered:
  the arms are distinct, so left→A/right→B differs from left→B/right→A.
- Each configuration restarts from the home pose, both arms IK to their own target
  tool-down, 4 s approach with contact monitoring every step, then a 1 s hold.
- Verdict precedence: `no IK` → `collision` (split arm-vs-arm / arm-vs-structure by
  whether both contact sides belong to arms) → `unstable` → `droop` → `drifted` → `PASS`.
- Props removed; `implicitfast` integrator (Euler chatters against these stiff servos and
  no pose reads as settled).
- 56 worker processes, one BLAS thread each, ~500–600 configs/s aggregate, ~7 min per
  layout. Chunks are retried with a smaller pool if a worker dies — MuJoCo took a process
  down hard twice during these runs, at different configurations each time.
- `error` verdicts (diverged physics or a crashed chunk) came out at **0** for all three
  layouts, so no result here rests on a failed configuration.

### Caveats

- **A `collision` verdict is evidence, not proof.** The IK is damped least squares
  warm-started from home, taking the first converging random restart on failure. It does
  not enumerate the ~8 analytic UR branches or prefer least motion, and there is no path
  planning — joints interpolate independently. A planner could clear some of these
  collisions, particularly in the same-side class. The crossed-pair result is unlikely to
  move: the arms must physically pass through each other's volume.
- **No task semantics.** Every pair is weighted equally, including pairs no sensible
  allocator would produce. The conditional tables, not the headline, are the usable result.
- **Static targets only** — no payload, no motion through the pair, no handovers.
- Simulation: no friction model, no gearbox compliance, no backlash.
