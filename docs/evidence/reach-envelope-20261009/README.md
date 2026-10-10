# Reach envelope study, reBot B601-RS URDF (B45, 2026-10-09)

`rebot_reachability.json` is the output of `scripts/reachability_study.py`
(CPU only; this run: 32 workers on a shared 128-core box, 689 s / 139 s /
187 s for the three passes). Write-up: [docs/REACH_ENVELOPE.md](../../REACH_ENVELOPE.md).

Layout:

- `provenance` — commit the run started from, the script's sha256, the URDF
  path and sha256, and every setting the result depends on (`home_q`,
  pregrasp offset 0.04 m, exemption radius 0.15 m, move duration, table z,
  table clearance, harness joint margin 0.02 vs IK limit margin 0.025, the
  velocity cap, the shipped workspace box, keep-outs).
- `grid` — x 0..0.80, y −0.80..0.80 (2.5 cm), grasp heights 0.02 / 0.05 /
  0.07 / 0.10 m with the object each stands for (`z_sources`), the ten
  approach families, the four jaw rolls, and the row layout.
- `passes.open|default_box|envelope_box.rows` — one row per grid point some
  family reaches: `[x, y, z, mask per family…, lowest joint origin (mm) per
  family…]`; mask bit i = roll `rolls_deg[i]` reached (selector flip twin
  included). Unreachable points are omitted. Passes 2 and 3 re-check only
  what pass 1 reached (a smaller box can only refuse more).
- `summary` — per pass, grasp height and family: reachable count (any roll and
  roll 0), bounding box incl. radius range, max x on the y = 0 line, what the
  family adds beyond top-down and outside the shipped box.
- `envelope` — the derived opt-in box, the rule that produced it and its
  verification under itself (`verified`: 2833 planner-family/roll-0 points vs
  953 top-down points in the shipped box, 0 lost, 0 outside the box, lowest
  joint origin 49 mm).

Reproduce (the evidence test re-derives the envelope from these rows and
re-evaluates four grid points through the same pipeline):

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 \
      python scripts/reachability_study.py --workers 32 \
      --out docs/evidence/reach-envelope-20261009/rebot_reachability.json

Not in this evidence: physics, contact, gripper-housing collision, the real
rig, any live pick.
