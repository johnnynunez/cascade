# July-2026 baselines

Verbatim snapshots of the 2026-07-07 gain-tuner (3.5.3) evidence produced
against the previously uploaded `RS-rebot-dev-arm` package
(`00-arm-rs_asm-v3-plus`), committed so `scripts/make_validation_md.py`
regenerates `VALIDATION.md` identically on any machine:

- `gt_pj_newasset_newton.json` / `gt_pj_newasset_physx.json` — full per-joint
  snap-to-limits matrix (predates the hybrid-collider fix: gripper rows are
  blocked there).
- `gt_grip_decomp_newton.json` / `gt_grip_decomp_physx.json` — gripper rerun
  with convexDecomposition colliders (the shipped-asset gripper baseline).

Absolute paths inside these JSONs are historical run metadata from the July
machine, not inputs read by any script.
