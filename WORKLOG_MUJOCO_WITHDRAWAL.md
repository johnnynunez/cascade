# MuJoCo release withdrawal

Base: `1db40c04e58e9f77e1a6882a96b72ad34a92c4b3`. The external frozen
`MUJOCO_MEMORY_DIAG` reproduces the final 70.582591 mm error: after release,
home contacts and displaces the object. The 60 mm original test remains.

Scope: separate grasp-height constraints from an actual model/IK-backed
withdrawal, plan clearance before opening, and refuse release/home when the
path is unavailable. Keep gains, physics, verifier thresholds and original
end-of-home assertions. No live GPU or physical robot and no publication.
Use actual MuJoCo collision geometry and private forward-kinematic scratch
state only; never mutate the live world to make a check succeed.

Workflow: diagnose frozen physics trace → geometry/IK contract → small CPU
adversarial controls → original rendered two-pick test after home → affected
regressions and independent review. The manipulation-IK skill's strict
contact-only and object-state validation rules apply; its Isaac stack examples
do not replace this MuJoCo backend or authorize an SDK launch.

Stage 1: actual first placement+retreat+home passes; an occupied second exact
point is rejected before transport, preserving the placed prefix. 213 tests
pass in 78.26 s, including the physical negative and cancellation/geometry
controls. Full source/store hashes and unmodified failed baseline are retained
in `benchmark/results/mujoco_release_withdrawal_20261003.json`. Original
two-pick success/60 mm assertions are untouched; no complete two-placement
success is claimed. A separate area-destination change is authorized next.
