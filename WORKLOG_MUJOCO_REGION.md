# Explicit MuJoCo destination region

Base: withdrawal22cac914 plus independent refusal verifierfc883721 (local
cherry-pickc62b239). Stage1 and its diagnostics remain frozen elsewhere.

Authorized scope: opt-in profile region, geometric free-location selection,
existing IK/carry/release/home validation, and a separate passive observation
window. Exact place_at does not shift coordinates. No physics/gain/contact,
6 cm original task assertion, kitchen profile or GPU changes.

Feature→skills: manipulation-ik governs contact-only manipulation and preserving
the already placed prefix. physics-simulation informs real body/collider and
solved-contact provenance. This backend is MuJoCo C, not an Isaac deployment.

Sequence: explicit region schema/geometry and pure planning → passive bounded
native-history observer → CPU malformed/missing/stale/occupied controls →
review of frozen plan and criteria → original rendered two-pick trial. No
observer may advance physics. The final window must be after measured opening,
span >=0.6 simulated seconds/6 distinct samples, and end at current state; it
may occur during withdrawal/home. Document batch-end samples, not all substeps.

Proposed region is90×80mm at x[.15,.24],y[-.16,-.08] on named existing support
planes; candidate erosion and spacing use observed geometry and existing
clearance. These bounds describe a table patch within the profile workspace;
IK/path checks still decide usable placements. No slots or color-specific rules.

Implementation checkpoint: pure placement IK extraction matches the exact
captured parent AST; region selector checks full preflight under one deadline.
Passive observer records last-solve input and native contacts separately from
integrated state, with complete compiled-model serialization and epoch. Prefix
failures remain sticky across later tasks; reset clears only the real world.
Static01 28PASS; selection01 1PASS; static02 80PASS. Static03 preserved136PASS/
1FAIL: stopped generation was not itself rejecting static planning. Explicit
halt gate now refuses; static04 37PASS and static05 41PASS. No dynamics/gains/
6cm assertion changed. Further controls and native CPU phase validation pending.

Final static contract checkpoint:66PASS8.27s plus affected placement regressions
100PASS15.61s; all inputs/protected stores unchanged. Root review identified
same-object passive rereads erasing a prior disturbance and stop racing the
explicit-goal retirement commit. Both are corrected: faults remain sticky and
only the short final ledger mutation shares a monotonic cancellation fence.
Three bounded-thread controls cover halt, estop and estop-then-reset during a
blocked audit; stop returns without waiting and no goal is retired. No physics
episode has run yet; native CPU journal microfixtures and unchanged two-pick
validation remain subject to the frozen-plan review.
