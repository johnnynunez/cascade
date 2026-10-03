# Interior region candidate: geometric capacity before carrying

Version2 of the optional MuJoCo C profile declares x[.13,.27],y[-.18,-.06]
metres in the fixed robot-base frame:140x120mm centered on the existing
(.20,-.12) destination. It declares a20mm **planned** boundary margin and
reserves capacity for every free body in the admitted model. Inter-object
separation retains the existing20mm table-clearance value. The exterior area
remains the measured-containment boundary; no observer tolerance was increased.

These dimensions address capacity, independently of the failed endpoint and
the original test's6cm radius. Two observed footprints36.574+34.834mm, one20mm
gap and two20mm border margins require131.408mm along their packing axis.
A120mm side cannot meet this contract;140mm can. For nominal35mm footprints,
allowed centers are x[.1675,.2325],y[-.1425,-.0975]. Their largest radius is
approximately.273m, within the profile's measured.12-.28m top-down band.
The exterior rectangle is inside the safety workspace and lies on the declared
horizontal support. Neither this arithmetic nor the band guarantees IK or
collision-free motion; each actual candidate still needs the existing full
carry/release/home checks.

The generic pure packing helper consumes opaque body IDs and measured bounds,
without labels, task-radius inputs or fixed color slots. It evaluates bounded
row arrangements along X and Y for at most six bodies. Unused space is spread
across the interior, and larger available border margin ranks first. Every
pair is separated along the row axis, a conservative sufficient condition.
Failure to find a row is not a proof that arbitrary2D packing is impossible.
Objects already inside or independently confirmed are fixed to measured poses;
the helper never repacks or moves them. Other free bodies receive geometric
reservations only. Future orientation, grasp and paths require new admission.

The selected payload's oriented footprint is recomputed from the actual full
IK/carry preview, checked against the planned20mm border, then treated as fixed
while checking remaining capacity again. The runtime exposes both layouts,
body identities, margin and chosen object target. Known prefix disturbances,
insufficient capacity, invalid binding, expired planning budget or unavailable
paths refuse before carry; they grant no automatic home or physical retry.
The ordinary exact-point placement still repeats actual-state preflight before
release. Explicit `place_at` never shifts coordinates.

The historical90x80mm version1 profile remains available as
`so101_mujoco_region_v1`; its wire descriptor remains exact. Its old physical
trial at07080fad remains FAIL: red crossed the maximum-x semantic boundary by
3.515mm, and the task did not attempt blue. Version2 has a different descriptor
and model identity and cannot inherit a task success from that episode.

Initial static evidence: from the previously saved held configuration, the
version2 candidate at[.1709486442,-.12] passes the existing complete geometric
route. Its projected footprint retains at least23.22mm border clearance while
reserving the other body's space. This is scratch-only planning, not new
physical success; no variant2 episode has run. Support, release, settling,
prefix integrity and the unchanged original task still need native evidence.
