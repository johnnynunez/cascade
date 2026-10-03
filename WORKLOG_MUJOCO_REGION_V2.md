# Interior placement planning with whole-inventory capacity

Basee353096 contains only the capture-cost correction after07080fa. The old
90x80mm episode remains failed/frozen. Root proposed a140x120mm area around
the existing (.20,-.12) nominal destination, with20mm planned boundary margin
and20mm inter-object separation, derived from model footprints and the existing
safety clearance. The original6cm test is not a planner input.

Feature-to-skills: manipulation-ik retains contact-only manipulation, measured
payload geometry and complete carry/release/home paths; physics-simulation
retains the passive phase-bound support verifier. No gains, contact parameters,
gravity, thresholds, pose writes, release assistance or kitchen changes.

Implement: strict version2 region schema -> bounded geometric row packing for
the full free-body inventory, keeping already placed bodies fixed -> measure
oriented payload at proposed release and recheck capacity/margin -> actual-state
ordinary preflight still owns all motion/release authority. Version1 remains
parseable and archived. Tests first, no new native episode before review.

Initial static controls18PASS0.95s, including real-model scratch IK/full carry/
release/home from a recorded held configuration. Combined02 retained109PASS/
1FAIL11.25s: the binding-negative correctly refused at the earlier observer
binding guard, but its assertion expected the word binding rather than bound.
Only that diagnostic regex was corrected; no production gate changed for it.
Historical fixture/profile remain explicit so the saved physical row and its
compiled-model identity remain reproducible in cost/legacy controls.

Final CPU controls110PASS11.29s across version1/2 contracts, packing, captured
journal/cost parity and shipped device/profile checks. Inputs/protected stores
unchanged; Ruff/diff checks pass. No native episode on version2, no source070
rewrite and no changes to the original two-pick task or its6cm assertion.

Independent review found that a fixed footprint near the exterior could extend
a free footprint's planning interval beyond the eroded interior when boundary
margin exceeds object separation. Freeze01 and all eleven files were preserved
outside the checkout before correction. Eight lower/upper, X/Y controls fail
on that source; the pure planner now intersects every free interval with the
same declared interior. The20mm/20mm profile and physical gates are unchanged.
Final02:118PASS11.25s, source/protected stores unchanged; no integration steps.
