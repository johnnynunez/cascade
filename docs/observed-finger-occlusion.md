# Observed finger endpoint occlusion veto

The orange attempt on source `7e02de70` stopped during descent settling when
the right jaw moved below its checked opening interval. It sent no closing
command. The retained nominal + x86 + ARM convex union already covered the
offending finger geometry: offline intersection with the measured pink box
exists at the selected endpoint, while no point of the captured depth surface
intersects that finger. The relevant side of the box is hidden behind its
visible depth surface. This geometric attribution uses witness data only in
the forensic analysis; planning never consumes simulator object poses.

The additional veto clips each convex finger envelope against discrete pixel
rays from the original depth/K/TF snapshot. At pregrasp and grasp endpoints,
an interval extending behind observed **non-target** depth rejects the candidate.
The pregrasp and open grasp use the complete original measured-to-open envelope;
the closing endpoint uses the independent full mechanical stroke of each finger.
The measured pose is checked again before each closing command, with the same
snapshot and immutable stroke bounds. No extra state or camera read is added.

This check uses depth Z, not Euclidean ray length. It retains the existing
integer pixel-coordinate calibration convention and real transform inverses.
The AABB projection only bounds the candidate pixels; every positive ray is
clipped against all component halfspaces. No margin or depth tolerance is
introduced. A bounding box crossing the camera plane is explicitly rejected.

This is **an additional endpoint rejection, not a free-space certificate**.
It makes no new claim for invalid depth, robot pixels, target pixels, locations
outside the view, space between sampled rays, or intermediate trajectory poses.
The existing target-inclusive surface checks still cover every discrete approach
sample. Only the existing closing surface check exempts the exact target mask;
target geometry is not erased from the approach scene. No masks are enlarged.
The scene is not refreshed during planning or closing, and there is no new
feedback brake for a commanded closure. Contact offsets and the fixed 0.1 mm
jaw tracking interval are unchanged.

Each symmetric wrist orientation now passes IK, harness, surface and endpoint
checks independently. A rejected original no longer hides a valid twin.
Quality, memory ranking, width filters and preference for lower joint travel
remain intact. Errors/cancellation propagate, and no actuator runs while these
alternatives are examined. The existing 3 s preflight and total 3-batch/8 s
planning limits remain; cancellation/deadline checks run between components and
bounded pixel blocks, including cached results. Each measured closing preflight
has a fresh 3 s limit capped by the task deadline.

The [offline evidence](evidence/observed-finger-occlusion/receipt.json) records:

- Orange07: the failed orientation is rejected; a separately checked twin of
  candidate 1 is selected in 1.105 s on the local x86 host.
- Green07: candidate 0 and its historical joint targets are preserved (0.600 s).
- Orange05: candidate 2's valid twin preserves the historical targets (1.608 s).
- NV12: candidate 1 preserves the historical targets (0.985 s); the replay does
  not reconstruct the native ESDF map or certify its live behavior.

These timings exclude model inference, transport and physical execution.
The failed07 archive remains negative; fresh native acceptance is still required.
