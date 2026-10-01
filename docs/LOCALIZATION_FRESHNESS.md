# Localization analysis freshness

Object localization applies the existing five-second `Frame.t` age limit both
before and after analysis. This uses the camera client's local monotonic receipt
time; it does not compare a remote capture clock with the client clock. The
separate Isaac camera capture contract remains two seconds.

The primary detector, secondary-camera detector and VLM fallback all reject an
analysis that outlives its image. A slow miss is also rejected: another view or a
remembered object cannot conceal that expiration. Several individually short
passes can exhaust the age of a retained image, so the primary image is checked
again before belief or VLM fallback. The existing belief policy still uses the
motion or re-observation epoch; an eligible belief cannot be paired with an
expired image.

VLM localization includes model initialization, earlier views and depth-to-3D
processing in its check. A successful secondary-view result carries that view's
image, receipt time, capture metadata and calibration. It never substitutes the
primary camera's image or retimestamps a result. The VLM configuration uses the
normal `Cfg` attribute interface.

`SlowPerceptionError` terminates the current native `pick_and_place` attempt
without automatic retry, home, placement or gripper recovery. Perception checks
do not connect `LazyArm`, request another simulator step or change an actuator
deadline. They reject delayed results after a synchronous detector returns;
they do not cancel a blocked detector or make a cold model load faster.

## Measured failure motivating the change

The retained NV13 run on `f9cb6b8e93e1ef720507e8b3e904b6aa5bfed67a`
entered native `pick_begin`, then Ultralytics downloaded `mobileclip_blt.ts`
into the runtime working directory. The configured `models/` copy already
existed. Both completed files were subsequently verified as identical
(599,764,649 bytes, SHA-256
`a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54`).
This was activity inside the case, not preparation before the case.

Localization returned after 104.046 seconds with the original image: its local
receipt age was about 103.861 seconds. The independent witness cameras remained
fresh (maximum server capture age 0.774482 seconds). A live stream therefore did
not establish freshness of the image retained by analysis.

The case then failed route preflight before any actuator command. That route
budget started separately after localization; this change does not establish
the cause of its overrun or fix route performance. The case remains a failure,
with no grasp, carry, release, home or reset validation.

Evidence identities retained outside the checkout:

- Native case receipt: `cabc311871044741f9d8c0bbe8f901bdfbbbfb517147e8571cdb320bed579b69`.
- Grasp attempt receipt: `b22b58a18609e3e515ae106aaf46c1e3e88bc5f85dfaacd070c13172165f39a7`.
- Independent partial-case review: `c14c23fbe3a566d37bd552659c6a805f6b314f24b8cad9beec9b21fc6267ea39`.

## Validation scope

CPU regressions exercise slow successful detections, slow misses, accumulated
view time, cold VLM construction, depth processing, invalid local timestamps,
foreign capture clocks, secondary-view calibration, eligible belief fallback
and terminal failure without materializing the arm. The existing configured
description, reference-resolution and motion-epoch tests remain relevant.
No new physical success is established by these tests; full-suite and physical
results must identify their tested source independently.
