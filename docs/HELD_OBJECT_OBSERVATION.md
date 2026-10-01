# Held-object observation after the NV08 failure

Merged in PR #42 on MAIN `477c88f`, together with preserved grasp-memory
ranking. The combined immutable suite passed 3,205 tests. New physical results
are tracked separately in [project status](PROJECT_STATUS_20261001.md); the NV08
failure analyzed here remains a failure.

The MAIN `6b5dad64eaa35b0ea6c0a0430a4098818626cfee` NV08 trial failed before
its planned post-release fault injection. The runtime reported that the orange
was 10 cm below the gripper and opened it. The passive witness instead observed
bilateral contact throughout the lift and until 10.7 ms before that opening.
The measured lift was approximately 93.57 mm; the final object-to-TCP vertical
offset was approximately −9.48 mm. Contact loss and the fall were observed
after the native opening. No extra recovery, reset, retry or normal-five
campaign was executed. The scene and original failed receipt were preserved.

The original receipt also contains a secondary diagnostic `AttributeError`:
the probe expected a release episode which this failed pick had never reached.
That error is separate from the incorrect carry-slip report. The skills trace
and independent witness retain the primary sequence.

The old cache subtracted a post-lift TCP from the pre-lift localization. The
retained NV08 inputs reproduce a −102.94 mm cached vertical offset. The detector
label `orange fruit` did not resolve through the old truth lookup, whereas the
canonical `orange` did. With a synthetically unavailable image, the old method
returns that cache and triggers the unchanged 60 mm slip threshold. The live
trial did not record the chosen offset branch, so this is a conditional
reproduction, not proof that the live image lookup failed. The offset method
was also identical in the earlier successful NV06 trial.

The correction separates aiming compensation from evidence that can authorize
opening:

- The cache uses the measured closed pose before lift. It remains an
  approximate compensation estimate: localization and closure are not one
  simultaneous measurement. Neither the cache nor a legacy vector pose API
  can authorize a slip opening.
- The optional truth observation reads object positions, articulation joints,
  robot base pose and physics clock in the same existing bridge request. It
  requires valid physical handles and an unchanged step. Canonical identity
  and aliases must be unambiguous against the complete dynamic inventory,
  including bodies whose reads fail. Legacy pose/fallback behavior remains
  available without release authority.
- Authoritative truth uses explicit asset-joint names/indices, the existing
  driver's conversion, and a world-to-robot-base transform. The completed
  lift's clock, producer and convention remain bound. Unsupported or missing
  measurements provide no release authority.
- Images use their captured articulation pose for TCP, rather than current
  TCP. Release authority additionally requires the exact measured instance,
  producer identity and a physical capture newer than the current-state check,
  acquired within two seconds of local elapsed observation time. No clocks
  from different hosts are subtracted. Historical images may compensate aim
  but cannot prove a current drop.
- A changed producer/backend/convention, contradictory same-step joints or
  regressed current clock terminates the composite place attempt without
  opening, retrying or moving home. A halt during observation or immediately
  before opening is checked against the original generation and terminates
  the attempt. Ordinary absence of a usable observation retains the held flag
  and the existing compensation/planning behavior.

The existing 60 mm slip threshold, 30 mm mapper clearance, unknown-space
policy and motion/barrier/mapper deadlines are unchanged. Isaac state and the
fresh truth request each use a one-second transport deadline. The two-second
observation check limits release authority at return; it is not a hard wall
limit on every possible legacy detector/backend call.

Memory now records the selected channel, numeric offset, identity/clock
evidence, authority and invalidation reason without another observation. The
offline tests include the retained NV08 joints with the real arm kinematics,
a real runtime close/lift cache regression, coherent genuine drops, ambiguous
identities, old images, base transforms, clock/convention changes and composite
halt/resume cases using the actual safety wrapper.

The compact retained evidence is in
`benchmark/results/nvblox-held-offset-false-slip-20261001.json`. That failure is
separate from the [two-object Spark proof on the later merged source](SPARK_DELIVERY.md#current-two-object-proof-1-october-2026).
Campaign and nvblox recovery results are tracked in the
[current acceptance index](PROJECT_STATUS_20261001.md). Runtime sources used by
the preserved NV08 scene were not edited.
