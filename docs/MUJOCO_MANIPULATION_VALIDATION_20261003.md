# MuJoCo placement, withdrawal and two-object validation

The unchanged original two-object memory task passed once on
`d273dc4f35d19b71c958f6445370ebea2a904888`: both ordinary `pick_and_place`
calls returned independent physics confirmation after release, withdrawal and
home, and the second placement preserved the first. The episode used the SO-101
MuJoCo C profile, MuJoCo 3.14.0, the pinned collision assets, CPU physics and Mesa
software rendering. It is not an Isaac, hardware, GPU or general-scene admission.

The publication branch extracts the manipulation changes onto main
`70d22deab7f4abb5e5e691386b4b5f952647abca`, then normally merges main
`67f0b61c490f1a83529ecd9143b0361737b3897b` without changing manipulation bytes.
The physical episode was not repeated
on the publication branch. Its changed placement methods and model are compared
below; software regression coverage is separate from that physical evidence.

## Result and physical evidence

| Measurement | Red after its home | Blue after its home |
| --- | ---: | ---: |
| Ordinary action wall time | 59.69 s | 93.84 s |
| Independent settling window | 11 samples / 0.600 s | 12 samples / 0.660 s |
| Final XY distance to original drop-zone point | 25.776 mm | 30.657 mm |
| Original test limit | 60 mm | 60 mm |
| Active arm contacts in settling window | 0 | 0 |
| Loaded support reaction, minimum | 0.595176 N | 0.595176 N |
| Actual support contacts per sample | 8 | 8 |

The blue proof also independently confirms red over the same 12-sample window.
Red's position changed approximately 1e-12 m between the two home observations;
the final X gap between the measured object footprints was 21.445 mm. Both full
footprints remained inside the declared region. Eight contacts refer to distinct
solver contact rows on the existing `floor` and `demo_floor` colliders; the audit
reconstructs the upward reactions from their raw frames and force vectors without
counting any row twice.

The region is explicitly `[0.13, 0.27] × [-0.18, -0.06]` m in the robot base
frame. The 20 mm planning border margin, 20 mm object separation and independent
outer-region containment are different conditions. The rectangle follows the
workspace, observed geometry and complete-inventory packing checks; it is not a
replacement for the original test's 60 mm point-distance assertion.

The journal is passive: it records the last solved interval of each existing
driver batch, with final integrated qpos/footprints carrying a separate timestamp.
It does not cover every substep or turn a frozen clock into a settling window.
The separate phase observer is diagnostic and does not establish synchronous
contact/final-state evidence. Actual initial, first-placement and final images
were inspected; images do not establish support forces or rest.

Pytest reported **1 passed in 157.36 s**. The supervisor completed naturally in
158.245 s, exit 0, with no termination signal, owned launcher or process-group
member left. All 1,693 source inputs, 26 pinned asset files and four protected-store states
matched. A separate bytes-only postcheck also matches the generated scene to its
prelaunch plan pin (27 asset/scene files total). Shared-host elapsed times are observations,
not controlled performance benchmarks.

Raw proofs, trace, keyframes, process closure and source hashes are bound by
[the physical audit](../benchmark/results/mujoco_two_pick_native_cpu_20261003.json).
The episode epoch is `f5afd9d89e8f43da8501413b691b9511`. Placement model plus
descriptor SHA-256 is `62060df4f3b0b8e86e8f0a06988858c705d8ea413fae77be0c3963cf2fbf2004`.
Withdrawal's empty-descriptor digest is
`ffcc49f162342c16ba3a1e86fde949ee96c00dd8da99cd27d73a3c8eed5ad945`;
these are intentionally different digest scopes over the same compiled model.

## Implementation and limits

The optional native-collider adapter plans carry, release, escape and home before
opening; it replans the empty-tool withdrawal from measured post-opening joints
and geometry. The generic safety fence retains a withdrawal obligation before
release and admits only its exact commands on the owning thread. Failed opening,
withdrawal or recovery does not authorize a fallback home sweep or another skill.

Placement predicts the observed body-to-tool transform in each destination
orientation and shares that calculation between region preview and ordinary
`place_at`. This is scratch geometry, never a weld, dynamic constraint or object
pose write. Snapshot/model/epoch checks and the original cancellation token reach
both actual stream and opening boundaries. The requested explicit point and z,
three-second planning budget, collision checks, actuator settings, gravity and
verifier limits remain intact.

A failed placement or a reported held object cannot receive positive placement
credit merely because its pose is near the target. Bounded diagnostic journals
preserve already-captured rows on failure without adding solver reads or a verdict.
Profiles without the explicit region retain their prior aiming semantics. Warp
withdrawal remains explicitly unadmitted; no hardware/Isaac capabilities are
inferred from this MuJoCo result.

Earlier failures remain retained: the original post-home displacement failure,
insufficient region-v1 capacity/edge clearance, region-v2 pre-opening route
rejection, and the later blue attachment-frame mismatch. The stop/reset handoff
race found after the 313-test checkpoint has its causal red/green control. None
of those receipts is relabelled as successful by the latest episode.

## Publication checks

The extraction uses the reviewed commits `22cac914`, `c62b239`, `07080fad`,
`e353096e`, `e2d89c81`, `0b73a48d`, `7a52d956`, `326720e7` and `d273dc4`.
Only conflict context for unrelated task-ledger and fastening branches was
excluded; no complete old runtime file was copied over main. Factory and unrelated
sensor work are not part of this topic.

**575 affected software tests passed in 68.84 s** on publication code
`1a9c231f927de4d36b85b477feb6b65eb6395a25`, with source and protected stores
unchanged. An earlier command named a nonexistent test file and ran zero tests;
that setup error is retained. Production Ruff F/E9 and diff checks passed.
A further **106 reader/destination checks passed in 0.88 s**. Their first run
retained 89 passes and 17 failures: seven needed the absent pinned kitchen
release assets, and ten used a partial fixture with no arm configuration.
The fixture now explicitly declares no region; the eight exact release files
were copied from the verified local bundle. An intermediate run retained the
seven asset errors. No production or assertion changed.

Thirty-three of 36 topic source/config/test files are byte-identical to the
physical checkpoint; the other three are runtime/effects context and that fixture. All common
placement/motion methods are AST-identical. A compiled-model-only check used the
ordinary profile and recomputed both exact model digests above with time 0 and
all step entry points prohibited. This verifies identity, not a new episode.
See [the composition receipt](../benchmark/results/mujoco_manipulation_publication_20261003.json).

The design and negative controls are detailed in
[withdrawal](MUJOCO_RELEASE_WITHDRAWAL.md),
[regions](MUJOCO_DESTINATION_REGION_V2.md),
[post-opening replanning](MUJOCO_POSTRELEASE_WITHDRAWAL.md) and
[measured attachment aiming](MUJOCO_PLACEMENT_ATTACHMENT.md).
