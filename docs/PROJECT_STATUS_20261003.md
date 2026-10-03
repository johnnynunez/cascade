# Project status — 3 October 2026

CASCADE implements modular robot domains, bounded tool execution, independent
observations and explicit outcome verification. Physical acceptance belongs to
an exact model, source and episode; a declaration, passing software test or
successful simulator startup does not admit a new robot. This index separates
implemented capabilities, measured results and remaining work. The table also
includes reviewed candidates in open PRs; it does not claim every capability is
already in main (including task-effect accounting in #90 and RGB-D in #91). The
[1–2 October index](PROJECT_STATUS_20261001.md) preserves earlier source history.

## Capability and evidence

| Area | Implemented and measured | Remaining boundary |
| --- | --- | --- |
| Manipulation | SafeArm control, observed grasp/release guards, task-effect accounting and measured withdrawal. Historical OVRTX/cuMotion PhysX pick/place/home passed; the original two-object MuJoCo delivery test now passes on the separate attachment-aim candidate. | Broader task/campaign reliability, hardware and acceptance of later source compositions. |
| Fastening | Mounted Factory domain, per-solve observation, exclusive owner and bounded startup/action contracts; explicit contact-writer compiler recipe passed native zero-solve preparation. A separate earlier mounted-socket experiment measured threading, seating and retention. | The configured domain has no successful native readiness/action episode. Tool pickup, initial engagement, withdrawal and calibrated preload remain unproved. |
| Locomotion | Optional measured `walk_distance`, cancellation and independent support/rest checks; historical fresh-start ±30 mm episodes passed. | Published candidate profiles have no new model/support admission. The 0.1 m software ceiling is not measured 0.1 m capability or general gait acceptance. |
| Sensing and spatial memory | Passive, bounded sensor providers; identity/epoch/age checks; observed calibrated RGB-D and retained pixel-to-surface annotations without constructing an arm. | Independent metric XY/general-3D accuracy, moving-camera calibration and per-AOV synchronization. No physical SLAM or `go_to` execution is admitted. |
| Speech | Browser/media gateway, real speech-provider path, original-intent deadlines, priority stop and bounded observed-result views. One continuous synthetic-input/native-motion recording exists. | That motion video contains no spoken robot reply. General dialogue/action reliability, microphone/speaker hardware and public hosted service remain unvalidated. |
| Structure and whole-body control | Typed embodiment and passive generalized-joint observations; domains expose only registered resources/tools. | Mixed physical mobile manipulation is refused. No admitted humanoid balance, dexterous-hand or generic whole-body controller. |
| Evaluation | Source-bound VAB and Arena integration/cancellation preflights ran real simulation. | Neither is a completed benchmark task or a predictor of general deployment success. |

## Manipulation and trial 12

The earlier [OVRTX/cuMotion integration](MANIPULATION_ASSEMBLY_20261002.md)
completed a PhysX GPU grasp, lift, placement and home in 126.19 s. That result
remains distinct from ordinary MuJoCo delivery, the mounted-socket experiment
and the complete kitchen campaign.

The original two-pick test on local source `d273dc4f` passed in **157.36 s**
using MuJoCo 3.14 and Mesa software rendering on CPU. Both placements were
independently confirmed after actual opening, withdrawal and home; the second
preserved the first. The red verification window contains 11 observations over
0.600 simulated seconds; blue and the retained red prefix have 12 over 0.660 s. The original 6 cm destination criterion remains intact.
The process exited naturally and source/assets/protected stores were unchanged.
Its [source-bound audit](evidence/project-status-20261003/mujoco-two-pick.json)
does not admit Isaac, GPU or hardware. Earlier occupied-region,
post-opening escape and rotated-attachment targeting failures remain historical
failures; this is one corrected two-object episode.

Scripted stdio MCP kitchen cases passed separately: green pick/place/home in
199.455 s and orange in 241.531 s, followed by separately timed reset and
complete owned closure. They used independent physical/camera witnesses but no
LLM host. [Green receipt](evidence/project-status-20261003/kitchen-mcp-300s.json),
[orange receipt](evidence/project-status-20261003/kitchen-mcp-orange-complete.json).

**The latest real OpenClaw/Qwen trial 12 remains FAIL at the unchanged 300 s
MCP limit.** On `f674fa64`, one green pick was attempted; no retry, reset or
orange task followed. Partial release near the destination was observed, but
it does not replace the missing completed placement/home/campaign result.
Return-home timing is an inference from joint readback and control flow:
there is no exact home-start command receipt. The previous private-path attempt
timed out during placement. Four units, eleven process births and four ports
closed; software containment does not establish physical rest or normal campaign
shutdown. External embedding requests returned 429 despite local Qwen inference.
[Latest source-bound report](evidence/project-status-20261003/realhost-trial12-observer.json).

[PR #86](https://github.com/johnnynunez/cascade/pull/86) adds optional target
receipts and preserves the launcher flag. Both diagnostics default **off**.
They distinguish queue acknowledgement, first setter return, repeats and an
available physics boundary; none is itself task success. The instrumented
real-host harness has CPU/preflight review, but no executed episode or replay
has supplied the missing command timing or demonstrated a speedup.

## Fastening and locomotion

The historical mounted-socket experiment measured 15.365 turns, 38.155 mm
advance, seating and two seconds of zero spindle-motor torque. The arm/socket
remained engaged; it was not tool withdrawal or preload calibration. Ordinary
`turn_screw` cannot claim verified tightening without an independent fastener
observer. [Experiment and limits](MANIPULATION_ASSEMBLY_20261002.md#physical-threading-seating-and-retention).

Factory's separate v2 runtime readiness failed its unchanged **10 s** budget:
the first advance took 9.364 s, with 20 completed steps before timeout and one
more drained at closure. No quiet startup or fastening action was admitted.
The v3 compiler recipe binds concrete contact-writer types and loaded symbols;
492 CPU checks passed, with seven exact-ABI checks separately passing. Native v3 preparation subsequently passed on `1f0d194f`, model `e712ac6c`:
39 loads and 22 predicates, with scene/native clocks and steps unchanged at zero.
Both owned processes closed naturally; six foreign process births and the
source/SDK/assets/stores were preserved. The observed 82.512 s on a shared GPU
is not a performance claim. A separate v3 readiness harness is prepared but has not run at this snapshot.
**Ordinary readiness and fastening remain unadmitted.**
Compilation success is not a solved task.
[Readiness failure](evidence/project-status-20261003/factory-readiness-v2-review.json),
[v3 CPU receipt](evidence/project-status-20261003/factory-contact-writer-cpu.json),
[v3 native preparation](evidence/project-status-20261003/factory-native-v3-prepare.json).

MicroDuck has four recorded fresh-start forward/reverse **30 mm** passes, a
separate small negative turn and priority cancellation with observed rest.
Other transition/contact and larger-turn failures remain retained. New source
composition requires a new identity: the distance profiles remain unpinned and
reject construction before transport. Same-solve reuse does not create advancing
observations or new travel credit; mocks cannot confirm physical outcomes.
[Historical distance receipt](evidence/project-status-20261003/microduck-distance-candidate.json).

## RGB-D geometry, speech and evaluation

A prior native static-camera RGB-D v2 episode delivered fresh captures over TCP
through the real MCP handler/SensorHub into 20 retained surface annotations.
Its selected ground-Z checks did not establish XY accuracy, general 3D accuracy
or independent timestamps for both AOVs.
[Capture/annotation receipt](evidence/project-status-20261003/rgbd-native02-review.json).

The later layout-A episode remains **FAIL**: all five native captures identified
the four tags, but all five failed the unchanged independent held-out geometry
gate. Four live geometry attempts produced **zero accepted annotations**.
Held-out RMS was 0.1648–0.1776 px against 0.15 px. The separate schema-5 CPU
checker evaluated a frozen 48-image synthetic corpus, excluding those native
failures: 19 accepted, **29 rejected** (16 missing detections, 13 residual
failures), with two gains and no admission losses against the prior detector.
Some admitted cases still had analytical-truth RMS 0.350258 px: a consistent
homography is not proof of absolute accuracy. No pixel gate was relaxed.
[Native result](evidence/project-status-20261003/rgbd-layout-a-native-summary.json),
[CPU comparison](evidence/project-status-20261003/rgbd-checker-accuracy-cpu.json).

The delivered **61.056 s** continuous video retains 915 frames. Labelled
synthetic speech entered the actual browser/Whisper-base/Qwen3-1.7B route and
selected `locomotion.walk_distance(0.03)`. Controller displacement was 25.249 mm;
an independent reconstruction with different endpoints measured 25.921 mm.
The supervisor selected stop, and the native audit reported motion and rest
confirmed. **No TTS reply was recorded.** This used conversation/RobotRuntime,
not MCP/OpenClaw or hardware. The original owner failure receipt remains intact;
the later bounded audit/delivery is recorded separately. File/media/source
consistency was independently reviewed without rerunning the physical auditor.
[Delivery](evidence/project-status-20261003/microduck-video-delivery.json),
[independent review](evidence/project-status-20261003/microduck-video-root-review.json).

VAB's real Panda move/cancel and Arena's PhysX zero-action/cancel preflights
validate limited integration. Arena ran 20 control steps/160 solves, and the
interrupted case stopped admission before step nine. No official benchmark task
success or continuous-world braking follows from those checks.
[VAB receipt](evidence/project-status-20261003/vab-native-preflight-20261002.json),
[Arena receipt](evidence/project-status-20261003/arena-native.json).

## Publication and reproducibility

The [GitHub snapshot](evidence/project-status-20261003/github-publication.json)
records the read-only API observation time and exact branch/commit state.
PRs [#79](https://github.com/johnnynunez/cascade/pull/79),
[#80](https://github.com/johnnynunez/cascade/pull/80),
[#81](https://github.com/johnnynunez/cascade/pull/81),
[#83](https://github.com/johnnynunez/cascade/pull/83) and
[#84](https://github.com/johnnynunez/cascade/pull/84) were merged into main.
[#82](https://github.com/johnnynunez/cascade/pull/82) merged into a feature
branch; [#85](https://github.com/johnnynunez/cascade/pull/85) is the separate
Factory integration into main and was still open at observation. PRs
[#86](https://github.com/johnnynunez/cascade/pull/86),
[#87](https://github.com/johnnynunez/cascade/pull/87),
[#88](https://github.com/johnnynunez/cascade/pull/88),
[#89](https://github.com/johnnynunez/cascade/pull/89),
[#90](https://github.com/johnnynunez/cascade/pull/90) and
[#91](https://github.com/johnnynunez/cascade/pull/91) were open. PR state is not
CI or physical acceptance; this documentation change performs no merges.

Historical receipt fields about a publication freeze or reserved video resources
refer to those past runs, not current publication or execution prerequisites.
The [copy manifest](evidence/project-status-20261003/sources.json) binds each
portable receipt to its unchanged original bytes. Original local artifact paths
inside receipts identify retained evidence, not downloadable repository files.
No new unit suite, benchmark, model inference or physical run was performed for
this documentation update.
