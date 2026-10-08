# Documentation index

Start with [current source and acceptance](PROJECT_STATUS_20261003.md). It
separates merged software, current physical runs and historical measurements.
The install examples pin the physical source; a later documentation-only commit
does not change which code produced those measurements.

## Operate and deploy

- [PAAI on NVIDIA DGX Spark](DGX_SPARK_SETUP.md)
- [Build-a-Claw presenter card](PRESENTER_QUICKSTART.md)
- [Spark delivery details](SPARK_DELIVERY.md)
- [Build a Claw with PAAI](BOOTH_GUIDE.md)
- [Booth runbook — hands-on expo session](BOOTH_RUNBOOK.md)
- [Opening the demo](QUICKSTART.md)
- [GraspGen-X on DGX Spark](GRASPGENX_SPARK.md)
- [HUG, an opt-in second grasp backend](HUG.md)
- [DGX Spark portability](DGX_SPARK_BREV.md)
- [Brev kitchen deployment](BREV.md)
- [Chrome camera companion](CHROME_EXTENSION.md)
- [Firefox camera companion](FIREFOX_EXTENSION.md)

## Runtime contracts

- [Architecture](ARCHITECTURE.md)
- [Composable robots, typed sensors, skill graphs and Arena/VAB validation](ROBOT_MODULARITY.md)
- [Local conversation gateway and typed robot speech tools](CONVERSATION.md)
- [Capture-time frames, spatial memory and planar route replay](SPATIAL_PROVIDERS.md)
- [Isaac motion clock](ISAAC_MOTION_CLOCK.md)
- [Prepare Isaac verification before native proof](ISAAC_STARTUP_READINESS.md)
- [Reject localization results after their image expires](LOCALIZATION_FRESHNESS.md)
- [Detector preparation, bounded reuse and local GPU comparison](DETECTOR_MODEL_REUSE.md)
- [Isaac camera frames bound to render history](isaac-render-frame-history.md)
- [Independent RGB/depth encoding: passive profiles 10/11 and CPU evidence](ISAAC_FRAME_ENCODING.md)
- [Observed-surface finger veto (experimental)](observed-finger-gate.md)
- [PhysX finger envelope, provenance and local admission](physx-finger-envelope.md)
- [Check the observed closing stroke before commanding it](observed-finger-closing.md)
- [Reject finger endpoints hidden behind observed non-target depth](observed-finger-occlusion.md)
- [Align detector masks with their RGB-D frame](detector-mask-letterbox.md)
- [Bounded grasp feasibility search](grasp-planning-search.md)
- [Preserve the planned grasp order](grasp-memory-ranking.md)
- [Held-object observation after the NV08 failure](HELD_OBJECT_OBSERVATION.md)
- [Native visitor turn budget](native-visitor-turn-budget.md)
- [Scoped retry evidence admission](DREAM_RSI_ADAPTATION.md)

## Mapping and engine experiments

- [nvblox and the kitchen cameras](NVBLOX.md)
- [Experimental nvblox manipulation](NVBLOX_PAYLOAD_EXPERIMENT.md)
- [Retained-payload capture refresh](NVBLOX_RETAINED_ANCHOR_REFRESH.md)
- [Retained withdrawal after release](NVBLOX_RELEASE_RETREAT.md)
- [Retained attachment during NV transport](NVBLOX_CARRY_ATTACHMENT.md)
- [Local RTX Pro validation and retained diagnostics](LOCAL_RTX_VALIDATION.md)
- [Optional Isaac bridge Python profiling](ISAAC_BRIDGE_PROFILING.md)
- [Profile 09: failed placement, qualified CPU window and verified closure](LOCAL_RTX_VALIDATION.md#profiling-attempt-09-python-spans-and-placement-timeout)
- [nvblox map reset and allocation history](NVBLOX_MAPPER_RESET.md)
- [Read-only construction of truth-pose views](NVBLOX_TRUTH_READONLY.md)
- [Optional OVRTX RGBD renderer: x86/ARM evidence and limits](OVRTX_RENDERER.md)
- [Optional cuMotion planner: native x86/ARM GPU evidence and candidate-only API](CUMOTION.md)
- [Screw manipulation research: SimReady/Factory assets and physical acceptance](SCREW_MANIPULATION_RESEARCH.md)
- [Newton physics engine: setup and validation](NEWTON_ENGINE.md)
- [Bridge degradation under sustained verification polling](BRIDGE_DEGRADATION.md)

## Historical measurements

These reports retain their original source, failed cases and scope. They do not
certify a later checkout.

- [Spark launch02: orange fails before close](SPARK_LAUNCH02_FORENSICS_20261001.md)
- [Spark launch03: physical green case passes, camera freshness fails](SPARK_LAUNCH03_FORENSICS_20261001.md)
- [Spark launch04: green task and cameras pass; orange grasp fails](SPARK_LAUNCH04_FORENSICS_20261001.md)
- [Idle Spark rendering comparison](SPARK_RENDER_COMPARISON_20261001.md)
- [Benchmarks](BENCHMARKS.md)
- [Layer attribution on LIBERO: which layer is actually failing?](LAYER_ATTRIBUTION_LIBERO.md)

## Research and future work

A design or paper comparison is not an enabled runtime feature. In particular,
Jev/Kev is offline evaluation. OVRTX is an implemented optional renderer listed
above; automatic physics producers and its kitchen acceptance remain future work.

- [Roadmap](ROADMAP.md)
- [Decision models in Cascade](JEV_DECISIONS.md)
- [Agentic upgrades (2026-07-31)](AGENTIC_UPGRADES.md)
- [cascade against the published agentic-manipulation results](COMPARISON_TO_PUBLISHED_WORK.md)
- [Where cascade can actually advance the state of the art](SOTA_CONTRIBUTION_ANALYSIS.md)
- [Perception and evaluation for an open-vocabulary booth agent](SOTA_PERCEPTION_AND_EVALUATION.md)
- [Perception and Execution: what to change, and what the measurements say](PERCEPTION_AND_EXECUTION_RESEARCH.md)
- [Synthetic-Augmented RGB-D → 3D Object Localization](SYNTHETIC_RGBD_PIPELINE.md)
- [Design: mobility, humanoids and navigation (`MobileBase`)](MOBILITY_AND_NAVIGATION_DESIGN.md)
- [HomeBody comparison: grounded memory, local recovery and whole-body ownership](HOMEBODY_COMPARISON.md)
- [Design brief: `type: ros2` backend for cascade — findings from ros-claw/rosclaw](ROS2_BACKEND_BRIEF.md)
- [OpenClaw 2.0 Integration Brief (for `scripts/openclaw_demo.sh`)](OPENCLAW_2.0_INTEGRATION_BRIEF.md)

## Other documentation locations

- [Offline presenter page](spark-presenter/index.html): copyable installation and
  operation instructions; its page does not connect to a robot.
- [Chrome companion](../extensions/chrome/README.md) and
  [Firefox companion](../extensions/firefox/README.md): browser packaging and
  permissions. A Live camera widget is not manipulation acceptance.
- [Optional Brev streaming](../deploy/brev/streaming/README.md) and
  [optional video viewer](../deploy/runtime/web/camera-player/README.md): separate
  x86 deployment contracts, not default Spark streaming claims.
- [September one-click design](superpowers/specs/2026-09-10-spark-oneclick-design.md)
  and [implementation plan](superpowers/plans/2026-09-10-spark-oneclick.md): retained
  historical design; current commands are in the setup guide.
- [reBot provenance](../assets/REBOT_PROVENANCE.md), per-asset `PROVENANCE.md`,
  [kitchen notice](../demo/scene/NOTICE.md), licenses and vendored validation
  reports: ownership and original asset evidence are preserved.
- [Repository development guide](../AGENTS.md): factual runtime conventions and
  existing contributor instructions. Locally installed authoring skills are
  not runtime dependencies.

## Documentation verification

The documentation-only PR #43 audit on 1 October inventoried tracked Markdown, readmes, historical plans,
asset documentation and the standalone presenter page. It inspected profile,
launcher and installer definitions before updating defaults and examples.
Runtime prompts, test fixtures, licenses, configuration and executable source
were kept unchanged. The vendored validation guide now identifies two upstream
references as unavailable in this snapshot instead of linking missing files;
no asset result or threshold was invented to replace them.

Offline checks cover local links/anchors, shell-block syntax and the existing
presenter/setup/consent/English tests. They do not execute installation, launch,
reset, motion or external commands copied from the guides. That audit compared
all non-documentary tracked files with physical MAIN `477c88f`, including Git LFS
content hashes and sizes; its presenter JavaScript and CSS were unchanged.
The later PhysX-envelope and OVRTX integration changes executable source and has
its own source-bound software and rendering evidence. This documentation update
distinguishes those contracts from the retained physical runs; it does not
extend the earlier documentation-only equivalence claim to new runtime code.
