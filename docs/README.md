# Documentation index

Start with the [roadmap](ROADMAP.md): what is measured and what remains.
The install examples pin the physical source; a later documentation-only commit
does not change which code produced those measurements.

## Operate and deploy

- [PAAI on NVIDIA DGX Spark](DGX_SPARK_SETUP.md)
- [Booth runbook — hands-on expo session](BOOTH_RUNBOOK.md)
- [Opening the demo](QUICKSTART.md)
- [GraspGen-X on DGX Spark](GRASPGENX_SPARK.md)
- [HUG, an opt-in second grasp backend](HUG.md)
- [Brev kitchen deployment](BREV.md)
- [Chrome camera companion](CHROME_EXTENSION.md)

## Runtime contracts

- [Architecture](ARCHITECTURE.md)
- [WRC control port for the reBot B601-RS: fault clear, park, gripper, Cartesian moves, onsite checks](WRC_CONTROL_PORT.md)
- [reBot RS gripper: opt-in contact-relative squeeze cap and its hardware protocol (B38)](REBOT_GRIP_SQUEEZE_CAP.md)
- [Composable robots, typed sensors, skill graphs and Arena/VAB validation](ROBOT_MODULARITY.md)
- [Local conversation gateway and typed robot speech tools](CONVERSATION.md)
- [Capture-time frames, spatial memory and planar route replay](SPATIAL_PROVIDERS.md)
- [Scoped retry evidence admission](DREAM_RSI_ADAPTATION.md)
- [Programs tier: authored and distilled task scripts under one admission rule (opt-in)](PROGRAMS_TIER.md)
- [Natural-language spatial commands: frames, vague sizes, undo, object-relative moves, wrist camera on top](SPATIAL_COMMANDS.md)

## Mapping and engine experiments

- [nvblox and the kitchen cameras](NVBLOX.md)
- [Optional OVRTX RGBD renderer: x86/ARM evidence and limits](OVRTX_RENDERER.md)
- [Optional cuMotion planner: native x86/ARM GPU evidence and candidate-only API](CUMOTION.md)
- [Newton physics engine: setup and validation](NEWTON_ENGINE.md)

## Historical measurements

These reports retain their original source, failed cases and scope. They do not
certify a later checkout.

- [Benchmarks](BENCHMARKS.md)

## Research and future work

A design or paper comparison is not an enabled runtime feature. In particular,
Jev/Kev is offline evaluation. OVRTX is an implemented optional renderer listed
above; automatic physics producers and its kitchen acceptance remain future work.

- [Roadmap](ROADMAP.md)

## Other documentation locations

- [Offline presenter page](spark-presenter/index.html): copyable installation and
  operation instructions; its page does not connect to a robot.
- [Chrome companion](../extensions/chrome/README.md) and
  [Firefox companion](../extensions/firefox/README.md): browser packaging and
  permissions. A Live camera widget is not manipulation acceptance.
- [Optional Brev streaming](../deploy/brev/streaming/README.md) and
  [optional video viewer](../deploy/runtime/web/camera-player/README.md): separate
  x86 deployment contracts, not default Spark streaming claims.
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
