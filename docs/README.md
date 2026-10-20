# Documentation index

Start here: [capability status](ROADMAP.md) says what is measured and what remains;
[Quickstart](QUICKSTART.md) opens the demo. A dated measurement keeps its original
source and scope; it does not certify a later checkout.

## Operate and deploy

- [PAAI on NVIDIA DGX Spark](DGX_SPARK_SETUP.md): install, READY checks, quick recovery
- [Booth runbook](BOOTH_RUNBOOK.md) and the offline [presenter page](spark-presenter/index.html) (copyable commands; connects to no robot)
- [Opening the demo](QUICKSTART.md), [Brev kitchen deployment](BREV.md), [OpenShell](OPENSHELL.md)
- [Chrome / Firefox camera companion](CHROME_EXTENSION.md), [Franka FR3 with OpenClaw](FRANKA_OPENCLAW.md)
- Grasp servers: [GraspGen-X on DGX Spark](GRASPGENX_SPARK.md), [HUG](HUG.md), [VLA executor](VLA_EXECUTOR.md)

## Real arm bring-up

- [Hand-eye calibration](HANDEYE_CALIBRATION.md)
- [WRC control port (reBot B601-RS)](WRC_CONTROL_PORT.md) and [WRC perception port](WRC_PERCEPTION_PORT.md)
- [reBot gripper squeeze cap](REBOT_GRIP_SQUEEZE_CAP.md), [contact stop](CONTACT_STOP.md)
- [Reach envelope](REACH_ENVELOPE.md), [lift-first clearance routes](CLEARANCE_LIFT_ROUTES.md)

## Runtime contracts

- [Architecture](ARCHITECTURE.md): tiers, `execute()`, safety path, module map, known limitations
- [Composable robots](ROBOT_MODULARITY.md), [spatial providers](SPATIAL_PROVIDERS.md), [spatial commands](SPATIAL_COMMANDS.md)
- [Conversation gateway](CONVERSATION.md), [programs tier](PROGRAMS_TIER.md), [retry evidence admission](DREAM_RSI_ADAPTATION.md)

## Simulators, mapping and other robots

- [Newton engine](NEWTON_ENGINE.md), [OVRTX renderer](OVRTX_RENDERER.md), [cuMotion](CUMOTION.md), [nvblox](NVBLOX.md)
- [MicroDuck](MICRODUCK.md) and its [agentic demos](MICRODUCK_AGENTIC_DEMOS.md), [Unitree H2](HUMANOID_H2.md)
- [Factory fastening](FACTORY_FASTENING_RUNTIME.md) and its [G1 humanoid variant](HUMANOID_FASTENING.md), [benchmarks](BENCHMARKS.md) (LIBERO layer attribution)

## Elsewhere in the repo

- [Chrome](../extensions/chrome/README.md) and [Firefox](../extensions/firefox/README.md) companion packaging and permissions
- [Optional Brev streaming](../deploy/brev/streaming/README.md) and the [video viewer](../deploy/runtime/web/camera-player/README.md) (x86 deployment contracts)
- [reBot provenance](../assets/REBOT_PROVENANCE.md), per-asset `PROVENANCE.md` and the [kitchen notice](../demo/scene/NOTICE.md)
- [AGENTS.md](../AGENTS.md): contributor and coding-agent conventions
