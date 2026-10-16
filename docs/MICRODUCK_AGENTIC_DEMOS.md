# Agentic MicroDuck demos (Isaac Lab 3.0 + Newton physics, Cascade agents, OVRTX replay)

LLM agents built on Cascade's `AgentOrchestrator` drive the **unchanged learned MicroDuck policies** in Isaac Lab 3.0
with Newton physics. Each robot has its own agent, its own episodic memory, its own ToF occupancy map and its own
trace. The recorded physics is then replayed through Cascade's `OvrtxRenderer` into 1080p review videos. The videos
show what the agent saw and decided, and they label what is illustrative.

## What runs where

```
run_episode.py (one process, Isaac Lab venv)
 ├─ microduck_lab_world.py   one Isaac Lab env (= one Newton world) per robot, same scene origin
 │    controllers per robot: walker ONNX | onboard_hero actor | get-up policy | depth_navigation_v4 | stand
 ├─ microduck_world_port.py  owner-thread bridge: agents ask, the main thread steps physics (lockstep clock)
 ├─ microduck_agent_runtime.py  per-robot skill runtime (tools below) + RobotMap + EpisodicMemory + trace
 └─ AgentOrchestrator x N    one LLM agent per robot (threads), fleet board for explicit messages
render_episode.py (Isaac Lab venv, after the run)
 └─ microduck_ovrtx_replay.py  canonical MicroDuck.usd + course -> OvrtxRenderer (RT2) -> 1080p composite
```

* **Physics and policies.** Every joint target comes from a learned policy. These are the handoff
  deliveries, used unchanged:
  * `velocity_flat_existing.onnx` (velocity walker);
  * `onboard_hero` (perception actor: proprioception + ToF 8x8 + head RGB 48x36);
  * `getup_kitfloor_ft_model_1100`;
  * `depth_navigation_v4_iter300`.

  No gait or pose is scripted or repaired. Idle robots run the walker at zero twist, so they balance rather than
  freeze. `go_to_waypoint` / `follow_route` only choose the *velocity command* that the learned walker tracks. This
  is a small heading/speed loop, and it is disclosed in every tool receipt.
* **Sensors.** The env is the gtceu_microduck onboard-sensor env (`sensors/onboard_env.py`). It provides a
  hardware-matched VL53L8CX 8x8 ToF ray model and a 48x36 head RGB (Newton Warp renderer, clip range 1.5 m). The
  head camera shows the course's own USD `displayColor`; this demo does not recolour anything the policy sees.
* **Lockstep time.** Physics advances only while a motion tool runs. LLM thinking time is logged as events. The video
  timeline is simulation time, and decision moments are shown as short holds with the reasoning text.
* **Per-robot map** (`memory/robot_map.py`): a log-odds grid built only from that robot's own ToF frames, with its
  own landmarks. Zone classes are floor, obstacle (hit above floor + 4 cm), free (no target) and drop. A drop is a
  no-target on a beam that should have met the floor, or a return more than 5 cm below the floor. The reported drop
  distance is an upper bound: it is where the beam would have met the robot's floor.
* **Fleet** (multi_room_fetch): robots coordinate only through an explicit fleet board (`share_with_fleet`,
  `read_fleet_board`). Each robot lives in its own Newton world. There is no inter-robot collision, and no robot sees
  another in its sensors. The videos state this.
* **Pose.** The pose comes from the simulator and stands in for SLAM. No cuVSLAM library is installed on this
  machine.
* **Object identity.** The 48x36 head RGB (the policy's input) renders USD colours, but only within its 1.5 m clip
  range, and the floor renders cyan-blue. It can show the green pad when the robot is close (backstage), but it
  cannot find a red or blue box across a room. Scenarios that need object identity declare `detectable` prims.
  The `detect_objects` stand-in then reports their simulator positions only when they are inside the head camera's
  field of view and range and not behind a wall. Receipts carry `objects_in_view`, and the end card discloses it.
* **Contact.** `no_contact: true` in a verify spec makes a robot fail when any recorded body origin comes within
  3 cm of an obstacle's bounding box (`sim/microduck_clearance.py`). This is an indicator, not a contact sensor.
  Every run stores the clearance in `episode.json`.

## Agent tools (per robot)

| Tool | Policy / source |
| --- | --- |
| `observe`, `scan_surroundings`, `my_map`, `bearing_to`, `mark_landmark`, `recall_memory` | ToF + head RGB, own map, own memory |
| `plan_route_on_my_map` | A* on the robot's own ToF map (obstacles/drop-offs inflated 0.14 m by default, unknown optimistic); proposes waypoints only |
| `detect_objects` (scenarios with `detectable`) | detector STAND-IN: simulator ground truth inside the head camera's 82 deg field of view, 2.5 m, walls occlude |
| `walk`, `turn` | learned walker, agent-chosen twist and duration, ToF guard on forward travel |
| `go_to_waypoint`, `follow_route` (1-6 waypoints) | learned walker; scripted heading/speed loop picks the command |
| `walk_with_onboard_perception` | onboard_hero perception actor (fixed 0.1 m/s forward, policy steers from ToF + RGB) |
| `navigate_to` | depth_navigation_v4 (goal-conditioned) |
| `get_up` | get-up policy |
| `share_with_fleet`, `read_fleet_board` | explicit messages between robots |
| `task_done` | agent's own claim; the scenario verifier checks the recorded final state separately |

The ToF guard stops forward travel when an obstacle is closer than 0.22 m ahead, or when a drop-off is closer than
about 0.45 m. Turning in place is always allowed.

## Scenarios

| Config | Robots | Goal | Verifier |
| --- | --- | --- | --- |
| `backstage_obstacles.yaml` | 1 | cross a backstage corridor with flight cases to a green marker | reach (2.6, 0.2) r 0.35, upright |
| `stage_edge.yaml` | 1 | walk to the front edge of a 0.30 m stage without falling off | inside x 1.70-2.10, abs(y) <= 0.70, z >= 0.36 |
| `multi_room_fetch.yaml` | 3 | duck_a: find the red box (room B); duck_b: explore and map room B, post objects; duck_c: starts lying on its back, finds the blue box | duck_a reach (3.8, -0.75) r 0.5; duck_b upright; duck_c reach (1.45, 0.85) r 0.5 |
| `creative_gym.yaml` | 1 | cross the gentle-ridge lane of the MicroDuck Gym course | reach (-0.8, 0.4) r 0.3 |

`creative_gym` uses the gym100 `course_single.usda`. The file is not vendored. Set `CASCADE_MICRODUCK_GYM_COURSE` to
the file; the run refuses any file whose sha256 differs from the pin in the config. Every mesh in that file has
purpose `guide`, which OVRTX does not render. The review video therefore renders its source, the Creative gym
collision asset (`CASCADE_MICRODUCK_GYM_VISUAL`, also pinned, same frame), and the end card says so.

## Run

```bash
WT=/path/to/cascade/worktree
LAB=/path/to/IsaacLab-microduck/.venv/bin/python   # Isaac Lab 3.0 + Newton + ovrtx 0.5.1 + pxr
export PYTHONPATH=$WT/src OMNI_KIT_ACCEPT_EULA=yes
# extra pure-python deps (anthropic, jsonschema, opencv-python-headless, onnxruntime) can be appended to sys.path
# without touching the Isaac Lab venv:
export CASCADE_APPEND_SYSPATH=/path/to/pydeps
python scripts/microduck_agentic/build_scenes.py --output demo/microduck_agentic
$LAB scripts/microduck_agentic/run_episode.py --scenario configs/microduck_agentic/stage_edge.yaml \
    --llm codex_astra --output runs/stage
$LAB scripts/microduck_agentic/render_episode.py --run runs/stage --output runs/stage/video.mp4
```

Outputs of a run:
* `recording.npz` / `.json`: 50 Hz; 16 body poses per robot, controller, ToF, head RGB.
* `events.jsonl`: every LLM reply, tool call and receipt.
* `<robot>/trace.jsonl`.
* `maps_<robot>.npz`.
* `episode.json`: reports, scenario verification, LLM call counts and wall time.

`reverify.py` re-runs the current verifier on the recorded final state. It keeps the old verdict in the file.
`clearance.py` recomputes the clearance of a finished run.

## OVRTX notes

1. `src/cascade/sim/microduck_ovrtx_replay.py`: `build_render_scene(...)` writes one flat Z-up USD with each of the
   16 bodies at `/World/DuckBodies/<robot>/<body>`; `ReplayPoses(...).transforms(frame)` feeds `SceneSnapshot`.
2. `OvrtxRenderer(..., sdk_versions=...)` accepts only `KNOWN_SDK` pairs (`sim/ovrtx_renderer.py`).
3. On ovrtx 0.5.1 / ovstage 0.2.1 the first single-prim write stops updating after frame 1, so the replay writes an
   identity `/World/ReplayWriteSentinel` first on every frame. Evidence and probe:
   [`evidence/microduck-ovrtx-replay-20261013/`](evidence/microduck-ovrtx-replay-20261013/).

## Honesty labels burned into the videos

* "illustrative camera view - OVRTX render of the recorded physics". The cameras are not robot sensors.
* "POLICY OBSERVATION (exact actor inputs)". These panels show the held ToF 8x8 frame and the 48x36 head RGB tensor.
* End card:
  * "Maps: each robot's own ToF occupancy grid with the simulator pose (no SLAM in this demo)."
  * "Robots are separate Newton worlds: no robot-robot contact, robots do not see each other."
  * The simulator check (PASSED/FAILED) next to the agent's own claim.
* The LLM name and the active controller/policy are shown on every frame.

## Tests

`tests/test_microduck_agentic.py` contains CPU-only tests that do not need Isaac Lab. They cover:
* ToF classification and drop detection on a synthetic sensor;
* per-robot maps;
* the lockstep port, guard stops and shutdown;
* two `AgentOrchestrator` agents with a scripted LLM on a fake kinematic world (not physics);
* scenario/verifier loading;
* the replay transform math, including the sentinel;
* the OVRTX SDK pair gate.

The physics paths are exercised by the episode runs themselves (see `episode.json`).
