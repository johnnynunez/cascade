# Agentic MicroDuck demos (Isaac Lab 3.0 + Newton physics, Cascade agents, OVRTX replay)

**Next action:** build the scenes, run one scenario and render its review video (needs the Isaac
Lab 3.0 venv with Newton, ovrtx 0.5.1 and `pxr`; one GPU; minutes per scenario):

```bash
WT=/path/to/cascade/worktree
LAB=/path/to/IsaacLab-microduck/.venv/bin/python   # Isaac Lab 3.0 + Newton + ovrtx 0.5.1 + pxr
export PYTHONPATH=$WT/src OMNI_KIT_ACCEPT_EULA=yes
# extra pure-python deps (anthropic, jsonschema, opencv-python-headless, onnxruntime) can be appended to sys.path
# without touching the Isaac Lab venv:
export CASCADE_APPEND_SYSPATH=/path/to/pydeps
$LAB scripts/microduck_agentic/build_scenes.py --output demo/microduck_agentic
$LAB scripts/microduck_agentic/run_episode.py --scenario configs/microduck_agentic/stage_edge.yaml \
    --llm codex_astra --output runs/stage
$LAB scripts/microduck_agentic/render_episode.py --run runs/stage --output runs/stage/video.mp4
```

LLM agents on Cascade's `AgentOrchestrator` drive the **unchanged learned MicroDuck policies** in
Isaac Lab 3.0 with Newton physics; each robot has its own agent, episodic memory, ToF occupancy map
and trace. The recorded physics is replayed through `OvrtxRenderer` into 1080p review videos that
show what the agent saw and decided and label what is illustrative. CPU tests:
`uv run --extra dev pytest tests/test_microduck_agentic.py -q` (no Isaac Lab).

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

1. **Policies, unchanged:** `velocity_flat_existing.onnx` (walker), `onboard_hero` (proprioception
   + ToF 8x8 + head RGB 48x36), `getup_kitfloor_ft_model_1100`, `depth_navigation_v4_iter300`. No
   gait or pose is scripted; idle robots run the walker at zero twist. `go_to_waypoint` /
   `follow_route` only choose the velocity command (a small heading/speed loop, disclosed in every
   receipt).
2. **Sensors:** the gtceu_microduck onboard-sensor env (`sensors/onboard_env.py`): a VL53L8CX 8x8
   ToF ray model and a 48x36 head RGB (Newton Warp renderer, 1.5 m clip) showing the course's own
   `displayColor`. **Lockstep time:** physics advances only while a motion tool runs; LLM thinking
   time is logged as events.
3. **Per-robot map** (`memory/robot_map.py`): log-odds grid from that robot's ToF only; zones floor,
   obstacle (hit above floor + 4 cm), free, drop (a no-target where the floor was expected, or a
   return more than 5 cm below the floor; the drop distance is an upper bound). The pose comes from
   the simulator and stands in for SLAM (no cuVSLAM here).
4. **Fleet** (multi_room_fetch): coordination only through the explicit fleet board; each robot is
   its own Newton world (no inter-robot collision or sensing; the videos say so).
5. **Object identity and contact:** scenarios needing identity declare `detectable` prims;
   `detect_objects` is a stand-in reporting simulator positions inside the head camera's view and
   range, not behind walls (`objects_in_view` in receipts). `no_contact: true` fails a robot when a
   body origin comes within 3 cm of an obstacle's bounding box (`sim/microduck_clearance.py`; an
   indicator, not a contact sensor; stored in `episode.json`).

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

The ToF guard stops forward travel when an obstacle is closer than 0.22 m ahead or a drop-off
closer than about 0.45 m; turning in place is always allowed.

## Scenarios

| Config | Robots | Goal | Verifier |
| --- | --- | --- | --- |
| `backstage_obstacles.yaml` | 1 | cross a backstage corridor with flight cases to a green marker | reach (2.6, 0.2) r 0.35, upright |
| `stage_edge.yaml` | 1 | walk to the front edge of a 0.30 m stage without falling off | inside x 1.70-2.10, abs(y) <= 0.70, z >= 0.36 |
| `multi_room_fetch.yaml` | 3 | duck_a: find the red box (room B); duck_b: explore and map room B, post objects; duck_c: starts lying on its back, finds the blue box | duck_a reach (3.8, -0.75) r 0.5; duck_b upright; duck_c reach (1.45, 0.85) r 0.5 |
| `creative_gym.yaml` | 1 | cross the gentle-ridge lane of the MicroDuck Gym course | reach (-0.8, 0.4) r 0.3 |

`creative_gym` needs `CASCADE_MICRODUCK_GYM_COURSE` set to the (not vendored) gym100
`course_single.usda`; a file whose sha256 differs from the config's pin is refused. Its meshes are
purpose `guide`, which OVRTX does not render, so the video renders the pinned Creative gym collision
asset (`CASCADE_MICRODUCK_GYM_VISUAL`, same frame) and the end card says so.

Run outputs: `recording.npz` / `.json` (50 Hz; 16 body poses per robot, controller, ToF, head RGB),
`events.jsonl` (every LLM reply, tool call and receipt), `<robot>/trace.jsonl`, `maps_<robot>.npz`
and `episode.json` (reports, verification, LLM call counts, wall time). `reverify.py` re-runs the
current verifier on the recorded final state (keeping the old verdict); `clearance.py` recomputes a
finished run's clearance. Physics paths are exercised only by the episode runs themselves.

The 13 October 2026 runs (18 attempts: verdict at run time and under the current verifier, agent
claim, clearance, LLM calls, wall time, a note per run) are recorded in
[`evidence/microduck-agentic-20261013/attempts.json`](evidence/microduck-agentic-20261013/attempts.json).

## OVRTX notes

1. `src/cascade/sim/microduck_ovrtx_replay.py`: `build_render_scene(...)` writes one flat Z-up USD with each of the
   16 bodies at `/World/DuckBodies/<robot>/<body>`; `ReplayPoses(...).transforms(frame)` feeds `SceneSnapshot`.
2. `OvrtxRenderer(..., sdk_versions=...)` accepts only `KNOWN_SDK` pairs (`sim/ovrtx_renderer.py`).
3. On ovrtx 0.5.1 / ovstage 0.2.1 the first single-prim write stops updating after frame 1, so the replay writes an
   identity `/World/ReplayWriteSentinel` first on every frame. Evidence and probe:
   [`evidence/microduck-ovrtx-replay-20261013/`](evidence/microduck-ovrtx-replay-20261013/).

## Honesty labels burned into the videos

1. "illustrative camera view - OVRTX render of the recorded physics": the cameras are not robot
   sensors.
2. "POLICY OBSERVATION (exact actor inputs)": the held ToF 8x8 frame and the 48x36 head RGB tensor.
3. End card: "Maps: each robot's own ToF occupancy grid with the simulator pose (no SLAM in this
   demo)." and "Robots are separate Newton worlds: no robot-robot contact, robots do not see each
   other."
4. The simulator check (PASSED/FAILED) next to the agent's own claim.
5. The LLM name and the active controller/policy on every frame.
