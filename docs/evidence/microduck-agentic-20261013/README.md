# Evidence: agentic MicroDuck demos (2026-10-13)

Workstation: 2x RTX PRO 6000 Blackwell. Isaac Lab 3.0 + Newton, in the Isaac Lab venv (Python 3.12). Every physics
episode and every render ran through `heavyrun` (memory-capped systemd units, TAG cascade). The LLM is GPT-6-Astra
through the Codex CLI (Cascade `codex_astra` profile), with 7-15 s per call. While the LLM thinks, the simulation is
paused (lockstep).

## Commands

```
# scenes (metre, Z-up USDA)
python scripts/microduck_agentic/build_scenes.py --output demo/microduck_agentic
# one episode (physics + agents + recording); <run> is any output folder
heavyrun -n cascade-ep-<run> -m 32G -g auto -l <runs>/<run>.log -- \
  bash tools/labpy.sh <worktree> scripts/microduck_agentic/run_episode.py \
    --scenario configs/microduck_agentic/<scenario>.yaml --llm codex_astra --output <runs>/<run>
# 1080p review video (OVRTX replay of the recording)
heavyrun -n cascade-render-<run> -m 32G -g auto -l <runs>/render_<run>.log -- \
  bash tools/labpy.sh <worktree> scripts/microduck_agentic/render_episode.py --run <runs>/<run> \
    --output <runs>/<run>/<scenario>_agentic_1080p.mp4
# post-hoc: current verifier + exact clearance on the recorded final state / recorded poses
python scripts/microduck_agentic/reverify.py  --run <runs>/<run> --scenario configs/microduck_agentic/<scenario>.yaml
python scripts/microduck_agentic/clearance.py --run <runs>/<run> --scenario configs/microduck_agentic/<scenario>.yaml
```

`tools/labpy.sh` sets `PYTHONPATH=<worktree>/src`, `OMNI_KIT_ACCEPT_EULA=yes` and `CASCADE_APPEND_SYSPATH=<pydeps>`
before it execs the Isaac Lab venv python. The pydeps are anthropic, opencv-python-headless, jsonschema and
onnxruntime, installed with `uv pip install --target` and appended to `sys.path`, so the shared venv is not
modified.

## Chosen runs (the videos)

| Scenario | Run | Result (simulator check) | Agent's own claim | LLM calls | Sim time | Tool calls |
| --- | --- | --- | --- | --- | --- | --- |
| stage_edge | stage2 | PASS: base (1.75, -0.12), z 0.42 on the stage, 0.35 m from the front edge, upright; clearance 0.247 m | failure (its ToF said the edge was 0.44 m away, outside its 0.25-0.40 m band) | 11 | 19.5 s | observe 3, go_to_waypoint 2, walk, turn, scan, my_map, task_done |
| backstage_obstacles | backstage9 | PASS: 0.247 m from the pad centre (recorded last frame; 0.244 m at episode end), upright, no possible contact (min clearance 0.056 m) | failure (could not verify the pad centre or no-contact) | 21 | 51.0 s | plan_route_on_my_map 6, follow_route 5, scan 3, observe 3, my_map, go_to_waypoint, task_done |
| multi_room_fetch | multiroom6 | PASS 3/3: duck_a 0.4 m from the red box surface (room B), duck_b upright after mapping room B, duck_c 0.37 m from the blue box after getting up from its back | success x3 | 70 | 178.3 s | scan 13, plan 13, follow_route 12, my_map 9, share_with_fleet 6, observe 4, read_fleet_board 2, get_up, detect_objects, ... |
| creative_gym (extra) | gym1 | FAIL: stuck tilted on the ridge lane, 1.78 m short | failure | 23 | 43.7 s | walk_with_onboard_perception 8 (fell once), turn 5, get_up 1, ... |

multiroom6 contact flags: duck_a's recorded body origins came within 0.011 m of the Crate's box, and duck_b's reached
CaseB's box. The multi-room task does not forbid contact, so this is not part of its check. The end card shows it.
Robots are separate Newton worlds: the minimum root distance between two robots was 0.008 m, which would have been a
collision with shared physics. The videos state that robots do not touch or see each other.

## All attempts

`attempts.md` and `attempts.json` list every episode with an `episode.json`: 18 runs, passes and failures alike. For
each run they give:
* the verdict at run time;
* the verdict under the current verifier configs;
* the agent's own claim;
* the clearance and LLM calls.

Verifier changes made after seeing results (stage_edge: point -> edge band; multi_room_fetch: centre radius ->
base-to-surface distance; backstage: `no_contact` added) are marked there. Each one was followed by a fresh run under
the new definition (stage2, multiroom6, backstage5 onward). Attempts per scenario with the final code:
* backstage: 2 runs in parallel, 1 PASS (backstage9) and 1 FAIL (backstage10, possible contact with the speaker box);
* multi_room_fetch: 1 run, 3/3;
* stage_edge: 1 run, PASS.

## Learned vs scripted

* Learned: every joint target. These come from the walker ONNX, the onboard_hero perception actor, the get-up policy
  and depth_navigation_v4, all unchanged handoff deliveries.
* LLM: tool choice and parameters, plans, fleet messages, and the task_done claim.
* Scripted:
  * the ToF stop guard;
  * the turn-angle and waypoint heading loops, which only choose the walker's velocity command;
  * A* route proposals on the robot's own map;
  * the detector stand-in;
  * scenarios and verifiers.
* Stand-ins:
  * pose from the simulator (no SLAM);
  * object identity from simulator ground truth inside the head camera's field of view (multi_room_fetch only).

## Tests

* `python -m pytest tests/test_microduck_agentic.py tests/test_microduck_ovrtx_replay.py -q`: 16 passed. These are
  CPU-only tests on a fake kinematic world, not physics.
* Full suite in a fresh uv venv (`.[dev,kinematics,llm,sim,sim-warp,grasping,grasp-safety,occupancy]`, pytest
  `-n 24`):
  * this branch: 146 failed, 11780 passed, 594 skipped;
  * origin/main in the same environment: 154 failed, 11756 passed, 594 skipped.
* No test fails only on this branch. The 8 extra failures on main are timing-flaky tests under xdist
  (`test_native_render_signal_fence`, `test_process_owner`, `test_mobile_effects`).
* All 146 shared failures are environmental. The kitchen release bundle (`demo/scene/cocina_asier/`, downloaded
  separately) is absent in a fresh worktree, so the placement and kitchen proof tests fail:
  * `test_kitchen_placement_verdict`: 58;
  * `test_spark_placement_proof`: 33;
  * `test_convex_support_geometry`: 30;
  * `test_spark_placement_proof_newton`: 18;
  * `test_placement_reader`: 7.
* `tests/test_kitchen_assets.py` passes. The replay PR re-pins `src/cascade/sim/ovrtx_renderer.py` in
  `demo/scene/own_assets.json`.
