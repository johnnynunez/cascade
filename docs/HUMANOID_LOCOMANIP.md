# Humanoid loco-manipulation through CASCADE (`kind: humanoid_locomanip`)

**Next action:** run the CPU tests (fake bridge speaking the real protocol; no GPU, about 5 s):

```bash
PYTHONPATH=$PWD/src:$PWD .venv/bin/python -m pytest tests/test_humanoid_locomanip.py tests/test_humanoid_harness2.py -q
```

CASCADE stays the agent orchestrator around a learned whole-body policy:

```
operator -> brain (AgentOrchestrator; local vLLM Qwen3.8 / Nemotron) -> tools, policies, harnessings -> verifier -> reply
                                       |
             carry.describe_scene (read)   carry.carry_object (motion)   carry.verify_placement (read)
                                       |
                 HumanoidSafetyGate: every control step (measured state + proposed action), STOP latches
                                       |
     bridge process (Isaac Lab-Arena env + GR00T policy server; its own Python environment, loopback JSON lines)
```

Status: candidate, simulation only. Profile `configs/robots/g1_arena_locomanip.yaml`; measured
results go in the PR, not here.

## Pieces

| piece | file | role |
| --- | --- | --- |
| domain | `src/cascade/skills/humanoid_locomanip.py` | tools, instruction from the arguments (`instruction_template`), segment loop, records |
| gate | `src/cascade/safety/humanoid_gate.py` | per-step PASS/STOP with value vs limit; latched hard stop; pre-motion admission |
| verifier | `src/cascade/skills/humanoid_verifier.py` | confirmed / refuted / unverified from the measured final state |
| client | `src/cascade/skills/humanoid_bridge.py` | protocol, busy/fault handling |
| bridge | `scripts/arena_locomanip_bridge.py` | runs in Arena's venv; owns env + GR00T client; applies an action only after `pass` |
| episodes | `scripts/humanoid_agent_episode.py` | one operator request through the agent, recorded (`trace.json`) |
| replay | `scripts/replay_gate_jsonl.py` | a recorded run's actions through today's gate, offline: first refused step per channel |

## One `carry_object(object, destination)` call

1. Arguments are enum-checked against the profile's `scene`; the instruction is
   `instruction_template` filled with the spoken labels (for brown box / blue bin it is the
   checkpoint's training instruction verbatim).
2. A latched stop refuses the call (`refused`: "stopped: ask the operator to clear the stop").
   Only the operator clears it, and no agent route can: `reset_stop` is not on the agent's tool
   list, the composed runtime refuses it without `operator=True`, and the MCP and fleet MCP
   servers neither list nor accept it for a robot behind this gate (an MCP host is an agent).
   Operator channels: SIGUSR1 to this robot's MCP server only, `kill -USR1 <pid>` for that one server (PID from `pgrep -af 'cascade.apps.mcp_server|cascade-mcp'`, checking the robot on that line, or from the `runs/mcp_<pid>` directory name), from a
   shell on the rig; the CLI console, `--operator-arm`, operator scripts (`operator=True`). Never
   signal by pattern (`pkill -USR1 -f ...`): every cascade MCP server on the host resumes, which
   clears an arm rig's latched e-stop (an fr3 also drives home). SIGUSR1 during a carry stops it
   (0 policy steps after) and closes the bridge session, so that reset is refused until a restart.
   Known limit: these channels are guarded by host access, not identity. An MCP host with a shell
   on the rig running as the robot user (Hermes, Codex, Claude Code via `setup_agents.py`) can send
   SIGUSR1 or restart the server, and a stdio host that starts a fresh server per session starts
   unlatched; the booth agent inside OpenShell has no host shell. Without an operator channel
   (the fleet MCP has none), the stop stays latched until the operator restarts the server, which
   rebuilds the domain. A stop that closed the bridge session mid-segment (a garbled line, a local
   fault) stays latched until that restart: the domain refuses every reset. Harness² advice may
   never name `reset_stop`. The gate vets the measured start state (pre-motion admission).
3. Per control step (50 Hz) the bridge sends the measured state and the policy's proposed action;
   the gate answers `pass` or `stop`. Gates: state finite, pelvis tilt <= 30 deg, pelvis height
   above ankle level >= 0.45 m, measured joints (fingers included) within articulation limits +
   0.05 rad, joint velocity <= limit, pelvis inside the workspace box, arm and waist targets within
   limits + 0.02 rad, leg targets (owned by the whole-body controller, but in the applied action)
   within limits + 0.02 rad, navigation and base-height commands within the checkpoint's training
   envelope (`dataset_statistics.json`) + margin (vx is asymmetric: -0.05..0.45 m/s, the data has
   no backward walking), torso roll/pitch/yaw commands within the G1 waist range.
   Finger targets are not a pass/stop criterion; they are clamped (see the finger rule below).
   Every commanded channel is checked or refused: a step without an action, an action channel
   the gate does not check, or any malformed step is a STOP that latches, and the bridge is still
   answered (it holds and ends the segment); the bridge refuses a policy action of another width.
4. The first STOP latches: the bridge switches to the hold command (zero navigation velocity,
   previous targets; the balance controller keeps running) and the result says `stopped`.
5. Arena's own terminations end a segment WITHOUT resetting the scene (auto-reset is suspended
   during segments); the bridge holds `rest_steps` (0.5 s), then reports the final snapshot.
6. The verifier judges that snapshot: object centre over the destination footprint and inside its
   height band, object speed <= 0.05 m/s, every hand point >= 0.03 m from the object's box, robot
   upright. Arena's success signal is written to `segments/NNN/segment.json` for comparison only;
   it never reaches the verdict or the agent.

Records per segment: `segments/NNN/gate.jsonl` (every step's rows and finger clamps), `segment.json` (instruction,
verdict, measurements, final state, gate summary, benchmark signal), and the bridge's frames
(`--bridge_record_dir`: observer + head camera JPEGs with an index).

## Finger rule

The Dex3 finger joints (`finger_joint_pattern`) are position-limited by the articulation itself:
a finger target beyond a joint stop cannot move the finger past that stop; its only effect is extra
drive force pressing the finger into its own stop. The rule is therefore defined from the
articulation, with no tunable tolerance:

- finger targets are clamped onto the joint range; the `pass` reply carries the clamps
  (`clamp: [[joint index, target], ...]`, protocol 2), the bridge applies them (and refuses a clamp
  that is not a target inside that joint's own range), `gate.jsonl` records each one (joint,
  commanded, applied) next to the unedited proposed action, and the tool result counts them
  (`gate.finger_targets_clamped`);
- fingers are judged on what they physically do: their measured positions and velocities go
  through the same joint gates as every other joint, so a finger that leaves its range by more than
  0.05 rad or exceeds its velocity limit still STOPs the segment;
- arm and waist targets are never edited: beyond range + 0.02 rad is a STOP.

## Start order

```bash
# 1. GR00T policy server (its own venv), e.g. port 5555
# 2. one bridge per port in the profile (8765, 8766, ...), Newton physics:
cd <IsaacLab-Arena> && .venv/bin/python <cascade>/scripts/arena_locomanip_bridge.py --bridge_port 8765 \
   --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
   --policy_config_yaml_path <policy yaml> --remote_host 127.0.0.1 --remote_port 5555 \
   --enable_cameras --device cuda:0 --presets newton --bridge_offline_assets [--bridge_preload <module>] \
   galileo_g1_locomanip_pick_and_place --object brown_box --embodiment g1_wbc_joint
# 3. the brain (scripts/serve_qwen_vllm.sh), then an agent episode or the MCP server:
CASCADE_ROBOT=g1_arena_locomanip python -m cascade.apps.mcp_server     # tools carry.*; no reset_stop (SIGUSR1 / restart)
python scripts/humanoid_agent_episode.py --robot g1_arena_locomanip --llm local_qwen --prompt "..." --out runs/...
```

`--bridge_preload` imports a module before the environment is built (Newton port patches for the
Galileo scene live outside this repository). `--bridge_offline_assets` serves already-cached
remote assets without the per-file freshness probe (Isaac Lab's offline path).

## Harness² on humanoid tasks

`benchmark/harness2/g1_humanoid_tasks.json` names a `robot` per task; `scripts/harness2_improve.py
--robot <profile>` runs that robot's tasks (`--config-dir` for a pinned fastening profile). Hidden
checks (`check.humanoid`): object `in` a destination or `unmoved_tol_m`, robot `upright`, fastening
`min_turns`/`max_turns` from the nut's axial advance; `expect_claim` for a request the agent must
decline. Harness text may use the local names of namespaced tools; it may never name `reset_stop`.

## Before a real Arena run

- **Replay a recorded run.** `python scripts/replay_gate_jsonl.py <run>/domains/carry/segments/001/gate.jsonl`
  checks every recorded proposed action with today's gate and the profile's limits (`--recorded-limits`:
  the run's own, from `session.json`) and prints, per commanded channel, the first refused step
  (`--json` for the full report; exit 1 when any channel refused, 2 when no action was replayed or on
  any error). `gate.jsonl` records actions, not measured states, so each action is paired with a
  neutral upright state: only commands decide. CPU only; run it on an earlier recording before the
  first real run on new gate code.
- **Leg target tolerance.** Leg targets (owned by the whole-body controller, but in the applied
  action) get the command tolerance, range + 0.02 rad, while a MEASURED joint may sit up to 0.05 rad
  past its range. A leg target 0.02-0.05 rad past its range is a STOP although the leg could stand
  there. Kept at 0.02 (fail closed); the replay shows whether the policy's actions hit it.
- **The episode script's brain.** `scripts/humanoid_agent_episode.py` without `--llm` uses
  `local_nemotron_bolt` (the launch default brain): it talks to the live local brain server on
  127.0.0.1:8080, as `local_qwen` does. Tests never reach it (fakes and a socket guard).

## Not validated / limits

- Simulation only; no hardware, no physical admission.
- The gate's limits are profile values (fall criteria, articulation limits, training envelope),
  not measured hardware limits. The workspace box is the scene's room interior.
- Two robot profiles (Arena loco-manipulation, Newton fastening) are two simulated worlds; CASCADE
  refuses composing two physical actuating domains without a whole-body contract, so the agent
  runs one episode per robot.
