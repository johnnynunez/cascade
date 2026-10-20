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

## One `carry_object(object, destination)` call

1. Arguments are enum-checked against the profile's `scene`; the instruction is
   `instruction_template` filled with the spoken labels (for brown box / blue bin it is the
   checkpoint's training instruction verbatim).
2. A latched stop refuses the call (`refused`; `reset_stop` is an operator action and a locked
   Harness² tool). The gate vets the measured start state (pre-motion admission).
3. Per control step (50 Hz) the bridge sends the measured state and the policy's proposed action;
   the gate answers `pass` or `stop`. Gates: state finite, pelvis tilt <= 30 deg, pelvis height
   above ankle level >= 0.45 m, measured joints within articulation limits + 0.05 rad, joint
   velocity <= limit, pelvis inside the workspace box, policy joint targets within limits + 0.02
   rad (legs belong to the whole-body controller), navigation and base-height commands within the
   checkpoint's training envelope (`dataset_statistics.json`) + margin.
4. The first STOP latches: the bridge switches to the hold command (zero navigation velocity,
   previous targets; the balance controller keeps running) and the result says `stopped`.
5. Arena's own terminations end a segment WITHOUT resetting the scene (auto-reset is suspended
   during segments); the bridge holds `rest_steps` (0.5 s), then reports the final snapshot.
6. The verifier judges that snapshot: object centre over the destination footprint and inside its
   height band, object speed <= 0.05 m/s, every hand point >= 0.03 m from the object's box, robot
   upright. Arena's success signal is written to `segments/NNN/segment.json` for comparison only;
   it never reaches the verdict or the agent.

Records per segment: `segments/NNN/gate.jsonl` (every step's rows), `segment.json` (instruction,
verdict, measurements, final state, gate summary, benchmark signal), and the bridge's frames
(`--bridge_record_dir`: observer + head camera JPEGs with an index).

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
CASCADE_ROBOT=g1_arena_locomanip python -m cascade.apps.mcp_server     # tools carry.*
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

## Not validated / limits

- Simulation only; no hardware, no physical admission.
- The gate's limits are profile values (fall criteria, articulation limits, training envelope),
  not measured hardware limits. The workspace box is the scene's room interior.
- Two robot profiles (Arena loco-manipulation, Newton fastening) are two simulated worlds; CASCADE
  refuses composing two physical actuating domains without a whole-body contract, so the agent
  runs one episode per robot.
