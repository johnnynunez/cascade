# VLA executor (`grasp.executor: vla`, opt-in)

Status: landed 2026-10-09 (B49, ROADMAP mid term "VLA policy backend").
Measured on the mock stack against the protocol stub only
(`tests/test_vla_executor.py`). No real policy weights have run through it.

`grasp_object(label)` normally runs the deterministic pipeline: localize,
plan candidates with the configured backend (GraspGen-X / HUG / OBB /
camera_frame), select, then a vetted approach, descent, close and lift. That
pipeline is the **analytic** executor and stays the default. Its behaviour,
capability matrix, banner and withheld-tool list are unchanged.

With `grasp.executor: vla`, the same tool is served by a language-conditioned
policy (an openpi or LingBot-VLA-v2 policy server) that returns **action
chunks**. The agent layer does not change:

- the tool is still `grasp_object(label)` with the same schema;
- every composite that grasps by label goes through it (`pick_and_place`,
  `sort_by_color`, handover and so on);
- the result has the same shape and is judged by the same verifier.

Pixel-addressed grasps (`grasp_at_pixel`) stay analytic, because a
language-conditioned policy needs a name to be conditioned on.

## Enabling it

```yaml
# configs/demo.yaml (or an arm profile's override)
grasp:
  executor: vla           # or CASCADE_GRASP_EXECUTOR=vla for one run
  vla:
    host: 127.0.0.1
    port: 8000            # or CASCADE_VLA_PORT
```

```bash
uv sync --extra vla      # websockets + msgpack; nothing else, no torch
```

Set in the launching shell, both variables reach the MCP server: `./run.sh` /
`scripts/launch.sh` copy them into the server entry they register, and so does
every entry `scripts/setup_agents.py` writes (B63, `cascade.apps.mcp_env`). An
MCP host starts the server with that entry's environment only; before B63 the
launcher registered neither variable, so `CASCADE_GRASP_EXECUTOR=vla ./run.sh`
ran the analytic executor.

The policy runs in its own environment and process, normally on the GPU
host. cascade only ships the client (`grasping/vla_client.py`).

At startup, `build_runtime` validates `grasp.vla` and probes the server
(connect plus metadata frame, within `connect_timeout_s`):

- **Malformed config:** an unknown key, a bad port, a prompt with fields
  other than `{label}`, a bad action space and so on fail the build.
- **No answer:** a WARNING goes to stderr and the capability matrix marks
  `vla_policy` as unavailable. The MCP catalog then withholds
  `grasp_object`, `pick_and_place` and `sort_by_color`, so the agent cannot
  plan a grasp that would be refused.

With the analytic executor, the `vla_policy` cell does not exist at all.

## Wire protocol

This is the openpi `WebsocketClientPolicy` protocol, which LingBot-VLA-v2's
`deploy/websocket_policy_server.py` uses unchanged.

| step | frame |
| --- | --- |
| connect | `ws://host:port`; the server sends ONE binary frame: the metadata dict |
| request | binary msgpack observation dict |
| reply | binary msgpack `{<action_key>: (H, D) float array, "server_timing": {...}}` |
| server error | a TEXT frame with the traceback, then close 1011 |
| health | `GET /healthz` → 200 |
| LingBot reset | an observation `{reset: true, robo_name: ...}` → `{action: None}` |

Arrays use openpi's msgpack-numpy extension:

- ndarrays: `{b"__ndarray__": True, b"data", b"dtype", b"shape"}`;
- numpy scalars: `{b"__npgeneric__": True, b"data", b"dtype"}`.

The client reimplements this. The PyPI `msgpack-numpy` package uses a
different layout, and the client never unpickles. It binds msgpack's own
`Packer`/`unpackb`, not the module attributes: the GraspGen-X, HUG and
occupancy clients call `msgpack_numpy.patch()` in the same process, and the
patched functions would send (and decode) msgpack-numpy's layout, which can
carry pickles. That layout is never produced or decoded on this wire.

The client refuses object, void and complex dtypes. A non-dict metadata
frame, a silent peer, an undecodable reply, a non-dict reply or a dropped
connection each produce a named error (`VLAUnavailable` before any motion,
`VLAServerError` during an episode). A late reply raises `VLATimeout` and
closes the session, so a chunk that arrives later is never read.

## One episode

1. **Route preconditions.** Some rigs have gates the waypoint stream does not
   implement, so the route refuses them before anything moves:
   - the observed-finger gate (`CASCADE_OBSERVED_FINGER_GATE=1`);
   - a native motion planner (cuMotion profiles);
   - a payload-tracking occupancy map.
2. **Connect.** If the server does not answer, or the `vla` extra is
   missing, the call fails with `VLAUnavailable` and nothing moves. There is
   **no analytic fallback**: the operator chose the policy, and a silent
   fallback would report the wrong executor's success.
3. **Open and home.** The jaws open, then the arm re-homes over the vetted
   route (`move_planned`), exactly like the analytic pipeline.
4. **Per chunk:**
   1. Build the observation: a fresh camera frame (RGB uint8, resized to
      `image_size`), the MEASURED joints plus the jaw opening fraction
      (float32), the prompt (`prompt.format(label=...)`) and `extra_obs`.
   2. If the jaw feedback is missing, the episode ends. The observation would
      otherwise invent the jaw state.
   3. The reply must arrive within `chunk_timeout_s`, and within what is left
      of `episode_timeout_s` and the task budget. A late chunk ends the
      episode and nothing more moves.
   4. The stop latch and the halt generation are checked when the reply
      arrives. A stop that landed during inference never reaches the arm.
   5. The chunk becomes joint targets:
      - `joint`: absolute local joint angles in this arm's convention;
      - `tcp`: base-frame `x y z roll pitch yaw`, solved by IK seeded from
        the previous target. A row with no IK solution, or whose solution
        leaves that IK branch, is refused.

      Malformed chunks are refused: a missing key, non-finite values, a
      wrong rank, too few columns, more than `max_chunk_len` rows, or a
      gripper value outside [0, 1] by more than 0.05 (the tolerance for
      numerical overshoot).
   6. **Whole-chunk admission:** `harness.vet_pose` runs on EVERY target
      before any of the chunk moves. There is no grasp exemption.
   7. **Streaming:** each target goes through `SafeArm.move_joints`, which
      calls `harness.approve()` on every sample (velocity cap, joint limits,
      workspace, table, keep-outs, neighbours, occupancy). SafeArm also
      checks the stop latch and halts before every waypoint and gripper
      command. The episode deadline is re-checked per waypoint.
   8. A gripper command is sent only when it moves by more than 0.02. Close
      uses the material profile's effort.
5. **End.** The episode ends `chunks_after_close` chunks after the first
   close command (the lift), or after `max_chunks`.

The executor never refuses motion on geometric grounds. **The harness is the
only motion authority.**

## What counts as a grasp

Only the configured `action_key` is read from a reply. A server's `success`,
`done`, `is_success` or reward is never evidence and never ends an episode.
The outcome is decided in this order:

- the policy never closed the jaws → failure, nothing is grasped;
- the jaws closed, then were commanded open again → failure, nothing is held;
- otherwise the analytic pipeline's own checks run: `_air_grasp` (the
  jaw-travel check, shared code), then `_promote_held`, then `execute()`'s
  unchanged three-state postcondition verifier (`agent/effects.py`, kind
  `holding`).

The held-object aiming offset is measured from the TCP's pose at the close
command. VLA outcomes do not train the analytic grasp-outcome memory,
because there is no candidate geometry to credit.

## Configuration (`grasp.vla`)

`configs/demo.yaml` documents every key. The main ones:

| key | default | meaning |
| --- | --- | --- |
| `host`, `port` | `127.0.0.1`, `8000` | policy server (openpi `serve_policy` default 8000; LingBot `--port`, default 8006) |
| `connect_timeout_s` | 1.0 | connect + metadata, also the startup probe |
| `chunk_timeout_s` | 2.0 | one inference round trip |
| `episode_timeout_s` | 60.0 | whole episode (also capped by the task budget) |
| `max_chunks`, `chunks_after_close` | 20, 1 | episode length |
| `max_chunk_len` | 64 | rows accepted per chunk |
| `action_key` | `actions` | reply key holding the chunk |
| `action_space` | `joint` | `joint` or `tcp` |
| `gripper`, `gripper_column` | `open_frac`, null | gripper convention (1 = open, or `close_frac`) and its column (default: right after the arm columns) |
| `waypoint_duration_s` | 0.2 | per action; SafeArm still stretches moves to respect the velocity cap |
| `prompt` | `pick up the {label}` | `{label}` is the only field |
| `image_key`, `state_key`, `prompt_key` | openpi DROID-style names | observation keys |
| `image_size` | 224 | square resize; null = native resolution |
| `reset` | null | sent once per episode (LingBot: `{reset: true, robo_name: <robot config>}`) |
| `extra_obs` | `{}` | static keys merged into every observation |

## The protocol stub

`scripts/serve_vla_stub.py` speaks the same wire and replays SCRIPTED
chunks. It is a test double, not a policy: it never looks at the image.

```bash
python scripts/serve_vla_stub.py --port 8000                       # holds: echoes the state
python scripts/serve_vla_stub.py --port 8000 --chunks chunks.json  # request N -> chunk N
```

`tests/test_vla_executor.py` uses it in-process (port 0, loopback). The tests
cover:

- the analytic golden;
- the default-off surface;
- the missing-extra refusal;
- codec round trips and refusals;
- protocol violations;
- whole-chunk admission and per-sample approval;
- the stop latch at inference time and mid-chunk;
- chunk, episode and task deadlines;
- the rule that the policy's own success flags are never evidence;
- re-open and never-close failures;
- the LingBot reset and `extra_obs`;
- the capability cell and withheld tools;
- environment overrides;
- the startup probe.

## Running a real policy (owed, GPU host)

None of this has run yet, and the stub is not evidence for any of it.

```bash
# openpi (its own checkout and env; serves on 8000)
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=<config> --policy.dir=<checkpoint dir>
# LingBot-VLA-v2 (its own checkout and env)
python -m deploy.lingbot_vla_v2_policy --model_path <ckpt> --port 8006 --use_length 50
# cascade (both variables are registered with the MCP server, B63)
CASCADE_GRASP_EXECUTOR=vla CASCADE_VLA_PORT=<port> ./run.sh <profile>
```

Match `action_key`, `action_space`, the gripper convention and the
observation keys to the checkpoint's training data. LingBot's reply key and
observation layout come from its `configs/robot_configs/<robo_name>.yaml`.

## Limits (not claimed)

- **No real policy has run through it.** No policy post-trained on this arm
  exists. The public LingBot and openpi checkpoints are other embodiments,
  so their actions here are expected to be refused by the harness or to miss.
- **Quasi-static execution.** Chunks run as harness-gated per-action moves
  that settle at every action, not at the policy's native control rate.
  There is no real-time chunk streaming or temporal ensembling.
- **One camera.** Only the manipulation camera goes into the observation; no
  wrist or side views. A checkpoint trained on several views needs them.
- **No grasp exemption.** A policy grasp of a very low object can be refused
  where the analytic descent (with its vetted exemption) is allowed.
- **Rigs the route refuses:** the observed-finger gate, native motion
  planners and payload-tracking maps.
- **No background re-probe.** The capability cell reflects the startup probe
  and the last connect. A server started after a failed probe needs a
  runtime restart.
- **Spark caveat (from the ROADMAP):** LingBot's flash-attn must build for
  aarch64+Blackwell, or its two hardcoded attention implementations must be
  patched to SDPA.
