# VLA executor (`grasp.executor: vla`, opt-in)

**Next action:** start the protocol stub, then point the executor at it (about 2 min, no GPU; the
stub replays scripted chunks and never looks at the image). `--extra vla` adds the client wire
(websockets + msgpack; nothing else, no torch):

```bash
uv run --extra vla python scripts/serve_vla_stub.py --port 8000                       # holds: echoes the state
uv run --extra vla python scripts/serve_vla_stub.py --port 8000 --chunks chunks.json  # request N -> chunk N
```

Status: landed 2026-10-09 (B49, ROADMAP mid term "VLA policy backend"), measured on the mock stack
against the protocol stub only (`tests/test_vla_executor.py`). **No real policy weights have run
through it.**

`grasp_object(label)` normally runs the deterministic **analytic** pipeline (localize, plan with
GraspGen-X / HUG / OBB / camera_frame, select, vetted approach, descent, close, lift); it stays the
default with unchanged capability matrix, banner and withheld tools. With `grasp.executor: vla` the
same tool is served by a language-conditioned policy server (openpi or LingBot-VLA-v2) returning
**action chunks**, with the same schema, the same result shape and the same verifier; every
label-grasping composite (`pick_and_place`, `sort_by_color`, handover) goes through it.
`grasp_at_pixel` stays analytic (a policy needs a name).

## Enable it

```yaml
# configs/demo.yaml (or an arm profile's override)
grasp:
  executor: vla           # or CASCADE_GRASP_EXECUTOR=vla for one run
  vla:
    host: 127.0.0.1
    port: 8000            # or CASCADE_VLA_PORT
```

1. Both variables, set in the launching shell, reach the MCP server: `./run.sh` /
   `scripts/launch.sh`, every entry `scripts/setup_agents.py` writes (B63, `cascade.apps.mcp_env`)
   and the `scripts/hermes_demo.sh` / `scripts/openclaw_demo.sh` registrations (B68) copy them.
   Checked live once on 10 Oct 2026 against the Isaac bridge with the stub
   ([evidence](evidence/w6-live-20261010/README.md)).
2. The policy runs in its own environment and process, normally on the GPU host; cascade ships
   only the client (`grasping/vla_client.py`).
3. At startup `build_runtime` validates `grasp.vla` and probes the server (connect plus metadata
   frame within `connect_timeout_s`). A malformed config (unknown key, bad port, a prompt field
   other than `{label}`, bad action space) fails the build.
4. No answer: a WARNING on stderr, the `vla_policy` capability cell unavailable, and the catalog
   withholds `grasp_object`, `pick_and_place` and `sort_by_color`. With the analytic executor the
   cell does not exist.

## Wire protocol

The openpi `WebsocketClientPolicy` protocol (LingBot-VLA-v2's `deploy/websocket_policy_server.py`
uses it unchanged):

| step | frame |
| --- | --- |
| connect | `ws://host:port`; the server sends ONE binary frame: the metadata dict |
| request | binary msgpack observation dict |
| reply | binary msgpack `{<action_key>: (H, D) float array, "server_timing": {...}}` |
| server error | a TEXT frame with the traceback, then close 1011 |
| health | `GET /healthz` → 200 |
| LingBot reset | an observation `{reset: true, robo_name: ...}` → `{action: None}` |

1. Arrays use openpi's msgpack-numpy extension (`{b"__ndarray__": True, b"data", b"dtype",
   b"shape"}`, scalars `{b"__npgeneric__": True, b"data", b"dtype"}`), reimplemented by the
   client.
2. The PyPI `msgpack-numpy` layout (which can carry pickles) is never produced or decoded: the
   client binds msgpack's own `Packer`/`unpackb`, because other clients call
   `msgpack_numpy.patch()` in the same process. The client never unpickles.
3. Object, void and complex dtypes are refused. A bad metadata frame, silent peer, undecodable or
   non-dict reply, or dropped connection raises a named error (`VLAUnavailable` before motion,
   `VLAServerError` during an episode); a late reply raises `VLATimeout` and closes the session.

## One episode

1. **Route preconditions** refused before anything moves: the observed-finger gate
   (`CASCADE_OBSERVED_FINGER_GATE=1`), a native motion planner (cuMotion profiles), a
   payload-tracking occupancy map.
2. **Connect:** no answer or a missing `vla` extra fails with `VLAUnavailable` and nothing moves.
   There is **no analytic fallback** (it would report the wrong executor's success).
3. **Open and home:** the jaws open, then the arm re-homes over the vetted route (`move_planned`).
4. **Per chunk:** build the observation (fresh RGB resized to `image_size`, MEASURED joints plus
   jaw opening fraction, `prompt.format(label=...)`, `extra_obs`; missing jaw feedback ends the
   episode); the reply must arrive within `chunk_timeout_s` and the remaining episode and task
   budget; the stop latch and halt generation are checked on arrival; the chunk becomes joint
   targets (`joint`: absolute local angles; `tcp`: base-frame `x y z roll pitch yaw` via IK seeded
   from the previous target, refusing rows without a solution or that leave the IK branch);
   malformed chunks (missing key, non-finite, wrong rank, too few columns, more than
   `max_chunk_len` rows, gripper outside [0, 1] by more than 0.05) are refused.
5. **Admission and streaming:** `harness.vet_pose` runs on EVERY target before any of the chunk
   moves (no grasp exemption); each target goes through `SafeArm.move_joints` with `approve()` on
   every sample and the stop latch checked before every waypoint and gripper command; a gripper
   command is sent only when it changes by more than 0.02 (close uses the material profile's
   effort). The episode ends `chunks_after_close` chunks after the first close, or after
   `max_chunks`.

The executor never refuses motion on geometric grounds. **The harness is the only motion
authority.**

## What counts as a grasp

Only the configured `action_key` is read; a server's `success`, `done`, `is_success` or reward is
never evidence. In order: the policy never closed the jaws → failure; the jaws closed and were
commanded open again → failure; otherwise `_air_grasp` (jaw travel), `_promote_held` and
`execute()`'s unchanged three-state verifier (`agent/effects.py`, kind `holding`). The held-object
aiming offset is measured at the close command. VLA outcomes do not train the analytic
grasp-outcome memory.

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

`tests/test_vla_executor.py` runs the stub in-process (port 0, loopback) and covers the analytic
golden, the default-off surface, codec and protocol refusals, whole-chunk admission, the stop latch
at inference time and mid-chunk, all deadlines, success flags never being evidence, re-open and
never-close failures, the LingBot reset and `extra_obs`, the capability cell, env overrides and the
startup probe.

## Running a real policy (owed, GPU host)

None of this has run yet, and the stub is not evidence for any of it.

```bash
# openpi (its own checkout and env; serves on 8000)
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=<config> --policy.dir=<checkpoint dir>
# LingBot-VLA-v2 (from its own checkout, with its own env's interpreter)
<lingbot-env>/bin/python -m deploy.lingbot_vla_v2_policy --model_path <ckpt> --port 8006 --use_length 50
# cascade (both variables are registered with the MCP server, B63)
CASCADE_GRASP_EXECUTOR=vla CASCADE_VLA_PORT=<port> ./run.sh <profile>
```

Match `action_key`, `action_space`, the gripper convention and the observation keys to the
checkpoint's training data (LingBot: `configs/robot_configs/<robo_name>.yaml`).

## Limits (not claimed)

1. **No real policy has run through it.** No policy is post-trained on this arm; public LingBot and
   openpi checkpoints are other embodiments, so the harness is expected to refuse or they miss.
2. **Quasi-static, one camera:** chunks run as harness-gated per-action moves that settle at every
   action (no real-time streaming or temporal ensembling); only the manipulation camera is observed.
3. **No grasp exemption:** a very low object can be refused where the analytic descent is allowed.
   The route refuses the observed-finger gate, native planners and payload-tracking maps.
4. **No background re-probe:** a server started after a failed probe needs a runtime restart.
5. **Spark caveat:** LingBot's flash-attn must build for aarch64+Blackwell, or its two hardcoded
   attention implementations must be patched to SDPA.
