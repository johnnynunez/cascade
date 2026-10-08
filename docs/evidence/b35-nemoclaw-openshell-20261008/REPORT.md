# B35 — the robot driven from an NVIDIA OpenShell sandbox (NemoClaw), on Isaac Sim

Date: 8 October 2026. Rig: x86 test box (not a Spark number). Isaac Sim
6.2.0-alpha.19 internal build, PhysX, the bare reBot scene (pink and green
cube, open box as the drop zone). GPU 0 runs the Isaac bridge, GraspGen-X and the
sandbox's llama-server. Model: Qwen3.8-27B UD-Q4_K_XL, the same GGUF as the
default brain. NemoClaw v0.0.124, OpenShell 0.0.116, OpenClaw 2026.7.1 in
the sandbox.

## Question

Can an agent confined by OpenShell (Landlock, seccomp, netns, deny-by-default
egress) drive the CASCADE robot through the new Streamable HTTP transport? Does
that route change what the robot does compared with calling the same server
directly from the host?

## Setup

- `cascade.apps.mcp_server --http 172.17.0.1:18740`: TLS from a private CA,
  bearer token in OpenShell's provider store. Registered with
  `scripts/nemoclaw_mcp.py register`. `nemoclaw cascade-robot mcp status
  cascade --json` reported `trustedPrivateTarget.state = match` with provider,
  policy and adapter ready.
- Inference stays local: `inference.local` → authenticated llama-server
  `127.0.0.1:8081` (bridged to `172.18.0.1:8081`).
- The robot side runs from `live/rig`, whose `src/` is byte-identical to the
  commit under test (`diff -r`). Its configs differ only in the private ports
  (bridge 18521, GraspGen-X 18522, occupancy 18523). Main's
  `load_demo_config` ignores the port env vars (B34 open), so the copy was
  needed.
- Every run starts from a **fresh Isaac stage and a fresh MCP server**
  (`fresh_rig.sh`).
- The two routes alternate H S H S H S (`ab_runs.sh`):
  - **Host-direct** (H): `probe_host.py` calls `get_observation`, then
    `pick_and_place(pink cube → drop zone)`, then `reset_scene` on the
    HTTPS endpoint.
  - **Sandbox agent** (S): `nemoclaw cascade-robot agent --session-id …`
    runs two turns in one session, "pick up the pink cube and place it in
    the box" and then "reset the scene and read world_state"
    (`proof_turns.sh`).

## What had to change in the sandbox first (measured failures)

The first sandboxed turn reached the server and called `get_observation`
twice and `camera_snapshot` once. It then ended in "Context overflow" for
three reasons:

1. NemoClaw's OpenClaw config enables tool search. MCP tools are reached
   through a generic `tool_call` whose result is JSON text capped at 16,000
   characters, so the camera image arrived as base64 text.
2. The `llama-cpp` provider registers the model as `input: ["text"]`, so no
   image would reach it in any case.
3. There is no per-call timeout for the MCP entry. OpenClaw's 60 s default
   is below a 74 s pick, and cancelling a motion latches the e-stop.

`scripts/nemoclaw_mcp.py configure-agent` sets these three like the stdio
profile, plus the connection timeout. After that, the next turn called
`get_observation` and described the scene from the image (pink cube, green
cube, the open box). It also flagged the tracker's duplicate "orange" box and
the gray "fighter jet" phantom on the arm. Both are perception issues already
on the backlog: B32b is open, and the B32a fix is in PR #257, not on this base.

## Results (`ab_summary.json`, raw rows in `manifest.json`)

| run | route | pick | pick s | physics postcondition | reset |
| --- | --- | --- | ---: | --- | --- |
| run_isaac (first, cold GraspGen-X) | sandbox | fail: already-holding retry loop | 144.1 | refuted, cube held at 0.44 m | fail: did not settle at home |
| hostA1 | host | **ok**, 1 attempt | 73.2 | confirmed, 2.1 cm | not called |
| sandboxB1 | sandbox | **ok**, 1 attempt | 73.9 | confirmed, 2.3 cm | **ok** 11.2 s |
| hostA2 | host | fail: place did not settle | 146.7 | unverified (0.6 cm) | fail: did not settle at home |
| sandboxB2 | sandbox | **ok**, 1 attempt | 69.4 | confirmed, 2.4 cm | **ok** 11.0 s |
| hostA3 | host | fail: already-holding retry loop | 153.5 | refuted, cube held at 0.43 m | fail: did not settle at home |
| sandboxB3 | sandbox | fail: dropped in carry (grip verified) | 63.5 | refuted, moved 7 cm | **ok** 11.8 s |

Fresh stages: host-direct 1/3 picks, sandbox agent 2/3. In two sandbox
sessions (B1, B2), a physics-confirmed pick and a physics-verified reset both
ran in the same chat session with zero failed tool calls. This is the same
shape as the launch proof, but it was not run through `scripts/demo_proof.py`.

## Reading

- **The route works.** A sandboxed agent called a 74 s motion tool through
  the OpenShell MCP proxy, and the server confirmed the result by physics. The
  300 s per-call budget held: no cancel, no e-stop latch. Reset and
  `world_state` followed in the same session.
- **The route does not explain the failures.** With n = 3 per arm there is no
  claim of a difference between the arms, and none is needed. Each failure
  signature appears on the host-direct route too or is independent of the
  caller:
  - already-holding: host A3 and the cold sandbox run;
  - place did not settle: host A2;
  - drop in carry: sandbox B3.

  All three are pick reliability on this x86 Isaac rig, filed as a new
  backlog item. The two host resets that failed were holding the cube
  ("did not settle at home").
- The already-holding signature: the first attempt's grasp lifts the cube,
  but the attempt is counted as failed. Every retry then refuses with
  "already holding 'pink cube'; place it first" until the persistence budget
  runs out, and the arm keeps the cube at home height. In the cold run, the
  first GraspGen-X inference took 15.5 s, longer than the 8 s client timeout,
  so the first attempt used the analytic planner.
- `nemoclaw <name> agent` exited 1 with `replayInvalid=true` on every turn,
  including the successful ones whose JSON says `status: ok`. Judge turns by
  the server log and the physics postcondition.

## Not covered

- The kitchen scene and its two proof cases. Repeated sessions beyond n = 3
  per route.
- `demo_proof.py` on the sandbox route.
- The DGX Spark: NemoClaw supports DGX OS; not attempted.
- A host firewall rule for the MCP port: there is no root on this box. The
  listener is protected by TLS and the token, and binds only docker0.
- NemoClaw gives the sandbox direct GPU access by default on this host. The
  agent does not need it; not yet turned off.

## Files

`serve_isaac.sh`, `fresh_rig.sh`, `ab_runs.sh`, `proof_turns.sh`,
`probe_host.py` and `tcp_forward.py` are the harness. `ab_summary.json` holds
the per-run rows. `manifest.json` holds the sha256 of the raw traces, server
logs and turn JSON, which stay in
`cascade-lab/HERMES_BACKLOG_20261007/nemoclaw-runtime/live/`. The bearer token
and the llama key were checked absent from every evidence and raw file.
