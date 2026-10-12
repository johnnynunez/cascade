# OpenShell: where the agent runs on Linux

OpenClaw in an [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell) 0.1.x
sandbox, **without NemoClaw**. On Linux this is the only place the agent runs:
`scripts/launch.sh` puts it there for every `--sim` (Isaac, MuJoCo, or the real
robot with `--sim none`; see [Every Linux launch](#every-linux-launch-scriptslaunchsh)),
and the real rig's operator route uses it too. One script,
[`scripts/openshell_rig.py`](../scripts/openshell_rig.py), owns the route:
a pinned and checksummed OpenShell, its own gateway, the agent image in
[`deploy/openshell/`](../deploy/openshell/), two provider profiles, the
sandbox and the chat forward.

Why a second route next to [NEMOCLAW.md](NEMOCLAW.md): NemoClaw v0.0.124 pins
OpenShell 0.0.116 and its managed `inference.local`. OpenShell 0.1.x replaced
that with **provider profiles**, which let the policy describe exactly what
cascade needs: two endpoints, two credentials and a per-tool MCP rule. The
NemoClaw route keeps working; both routes can be installed side by side
(separate gateways, separate CLI state).

## What runs where

```
 host (all on 127.0.0.1)                          OpenShell sandbox `cascade-agent`
 ───────────────────────────────────────────      ─────────────────────────────────
 brain  :8081  vLLM / llama-server, bearer key     OpenClaw 2026.9.3 (node 24, uid node)
        ▲                                            gateway + chat on its own :18789
        │  CASCADE_BRAIN_API_KEY (real)                 │ $CASCADE_BRAIN_API_KEY = placeholder
        │                                               │ $CASCADE_MCP_TOKEN     = placeholder
 robot MCP :18741  cascade.apps.mcp_server              ▼
        ▲   plain HTTP + bearer (loopback only)    OpenShell supervisor (proxy, L7)
        │  CASCADE_MCP_TOKEN (real)                  - only host.openshell.internal:8081
        └──────────── host.openshell.internal ◄──    - only host.openshell.internal:18741/mcp
                                                     - only for /usr/local/bin/node
 chat  127.0.0.1:18799 ◄── openshell forward ──      - swaps placeholder -> real secret
       (systemd: cascade-openshell-chat)             - 403 on tools/call reset_stop
                                                     - (observe-only) 403 on every motion tool
 gateway 127.0.0.1:17670 (mTLS, Docker driver)
       (systemd: cascade-openshell-gateway)
```

On native Linux the supervisor maps `host.openshell.internal` to the host's
loopback. So neither the brain nor the robot server listens on a Docker
bridge or on the LAN. The NemoClaw route, by contrast, needs
`172.17.0.1:18740` plus TLS from a private CA.

## What OpenShell enforces (verified live on the Spark, 2026-10-11)

These were checked from inside the sandbox:

- **Credentials never enter the sandbox.** `CASCADE_MCP_TOKEN` in the agent's
  environment is an `openshell:resolve:…` placeholder. The supervisor
  substitutes the real token only on requests that match the profile's
  endpoint.
- **Binary binding.** The credentials and endpoints are granted to
  `/usr/local/bin/node` only. `curl` from the same sandbox to the same MCP URL
  is refused at connect (`binary '/usr/bin/curl' not allowed in policy`).
- **Per-tool MCP rules.** `tools/call reset_stop` returns `403 policy_denied`
  from OpenShell, and the cascade server never sees the request. Clearing an
  e-stop stays staff-only.
- **Deny-by-default egress.** The agent cannot reach the internet, the live
  view (:8090), the occupancy bridge (:5557) or the NemoClaw MCP listener.
- **Observe-only, live.** `motion lock` adds every motion tool to the robot
  profile's deny rules, and `motion unlock` removes them. The full set is
  `_MOTION_SKILLS` plus `run_program`, read from the source so a new skill is
  locked the day it lands. A profile update reaches the running sandbox in
  under 10 s, so the agent and its chat token are not recreated. The two stop
  tools (`emergency_stop`, `halt_motion`) can never be denied.
- **Locked by default; every unlock expires.** `up` starts observe-only
  unless `--allow-motion` is given, and a saved unlock never carries over to
  a new session. `motion unlock` lasts 20 min by default (`--for 90s|15m|8h`,
  max 12 h, a presentation day; zero or unbounded is refused). Before it unlocks, it schedules its
  own re-lock, a transient systemd user timer (`cascade-openshell-relock`)
  that runs `motion lock`. If that cannot be scheduled, the arm stays locked.
  `motion lock` cancels a pending re-lock, and `motion status` shows the time
  left. Tools denied with `up --deny-tool` stay denied across every lock and
  unlock.

## Bring-up

```bash
# 0. the brain on 127.0.0.1:8081 with a bearer key read from a 0600 file (default
#    ~/.cascade/nemoclaw/llama.key, or --brain-key-file / $CASCADE_BRAIN_KEY_FILE)
#    and the bare served id qwen3.8-27b. vLLM, Qwen3.8 NVFP4 + DFlash2 (measured 2-3x
#    llama.cpp on the Spark); the key goes to VLLM_API_KEY, never argv:
systemd-run --user --unit=cascade-brain-vllm --collect -p MemoryMax=infinity \
    -E PORT=8081 -E API_KEY_FILE="$HOME/.cascade/nemoclaw/llama.key" \
    "$PWD/scripts/serve_qwen_vllm.sh"
#    or llama.cpp: PORT=8081 ALIAS=qwen3.8-27b API_KEY_FILE=... scripts/serve_qwen_llamacpp.sh

# 1. OpenShell 0.1.3: downloaded, sha256-checked against the pinned digests, extracted
#    under ~/.cascade/openshell (its own XDG config/state; NemoClaw's is untouched)
scripts/openshell_rig.py install
scripts/openshell_rig.py gateway start     # systemd user unit, 127.0.0.1:17670

# 2. cascade's MCP server on loopback (the robot runtime; one owner of the CAN bus
#    and cameras: stop the NemoClaw route's `nemoclaw_mcp.py serve` first)
systemd-run --user --unit=cascade-openshell-mcp --collect -p MemoryMax=infinity \
    -E CASCADE_ARM=rebot_rs -E CASCADE_CAMERAS=d455f_scene,d455f_wrist -E CASCADE_BOOTH=1 \
    "$PWD/.venv/bin/python" "$PWD/scripts/openshell_rig.py" serve-mcp

# 3. image (rebuilt when deploy/openshell changes), profiles, providers,
#    sandbox, chat forward
scripts/openshell_rig.py up                  # observe-only by default: look, don't touch
scripts/openshell_rig.py status
scripts/openshell_rig.py open                # the demo's screen: OpenClaw's own web UI in the
                                             # Chromium profile with the cascade camera extension
                                             # (click it, then "Connect cameras"); token never printed
scripts/openshell_rig.py chat-url            # or print the tokenized chat URL: a secret

# 4. when the operator is ready to let the arm move (e-stop in hand, space clear)
scripts/openshell_rig.py motion unlock --for 15m   # re-locks itself; default 20m, max 12h
scripts/openshell_rig.py motion status              # "allowed for 14 min more ..."
scripts/openshell_rig.py motion lock                # done: observe-only again now
```

To ask the agent from a shell (for example a smoke test):

```bash
~/.cascade/openshell/0.1.3/bin/openshell sandbox exec --name cascade-agent -- \
    openclaw agent --session-id smoke -m "Look through the cameras and describe the table. Do not move."
```

Run this with the route's environment. `openshell_rig.py` sets
`XDG_CONFIG_HOME`/`XDG_STATE_HOME` under `~/.cascade/openshell` for every call
it makes. A bare `openshell` from a shell would talk to NemoClaw's
registry.

`down` stops the chat forward and deletes the sandbox; `down --gateway` also
stops the gateway.

## Instances

Everything above describes the **default instance**, which is the live rig's
route. `--instance NAME` (or `CASCADE_OPENSHELL_INSTANCE=NAME`) runs another
route next to it, fully isolated, for example a simulator agent beside the
real robot. The flag wins over the variable, and `--instance ''` selects the
default instance. A name is 1 to 16 characters from `a-z`, `0-9` and `-`,
starting with a letter. `default` is reserved.

| | default instance | named instance `<name>` | example: `sim` |
|---|---|---|---|
| home | `~/.cascade/openshell` | `~/.cascade/openshell-<name>` | `~/.cascade/openshell-sim` |
| gateway | `cascade`, 127.0.0.1:17670 | `cascade-<name>`, 17670 + slot | `cascade-sim`, 17710 |
| systemd units | `cascade-openshell-{gateway,chat,relock,mcp}` | `cascade-openshell-<name>-{gateway,chat,relock,mcp}` | `cascade-openshell-sim-gateway`, … |
| providers = profile ids | `cascade-brain`, `cascade-robot` | `cascade-<name>-brain`, `cascade-<name>-robot` | `cascade-sim-brain`, `cascade-sim-robot` |
| sandbox, Docker label | `cascade-agent`, `cascade-openshell` | `cascade-<name>-agent`, `cascade-openshell-<name>` | `cascade-sim-agent`, `cascade-openshell-sim` |
| MCP port | 18741 | 17770 + slot | 17810 |
| chat port | 18799 | 17870 + slot | 17910 |

`slot = 1 + sha256(name) mod 99`, so `sim` gets slot 40. The three port bands
(17671-17769, 17771-17869, 17871-17969) stay clear of the default route,
NemoClaw (18740, 18789) and OpenClaw (18790, 18791). `scripts/openshell_rig.py
--instance sim instance` prints a name's whole scheme as JSON and touches
nothing. Two names can land on the same slot (about 1 in 99). Check with
`instance` and pick another name: `--mcp-port` and `--chat-port` move those
two ports, but the gateway port has no flag.

- **Shared on purpose:** the brain (one host LLM: `--brain-port`, the key
  file), the model and the sandbox image.
- **Explicit flags still win.** `--sandbox`, `--mcp-port`, `--chat-port`,
  `--brain-port`, `--model` and the token/key files override the instance's
  defaults. They are remembered in that instance's `deployed.json` only.
- **Each instance has its own re-lock.** `motion unlock` schedules that
  instance's timer, which re-runs `--instance <name> motion lock` with that
  instance's home. `motion lock` of one instance never cancels another
  instance's timer. The deny list (every motion tool when locked,
  `reset_stop` always, the stop tools never) is the same for every instance.
- **One home, one instance.** A named instance writes an `instance` file into
  its home. It refuses a home that holds another instance's state, for
  example when `CASCADE_OPENSHELL_HOME` still points at the live route's
  home. The default route, whose home carries no such file, refuses a marked
  home.
- **The MCP server behind a named instance is a second robot runtime.** Only
  one process may own the CAN bus and the cameras, so give it the mock arm
  and cameras (or other hardware). Give it its own state too:
  `CASCADE_STREAM=0` (the live view's 8090 and the Spark camera service's
  8091 belong to the live rig, and the camera extension always shows 8090),
  `CASCADE_OCCUPANCY=0`, and private `CASCADE_BELIEFS_PATH`,
  `CASCADE_GRASP_MEMORY_PATH`, `CASCADE_ENVELOPE_PATH`, `CASCADE_EPISODIC_PATH`
  and `CASCADE_RUN_DIR`, so simulated outcomes never reach the real robot's
  learned stores.

```bash
I=sim; RIG=(scripts/openshell_rig.py --instance "$I"); S=~/.cascade/openshell-$I
"${RIG[@]}" install && "${RIG[@]}" gateway start      # 127.0.0.1:17710, unit cascade-openshell-sim-gateway
systemd-run --user --unit=cascade-openshell-$I-mcp --collect -p MemoryMax=infinity \
    -E CASCADE_OPENSHELL_INSTANCE=$I -E CASCADE_ARM=mock -E CASCADE_CAMERA=mock \
    -E CASCADE_STREAM=0 -E CASCADE_OCCUPANCY=0 -E CASCADE_RUN_DIR=$S/runs \
    -E CASCADE_BELIEFS_PATH=$S/beliefs.json -E CASCADE_GRASP_MEMORY_PATH=$S/grasp_memory.json \
    -E CASCADE_ENVELOPE_PATH=$S/envelope.json -E CASCADE_EPISODIC_PATH=$S/episodic.json \
    "$PWD/.venv/bin/python" "$PWD/scripts/openshell_rig.py" serve-mcp   # MCP on 127.0.0.1:17810
"${RIG[@]}" up && "${RIG[@]}" status                     # sandbox cascade-sim-agent, chat on 17910
```

## Every Linux launch (`scripts/launch.sh`)

On Linux, `scripts/launch.sh` always runs the agent inside OpenShell, for
`--sim isaac`, `--sim mujoco` and the real robot (`--sim none`). The platform
decides (`uname -s`); there is no flag or variable that puts the agent back on
a Linux host. macOS has no OpenShell Docker driver, so OpenClaw runs on the
host there, as before. The simulator and its sidecars start exactly as before;
steps 4-6 are [`scripts/launch_openshell.sh`](../scripts/launch_openshell.sh):

| | what happens |
|---|---|
| instance | `sim` for a simulator, the default instance (the real rig's) for `--sim none`; `CASCADE_OPENSHELL_INSTANCE` overrides |
| brain | a model served on this host's loopback (`--brain qwen\|cosmos\|cosmos-sglang`). `keep` (a cloud model the host's OpenClaw is logged into) is refused: the sandbox reaches only the loopback brain. The served id must be bare (no `/`), and its context window reaches the sandbox (`--context-window`) |
| brain key | `CASCADE_BRAIN_KEY_FILE`, else `CASCADE_MODEL_API_KEY` written to a 0600 file in the launch state, else a minted one (a server without `--api-key` ignores it) |
| robot server | built once (`runtime_check`), then `openshell_rig.py mcp-start`: `serve-mcp` as the instance's unit, bound to this launch's owner, with the registration's environment from a 0600 file. A simulator's server learns into the launch state's `memory/` (grasp memory, envelope, beliefs, episodic), never into `~/.cascade` (the real robot's stores), and it starts empty on every launch: the world is new, so a recalled position is wrong by construction. Explicit `CASCADE_*_PATH` values win |
| sandbox | `openshell_rig.py up`. A simulator starts with motion allowed for 12 h; the real robot starts observe-only (`motion unlock`) |
| proof | the sandbox's own OpenClaw (`openshell_rig.py exec -- openclaw ...`, through `CASCADE_OPENCLAW_COMMAND`): its gateway must be healthy, `mcp probe` must list the robot tools, the brain must answer, and in sim modes one pick is checked against physics and the same world is reset (`demo_proof.py --persistent-mcp`: the world is the instance's one long-lived server) |
| chat | `http://127.0.0.1:<chat port>/` (`open` or `chat-url`); `launch.sh --down` runs `teardown` of the recorded instance (a named one with its gateway) |

The Spark kitchen (`install.sh --profile spark`) still uses the host route on
Linux for now: its chat setup (the kitchen prompt plugin, the visitor tool
list and instructions, the demo UI on :8092) lives in a host OpenClaw profile
and moves into the sandbox next.

## The demo's screen

OpenClaw's own web UI (the sandbox's gateway, forwarded to 127.0.0.1) in the
Chromium profile that loads the cascade camera extension: `open`. The
extension finds the rig's cameras from that loopback page through the
live-view server (`127.0.0.1:8090/state`, `extensions/chrome/discovery.mjs`).
That server stays up as the cameras' backend; its own dashboard page is not
the demo UI. `serve-mcp` runs headless (`CASCADE_VIEW=0` unless set): a
systemd user unit has `DISPLAY=:0`, and the server's default OpenCV window
would otherwise open next to the demo.

## Two Landlock findings baked into the image

OpenShell's default filesystem policy lets the agent read `/usr` and write
`/tmp`. Two things in the image depend on that:

- The config template and `render-config.mjs` live under
  `/usr/local/share/cascade`. A copy under `/opt` was `Permission denied` at
  start, and the sandbox went to phase `Error`.
- `TMPDIR=/tmp` and `SQLITE_TMPDIR=/tmp`. Without them, OpenClaw's session
  store failed with `fts5: error creating shadow table … unable to open
  database file`.

## Gotchas found against 0.1.3

- **MCP revisions.** OpenShell reads a request without an
  `MCP-Protocol-Version` header as `2025-03-26` (that includes `initialize`).
  Cascade answers `2025-06-18`, and the OpenClaw SDK offers `2025-11-25`. The
  robot profile admits all three; missing any one of them 403s the session.
- **`provider profile lint` treats an existing id as an error**, and
  `provider profile update` needs the stored `resource_version` (read from
  `export`). `up` therefore lints only new profiles.
- **OpenClaw 2026.9.3 needs node >= 24.16.** npm refuses it on node 22, so the
  base image is `node:24-bookworm-slim`, pinned by digest.
- **`openshell status` exits 0 with no gateway registered** ("No gateway
  configured"). Health is this instance's gateway named in the output and
  `Status: Connected`; `gateway start` registers it on a fresh home (`gateway
  add` also selects it) or selects it when another gateway is active.
- **OpenClaw's own tools in the robot's prompt.** With the template's former
  `alsoAllow: ["bundle-mcp"]`, the agent also saw OpenClaw's built-in tools and
  skills: "What is the session status?" called OpenClaw's `session_status`
  instead of cascade's `world_state`, and a two-call turn sent 73k prompt
  tokens. `tools.allow: ["cascade__*"]` and no skills: `world_state`, 22.5k
  (MuJoCo, Qwen3.8-27B, 2026-10-12). OpenClaw refuses `allow` and `alsoAllow`
  in one scope.
- **The chat token** is minted inside the sandbox at first start, in a 0600
  file. It is never on a host argv. A restart keeps it, so an open tab stays
  valid.

## Files

| Path | What |
|---|---|
| `scripts/openshell_rig.py` | the route: install, gateway, serve-mcp, mcp-start, up, motion, open, chat-url, exec, status, down, teardown, instance |
| `scripts/launch_openshell.sh` | `scripts/launch.sh` steps 4-6 on Linux: brain, robot server, sandbox, proof inside it, chat |
| `deploy/openshell/Dockerfile` | agent image (node 24 by digest, OpenClaw pinned, Landlock-safe layout) |
| `deploy/openshell/openclaw.json` | OpenClaw config template: placeholders, never secrets |
| `deploy/openshell/render-config.mjs` | first-start render: ports, model, chat origin, chat token |
| `deploy/openshell/cascade-openclaw` | entrypoint: render, then `openclaw gateway` on the sandbox loopback |
| `tests/test_openshell_rig.py` | the contract above, including the default instance's golden values and the isolation of a named one (runs `render-config.mjs` with the host's node) |
| `tests/test_launch_openshell_route.py` | the real `launch.sh` on Linux end to end, with the rig, Docker, the model and OpenClaw doubled |
| `~/.cascade/openshell/` | OpenShell binaries, gateway config + mTLS, profiles, `mcp.token`, `deployed.json` (a named instance: `~/.cascade/openshell-<name>/`, plus its `instance` file) |
