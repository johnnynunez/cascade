# OpenShell: where the agent runs on Linux

**Next action:** start the brain and the route with [Bring-up](#bring-up) (about 10 min on a
host with the model already downloaded). The arm stays observe-only until you run
`motion unlock`.

OpenClaw in an [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell) 0.1.x sandbox, **without
NemoClaw**. On Linux this is the only place the agent runs: `scripts/launch.sh` puts it there
for every `--sim` (Isaac, MuJoCo, real robot with `--sim none`; see
[Every Linux launch](#every-linux-launch-scriptslaunchsh)), and the real rig's operator route
uses it too. [`scripts/openshell_rig.py`](../scripts/openshell_rig.py) owns the route: a pinned,
checksummed OpenShell, its own gateway, the agent image in
[`deploy/openshell/`](../deploy/openshell/), two provider profiles, the sandbox and the chat
forward. The older [NemoClaw route](#nemoclaw-route-opt-in) (pinned to OpenShell 0.0.116 and
`inference.local`) still works and can be installed side by side.

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
        │  CASCADE_MCP_TOKEN (real)                  - only host.openshell.internal:8081/v1/* (GET, POST)
        └──────────── host.openshell.internal ◄──    - only host.openshell.internal:18741/mcp
                                                     - only for /usr/local/bin/node
 chat  127.0.0.1:18799 ◄── openshell forward ──      - swaps placeholder -> real secret
       (systemd: cascade-openshell-chat)             - 403 on tools/call reset_stop
                                                     - (observe-only) 403 on every motion tool
 gateway 127.0.0.1:17670 (mTLS, Docker driver)
       (systemd: cascade-openshell-gateway)
```

On native Linux `host.openshell.internal` maps to the host's loopback, so neither the brain nor
the robot server listens on a Docker bridge or the LAN (NemoClaw needs `172.17.0.1:18740` plus
TLS from a private CA).

## What OpenShell enforces (verified live on the Spark, 2026-10-11)

1. **Credentials never enter the sandbox.** `CASCADE_MCP_TOKEN` there is an
   `openshell:resolve:…` placeholder; the supervisor substitutes the real token only on requests
   matching the profile's endpoint. Credentials and endpoints are bound to `/usr/local/bin/node`:
   `curl` to the same URL is refused (`binary '/usr/bin/curl' not allowed in policy`).
2. **Per-tool MCP rules.** `tools/call reset_stop` returns `403 policy_denied` before cascade
   sees it: clearing an e-stop stays staff-only.
3. **Deny-by-default egress.** No internet, no live view (:8090), no occupancy bridge (:5557),
   no NemoClaw MCP listener.
4. **Observe-only, live.** `motion lock` adds every motion tool (`_MOTION_SKILLS` plus
   `run_program`, read from source, so a new skill is locked the day it lands) to the robot
   profile's deny rules; `motion unlock` removes them. The update reaches the running sandbox in
   under 10 s. `emergency_stop` and `halt_motion` can never be denied.
5. **Locked by default; every unlock expires.** `up` starts observe-only unless
   `--allow-motion`; a saved unlock never carries over. `motion unlock` lasts 20 min by default
   (`--for 90s|15m|8h`, max 12 h; zero or unbounded is refused) and first schedules its own
   re-lock (transient user timer `cascade-openshell-relock`); if that cannot be scheduled the arm
   stays locked. `motion lock` cancels a pending re-lock, `motion status` shows the time left,
   and tools denied with `up --deny-tool` stay denied across every lock and unlock.

## Brain warm start: what it opens

The Bolt brain prefills the agent prefix OpenClaw last sent after every restart
(`scripts/serve_nemotron_bolt_llamacpp.sh`, `WARM_START=1` by default). To record that prefix,
llama-server runs with two features that widen its API for **every holder of the brain key**:

1. `LLAMA_SERVER_SLOTS_DEBUG=1`: `GET /slots` returns each slot's last prompt and output text
   (agent chats, perception), and a request reusing part of a slot logs the 2 tokens where it
   diverges. llama.cpp v0.7.0 reads it once at start: it cannot be on only while the helper records.
2. `--slot-save-path <state>/bolt-slots-PORT`: `POST /slots/N?action=save|restore|erase` work
   (one save writes about 0.5 GB).

The limits:

1. Only behind `API_KEY_FILE`. Without a key the script warns and serves as `WARM_START=0`; the
   server binds `127.0.0.1` either way. The same fallback covers a missing helper, a llama-server
   without `--slot-save-path` and a slot directory that cannot be private.
2. The slot directory is the user's own, mode 0700, never a symlink. A save needs 4 GiB free
   there before it starts (`--min-free-gb`; its file comes on top), else it waits; the file is
   deleted right after the read.
3. The sandbox reaches only `GET` and `POST /v1/*` of every brain (`brain_profile`, OpenShell
   REST rules): `/slots`, `/completion`, `/props` and `/tokenize` (llama-server), `/metrics`
   (vLLM) get 403. The profile names no model: Bolt under a custom `ALIAS`, `--model
   qwen3.8-27b` and Cosmos get the same one (OpenClaw only calls `<brain>/v1`).
4. Proven live only by the [live check](#live-check-of-the-brain-profile) after an `up` that
   changes the profile (~1 min per brain); if it fails, [roll back](#rollback-of-the-brain-profile).
5. Still trusted with the key: host processes that read `llama.key` (the robot server's
   perception, the helper). `WARM_START=0` closes all of the above.

No inherited variable configures a brain server; the one exception is the explicit
`CASCADE_LLAMA_CONFIG_INI=1` (below). The llama.cpp scripts drop every inherited
`LLAMA_ARG_*` (llama-server reads one per flag), `LLAMA_API_KEY` (an extra accepted key), `AIP_*`
(`AIP_MODE=PREDICTION` moves the port and adds a route to every endpoint) and
`LLAMA_SERVER_SLOTS_DEBUG` (Bolt sets it itself, for the warm start only); a bash without `compgen`,
which lists them, refuses to start. The Brev runtime's `deploy/runtime/brain.py` drops the same
names. The native vLLM brains (Qwen, Cosmos) drop `VLLM_SERVER_DEV_MODE`,
`VLLM_ALLOW_RUNTIME_LORA_UPDATING`, `VLLM_ENABLE_COHERE_API`, `VLLM_USE_RUST_FRONTEND` and
`VLLM_RUST_FRONTEND_PATH` (each adds routes or swaps the HTTP server), and
`VLLM_ENABLE_RESPONSES_API_STORE` (memory that grows with every stored `/v1/responses` call); Bolt's
vLLM container gets only the variables its script names. The SGLang Cosmos brain: [below](#cosmos-on-sglang).

llama.cpp's config files act like those variables. On every run, `--help` included, llama-server
v0.7.0 applies `/etc/llama.cpp/config.ini`, then `${XDG_CONFIG_HOME:-~/.config}/llama.cpp/config.ini`,
before the environment and argv. Either can set any option: `tools = all` serves a shell at `/tools`,
`api-prefix = /v1` puts every route inside the sandbox's `/v1/**`, `log-file = <path>` truncates that
file. So:

1. both llama.cpp scripts and `brain.py` run every llama-server, the `--help` capability probe too,
   with `XDG_CONFIG_HOME=/dev/null`. No `llama.cpp/config.ini` can exist under it, so the caller's
   `XDG_CONFIG_HOME`, `~/.config` and the deployment's `$PAAI_DATA_DIR/config` are never read, and no
   state directory is needed;
2. the scripts refuse to start, in `--check` and `--setup-only` too, while `/etc/llama.cpp/config.ini`
   exists, before any llama-server run (llama-server has no flag that skips it). `brain.py` runs in an
   image whose `/etc` is the image's.

`CASCADE_LLAMA_CONFIG_INI=1` runs with both files anyway and says so on stderr.

### Cosmos on SGLang

`--brain cosmos-sglang` reaches `scripts/serve_cosmos_sglang.sh` (SGLang 0.5.22, pinned; port 8083):

```bash
PORT=8083 API_KEY_FILE="$HOME/.cascade/nemoclaw/llama.key" scripts/serve_cosmos_sglang.sh
# then your usual launch; its brain checks and the agent's brain provider send the same key:
#   K="$HOME/.cascade/nemoclaw/llama.key"
#   CASCADE_MODEL_API_KEY_FILE="$K" CASCADE_BRAIN_KEY_FILE="$K" scripts/launch.sh <flags> --brain cosmos-sglang
```

1. **Keyed.** It refuses without `API_KEY_FILE` (0600, one key). SGLang takes a key only as
   `--api-key KEY`, so the script reads the file inside the server process: `ps` shows the path,
   never the key. SGLang still logs its arguments, key included, at start, and `/get_server_info`
   returns them to key holders.
2. **Loopback, no remote code.** It binds `127.0.0.1`. There is no `--trust-remote-code`:
   Cosmos3-Edge is native in SGLang and transformers, and `nvidia/Cosmos3-Edge` ships no code.
3. **Scrubbed.** It drops `SGLANG_GRPC_PORT` (a gRPC listener the key does not cover),
   `SGLANG_RUST_SERVER` (another HTTP server), `SGLANG_OLLAMA_*` and `AIP_*` (routes moved anywhere,
   `/v1/**` too), `DUMPER_*` (a keyless control server on `0.0.0.0`),
   `SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES` and `EXA_API_KEY` (a web search tool behind `/v1/responses`).
4. **Inside `/v1/**`** (what the sandbox reaches): chat/completions, completions, models, responses
   (the store is off), messages, embeddings, classify, rerank, score, decisions, systemone, tokenize,
   detokenize, loads, audio/transcriptions, all keyed. The WebSocket `/v1/realtime` skips the key
   (SGLang checks HTTP only); Cosmos has no streaming ASR, so it answers `not_supported` and closes.
   None returns another caller's text or changes the server.
5. **Outside `/v1`** (the profile denies them; key holders on loopback only): `/generate`,
   `/get_server_info`, `/flush_cache`, `/update_weights_*`, `/load_lora_adapter`, profiling, Ollama
   `/api/*` and more. `/health*`, `/ready*` and `/metrics*` answer without a key (metrics are off).

## Live check of the brain profile

Run it after every `up` that changes the brain profile, once per brain (about 1 min each). It
sends read-only requests from inside the sandbox with raw `http.request` (`fetch()` rewrites
`/v1/../slots` to `/slots` before sending) and prints PASS or FAIL per row; exit 0 = all pass.

1. **Bolt.** `scripts/openshell_rig.py up` exits 0 with no `openshell provider profile update
   failed` line. Then:
   `scripts/openshell_rig.py exec -- node -e "$(cat scripts/brain_v1_live_check.js)" nemotron-3.5-bolt`
2. **Qwen.** Serve it on :8081 with the same key file (vLLM: `PORT=8081 API_KEY_FILE=...
   scripts/serve_qwen_vllm.sh`; llama.cpp: `PORT=8081 ALIAS=qwen3.8-27b API_KEY_FILE=...
   scripts/serve_qwen_llamacpp.sh`), run `scripts/openshell_rig.py up --model qwen3.8-27b` (exits
   0), then the same `exec` with `qwen3.8-27b`.
3. **Cosmos on SGLang.** Serve it as in [Cosmos on SGLang](#cosmos-on-sglang), launch with
   `--brain cosmos-sglang`, then the same `exec` with `cosmos3-edge`.
4. Send one chat turn in the web UI: it answers, with no `policy_denied` in the sandbox log.

| Row | Expected |
|---|---|
| `GET /v1/models`, `POST /v1/chat/completions` | 200 |
| `POST /v1/chat/completions` with `stream: true` | 200 `text/event-stream` |
| `GET /slots`, `GET /props`, `GET /metrics`, `POST /tokenize` | 403 `policy_denied` |
| `GET /v1/../slots`, `GET /v1/%2e%2e/slots` | 403 `policy_denied` (400 also passes) |

Reading a FAIL:

1. 401 on a `/v1` row: the key placeholder was not swapped (brain provider or profile not in force).
2. 200, 401, 404 or 400 on a non-`/v1` row: the request reached the brain, so the profile is not in
   force. On vLLM the `/metrics` row decides: vLLM answers it without a key.
3. Any FAIL in steps 1-4: [roll back](#rollback-of-the-brain-profile).

## Rollback of the brain profile

From the live checkout, about 2 min. It replaces only the brain profile; the sandbox keeps running.

1. If Bolt is the brain, restart it without the warm start (read-write plus the warm start would
   show `/slots` prompt text to the sandbox; step 2 refuses while it runs). Keep it on
   `WARM_START=0` until `/v1`-only is back (step 3): step 2 checks only at the moment it runs, and
   the [bring-up](#bring-up) command starts the warm start by default.
   ```bash
   systemctl --user stop cascade-brain
   systemd-run --user --unit=cascade-brain --collect -p MemoryMax=infinity \
       -E PORT=8081 -E API_KEY_FILE="$HOME/.cascade/nemoclaw/llama.key" -E WARM_START=0 \
       "$PWD/scripts/serve_nemotron_bolt_llamacpp.sh"
   ```
2. `scripts/openshell_rig.py brain-profile --read-write`: main's old read-write brain profile
   (sha256 `7e81c98a…`), nothing else. Like `up`, it refuses while cascade's gateway is down.
3. Back to `/v1`-only once fixed: `scripts/openshell_rig.py brain-profile`, or the next `up`. Then
   restart Bolt with the warm start: the [bring-up](#bring-up) unit (step 0), without `WARM_START=0`.

## Bring-up

```bash
# 0. the brain on 127.0.0.1:8081 with a bearer key read from a 0600 file (default
#    ~/.cascade/nemoclaw/llama.key, or --brain-key-file / $CASCADE_BRAIN_KEY_FILE)
#    and the bare served id nemotron-3.5-bolt: Nemotron 3.5 Bolt 74B-A9B 24GB (GGUF)
#    on llama.cpp v0.7.0 with the LiLiCorr drafter, two slots; llama-server reads the key
#    file itself, never argv. Once per host: scripts/serve_nemotron_bolt_llamacpp.sh
#    --setup-only (pinned GGUF + projector + drafter, 24.8 GB, and the pinned llama.cpp
#    build). Restart this unit after pulling a new revision. After a restart it prefills the
#    agent prefix OpenClaw last sent (scripts/brain_warm_start.py; only with API_KEY_FILE;
#    WARM_START=0 turns it off; what it opens: "Brain warm start" below).
systemd-run --user --unit=cascade-brain --collect -p MemoryMax=infinity \
    -E PORT=8081 -E API_KEY_FILE="$HOME/.cascade/nemoclaw/llama.key" \
    "$PWD/scripts/serve_nemotron_bolt_llamacpp.sh"
#    Opt-in, NOT beside Isaac Sim: Bolt on vLLM (scripts/serve_nemotron_bolt_vllm.sh, the
#    model card's container) faulted the GPU (Xid 31) beside Isaac; see its header.
#    Qwen3.8 NVFP4 + DFlash2 instead: the same unit with scripts/serve_qwen_vllm.sh, and
#    `scripts/openshell_rig.py up --model qwen3.8-27b` (or launch.sh --brain qwen);
#    or llama.cpp: PORT=8081 ALIAS=qwen3.8-27b API_KEY_FILE=... scripts/serve_qwen_llamacpp.sh

# 1. OpenShell 0.1.3: downloaded, sha256-checked against the pinned digests, extracted
#    under ~/.cascade/openshell (its own XDG config/state; NemoClaw's is untouched)
scripts/openshell_rig.py install
scripts/openshell_rig.py gateway start     # systemd user unit, 127.0.0.1:17670; registers the
                                           # gateway on a fresh home (`gateway add` also selects
                                           # it) or selects it when another gateway is active

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

Ask the agent from a shell (smoke test). `openshell_rig.py` sets `XDG_CONFIG_HOME` /
`XDG_STATE_HOME` under `~/.cascade/openshell` for its own calls; a bare `openshell` from your
shell talks to NemoClaw's registry, so use the route's binary with the same environment:

```bash
~/.cascade/openshell/0.1.3/bin/openshell sandbox exec --name cascade-agent -- \
    openclaw agent --session-id smoke -m "Look through the cameras and describe the table. Do not move."
```

`down` stops the chat forward and deletes the sandbox; `down --gateway` also stops the gateway.

**The demo's screen** is OpenClaw's own web UI (the sandbox gateway forwarded to 127.0.0.1) in
the Chromium profile with the cascade camera extension (`open`). The extension finds the cameras
through the live-view server (`127.0.0.1:8090/state`, `extensions/chrome/discovery.mjs`), whose
own dashboard page is not the demo UI. `serve-mcp` runs headless (`CASCADE_VIEW=0` unless set),
because a systemd user unit has `DISPLAY=:0` and OpenCV would open a window.

## Instances

Everything above is the **default instance**: the live rig's route. `--instance NAME` (or
`CASCADE_OPENSHELL_INSTANCE=NAME`; the flag wins, `--instance ''` selects the default) runs a
fully isolated route beside it, e.g. a simulator agent next to the real robot. Names: 1–5
characters from `a-z`, `0-9` and single `-` between them, starting with a letter; `default` is
reserved. Five because OpenShell takes sandbox names of at most 19 characters and a named
instance's sandbox is `cascade-<name>-agent`: `openshell_rig.py` and `launch.sh` refuse any other
name before they start anything (an explicit `up --sandbox` is held to the same 19 characters).

| | default instance | named instance `<name>` | example: `sim` |
|---|---|---|---|
| home | `~/.cascade/openshell` | `~/.cascade/openshell-<name>` | `~/.cascade/openshell-sim` |
| gateway | `cascade`, 127.0.0.1:17670 | `cascade-<name>`, 17670 + slot | `cascade-sim`, 17710 |
| systemd units | `cascade-openshell-{gateway,chat,relock,mcp}` | `cascade-openshell-<name>-{gateway,chat,relock,mcp}` | `cascade-openshell-sim-gateway`, … |
| providers = profile ids | `cascade-brain`, `cascade-robot` | `cascade-<name>-brain`, `cascade-<name>-robot` | `cascade-sim-brain`, `cascade-sim-robot` |
| sandbox, Docker label | `cascade-agent`, `cascade-openshell` | `cascade-<name>-agent`, `cascade-openshell-<name>` | `cascade-sim-agent`, `cascade-openshell-sim` |
| MCP port | 18741 | 17770 + slot | 17810 |
| chat port | 18799 | 17870 + slot | 17910 |

`slot = 1 + sha256(name) mod 99` (`sim` = 40). The bands 17671-17769, 17771-17869 and
17871-17969 avoid the default route, NemoClaw (18740, 18789) and OpenClaw (18790, 18791).
`scripts/openshell_rig.py --instance sim instance` prints a name's scheme as JSON and touches
nothing; two names can share a slot (~1 in 99): pick another name (`--mcp-port` / `--chat-port`
move two ports, the gateway port has no flag).

1. **Shared on purpose:** the brain (`--brain-port`, the key file), the model and the image.
2. **Explicit flags win** (`--sandbox`, `--mcp-port`, `--chat-port`, `--brain-port`, `--model`,
   token/key files) and are remembered in that instance's `deployed.json` only.
3. **Each instance has its own re-lock timer**: `motion unlock` schedules one that re-runs
   `--instance <name> motion lock` with that instance's home; `motion lock` of one never cancels
   another's. The deny list (motion tools when locked, `reset_stop` always, stop tools never) is
   the same.
4. **One home, one instance.** A named instance writes an `instance` file into its home and
   refuses a home holding another instance's state (e.g. `CASCADE_OPENSHELL_HOME` still pointing
   at the live route); the default route refuses a marked home.
5. **A named instance's MCP server is a second robot runtime.** Only one process may own the CAN
   bus and cameras: give it the mock arm and cameras (or other hardware), `CASCADE_STREAM=0`
   (the live view's 8090 and the Spark camera service's 8091 belong to the live rig, and the
   camera extension always shows 8090), `CASCADE_OCCUPANCY=0` and private
   `CASCADE_BELIEFS_PATH`, `CASCADE_GRASP_MEMORY_PATH`, `CASCADE_ENVELOPE_PATH`,
   `CASCADE_EPISODIC_PATH`, `CASCADE_RUN_DIR`, so simulated outcomes never reach the real
   robot's learned stores:

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

On Linux `scripts/launch.sh` always runs the agent inside OpenShell (`--sim isaac`,
`--sim mujoco`, `--sim none`); `uname -s` decides and no flag or variable puts it back on a
Linux host (macOS has no OpenShell Docker driver and keeps the host route). Simulators and
sidecars start as before; steps 4-6 are [`scripts/launch_openshell.sh`](../scripts/launch_openshell.sh):

| | what happens |
|---|---|
| instance | `sim` for a simulator, the default instance (the real rig's) for `--sim none`; `CASCADE_OPENSHELL_INSTANCE` overrides |
| brain | a model on this host's loopback (`--brain nemotron-bolt\|qwen\|cosmos\|cosmos-sglang`; `auto` finds a local Cosmos, else the brain on :8080, Nemotron 3.5 Bolt by default); `keep` (a cloud model) is refused. The served id must be bare (no `/`); its context window reaches the sandbox (`--context-window`) |
| brain key | `CASCADE_BRAIN_KEY_FILE`, else `CASCADE_MODEL_API_KEY` written to a 0600 file in the launch state, else a minted one (a server without `--api-key` ignores it) |
| robot server | built once (`runtime_check`), then `openshell_rig.py mcp-start`: `serve-mcp` as the instance's unit, bound to this launch's owner, env from a 0600 file. A simulator learns into the launch state's `memory/` (grasp memory, envelope, beliefs, episodic), never `~/.cascade`, and starts empty every launch; explicit `CASCADE_*_PATH` values win |
| sandbox | `openshell_rig.py up`: a simulator starts with motion allowed for 12 h; the real robot starts observe-only (`motion unlock`) |
| proof | the sandbox's OpenClaw (`openshell_rig.py exec -- openclaw ...`, via `CASCADE_OPENCLAW_COMMAND`): healthy gateway, `mcp probe` lists the robot tools, the brain answers, and in sim one pick is checked against physics and the world reset (`demo_proof.py --persistent-mcp`) |
| chat | `http://127.0.0.1:<chat port>/` (`open` or `chat-url`); `launch.sh --down` runs `teardown` of the recorded instance (a named one with its gateway) |

The Spark kitchen (`install.sh --profile spark`) still uses the host route on Linux: its chat
setup (kitchen prompt plugin, visitor tool list and instructions, demo UI on :8092) lives in a
host OpenClaw profile.

## Gotchas found against 0.1.3

1. **Landlock layout.** The default policy allows reading `/usr` and writing `/tmp`, so the
   config template and `render-config.mjs` live under `/usr/local/share/cascade` (under `/opt`
   the sandbox went to phase `Error`), and the image sets `TMPDIR=/tmp` and `SQLITE_TMPDIR=/tmp`
   (else `fts5: error creating shadow table … unable to open database file`).
2. **MCP revisions.** A request without `MCP-Protocol-Version` (including `initialize`) reads as
   `2025-03-26`; cascade answers `2025-06-18`, the OpenClaw SDK offers `2025-11-25`. The robot
   profile admits all of them; a missing one 403s the session.
3. **Profiles and status.** `provider profile lint` treats an existing id as an error and
   `provider profile update` needs the stored `resource_version` (from `export`), so `up` lints
   only new profiles. `openshell status` exits 0 with "No gateway configured": health is this
   instance's gateway named in the output plus `Status: Connected`.
4. **Images and tokens.** OpenClaw 2026.9.3 needs node >= 24.16 (base `node:24-bookworm-slim`,
   pinned by digest). The chat token is minted inside the sandbox in a 0600 file, never on a
   host argv; a restart keeps it, so an open tab stays valid.
5. **Only the robot's tools.** With `alsoAllow: ["bundle-mcp"]` the agent saw OpenClaw's
   built-ins ("What is the session status?" called `session_status`, 73k prompt tokens).
   `tools.allow: ["cascade__*"]` and no skills: `world_state`, 22.5k tokens. OpenClaw refuses
   `allow` and `alsoAllow` in one scope.

## NemoClaw route (opt-in)

NemoClaw v0.0.124 reaches the robot through `cascade.apps.mcp_server --http` (TLS from a private
CA, bearer token in OpenShell's provider store). `scripts/nemoclaw_mcp.py` does the operator side,
run from the repo root with the rig venv's interpreter (`serve` starts the robot server with that
same interpreter; `uv run` would re-sync the live `.venv`):

1. `.venv/bin/python scripts/nemoclaw_mcp.py certs`, then
   `export NEMOCLAW_CORPORATE_CA_BUNDLE="$(.venv/bin/python scripts/nemoclaw_mcp.py ca-path)"`.
2. `nemoclaw onboard` (an existing sandbox: `nemoclaw <name> rebuild`), so the sandbox trusts that CA.
3. `.venv/bin/python scripts/nemoclaw_mcp.py serve` with the usual `CASCADE_*` env.
4. `.venv/bin/python scripts/nemoclaw_mcp.py register --sandbox <name>`, then
   `configure-agent --sandbox <name>`.

Keep the model id bare (`nemotron-3.5-bolt`, `qwen3.8-27b`): OpenClaw splits provider/model on `/`.
The chat's model label comes from the served id (`deploy/openshell/render-config.mjs`):
"Nemotron 3.5 Bolt 74B-A9B (cascade brain)", "Qwen3.8-27B (cascade brain)", or the id itself
for any other brain. For Nemotron the sandbox's OpenClaw also sends
`chat_template_kwargs: {enable_thinking: false}` with every request: the model card's ultra_v3
reasoning parser ignores the server's default for non-streaming requests.

## Files

| Path | What |
|---|---|
| `scripts/openshell_rig.py` | the route: install, gateway, serve-mcp, mcp-start, up, motion, open, chat-url, exec, status, down, teardown, instance, brain-profile |
| `scripts/brain_v1_live_check.js` | the [live check](#live-check-of-the-brain-profile) of the brain profile, run inside the sandbox |
| `scripts/launch_openshell.sh` | `scripts/launch.sh` steps 4-6 on Linux: brain, robot server, sandbox, proof inside it, chat |
| `deploy/openshell/` | `Dockerfile` (node 24 by digest, OpenClaw pinned, Landlock-safe), `openclaw.json` (placeholders, never secrets), `render-config.mjs` (first-start render: ports, model, chat origin, token, agent profile), `cascade-openclaw` (entrypoint), `franka-plugin/` (the FR3 operator profile, `up --agent-profile franka`: [FRANKA_OPENCLAW.md](FRANKA_OPENCLAW.md)) |
| `tests/test_openshell_rig.py`, `tests/test_launch_openshell_route.py` | the contract above incl. the default instance's golden values; the real `launch.sh` on Linux end to end with rig, Docker, model and OpenClaw doubled |
| `~/.cascade/openshell/` | binaries, gateway config + mTLS, profiles, `mcp.token`, `deployed.json` (named: `~/.cascade/openshell-<name>/` plus its `instance` file) |
