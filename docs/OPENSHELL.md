# Standalone OpenShell profile (opt-in)

OpenClaw in an [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell) 0.1.x
sandbox, driving the real rig, **without NemoClaw**. One script,
[`scripts/openshell_rig.py`](../scripts/openshell_rig.py), owns the whole route:
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

## Bring-up

```bash
# 0. the brain: OpenAI-compatible on 127.0.0.1:8081, a bearer key read from a 0600
#    file (default ~/.cascade/nemoclaw/llama.key, or --brain-key-file /
#    $CASCADE_BRAIN_KEY_FILE) and a bare served model id, e.g. vLLM with
#      --host 127.0.0.1 --port 8081 --served-model-name qwen3.8-27b
#      (VLLM_API_KEY from that file; never --api-key on argv)
#    scripts/serve_qwen_{vllm,llamacpp}.sh do not take a key yet, and the vLLM one
#    binds 0.0.0.0: do not use them unchanged for this route.

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
scripts/openshell_rig.py up --observe-only   # first bring-up: look, don't touch
scripts/openshell_rig.py status
scripts/openshell_rig.py open                # the demo's screen: OpenClaw's own web UI in the
                                             # Chromium profile with the cascade camera extension
                                             # (click it, then "Connect cameras"); token never printed
scripts/openshell_rig.py chat-url            # or print the tokenized chat URL: a secret

# 4. when the operator is ready to let the arm move
scripts/openshell_rig.py motion unlock
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
- **The chat token** is minted inside the sandbox at first start, in a 0600
  file. It is never on a host argv. A restart keeps it, so an open tab stays
  valid.

## Files

| Path | What |
|---|---|
| `scripts/openshell_rig.py` | the route: install, gateway, serve-mcp, up, motion, open, chat-url, status, down |
| `deploy/openshell/Dockerfile` | agent image (node 24 by digest, OpenClaw pinned, Landlock-safe layout) |
| `deploy/openshell/openclaw.json` | OpenClaw config template: placeholders, never secrets |
| `deploy/openshell/render-config.mjs` | first-start render: ports, model, chat origin, chat token |
| `deploy/openshell/cascade-openclaw` | entrypoint: render, then `openclaw gateway` on the sandbox loopback |
| `tests/test_openshell_rig.py` | the contract above (33 tests; runs `render-config.mjs` with the host's node) |
| `~/.cascade/openshell/` | OpenShell binaries, gateway config + mTLS, profiles, `mcp.token`, `deployed.json` |
