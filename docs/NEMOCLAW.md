# NemoClaw / OpenShell profile (opt-in)

[NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell) runs an agent in a
policy-enforced sandbox (Landlock filesystem rules, seccomp, a network
namespace with deny-by-default egress through an inspecting proxy, credentials
resolved by the gateway at egress). [NVIDIA NemoClaw](https://github.com/NVIDIA/NemoClaw)
is the reference stack that installs OpenShell, builds the sandbox for an agent
(OpenClaw, Hermes or Deep Agents Code), routes inference through
`https://inference.local` and manages MCP servers for it.

CASCADE uses it as an **opt-in agent host**. Plain OpenClaw over stdio
(`run.sh` → `scripts/launch.sh`) stays the default until this profile passes
the same READY proof (tools listed, a brain turn, a real pick and reset in one
chat session).

## What runs where

```
 host                                               OpenShell sandbox (k3s pod)
 ─────────────────────────────────────────────      ─────────────────────────────
 robot runtime: SafetyHarness, perception,           OpenClaw agent (NemoClaw image)
 Isaac/MuJoCo bridge, GPU, grasp planners              │ tools/call cascade.*
        ▲                                              ▼
 cascade.apps.mcp_server --http 172.17.0.1:18740  ◄── OpenShell MCP proxy (policy:
   TLS (private CA) + bearer token                    pinned host/IP, method profile,
   same serial worker + stop channel as stdio         131,072-byte body, --deny-tool)
                                                       │ inference
 llama-server :8081 (authenticated, GPU 0)  ◄──────── inference.local → gateway
```

The robot never runs inside the sandbox. The sandbox gets the tool surface and
nothing else: no robot process, no GPU, no files, no other network.

## Why an HTTP transport

NemoClaw registers only authenticated Streamable HTTP MCP servers; it does not
start, wrap or translate stdio servers. `cascade.apps.mcp_server --http
HOST:PORT` serves the same 45 tools over Streamable HTTP. Only the framing
changes:

- One process, one robot runtime, one **serial worker**, any number of MCP
  sessions (the NemoClaw readiness probe opens its own).
- The **stop channel is unchanged**: `emergency_stop`, the capability stop
  tools, `notifications/cancelled` for the in-flight motion and `ping` are
  handled on receipt (`_admit`, shared with stdio), never queued behind a
  running pick.
- JSON-RPC ids are namespaced per session, so a cancel from one session cannot
  freeze another session's motion.
- Tool calls answer as JSON or SSE (`Accept`); SSE sends `: keepalive`
  comments every 15 s (`CASCADE_MCP_SSE_KEEPALIVE_S`) so a 150 s pick survives
  proxy idle timeouts. A dropped connection is **not** a cancel (MCP spec);
  stop and cancel stay explicit.
- `GET` answers 405 (no server-to-client stream). A trimmed tool catalog
  therefore is not pushed; a withheld tool is still refused with its reason.

Guards (all refuse before any robot code runs):

| Guard | Behaviour |
| --- | --- |
| Bearer token | Required at start-up (`CASCADE_MCP_TOKEN` or `--token-file`, ≥ 24 chars); every request is compared in constant time; 401 otherwise |
| Listener | A non-loopback address requires `--tls-cert/--tls-key` |
| `Origin` header | 403 (browsers send it, MCP hosts do not: DNS rebinding) |
| `Content-Type` | 415 unless `application/json` |
| `MCP-Protocol-Version` | 400 for unknown versions |
| Body | 413 above 131,072 bytes (OpenShell's own cap); 400 for non-JSON or batches |
| Session | `initialize` issues `Mcp-Session-Id`; unknown/deleted sessions get 404 |

## Bring-up

`scripts/nemoclaw_mcp.py` does the operator side:

```bash
# 1. private CA (CA:TRUE), server cert with the IP SAN, 32-byte bearer token;
#    default host = Docker bridge gateway, state in ~/.cascade/nemoclaw (0700)
python scripts/nemoclaw_mcp.py certs
# 2. the sandbox must trust that CA: set it BEFORE onboarding (or rebuild)
export NEMOCLAW_CORPORATE_CA_BUNDLE="$(python scripts/nemoclaw_mcp.py ca-path)"
nemoclaw onboard                      # existing sandbox: nemoclaw <name> rebuild
# 3. the robot-side MCP server (same CASCADE_* env as launch.sh would give it)
CASCADE_ARM=isaac CASCADE_CAMERAS=isaac,isaac_side python scripts/nemoclaw_mcp.py serve
# 4. register it; the token goes to OpenShell's provider store, never to argv
python scripts/nemoclaw_mcp.py register --sandbox <name>
nemoclaw <name> mcp status cascade --json   # trustedPrivateTarget.state == "match"
# 5. give the sandbox's OpenClaw the stdio profile's settings (see below)
python scripts/nemoclaw_mcp.py configure-agent --sandbox <name>
nemoclaw <name> agent --session-id proof -m "Pick up the pink cube and place it in the box"
```

A gateway on a non-default port needs `NEMOCLAW_GATEWAY_PORT` in the
environment of every later `nemoclaw` command (otherwise: "Sandbox not
found"); on the x86 rig the default 8080 is the `cascade-qwen` brain, so the
gateway runs on 18750.

### Agent settings NemoClaw does not set

`configure-agent` runs `openclaw config set` inside the sandbox for four
settings that `launch.sh` gives the stdio profile and NemoClaw's generated
config lacks. Each one was a measured failure on 8 October 2026:

| Setting | Without it |
| --- | --- |
| `tools.toolSearch.enabled=false` | MCP tools sit behind `tool_search`/`tool_call`; the call result is JSON text capped at 16,000 chars, so `get_observation`'s image arrived as base64 text and the turn ended in "Context overflow" after three image tools |
| model `input: ["text","image"]` | the `llama-cpp` provider registers Qwen3.8-27B as text-only: no image reaches the model |
| `mcp.servers.cascade.requestTimeoutMs=300000` | OpenClaw's 60 s per-call default cancels a 74 s pick; a cancelled motion latches the e-stop |
| `mcp.servers.cascade.connectionTimeoutMs=120000` | first connect races the lazy runtime build |

These live in the sandbox's `openclaw.json`; re-run `configure-agent` after a
`rebuild` or a re-onboard.

Address choice. NemoClaw rejects loopback, `host.openshell.internal` and
public addresses for a trusted-private server; it needs HTTPS on a stable
RFC1918/CGNAT address the OpenShell gateway can route to, with a certificate
whose SAN matches the URL host. The Docker bridge gateway (`172.17.0.1` by
default) satisfies that and is not published on the LAN. Without root there is
no host firewall rule in this setup; the listener is protected by TLS and the
bearer token only. Add a firewall rule restricting the port to the OpenShell
subnet where you can.

Local inference. NemoClaw's `llama-cpp` provider attaches an operator-run,
**authenticated** `llama-server` on `127.0.0.1:8081` that is also reachable as
`host.openshell.internal:8081`; the served alias must not look like a path
(`qwen3.8-27b`, not `Qwen/Qwen3.8-27B`). The default CASCADE brain
(`cascade-qwen`, `:8080`, unauthenticated, loopback) is left untouched; the
profile runs a second instance with `--api-key-file`.

## Hazards specific to this host

- **Cold GraspGen-X.** The first learned-grasp inference after the server
  starts took 15.5 s on the x86 rig, past the client's 8 s timeout, so the
  first attempt fell back to the analytic planner; in that run the cube ended
  held at 0.44 m and the retries refused with "already holding". The same
  failure does not depend on the transport; warm the planner before a proof.
- **The `nemoclaw <name> agent` wrapper exits 1** with `replayInvalid=true`
  even when the turn's JSON reports `status: ok` and every tool call
  succeeded. Judge a turn by the server's per-call log and the physics
  postcondition, not by that exit code.
- **Transient units and the gateway.** A `systemd-run` unit that runs
  `nemoclaw onboard` kills the OpenShell gateway it started when the unit
  ends (the sandbox container exits 137). Use `-p KillMode=process`. After a
  reboot, `nemoclaw onboard` (same env) brings the gateway and sandbox back;
  `<name> start`/`recover` cannot restart the shared gateway.
- **`--deny-tool` matches names only.** It cannot restrict arguments.
- **Stdio is still the default.** Nothing in `launch.sh` changes.

## Status

Measured on the x86 rig on Isaac Sim (8 Oct 2026,
[evidence](evidence/b35-nemoclaw-openshell-20261008/REPORT.md)): a sandboxed
OpenClaw agent ran a physics-confirmed `pick_and_place` (≈74 s, no cancel)
followed by a verified `reset_scene` in the same session, 2 of 3 fresh stages;
host-direct calls to the same server 1 of 3. Every failure signature also
occurs without the sandbox (pick reliability on this rig, a separate item).
Not yet: the kitchen proof cases, `demo_proof.py` on this route, the Spark.
See the dated entry in [ROADMAP](ROADMAP.md).
