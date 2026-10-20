# shellcheck shell=bash
# Steps 4-6 of scripts/launch.sh on Linux, where the agent ALWAYS runs inside
# an NVIDIA OpenShell sandbox (docs/OPENSHELL.md), with a simulator or the
# real robot. Sourced by launch.sh once the simulator and its sidecars are up:
# it uses launch.sh's variables and helpers (log, warn, die, runtime_check,
# judge_pass, MCP_JSON, LAUNCH_OWNER, STATE_DIR, OS_INSTANCE, ...) and ends
# the launch. Nothing of the agent runs on this host:
#
#   4. brain    a model served on this host's loopback; the sandbox reaches only
#               that, through the instance's brain provider (the key stays here)
#      robot    the runtime is built once (runtime_check), then cascade's MCP
#               server runs as the instance's user unit on its loopback port,
#               bound to this launch's owner (one world for every chat session)
#      sandbox  `openshell_rig.py up`: image, provider profiles, the sandbox and
#               its chat forward. A simulator starts with motion allowed for
#               12 h; the real robot starts observe-only, as on the rig
#   5. proof    OpenClaw INSIDE the sandbox (`openshell_rig.py exec --`): the robot
#               tools must list, the brain must answer, and in sim modes one pick
#               is checked against physics and the same world reset
#   6. chat     OpenClaw's web UI through the instance's loopback forward

OS_RIG=("$PY" "$REPO/scripts/openshell_rig.py" --instance "$OS_INSTANCE")
OS_NAME="${OS_INSTANCE:-default}"
# How an operator addresses this instance (the default one needs no flag).
OS_HINT="scripts/openshell_rig.py${OS_INSTANCE:+ --instance $OS_INSTANCE}"

docker info >/dev/null 2>&1 \
    || die "OpenShell runs the agent in a Docker container: install Docker and add $USER to the docker group (then log in again)"

# ── 4. brain: a model on this host's loopback ──────────────────────────────
BRAIN_INFO="$("$PY" "$REPO/scripts/demo_proof.py" --check-brain "$BRAIN")" \
    || die "brain resolution failed; configure the selected model/endpoint explicitly"
log "brain resolution: $BRAIN_INFO"
BRAIN_FIELDS="$(printf '%s' "$BRAIN_INFO" | "$PY" -c '
import json, sys
from urllib.parse import urlsplit
d = json.load(sys.stdin)
if d.get("brain") == "keep":
    print("keep - - - -")
    raise SystemExit
u = urlsplit(d["base_url"])
loopback = u.hostname in ("127.0.0.1", "localhost", "::1") and u.port and u.path.rstrip("/") == "/v1"
print(d["brain"], d["base_url"], d["model"], d["context_window"], u.port if loopback else "-")
')" || die "unexpected brain resolution: $BRAIN_INFO"
read -r RESOLVED_BRAIN BRAIN_BASE BRAIN_MODEL BRAIN_CTX BRAIN_PORT <<<"$BRAIN_FIELDS"
[[ "$RESOLVED_BRAIN" != keep ]] \
    || die "the agent's sandbox reaches only a model served on this host: pass --brain nemotron-bolt|qwen|cosmos|cosmos-sglang (a cloud model OpenClaw is logged into never enters the sandbox)"
[[ -n "$BRAIN_PORT" && "$BRAIN_PORT" != - ]] \
    || die "brain $BRAIN_BASE is not http://127.0.0.1:<port>/v1; the sandbox reaches only this host's loopback"
[[ "$BRAIN_MODEL" != */* ]] \
    || die "served model id '$BRAIN_MODEL' contains '/'; serve it under a bare name (vLLM --served-model-name, llama-server --alias)"
"$PY" "$REPO/scripts/demo_proof.py" --probe-native "$BRAIN_BASE" --model "$BRAIN_MODEL" \
    || die "requested model lacks working native tool calls; refusing to substitute another brain"
# The brain provider's bearer key. The supervisor sends it only to this
# brain; a server without --api-key ignores it, so a fresh one is minted.
# scripts/serve_cosmos_sglang.sh serves only with a key: that brain's provider
# reuses the key file its brain checks just passed with (launch.sh defaults
# CASCADE_MODEL_API_KEY_FILE to the brain key file), never a minted key.
if [[ -n "${CASCADE_BRAIN_KEY_FILE:-}" ]]; then
    BRAIN_KEY_FILE="$CASCADE_BRAIN_KEY_FILE"
elif [[ "$RESOLVED_BRAIN" == cosmos-sglang && -z "${CASCADE_MODEL_API_KEY:-}" && -n "${CASCADE_MODEL_API_KEY_FILE:-}" ]]; then
    BRAIN_KEY_FILE="$CASCADE_MODEL_API_KEY_FILE"
else
    BRAIN_KEY_FILE="$STATE_DIR/openshell-brain.key"
    (
        umask 077
        if [[ -n "${CASCADE_MODEL_API_KEY:-}" ]]; then printf '%s\n' "$CASCADE_MODEL_API_KEY"
        else "$PY" -c 'import secrets; print(secrets.token_urlsafe(24))'; fi > "$BRAIN_KEY_FILE"
    ) || die "cannot write $BRAIN_KEY_FILE"
    chmod 600 "$BRAIN_KEY_FILE"
fi

# ── 4. robot server: built once, then the instance's unit ──────────────────
log "warming the MCP server's imports (first run in a fresh venv compiles bytecode; up to ~60 s)"
(cd "$REPO/models" && "$PY" -c "import cascade.apps.mcp_server, cascade.skills.runtime" >/dev/null 2>&1) \
    || warn "import warm-up failed -- the robot server's start below will show why"
runtime_check

log "OpenShell instance $OS_NAME: install (pinned, sha256-checked) and gateway"
"${OS_RIG[@]}" install 2>&1 | sed 's/^/        /' || die "OpenShell installation failed"
# From here on `launch.sh --down` tears this instance down, even after a failure.
printf '%s\n' "$OS_INSTANCE" > "$STATE_DIR/openshell-instance"
"${OS_RIG[@]}" gateway start 2>&1 | sed 's/^/        /' || die "OpenShell gateway start failed; see $OS_HINT status"

# The registration's environment, from a private file (never a unit's argv).
# A named instance belongs to this launcher and is replaced; the default
# instance is the real robot's, and a running server there is never touched.
# A simulator's outcomes must never teach the real robot: grasp memory and
# the learned envelope default to ~/.cascade, the real rig's stores, so a
# simulator's server learns into this launch's state (explicit paths win).
# Its world is new on every launch, so that memory starts empty each time:
# a position recalled from an earlier launch is wrong by construction.
MCP_ENV_FILE="$STATE_DIR/openshell-mcp-env.json"
rm -f "$MCP_ENV_FILE"
if [[ "$SIM" != none ]]; then
    rm -rf "$STATE_DIR/memory" && mkdir -p "$STATE_DIR/memory" || die "cannot reset the simulator's memory"
fi
(
    umask 077
    printf '%s' "$MCP_JSON" | "$PY" -c '
import json, os, sys
env = dict(json.load(sys.stdin)["env"])
sim, memory = sys.argv[1], os.path.join(sys.argv[2], "memory")
if sim != "none":
    for key, name in (("CASCADE_GRASP_MEMORY_PATH", "grasp_memory.json"), ("CASCADE_ENVELOPE_PATH", "envelope.json"),
                      ("CASCADE_BELIEFS_PATH", "beliefs.json"), ("CASCADE_EPISODIC_PATH", "episodic.json")):
        env.setdefault(key, os.path.join(memory, name))
if "DISPLAY" in env and os.environ.get("XAUTHORITY"):
    env.setdefault("XAUTHORITY", os.environ["XAUTHORITY"])  # the viewer, from a user unit
print(json.dumps(env))' "$SIM" "$STATE_DIR" > "$MCP_ENV_FILE"
) || die "cannot write the robot server's environment"
REPLACE=()
[[ -z "$OS_INSTANCE" ]] || REPLACE=(--replace)
# A native-controller arm (the Franka FR3) runs its server in the ROS 2
# environment; CASCADE_MCP_CPUS (or the Franka launcher's own taskset) pins it.
[[ ${FRANKA_ROS:-0} == 0 ]] || REPLACE+=(--ros-env)
[[ -z "${MCP_CPUS:-}" ]] || REPLACE+=(--cpus "$MCP_CPUS")
log "starting cascade's robot MCP server as instance $OS_NAME's unit (cameras=$CAMERAS arm=$ARM)"
"${OS_RIG[@]}" mcp-start --env-file "$MCP_ENV_FILE" ${REPLACE[@]+"${REPLACE[@]}"} --wait 300 \
    -- --launch-owner "$LAUNCH_OWNER" --launch-state-dir "$STATE_DIR" \
    || { rm -f "$MCP_ENV_FILE"; die "the robot MCP server did not start (see above)"; }
rm -f "$MCP_ENV_FILE"
if [[ ${FRANKA:-0} == 1 ]]; then
    # The Franka operator contract, read from the server's own live view: the
    # camera streams, the software STOP is latched and Resume & Home exists.
    log "franka: waiting for the robot server: ${CAMERAS%%,*} streaming, STOP latched, Resume & Home on :$STREAM_PORT"
    FRANKA_READY="$("$PY" "$REPO/scripts/franka_rig.py" ready --port "$STREAM_PORT" --camera "${CAMERAS%%,*}" \
        --timeout "${CASCADE_FRANKA_READY_WAIT_S:-300}")" \
        || die "the robot server did not reach the Franka operator contract (see above); log: $OS_HINT status"
    log "franka: robot server ready: $FRANKA_READY"
fi

# ── 4. sandbox ─────────────────────────────────────────────────────────────
if [[ "$SIM" == none ]]; then
    MOTION=(--observe-only)
    MOTION_NOTE="LOCKED by OpenShell (observe-only): $OS_HINT motion unlock --for 20m"
else
    MOTION=(--allow-motion --for 12h)
    MOTION_NOTE="allowed for 12 h (simulator), then re-locked by OpenShell"
fi
# The agent's profile: the Franka FR3 operator agent (command router +
# instructions, deploy/openshell/franka-plugin), the Cosmos3-Edge brain's
# compact agent (deploy/openshell/cosmos-plugin: short prompt, no JSON
# envelopes, 10 tools; the 2.4B model writes JSON prose instead of tool calls
# under the generic one) or the generic one. Always named, so a profile saved
# by an earlier launch never carries over.
AGENT_PROFILE=generic
[[ "$RESOLVED_BRAIN" != cosmos && "$RESOLVED_BRAIN" != cosmos-sglang ]] || AGENT_PROFILE=cosmos
[[ ${FRANKA:-0} == 0 ]] || AGENT_PROFILE=franka
log "bringing up the OpenClaw sandbox (brain :$BRAIN_PORT $BRAIN_MODEL, context $BRAIN_CTX)"
case "$AGENT_PROFILE" in
    franka) log "agent profile: franka (deploy/openshell/franka-plugin: command router + FR3 instructions)" ;;
    cosmos) log "agent profile: cosmos (deploy/openshell/cosmos-plugin: short prompt, no JSON envelopes, 10 tools)" ;;
esac
"${OS_RIG[@]}" up --brain-port "$BRAIN_PORT" --model "$BRAIN_MODEL" --context-window "$BRAIN_CTX" \
    --brain-key-file "$BRAIN_KEY_FILE" --agent-profile "$AGENT_PROFILE" "${MOTION[@]}" 2>&1 | sed 's/^/        /' \
    || die "the OpenClaw sandbox did not come up; see $OS_HINT status"
OS_DEPLOYED="$("${OS_RIG[@]}" instance | "$PY" -c '
import json, pathlib, sys
inst = json.load(sys.stdin)
saved = json.loads((pathlib.Path(inst["home"]) / "deployed.json").read_text())["settings"]
print(saved["sandbox"], saved["chat_port"])
')" || die "cannot read instance $OS_NAME's deployed settings"
read -r OS_SANDBOX OS_CHAT_PORT <<<"$OS_DEPLOYED"

# ── 5. proof: OpenClaw inside the sandbox ──────────────────────────────────
# demo_proof.py drives the sandbox's own CLI; its MCP server is `cascade`
# (deploy/openshell/openclaw.json).
CASCADE_OPENCLAW_COMMAND="$("$PY" -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "${OS_RIG[@]}" exec -- openclaw)"
export CASCADE_OPENCLAW_COMMAND
export CASCADE_MCP_NAME=cascade
"$PY" "$REPO/scripts/demo_proof.py" --wait-gateway 120 >/dev/null \
    || die "the sandbox's OpenClaw gateway did not become healthy; see $OS_HINT status"
log "probing MCP tools from inside the sandbox"
PROBE_JSON="$("${OS_RIG[@]}" exec -- openclaw mcp probe cascade --json 2>/dev/null || true)"
read -r NTOOLS HAVE_ROBOT <<<"$(printf '%s' "$PROBE_JSON" | "$PY" -c '
import json, sys
try:
    d = json.loads(sys.stdin.read() or "{}")
except ValueError:
    print("0 0")
    raise SystemExit
tools = [t.split("__", 1)[-1] for t in d.get("tools", [])]
print(len(tools), int({"pick_and_place", "get_observation", "analyze_scene"} <= set(tools)))
' 2>/dev/null || echo "0 0")"
[[ "$HAVE_ROBOT" == 1 ]] \
    || die "the sandbox's MCP probe did not list the robot tools ($NTOOLS found); see $OS_HINT status"
log "tools reachable from the sandbox ($NTOOLS listed, incl. pick_and_place / get_observation / analyze_scene)"
log "proving brain, manipulation and reset in one persistent chat session, inside the sandbox"
PROOF_FLAGS=(--persistent-mcp)
[[ $ROBOT_TURN == 1 ]] || PROOF_FLAGS+=(--no-robot-turn)
PROOF_JSON="$("$PY" "$REPO/scripts/demo_proof.py" --run --repo "$REPO" \
    --state-dir "$STATE_DIR" --sim "$SIM" "${PROOF_FLAGS[@]}")" \
    || die "demo verification failed; NOT READY. Evidence: $STATE_DIR/cascade-proof-*/"
printf '%s\n' "$PROOF_JSON"
PROOF_VERIFIED="$(printf '%s' "$PROOF_JSON" | "$PY" -c 'import json,sys; print(int(json.load(sys.stdin)["verified"]))')"
BRAIN_DESC="$(printf '%s' "$PROOF_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["model"])')"
judge_pass

# ── 6. chat ────────────────────────────────────────────────────────────────
OPERATOR_NOTE=""
if [[ ${FRANKA:-0} == 1 ]]; then
    OPERATOR_NOTE=$'\n'"         operator:  STOP / Resume & Home http://127.0.0.1:$STREAM_PORT/ (STOP latched at start; only Resume & Home clears it)"
fi
LAUNCH_STATUS="STARTED (UNVERIFIED: robot proof skipped)"
[[ "${PROOF_VERIFIED:-0}" == 1 ]] && LAUNCH_STATUS="READY"
cat <<EOF
[launch] $LAUNCH_STATUS   sim=$SIM  arm=$ARM  cameras=$CAMERAS
         agent:     OpenClaw inside OpenShell (instance $OS_NAME, sandbox $OS_SANDBOX); no agent on this host
         brain:     ${BRAIN_DESC:-not verified} (127.0.0.1:$BRAIN_PORT, through the instance's brain provider)
         chat:      http://127.0.0.1:$OS_CHAT_PORT/   ($OS_HINT open | chat-url)
         motion:    $MOTION_NOTE$OPERATOR_NOTE
         evidence:  $STATE_DIR/proof.json
         headless:  $OS_HINT exec -- openclaw agent --session-id cascade-demo -m "describe the scene"
         try:       "what do you see?"  "pick and place the red object"  "did it actually move?"
         reset:     "reset the scene"  (between visitors: props restored, scene beliefs cleared)
$( [[ -n "${JUDGE_NOTE:-}" ]] && echo "         judge:     $JUDGE_NOTE" )
$( [[ "$SIM" == "isaac"  ]] && echo "         isaac:     bridge :$BRIDGE_PORT, log $STATE_DIR/isaac_bridge.log" )
$( [[ "$OCCUPANCY" != "none" ]] && echo "         occupancy: :$OCC_PORT ${OCC_DESC:-}" )
$( [[ "$GRASPGENX" != "none" ]] && echo "         graspgenx: $GGX_HOST:$GGX_PORT ($GRASPGENX)" )
         stop:      ./scripts/launch.sh --down
EOF
if [[ $OPEN_CHAT == 1 ]]; then
    "${OS_RIG[@]}" open || warn "could not open the browser; use the chat URL above ($OS_HINT chat-url)"
fi
