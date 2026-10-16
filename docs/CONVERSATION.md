# Local conversation and robot tools

**Next action:** start a private run against a running Realtime provider (2 min; `--extra
conversation` adds the optional transport):

```bash
uv run --extra conversation cascade-conversation \
  --robot conversation_mock \
  --provider-url ws://127.0.0.1:8765/v1/realtime \
  --allow-tool sensing.list_sensors \
  --allow-tool sensing.read_sensor \
  --run-dir runs/conversation-01
```

CASCADE has an optional browser/media gateway and a robot-agnostic conversation supervisor. Audio
travels through a configured Realtime provider; selected typed tool intents enter the existing
`RobotRuntime`. The conversation layer has no joint writer, hardware identity, locomotion policy or
implicit authority to reset stops, and does not import or emulate the Reachy Mini SDK. Fleet
conversation routing is separate work ([ROBOT_MODULARITY](ROBOT_MODULARITY.md)).

What works today (simulation and loopback only):

1. Real loopback HTTP/WebSocket gateway, generated PCM/WAV audio, the synthetic sensor profile and
   the CLI; Chromium 153 with AudioWorklet, Stop, reset, reconnect and disconnect
   ([receipt](evidence/robot-modularity/conversation-chromium.json),
   [screenshot](evidence/robot-modularity/conversation-chromium.png)).
2. A split deployment where the robot runs in its own process, on loopback or over TLS
   ([below](#deploy-the-conversation-service-separately)).
3. [Two complete native speech cases (VOICE13)](#two-complete-native-speech-cases--voice13):
   synthetic speech → provider → `index_flex` on the simulated LEAP hand → verified rest → spoken
   reply, plus stop/reset/reconnect.
4. **Not established:** hardware microphone/speaker, general listening accuracy, CPU speech tools
   at interactive latency (one CPU greeting took 68.923 s, its 61.580 s LLM interval past the 60 s
   intent deadline: [receipt](evidence/robot-modularity/conversation-native-chromium.json)),
   physical task success, speech-initiated native MicroDuck walking with a spoken reply.

## Run the gateway

1. The provider must already be running. Open the printed `http://127.0.0.1:8780/#…` URL (the
   fragment is a temporary access credential), click **Connect**, then type or **Start
   microphone**. The browser needs AudioWorklet and a 24 kHz AudioContext (unsupported rates fail
   visibly); capture is mono s16le, 960 samples per chunk. Remote providers need `wss`; local `ws`
   must be loopback. `--token-env YOUR_PROVIDER_TOKEN_VARIABLE` reads a bearer credential in the
   gateway process, never sent to the browser.
2. **Tools.** `conversation_mock` declares only a synthetic IMU (`synthetic`, `software_only`).
   Provider tools default to an empty allowlist. Motion also needs `--allow-motion` plus explicitly
   listed curated skills (e.g. `locomotion.walk_velocity`, `manipulation.pick_and_place`); the
   runtime's schemas, capabilities, ownership, collision/lease gates and verification still apply.
   Raw joint/pose setters and permission/reset controls can never be speech tools, and a
   successful function call never upgrades an unverified physical postcondition.
3. **Stop and reset.** Stop is an authenticated HTTP route independent of the provider, playback
   and the action worker; disconnect also stops. Then **Explicitly reset stop** and connect a new
   session: reset closes the old conversation, reads the generation from `/api/status` and posts
   exactly `{"generation": <observed integer>}` to `/api/reset`, checked under the runtime's
   admission lock. A delayed request cannot clear a newer stop; runtime refusal is preserved.
   Shutdown writes `closure.json` (a running action is reported pending, never "cancelled").
4. **Bounded LEAP hand.** `--robot-lifecycle bounded_hand` (JSON `robot_lifecycle:
   "bounded_hand"`) defers construction until the operator clicks **Start robot session** (one
   construction, the stopped-readiness check, a real reset at the observed generation, a new
   provider conversation; only that session gets microphone PCM; nothing is replayed). Supply the
   prepared hand profile and its exact model pin. A disconnect or stop during construction cancels
   admission (a late constructor is kept until its cleanup finishes). The hand keeps its 30 s
   lifetime from owner start; Start has its own 10 s deadline; neither renews; an expired episode
   needs a new service run.
5. **MicroDuck.** `--robot microduck_conversation_mock` is a base-only `kinematic_mock` fixture
   (allow only `locomotion.get_base_state` for a passive first connection).
   `microduck_conversation_native` composes `microduck_isaac` and needs every pin in
   [MicroDuck](MICRODUCK.md#running-the-candidate); missing pins fail before connecting. Start with
   observation-only tools. No speech timing enters a control loop; the locomotion policy keeps its
   head and leg actions.

### Installed service

A versioned JSON configuration can live outside the checkout:

```json
{
  "version": 1,
  "robot": "conversation_mock",
  "config_dir": "profiles",
  "provider_url": "ws://127.0.0.1:8765/v1/realtime",
  "allow_tools": ["sensing.list_sensors", "sensing.read_sensor"],
  "start_stopped": true,
  "port": 8780,
  "run_root": "runs"
}
```

1. Run `uv run --extra conversation cascade-conversation --config /absolute/path/service.json`.
   Paths are relative to the file; `profiles` holds the selected `robots/*.yaml` and `llm/*.yaml`.
   CLI options override fields. Choose exactly one of `run_dir` / `run_root` (a new private child
   per start). Unknown, duplicate or ambiguous fields are rejected.
2. `intent_timeout_s` / `execution_timeout_s` (also CLI): defaults 10/30 s, maxima 60/300 s. The
   intent budget starts at the input origin and is never renewed by a model response or tool
   continuation; it does not extend any actuator's own admission deadline.
3. `token_env` names a credential variable (never expanded into config or receipts). The loopback
   bind stays.
4. `ready.json` records listener publication, process/profile identity and whether the runtime
   started stopped (no token; not provider readiness or a physical result). With
   `start_stopped`, reset the stop before connecting. A restart never replays the previous
   session, restores physical state or clears an external controller's stop.
5. Shutdown attempts every cleanup even when one fails; `closure.json` keeps partial cleanup and
   the process exits nonzero on startup, teardown or receipt failure. A failed session close
   blocks a replacement session.

## Deploy the conversation service separately

By default `cascade-conversation` builds the selected robot runtime inside its
own process. The opt-in split deployment (B51) runs the robot in its own process
instead, so the speech layer can be deployed, restarted or upgraded without
restarting the robot, and the robot side needs neither `aiohttp` nor any speech
dependency:

```bash
export CASCADE_ROBOT_ENDPOINT_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
# Robot side (base install): owns the runtime, harnesses, verifiers and trace.
uv run cascade-robot-service --robot conversation_mock --port 8781 \
  --token-env CASCADE_ROBOT_ENDPOINT_TOKEN --run-root runs/robot
# Conversation side (`conversation` extra): builds no robot.
uv run --extra conversation cascade-conversation --config configs/conversation/remote_robot.json
```

1. [`configs/conversation/remote_robot.json`](../configs/conversation/remote_robot.json) is the
   example. `robot_endpoint` (`http://<loopback>:<port>`, normalized; no path, query or
   credentials) and `robot_token_env` (variable holding the shared bearer token, ≥ 32 characters)
   select split mode; `robot` names the robot_id the service must report (any other identity
   refuses startup); `robot_lifecycle` and `config_dir` are refused. CLI: `--robot-endpoint`,
   `--robot-token-env`. Both processes bind loopback by default; the robot service binds another
   address only over TLS ([across hosts](#across-hosts-opt-in-tls-b71)).
2. `cascade.robot-runtime/1` is HTTP/JSON
   ([`robotics/endpoint.py`](../src/cascade/robotics/endpoint.py), stdlib only). Requests carry
   `Authorization: Bearer …` and `X-Cascade-Protocol: cascade.robot-runtime/1`; a peer on another
   version is refused before any lease or command.
3. **Operator stop wins.** `/api/stop` reaches `/v1/stop` on its own request thread while a motion
   executes; any token holder may stop without a lease. Execute can never clear a stop; reset needs
   the lease and the exact observed generation.
4. **Lease.** One supervision lease (default 3 s, `--lease-ttl-s`, 0.2 < ttl ≤ 30) renewed every
   third of that; when renewal lapses the robot latches `stop()` (recorded under
   `lease_expiry_stops` in `closure.json`). Release is neither stop nor reset; a lost lease is
   terminal for that conversation process (restart, then reset explicitly).
5. **Uncertainty is explicit.** Unknown robot state reads as stopped; a transport failure during
   execute is `delivery_uncertain`; with the robot gone `/api/status` answers 500 and shutdown exits
   nonzero. Tool results stay the robot's verdicts. Unauthenticated probes: `GET /healthz`
   (`{"alive": true}`), `GET /readyz` (200 with endpoint answering and lease held, else 503).
   `ready.json` gains `robot_endpoint: {origin, protocol, catalog_sha256, lease_ttl_s}` (the robot's
   own `ready.json` carries the same `catalog_sha256`).

| Route | Needs | Effect |
| --- | --- | --- |
| `GET /v1/health` | nothing | liveness |
| `GET /v1/describe` | token | robot_id, resources, tool descriptors, `trace_recorded`, `lease_ttl_s` |
| `GET /v1/state` | token | generation, stopped, lease held, lease-expiry stops, executing requests |
| `POST` / `DELETE /v1/lease` | token | acquire (one supervisor, 409 while held), renew, release |
| `POST /v1/execute` | token + lease | `RobotRuntime.execute` with the episode generation and remaining deadline (at most 300 s) |
| `POST /v1/stop` | token only | `RobotRuntime.stop()`, never refused for lease reasons |
| `POST /v1/reset` | token + lease | `reset_stop` with the exact observed generation |
| `POST /v1/record` | token + lease | trace row for an already delivered stop-effect tool |

Measured on CPU (`tests/test_conversation_split_service.py`, real child processes, a Realtime stub,
`microduck_conversation_mock`): a text turn ran to `locomotion.walk_velocity` in the robot process
and returned `unverified`; an operator stop superseded a 3 s walk; a stale-generation reset was
refused; `SIGKILL` of the conversation mid-walk latched the robot stop through a 1 s lease and a
new conversation re-attached with the stop still latched. Transport and authority evidence only.

### Across hosts: opt-in TLS (B71)

`cascade-robot-service --tls-cert <pem> --tls-key <pem>` serves HTTPS (TLS 1.2+, stdlib `ssl`) and
may then bind any explicit unicast IP. Without TLS a non-loopback `--host` is refused before any
runtime is built (exit 2); `0.0.0.0`, `::` and multicast are always refused. Token, lease, stop
and reset rules are unchanged: TLS adds who the robot is, not who may command it.

```bash
# Robot host: a private CA (stays here) and a certificate for the address the conversation dials.
uv run python scripts/robot_endpoint_certs.py --out runs/robot-tls --san IP:192.0.2.10
uv run cascade-robot-service --robot conversation_mock --host 192.0.2.10 --port 8781 \
  --tls-cert runs/robot-tls/server.pem --tls-key runs/robot-tls/server.key \
  --token-env CASCADE_ROBOT_ENDPOINT_TOKEN --run-root runs/robot
# Conversation host: copy ca.pem only (never ca.key or server.key) and set the token.
uv run --extra conversation cascade-conversation --config configs/conversation/remote_robot_tls.json
```

1. [`scripts/robot_endpoint_certs.py`](../scripts/robot_endpoint_certs.py) (the B35 openssl recipe
   of `scripts/nemoclaw_mcp.py certs`): RSA-3072 CA once per directory (`--force` rotates),
   RSA-2048 server cert valid for exactly the `--san` entries (`IP:<address>` / `DNS:<name>`, no
   wildcard), keys 0600 in a 0700 directory, SHA-256 fingerprint printed (the service prints
   `TLS certificate sha256 …` and adds `tls_certificate_sha256` to `ready.json`).
2. Conversation keys ([example](../configs/conversation/remote_robot_tls.json)): `robot_endpoint`
   `https://<host>:<port>` plus at least one trust anchor: `robot_tls_ca` (`--robot-tls-ca`, the
   only CA trusted, cert must name the URL host) and/or `robot_tls_fingerprint`
   (`--robot-tls-fingerprint`, 64 hex digits, checked before a single request byte; alone it trusts
   exactly that cert; with a CA both must hold).
3. TLS is explicit both ways: https without an anchor, an anchor with http, or plaintext off
   loopback are config errors. `ready.json` adds `tls: {certificate_sha256, trust}`. The plaintext
   default is byte-identical (golden-pinned). Handshakes run per connection thread with a 10 s
   limit, so a silent peer cannot delay a stop; every call is a fresh HTTP/1.0 connection (full
   handshake, no resumption).
4. Tested (`tests/test_robot_endpoint_tls.py`, skipped without `openssl`): a TLS text turn and stop;
   wrong CA, wrong fingerprint, plaintext to TLS, stale or missing token and wrong host name all
   refused before the lease; `SIGKILL` mid-walk latched the stop; a silent TCP peer delayed nothing.
5. **Not claimed:** a two-host deployment (loopback and one private interface only), real-network
   or loaded latency, mutual TLS (the client proves itself with the token), token expiry or
   rotation, certificate revocation or renewal, firewall rules.

Extra-hop latency ([`scripts/bench_robot_endpoint.py`](../scripts/bench_robot_endpoint.py),
[evidence](evidence/b71-robot-runtime-tls-20261010/latency.json)): N = 500
interleaved calls per mode after 20 warm-up calls, mock MicroDuck robot, one
x86 host with 128 hardware threads at load average 30–33, loopback, TLSv1.3
(TLS_AES_256_GCM_SHA384, RSA-2048 leaf). Milliseconds, p50 / p95:

| Call | In process | Split, plaintext | Split, TLS |
| --- | --- | --- | --- |
| state read (`GET /v1/state`) | 0.003 / 0.004 | 0.53 / 0.67 | 1.57 / 1.89 |
| read tool (`locomotion.get_base_state`) | 0.79 / 1.12 | 1.51 / 1.85 | 2.37 / 2.75 |

The hop adds about 0.5 ms (state) to 0.7 ms (tool) at p50 in clear and about 1.6 ms with TLS (a
full handshake per call). One host's loopback numbers with other suites running.

## Boundaries and lifecycle

1. **Media.** `MediaIO` separates capture, playback, flush and close; `QueueMediaIO` bounds event
   count, queued PCM bytes and capture age; browser sequence numbers must advance exactly; gateway
   timestamps use its local monotonic clock, never the actuator clock. Browser playback holds at
   most 15 s and 512 audio objects with 2 s scheduled ahead; overflow visibly stops the session;
   flush discards queued audio; a permission prompt finished after stop releases its tracks.
2. **Authority.** `ConversationSession` binds session ID, robot ID, response origin, request ID,
   the runtime cancellation generation and a local deadline. Audio authority starts at local
   receipt of `speech_started` (unique item ID); text at local submission; continuations keep the
   original deadline. `response.create` carries a nonce the response must echo; arguments are
   staged until `response.done` is `completed`, and cancelled, failed or incomplete responses never
   run staged calls. Requests are deduplicated at session and dispatch. Budgets per session: 256
   responses, 256 calls, 8 calls per response, 30 min; an exhausted budget needs a new session.
   After an external stop/reset, a new response ID gets no fresh authority in the old connection.
   `RealtimeWebSocket` lazily imports `aiohttp`, bounds sizes and send/open time, never reconnects
   silently.
3. **Dispatch.** `ConversationDomain` owns at most one action thread; timeout asks the runtime to
   stop and reports uncertain delivery. A tool declared `effect="stop"` takes the coalesced
   priority stop path even while that thread is busy, is traced like `RobotRuntime.execute`, and
   its ACK stays `physical_stop_verified=False`. A completed response may stage a motion plus an
   adjacent stop: the stop is dispatched before either result goes to a slow provider, and the
   reply continuation after that stop cannot renew motion authority. A successful close is
   terminal: a later stop is refused (`conversation domain closed; no new stop delivered`), never
   answered with a cached ACK. `close()` revokes authority and discards pending input before
   teardown.
4. **Interruption.** Default barge-in invalidates pending speech/tool contexts and stops active or
   pending robot work (idle initial speech does not latch a stop). `speech_only` interruption is
   allowed only without motion authority. Ambiguous VAD ownership after an interruption revokes
   tool authority until reconnect. Tool output to the provider is bounded to 32,768 UTF-8 bytes;
   an oversized traced result may omit only image attachments, `measured.samples` and
   `postcondition.evidence` (listed with size and SHA256 under `speech_transport`).
5. **Access.** The gateway binds an explicit loopback IP with a random bearer token, exact Origin
   checks and a single-use media ticket; provider config is operator-owned; one active session;
   access logging off; no credentials in receipts. Public hosting and multi-user auth are out of
   scope.

## Provider contract and self-hosting

The target is the HF speech-to-speech GA WebSocket subset at revision
`411399d34555b2169823a6eaeb7f8ff192db89db` (`session.update`, `input_audio_buffer.append`, input
text, `response.create`, `response.cancel`, output PCM, function outputs; `response_id` and
`output_index` are bound), not full OpenAI API or WebRTC
([protocol](https://github.com/huggingface/speech-to-speech/blob/411399d34555b2169823a6eaeb7f8ff192db89db/src/speech_to_speech/api/openai_realtime/README.md),
[response handler](https://github.com/huggingface/speech-to-speech/blob/411399d34555b2169823a6eaeb7f8ff192db89db/src/speech_to_speech/api/openai_realtime/handlers/response.py)).
Its [audio handler](https://github.com/huggingface/speech-to-speech/blob/411399d34555b2169823a6eaeb7f8ff192db89db/src/speech_to_speech/api/openai_realtime/handlers/audio.py)
may defer `speech_stopped`, its automatic response has no input correlation and it ignores
`turn_detection.create_response=False`: CASCADE admits first uninterrupted VAD input and requires
reconnection after ambiguity. No provider-to-capture clock mapping is claimed.

**CPU provider** (Linux x86_64, Python 3.12+, `uv`, `git`; its own env and models, nothing installed
into Cascade's env; use a **new** private state dir):

```bash
python3.12 scripts/conversation_provider.py prepare --state-dir ~/.cascade-speech-schema-eof-cancel
python3.12 scripts/conversation_provider.py verify --state-dir ~/.cascade-speech-schema-eof-cancel
python3.12 scripts/conversation_provider.py serve --state-dir ~/.cascade-speech-schema-eof-cancel \
  --run-dir runs/speech-provider-01 --port 18878 --timeout-s 900
```

**CUDA LLM profile** (`--profile cuda-llm-fp32`: hash-pinned PyTorch 2.11.0 CUDA 13.0, FP32 LLM
on CUDA with an explicit `{"": "cuda:0"}` map, Whisper and Kokoro on CPU; set `GPU_UUID` to the full
`GPU-…` UUID from `nvidia-smi`):

```bash
python3.12 scripts/conversation_provider.py prepare --profile cuda-llm-fp32 \
  --state-dir ~/.cascade-speech-cuda-fp32
python3.12 scripts/conversation_provider.py verify --profile cuda-llm-fp32 \
  --state-dir ~/.cascade-speech-cuda-fp32
python3.12 scripts/conversation_provider.py serve --profile cuda-llm-fp32 \
  --state-dir ~/.cascade-speech-cuda-fp32 --cuda-device-uuid "$GPU_UUID" \
  --run-dir runs/speech-cuda-01 --port 18878 --timeout-s 900
```

1. Preparation downloads the pinned dependencies, Whisper, Qwen and Kokoro; the
   [source manifest](../configs/conversation/provider/manifest.json) binds the upstream archive and
   exact patch bytes (schema-derived tool names, bounded decoded-output observation, EOF
   finalization keeping cancellation and the worker join). A recipe mismatch refuses an old state
   dir; nothing is migrated.
2. Serving binds unauthenticated loopback only, owns its process group and records closure in the
   exclusive run dir; `--trace-generation` records bounded decoded chunks (no prompts or media).
   The host restores the declared Torch intra-op threads after warmup and refuses an unexpected
   inter-op value.
3. CUDA serving requires that UUID as the only visible device, capability 12.0, an `sm_120` binary,
   the exact Torch/CUDA versions and ≥ 16 GiB free; TF32 off; after warmup all LLM tensors must be
   FP32 on `cuda:0` and STT/TTS on CPU. No automatic device fallback.
4. CUDA startup shares one absolute `time.monotonic` deadline (≤ 150 s;
   `--startup-deadline-monotonic-s` from a coordinating owner) across imports, loading and both
   warmups; warmups stop admitting work 5 s early. Conversation streaming keeps its 10 s queue wait.
5. Upstream `speech-to-speech serve` can also be run in its own env (`speech-to-speech serve -h`;
   choose local models explicitly, the default recipe may call an external LLM API). CASCADE does
   not install or start it ([upstream setup](https://github.com/huggingface/speech-to-speech/blob/411399d34555b2169823a6eaeb7f8ff192db89db/README.md)).
   The Reachy Space allocator is not a generic endpoint and is never called.

### Waiting for a provider pipeline to be released

`provider_release_contract: "hf_pool"` (CLI `--provider-release-contract hf_pool`) binds the
Realtime `session.created.id` to the same authority's `/v1/pool` (`active`/`draining`/`stuck`/
`idle`). Disconnect then succeeds only after the unit is observed idle within one I/O deadline
(default 5 s, starting before socket close); a missing, changed, stuck or late observation keeps
closure failed and blocks a replacement session. Nothing is force-released or retried. The default
transport makes no pool claim ([CPU record](evidence/robot-modularity/hf-pool-reconnect-cpu-20261004.json):
reproduces the old immediate-reconnect HTTP 502).

## Verification

```bash
uv run --extra dev --extra conversation pytest tests/test_conversation_contracts.py \
  tests/test_conversation_protocol.py tests/test_conversation_input_origin.py \
  tests/test_conversation_frontend.py tests/test_conversation_split_service.py -q
```

The protocol suite opens owned ephemeral loopback ports and covers audio byte identity,
cancelled/duplicate/stale tools, stop/reset generation fencing, blocked workers, stop during
handshake and before blocked playback flush, reconnect isolation, concurrent admission and a real
CLI subprocess. The Node worklet test checks PCM saturation, endianness and chunking; playback tests
replay three native provider timing traces. CI enables the conversation extra.

## Voice episodes (simulated hand and MicroDuck)

| Episode | Result | Record |
| --- | --- | --- |
| VOICE-HAND02/05, LLM components | negative: no action dispatched (a rejected `function_name` call); later component reruns produced the right `index_flex` call | [attempts](evidence/robot-modularity/voice-hand-negative-20261004.json), [component](evidence/robot-modularity/voice-hand-llm-component-20261004.json), [EOF rerun](evidence/robot-modularity/voice-hand-llm-eof-20261004.json) |
| VOICE03/04 MicroDuck | VOICE04's real walk was **refuted** (heading drift −0.19063 rad past 0.08 rad); operator fallback stop, no reply | [recovery receipt](evidence/voice-reply-recovery-20261004/receipt.json), [guarded continuation](evidence/voice-reply-recovery-20261004/guarded-continuation.json), [owner patch](evidence/voice-reply-recovery-20261004/external-owner-final-drain.patch) |
| voice06/07 (installed provider) | real `index_flex` action and rest, but no reply PCM within the 30 s owner budget | [voice06](evidence/robot-modularity/voice-hand-installed-negative-20261004.json), [voice07](evidence/robot-modularity/voice-hand-cancellation-negative-20261004.json), installs [1](evidence/robot-modularity/private-speech-provider-install-20261004.json) [2](evidence/robot-modularity/private-speech-provider-cancel-install-20261004.json) |
| VOICE08/09/10 (CUDA) | startup failures (RSS watchdog, streamer `_queue.Empty`), then a reconnect HTTP 502 while the provider slot drained | [VOICE08/09](evidence/robot-modularity/voice-direct-cuda-negative-20261004.json), [VOICE10](evidence/robot-modularity/voice-startup-and-negative-20261004.json) |
| VOICE12 | failed the terminal-authority assertion (fixed: `close()` now revokes authority before teardown) | [record](evidence/robot-modularity/voice-terminal-authority-20261004.json) |

### Two complete native speech cases — VOICE13

The [closed VOICE13 record](evidence/robot-modularity/voice-complete-two-case-20261004.json) passes
the unchanged full auditor in both cases: synthetic speech entered the real Chromium microphone API,
the provider transcribed it, the model requested `index_flex`, the real tool route moved the
simulated LEAP hand and verified approach plus a 0.2 s post-ACK rest, and the tool result produced a
reply (186,624 PCM samples per case reached the browser and drained). The second case also exercised
operator stop during queued audio, flush, authority revocation, reset at the observed generation,
provider-slot release and a new conversation, within the original 30 s case limit (LLM FP32 on CUDA,
ASR/TTS on CPU, pool ready at 34.716 s of the 150 s window). Two cases in one run: not general
reliability, hardware audio or hand hardware. VOICE12 stays failed.
