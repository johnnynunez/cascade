# Local conversation and robot tools

CASCADE has an optional browser/media gateway and a robot-agnostic conversation
supervisor. Audio travels through a configured Realtime provider; selected typed
tool intents enter the existing `RobotRuntime`. The conversation layer has no
joint writer, hardware identity, locomotion policy or implicit authority to reset
stops. It does not import or emulate the Reachy Mini SDK.

The implementation has been exercised through real loopback HTTP/WebSocket
servers, generated PCM/WAV audio, a synthetic sensor profile and the actual CLI.
A separate owned GPU run also exercised local HF speech-model inference and
readonly tools. Its small three-phrase recipe completed speech input/output for
all three phrases but satisfied only two requested intents; an incorrect
success narration remains a recorded failure. See the
[native speech evidence and recipes](CONVERSATION_NATIVE_20261002.md).
A separate Chromium 153 run on source `4d8c350` exercises the actual browser,
AudioWorklet, AudioContext, HTTP/WebSocket gateway and composed sensor runtime.
A fake microphone WAV produced ten nonzero PCM chunks; an explicit tool call
listed the synthetic IMU. A 6.784-second audio burst drained all 68 buffers,
with at most a two-second scheduling horizon; Stop cleared a subsequent burst,
latched the runtime, and explicit reset/reconnect/disconnect passed. See the
[Chromium receipt](evidence/robot-modularity/conversation-chromium.json) and
[screenshot](evidence/robot-modularity/conversation-chromium.png).
A subsequent same-episode CPU HF/Chromium greeting also completed actual
ASR, LLM, TTS and playback in 68.923 s; its [native browser receipt](
evidence/robot-modularity/conversation-native-chromium.json) preserves exact
source/model bindings. The 61.580 s LLM interval exceeds the 60 s tool-intent
deadline, so this does not admit CPU speech tools or practical interactive
latency. Hardware audio, general listening accuracy and physical task success
remain unvalidated.

The later [voice-to-MicroDuck recording](PROJECT_STATUS_20261003.md#rgb-d-geometry-speech-and-evaluation)
binds synthetic input speech, a real provider tool decision and a short native
motion in one episode. That recording has no spoken robot reply and does not
establish general walking or dialogue reliability. The [architecture diagram](ROBOT_MODULARITY.md)
places this supervisor above a single robot runtime; fleet conversation routing
remains separate work.

The later [bounded LEAP voice attempts](evidence/robot-modularity/voice-hand-negative-20261004.json)
remain negative. The named hand postures already had separate native motion and
retained-rest evidence, but neither voice attempt dispatched an action. The
first stopped after 154 zero-target solves because the test harness used an
unsupported file-opening argument. After that correction and two browser-only
regressions, the next attempt recorded 12,969 zero-target solves and exhausted
the original 30-second owner budget, with two seconds reserved for browser
closure. Its internal provider log contains the correct transcription, then a
rejected call to unavailable `function_name`; that identifier also appears in
the provider's generic prompt example. This correspondence does not establish
causation, and full generated text or per-token timings were not retained.

The provider withheld speculative speech-stop/transcription events from the
external wire; their absence does not mean recognition or generation never ran.
No tool call or reply PCM reached the gateway, and the stop/reset/reconnect case
was withheld after the first failure. All owned process groups closed without
force escalation or remaining births; the hand and provider used owned TERM15
in the second attempt. Source/model inputs stayed intact. Captured microphone
chunks retain hashes and byte counts, while only the original and derived WAVs
retain full PCM. These results establish neither voice-controlled hand motion
nor spoken acknowledgement of physical rest; prior motion and voice negatives
remain separate evidence.

A subsequent [isolated LLM component](evidence/robot-modularity/voice-hand-llm-component-20261004.json)
used the retained transcription and actual tool schema with a separate provider
candidate. Its prompt names the available functions instead of presenting the
unavailable `function_name` example. One real CPU generation produced exactly
`robot_tool_0(posture="index_flex")` and a successful terminal; it had no
dispatcher, supplied tool result, ASR, TTS or robot. The original two model
warmups and loading took 53.276 s; the evaluated response took 26.344 s. Its
47 decoded characters arrived by 16.341 s, followed by a measured 10.002 s gap
before generation finalization. The bounded private trace retained all decoded
chunks and reported token counts without dropped events, and the owned scope
closed normally. This new stochastic response does not replay the historical
RNG state, prove the cause of VOICE-HAND05, establish repeatability, or show that
speech, hand motion and spoken confirmation fit the unchanged 30-second owner.

## Run the gateway

From a checkout, install the optional transport and start a new private run:

```bash
uv sync --extra conversation
uv run cascade-conversation \
  --robot conversation_mock \
  --provider-url ws://127.0.0.1:8765/v1/realtime \
  --allow-tool sensing.list_sensors \
  --allow-tool sensing.read_sensor \
  --run-dir runs/conversation-01
```

The provider must already be running. The printed `http://127.0.0.1:8780/#…`
URL opens the local interface; the fragment is its temporary access credential.
Click **Connect** to establish the provider session, then use typed text or
**Start microphone**. The browser must support AudioWorklet and admit a 24 kHz
AudioContext. Unsupported sample rates fail visibly; the client does not play
speech at an assumed rate. PCM capture is mono, signed little-endian 16-bit,
960 samples per browser chunk. Remote provider connections require `wss`; local
`ws` endpoints must be loopback. `--token-env YOUR_PROVIDER_TOKEN_VARIABLE`
reads a bearer credential in the gateway process without returning it to the
browser.

`conversation_mock` declares only a synthetic IMU. Its reading is labelled
`synthetic` and `software_only`. Any configured composed robot profile can be
selected, but provider tools default to an empty allowlist. Motion additionally
requires `--allow-motion` and explicitly listed curated semantic skills, such as
`locomotion.walk_velocity` or `manipulation.pick_and_place`. The runtime's own
schemas, capabilities, ownership, collision/lease gates and outcome verification
still apply. Raw joint/pose setters and permission/reset controls cannot be
advertised as speech tools. A successful function call does not upgrade an
unverified physical postcondition.

Stop uses an authenticated HTTP route independent of provider completion,
playback flushing and the action worker. Disconnect also requests stop. After a
stop, use **Explicitly reset stop**, then connect a new session. Reset closes the
old conversation first, reads the current generation from `/api/status`, and
posts exactly `{"generation": <observed integer>}` to `/api/reset`. The runtime
checks this generation under its admission lock before resetting any domain.
A delayed request cannot clear a newer stop; a stop during reset still triggers
the existing relatch behavior. A delayed successful reply does not reactivate
the browser or replay a command. Runtime reset refusal is preserved.
Shutdown writes `closure.json`; a still-running action is reported as
pending rather than falsely reported as cancelled.

For an installed service, supply profiles explicitly and keep each process's
records separate. A versioned JSON configuration can live outside the checkout:

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

Run `cascade-conversation --config /absolute/path/service.json` with the
conversation extra installed. Paths in the JSON are relative to that file;
`profiles` contains the selected `robots/*.yaml` and `llm/*.yaml` configuration
files. Explicit CLI options override matching fields. Select exactly one of
`run_dir` and `run_root`; the latter creates a new private child on each start.
Unknown or duplicate fields and ambiguous permission values are rejected.
Optional `intent_timeout_s` and `execution_timeout_s` (also CLI options) select
the existing domain budgets: defaults 10/30 seconds, maxima 60/300 seconds.
The first budget starts at the input origin and is never renewed by a model
response, a tool continuation or a configured longer execution budget. Effective
values are recorded in `ready.json`; this configuration does not extend any
actuator's own admission deadline.
`token_env` names a credential environment variable; credentials are not expanded
into the configuration or readiness record. The loopback bind stays unchanged.

`ready.json` records when the listener was published, the process and profile
identity, and whether the runtime started stopped. It contains no access token
and does not establish provider readiness or a physical result; authenticated
`/api/status` reports the current session. With `start_stopped`, explicitly reset
the stop before connecting. A restart creates new authority and never replays
the previous session. It does not restore a physical state or clear an external
controller's stop.

Shutdown attempts provider, media, HTTP, conversation and runtime cleanup even
when another owner fails. `closure.json` retains partial cleanup and the process
exits nonzero on startup, teardown or receipt-persistence failure. A failed session
close blocks a replacement session in that gateway. External supervisors still
own their process deadlines; these receipts do not prove physical rest or
forcible termination of an outstanding action. CPU tests exercise real child
HTTP/WebSocket processes, explicit reset/stop, restart, and failed cleanup using
a protocol fixture and the synthetic sensor profile, without speech inference.

## Boundaries and lifecycle

- `MediaIO` separates capture, playback, flush and close. `QueueMediaIO` bounds
  event count, queued PCM bytes and capture age. Browser sequence numbers must
  advance exactly. Gateway receipt timestamps use its local monotonic clock;
  no browser/provider timestamp is treated as the actuator clock.
- Browser playback holds at most 15 seconds of audio and 512 audio objects,
  with only two seconds scheduled ahead. Faster-than-playback TTS bursts can
  drain without accumulating an unlimited queue. Overflow visibly stops the
  session. Flush discards queued and scheduled audio; local stop and reconnect
  also invalidate audio waiting for `AudioContext.resume()`.
  Microphone setup is likewise bound to the current session and local capture
  revision: a permission prompt completed after stop/disconnect immediately
  releases its tracks instead of restarting capture.
  Socket callbacks retain their session owner. A stopped pending connection
  cannot become active later, and a superseded reset cannot undo a newer stop.
- `RealtimeProvider` separates protocol transport. `RealtimeWebSocket` lazily
  imports `aiohttp`, negotiates PCM formats, limits message sizes and send/open
  time, and does not allocate endpoints or reconnect silently.
- `ConversationSession` binds session ID, configured robot ID, response origin,
  request ID, the connection's runtime cancellation generation and a local
  deadline. Audio authority starts at local receipt of `speech_started`,
  with an exact, unique speech item ID; a delayed `speech_stopped` only confirms
  that input. Text authority starts at local submission. Responses and tool
  continuations retain that original deadline, with no model-driven renewal.
  Explicit `response.create` requests carry a unique metadata nonce which the
  response must echo; unsolicited or mismatched responses have no authority.
  Argument-complete events are staged until `response.done` is `completed`.
  Cancelled, failed or incomplete responses do not execute staged calls.
- Requests are deduplicated at both session and dispatch boundaries. Session
  budgets are 256 responses, 256 calls, eight calls per response and 30 minutes;
  exhausting a budget requires a new session. A new response ID after an
  external stop/reset cannot obtain fresh authority in the old connection.
- `ConversationDomain` owns at most one in-flight action thread. Timeout asks
  the runtime to stop and reports uncertain delivery. A blocked worker prevents
  another session from claiming action ownership; Python threads are not
  reported as forcibly cancelled. Generation and deadline are checked under
  runtime admission and just before domain dispatch. Those checks are not a
  real-time guarantee; actuator owners retain their final backend checks.
- An allowlisted tool declared `effect="stop"` uses the coalesced priority stop
  path even while that action thread is busy; it does not replace the pending
  action's future or claim that worker finished. Session/robot identity,
  arguments, catalog, request deduplication, generation and local deadline still
  gate model-originated stop admission. Once admitted, stop delivery is not
  cancelled by later expiry. The independent operator stop has no speech token.
  Each admitted stop tool uses the same internal trace recorder as
  `RobotRuntime.execute`, after receiving the coalesced stop result. Its name,
  arguments, result, duration and context remain in the trace even when joining
  an operator's pending stop. Logging is separate from the priority stop task;
  a dedicated single worker keeps stop delivery available even if loggers fill
  asyncio's default executor. Each admitted tool retains its bounded recording
  obligation after caller cancellation, with an independent result snapshot.
  Pending/failed records prevent successful closure or a new session; closure
  joins the idle stop worker only after records and the latest stop delivery
  have drained. A newer concurrent operator stop keeps closure incomplete.
  Successful close is terminal and idempotent; a later stop request is rejected
  rather than presented with a cached ACK. The ACK remains
  `physical_stop_verified=False`.
  This dispatch contract does
  not make the speech session execute concurrent response streams: response
  ordering, interruption and the independent HTTP stop remain unchanged.
  An actually completed response may stage a motion followed by an adjacent
  stop. The stop is dispatched before either result is sent to a slow provider.
  Its correlated continuation can speak about the returned evidence while
  retaining the original input deadline and generation; it cannot renew motion
  authority after that stop. An operator interrupt still revokes the origin and
  suppresses its remaining audio. CPU protocol controls cover both paths; they
  are not model inference, physical motion or a recorded spoken robot reply.
- Tool output to the speech provider remains bounded to 32,768 UTF-8 bytes.
  Small results retain their exact JSON representation. For an oversized result
  recorded by the runtime trace, the speech view may omit only image attachments,
  `measured.samples` and `postcondition.evidence`. It wraps the retained fields
  under `result` and lists each omitted path, size and canonical JSON SHA256
  under `speech_transport`. Verdicts, errors, metrics, limits, model/epoch
  bindings, ACKs and task/receipt IDs are not rewritten. Digests reference the
  full trace; they do not prove a physical outcome. If unrecognized or retained
  fields still exceed the bound, or no trace is configured, the existing explicit
  transport error is returned. Projection runs outside the audio event loop,
  after any adjacent staged stop, and cancellation is checked again before send.
  It never changes the task ledger, command deadline or stop latch.
- Default barge-in invalidates pending speech/tool contexts and stops active or
  pending robot work. Initial idle speech does not latch a stop. Optional
  `speech_only` interruption is available only without motion authority.
  After an interruption makes automatic VAD response ownership ambiguous, the
  session revokes tool authority and asks the operator to disconnect/reconnect.
  In `speech_only` mode this revocation itself does not stop the robot or reset
  any latch. The UI mutes capture and disables input until reconnection.
- The gateway binds an explicit loopback IP. HTTP uses a random bearer token,
  exact browser Origin checking and a short-lived, single-use media ticket.
  Provider configuration is operator-owned, not accepted from browser requests.
  Abandoned tickets close their provider session. There is one active session;
  simultaneous HTTP requests cannot overwrite its owner. Access logging is
  disabled, and credentials are absent from run receipts. Public hosting and
  multi-user authentication are outside this interface.

No speech timing enters a robot's high-rate control loop. MicroDuck's locomotion
policy retains ownership of its head and leg actions; no speech head/mouth
controller is created by this feature.

`--robot microduck_conversation_mock` selects a base-only MicroDuck software
fixture for the same gateway. It exposes `locomotion.get_base_state` and the
existing bounded mobile tools, with no arm/gripper claims. The state is explicitly
`kinematic_mock`; successful transport cannot turn it into physical evidence.
For a passive first connection, allow only `locomotion.get_base_state`.

`microduck_conversation_native` composes the existing `microduck_isaac` candidate.
It requires the reviewed asset/policy/effective-model hashes, device identity,
support contract and verifier configuration documented in [MicroDuck](MICRODUCK.md).
Missing pins fail before connecting. The profile adds no gait admission and does
not use speech to circumvent the mobile verifier. Start with observation-only
tools; no speech-initiated native movement was validated in this increment.

The [2026-10-04 recovery receipt](evidence/voice-reply-recovery-20261004/receipt.json)
retains an attempted spoken-reply candidate on source `a8714b8`. Its guard
requires the real completed model response to contain the 30 mm walk and its
adjacent stop before motion, followed by independently observed rest and a
completed same-origin spoken reply. It preserves the original 60 s intent and
30 s execution limits. Protocol tests exercise that narration authority and
operator revocation; they do not establish a native spoken reply.

A separate zero-command probe recorded 800 solves, 200 policy commits and 41
frames with ordinary closure. Its original harness result remains failed because
an RGB-D diagnostic expected 80 rows; a separate audit checked all 800 against
the frozen budget and admitted only the model/recording identity. The speech
candidate first refused a changed foreign GPU baseline, then reached the
unchanged 65 s startup deadline before the microphone or any task response.
Its forced native closure remains failed; a later audit found all owned births
gone without additional signals. A separate
[external-owner correction](evidence/voice-reply-recovery-20261004/external-owner-final-drain.patch)
waits for unit, members and observed births together, with seven CPU ownership
and drain cases passing. The historical motion
recording still has no spoken robot reply. A retained detailed Kit log narrows
the startup failure to RtPso asynchronous compilation waits, continuing through
84.9 s around viewport initialization/destruction. It does not establish the
underlying cause. The 422 copied cache files were CUDA, Warp and XDG data;
writable RTX/PSO caches still used shared Kit paths and were not frozen or
audited. The subsequent isolation change selects both writable shader cache
settings before bootstrap and binds the policy to a new model/episode. The
subsequent [private-cache probe](MICRODUCK.md#running-the-candidate) closed with
800 solves and a separately admitted cache seed and model identity.

The [guarded continuation](evidence/voice-reply-recovery-20261004/guarded-continuation.json)
retains four failed episodes. The first two stopped before microphone input
(missing browser dependency, then missing curated `walk_distance` admission).
VOICE03 reached the real provider, which returned only the walk call; the
unchanged two-call guard refused motion. Two new CPU generations then isolated
the trial instructions: the original instructions produced one call, while an
explicit same-response walk/stop suffix produced both. Raw text and parser
output agree in those two cases; the historical raw output was not retained.

VOICE04 used that suffix only for the trial, with its exact `session.updated`
acknowledgement before microphone input. The real model returned both calls
within the original 60 s intent. The walk executed but its physical outcome was
refuted: heading drift crossed the unchanged 0.08 rad limit after command
completion and reached −0.19063 rad. The guard therefore invoked operator
fallback instead of forwarding the staged model stop or claiming success.
Its stop acknowledgement remained physically unverified; no reply PCM was
produced. The 47.482 s recording also missed the 60–90 s gate. All three owned
scopes closed without force or remaining births, and both ports were free;
the failed episode stays failed. Source, model, seed, freshness, 30 s execution
and physical verification limits were unchanged.

## Provider contract and self-hosting

The target is the HF speech-to-speech GA WebSocket subset at revision
`411399d34555b2169823a6eaeb7f8ff192db89db`: `session.update`,
`input_audio_buffer.append`, input text, `response.create`, `response.cancel`,
output PCM events and function outputs. It does not claim complete OpenAI API
or WebRTC compatibility. The upstream handler includes `response_id` and
`output_index` on completed function arguments; the gateway binds both.
[Protocol](https://github.com/huggingface/speech-to-speech/blob/411399d34555b2169823a6eaeb7f8ff192db89db/src/speech_to_speech/api/openai_realtime/README.md),
[response handler](https://github.com/huggingface/speech-to-speech/blob/411399d34555b2169823a6eaeb7f8ff192db89db/src/speech_to_speech/api/openai_realtime/handlers/response.py).

The pinned [audio handler](https://github.com/huggingface/speech-to-speech/blob/411399d34555b2169823a6eaeb7f8ff192db89db/src/speech_to_speech/api/openai_realtime/handlers/audio.py)
may defer `speech_stopped` until model output. Its automatic response lacks an
input-item correlation field, and its runtime does not honor
`turn_detection.create_response=False`. CASCADE therefore does not pretend
manual VAD disables those responses: first uninterrupted VAD input is admitted,
while ambiguous responses after interruption require reconnection. The local
onset deadline does not establish microphone capture age: a provider could
itself delay `speech_started`; no provider-to-capture clock mapping is claimed.

The optional CPU provider has a reproducible Linux x86_64 recipe in
[`scripts/conversation_provider.py`](../scripts/conversation_provider.py).
It creates its own Python environment and model copies; it does not install
speech dependencies in Cascade's environment. With Python 3.12+, `uv` and `git`
available, use a **new**, private state directory:

```bash
python3.12 scripts/conversation_provider.py prepare --state-dir ~/.cascade-speech-schema-eof
python3.12 scripts/conversation_provider.py verify --state-dir ~/.cascade-speech-schema-eof
python3.12 scripts/conversation_provider.py serve --state-dir ~/.cascade-speech-schema-eof \
  --run-dir runs/speech-provider-01 --port 18878 --timeout-s 900
```

Preparation downloads the pinned CPU dependencies, Whisper, Qwen and Kokoro.
The [source manifest](../configs/conversation/provider/manifest.json) binds the
upstream archive and exact patch bytes: schema-derived tool names, optional
bounded decoded-output observation, and EOF finalization that retains
cancellation and the worker join. Preparation checks every changed source file;
serving checks the complete effective source and the normal installation's
origin and runtime files before constructing models. A recipe mismatch refuses
an old state directory; no existing environment or weights are migrated.
Model selection, provider lifetime and robot deadlines remain separate.

Serving binds unauthenticated loopback only, owns its child process group,
and records closure in the exclusive run directory. `--trace-generation`
optionally records private bounded decoded chunks after upstream warmup;
it does not record prompts or media, and chunks are not token boundaries.
Cancellation can produce interrupted spans. Lost events, observation errors or
in-flight spans make diagnostic accounting incomplete; a complete trace is
not proof of a tool result or physical action. Fresh installation and the full
voice-to-hand chain on this packaged source still require separate validation.

Run the speech stack in a separate environment. Its `speech-to-speech serve`
command exposes `/v1/realtime` and supports selecting STT, LLM and TTS backends.
Use `speech-to-speech serve -h` at the pinned revision and explicitly choose
local models/endpoints if no paid service is intended: the default recipe can
use an external LLM API. CASCADE does not install its model stack or start its
server. Model downloads, licenses and hardware requirements remain separate
from the lightweight gateway installation.
[Upstream setup and backend selectors](https://github.com/huggingface/speech-to-speech/blob/411399d34555b2169823a6eaeb7f8ff192db89db/README.md).

The referenced Reachy Space is an installation/source landing page. Its app's
allocator path is not a generic free endpoint, and this implementation does not
call it or present a fabricated Reachy identity. The earlier
[conversation research](MICRODUCK_CONVERSATION_DESIGN.md) preserves the exact
application, SDK and allocator pins and the distinction between hosting and
inference.

## Verification

```bash
uv sync --extra dev --extra conversation
uv run pytest tests/test_conversation_contracts.py \
  tests/test_conversation_protocol.py tests/test_conversation_input_origin.py \
  tests/test_conversation_frontend.py -q
```

The protocol suite opens owned ephemeral loopback ports. It verifies audio byte
identity, cancelled/duplicate/stale tools, stop/reset generation fencing,
blocked workers, stop during provider handshake, stop before blocked playback
flush, reconnect isolation, concurrent session admission, and a real CLI
subprocess using the synthetic sensor profile. The Node worklet test verifies
PCM saturation, endianness and chunking without claiming microphone validation.
Playback tests replay the three native provider timing traces with generated
sample bytes: the previous two-second limit rejects each burst, while the
bounded queue preserves every sample. Overflow, flush, stale generations and
pending audio-context resumes have separate controls.
The regular three-platform CI enables the optional conversation extra; the
minimal-install job remains unchanged.

The input-origin regressions preserve selected events and hashes from two native
Kokoro greeting traces. A session-local clock replays their observed spacing;
the actual domain/runtime rejects a separately injected expired readonly tool.
The archived pre-fix source instead executes that injected tool in both cases.
Those original native greetings requested no tool and establish no native tool
admission result. Their speech input/output evidence remains unchanged.
