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

The subsequent [EOF component rerun](evidence/robot-modularity/voice-hand-llm-eof-20261004.json)
used provider source `66b7233` with the same 581-token prompt, weights, generation
settings and recorded pre-response RNG state. Nine real streamer/queue/thread
regressions cover natural EOF, cancellation, consumer close, parser errors and
worker finalization; the related focused suite passed 43 tests. The real model
again emitted the same single `index_flex` call. Evaluated generation took
16.363 seconds; the interval from the last decoded chunk to response completion
was 0.007607 seconds, versus 10.002136 seconds in the retained earlier component.
Construction and two ordinary warmups took 53.153 seconds. The bounded trace,
source/assets and ordinary owned closure passed, with no dispatched tool.
This is one component rerun, without ASR, TTS, hand or browser. It does not
establish the complete voice chain, repeated response latency or its 30-second
hand-owner budget; VOICE-HAND02 and VOICE-HAND05 remain negative.

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
python3.12 scripts/conversation_provider.py prepare --state-dir ~/.cascade-speech-schema-eof-cancel
python3.12 scripts/conversation_provider.py verify --state-dir ~/.cascade-speech-schema-eof-cancel
python3.12 scripts/conversation_provider.py serve --state-dir ~/.cascade-speech-schema-eof-cancel \
  --run-dir runs/speech-provider-01 --port 18878 --timeout-s 900
```

Preparation downloads the pinned CPU dependencies, Whisper, Qwen and Kokoro.
The [source manifest](../configs/conversation/provider/manifest.json) binds the
upstream archive and exact patch bytes: schema-derived tool names, optional
bounded decoded-output observation, and EOF finalization that retains
cancellation and the worker join. Queue polling uses waits capped at 50 ms
while retaining the live response's original timeout; scheduling can delay
cancellation observation. Cleanup drain and join share five seconds. If the producer
remains alive, that handler rejects every subsequent generation before resetting
cancellation or reusing its queue; this does not make native model work interruptible.
Preparation checks every changed source file; serving checks the complete
effective source and the normal installation's origin and runtime files before
constructing models. A recipe mismatch refuses
an old state directory; no existing environment or weights are migrated.
Model selection, provider lifetime and robot deadlines remain separate.
After the upstream builder and warmups, the host restores the recipe's declared
Torch intra-op thread count and records the before/after values. It verifies
inter-op configuration without changing it after construction; an unexpected
value refuses serving. This configuration check does not establish response latency.

Serving binds unauthenticated loopback only, owns its child process group,
and records closure in the exclusive run directory. `--trace-generation`
optionally records private bounded decoded chunks after upstream warmup;
it does not record prompts or media, and chunks are not token boundaries.
Cancellation can produce interrupted spans. Lost events, observation errors or
in-flight spans make diagnostic accounting incomplete; a complete trace is
not proof of a tool result or physical action.

A [fresh installation](evidence/robot-modularity/private-speech-provider-install-20261004.json)
on source `586fdb6` completed with Python 3.12.13 and a new private environment.
All 116 installed provider runtime files matched the pinned candidate; 234
model/resource files and 132 dependency versions matched the retained recipe.
The owned scope closed ordinarily with no signals, forced cleanup or remaining
births. Preparation imported package classes and libraries but constructed no
speech handler or pipeline and ran no warmup, inference or service. The host's
configured thread count is distinct from the earlier LLM-only measurement;
effective serving threads must be observed after pipeline construction.

A separate [cancellation-package installation](evidence/robot-modularity/private-speech-provider-cancel-install-20261004.json)
on `106d485` prepared a new private environment with provider candidate
`eee2937`. All 116 normal-installed runtime files matched that source; the
234 model/resource hashes and 132 dependency versions matched the prior
installation. The scope closed with exit zero, no signals or force, and all
11 observed process births disappeared. This prepare/import step constructed
no speech handler or model and ran no inference, warmup or service. Host
thread restoration and voice episodes remain separately bound evidence.

The subsequent [voice06 episode](evidence/robot-modularity/voice-hand-installed-negative-20261004.json)
reached the real `hand.set_hand_posture(index_flex)` action from the Chromium
microphone. Its actual tool result was returned to the same provider origin:
12,944 native solves retained the approach, the 0.2 s simulated post-ACK rest
window, and quiet targets through the final solve. This verifies one simulated
speech-to-action execution, not repeatability or hardware behavior.

The complete voice case failed. The physical result returned 24.820 s after the
hand owner started; the real spoken continuation began, but no output PCM
arrived before the original 30 s owner budget (including browser closure
reserve). The second stop/reset/reconnect case was withheld. Provider shutdown
required owned SIGTERM followed by SIGKILL; all owned processes disappeared,
but that forced closure remains a failure. Its bounded trace retained a
16.561 s first generation and an open second span, without final footer/status.
The wire evidence is complete; provider diagnostic accounting is incomplete.
The configured intra-op count was two, while post-construction attestation
observed one. No deadline, physical gate, fixed reply or synthetic tool result
was changed to obtain this partial result. Full voice completion remains open.


A later [voice07 episode](evidence/robot-modularity/voice-hand-cancellation-negative-20261004.json)
used the separately installed cancellation fix and the host correction that
restores its declared two intra-op threads after pipeline construction. The
same spoken request produced one real hand action and 13,005 native solves;
the 0.2 s post-ACK rest window remained valid through the final solve. The
actual result returned 18.939 s after hand startup and reached its original
provider session before the spoken continuation was requested at 18.941 s.

This voice case also exhausted the unchanged 30 s owner budget with zero reply
PCM; stop/reset/reconnect remained unexecuted. Its first generation took
11.584 s. The second produced no decoded chunks and was explicitly cancelled
after 14.125 s, ending during cleanup at 33.068 s after hand startup. That
cleanup timestamp grants no late task credit. The provider closed with owned
SIGTERM and exit zero, without SIGKILL; both generation endings, footer and
status were retained with complete accounting and no dropped events. Complete
accounting does not make the interrupted response successful. The hand also
used owned TERM; all owned births disappeared. Cancellation handling and thread
restoration both changed from voice06, so this run does not isolate their
latency effects or establish repeatability. Earlier failures remain intact.

The separate `--profile cuda-llm-fp32` recipe prepares a **new** Linux x86_64 /
Python 3.12 state with hash-pinned official PyTorch 2.11.0 CUDA 13.0 wheels.
Model revisions, prompt handling, sampling, two CPU threads and the FP32 LLM
policy remain the same. Only the
LLM moves to CUDA; Whisper and Kokoro stay on CPU. Preparation imports classes
with CUDA hidden and does not construct models or establish GPU compatibility.

```bash
python3.12 scripts/conversation_provider.py prepare --profile cuda-llm-fp32 \
  --state-dir ~/.cascade-speech-cuda-fp32
python3.12 scripts/conversation_provider.py verify --profile cuda-llm-fp32 \
  --state-dir ~/.cascade-speech-cuda-fp32
python3.12 scripts/conversation_provider.py serve --profile cuda-llm-fp32 \
  --state-dir ~/.cascade-speech-cuda-fp32 --cuda-device-uuid "$GPU_UUID" \
  --run-dir runs/speech-cuda-01 --port 18878 --timeout-s 900
```

Set `GPU_UUID` to the complete `GPU-…` UUID reported by `nvidia-smi`.
Before constructing models, serving requires that UUID as the sole visible
logical device zero, capability 12.0, a compiled `sm_120` binary, the exact
Torch/CUDA versions and at least 16 GiB free GPU memory. It selects highest
FP32 matmul precision with TF32 disabled. After upstream warmup it inspects
the actual parameters and buffers: all LLM floating tensors must be FP32 on
`cuda:0`, and all STT/TTS tensors must remain on CPU. Admission failures refuse
serving; there is no automatic device fallback. The admission and runtime
receipts record these observations when their respective stages complete.
The two historical startup trials below did not establish GPU serving, response
latency or a spoken reply within a robot trial deadline. Provider lifetime and the existing
voice/actuator deadlines remain separate and unchanged.

The CUDA recipe loads the FP32 LLM directly with the explicit single-device map
`{"": "cuda:0"}`. Its normally installed provider patch enables this only for
the CUDA profile; the default CPU loading path stays unchanged. It does not
request automatic placement, CPU/disk offload or a second model-wide transfer.
Hash-pinned Accelerate 1.15.0 and psutil 7.2.2 wheels are additional CUDA-profile
dependencies. Readiness must attest the actual loader flag, pipeline device,
model placement and tensor dtypes after warmup.

The [installation and startup record](evidence/robot-modularity/voice-direct-cuda-negative-20261004.json)
binds host `064a2ce` and provider `c2302a6`. A fresh normal installation reproduced
all 116 runtime files, with exactly two declared changes from the prior provider;
234 model/resource hashes and 151 prior dependency versions stayed identical,
and only the two declared dependencies were added. Installation constructed no
models or service, kept CUDA hidden, and closed with exit zero and all 12
observed process births absent. CPU validation passed 82 host, 12 provider-stub,
11 installer and 59 voice-harness tests; these do not establish serving readiness.

Both GPU startup episodes remain failed. VOICE08 exceeded the unchanged 12 GiB
RSS watchdog with a sampled peak of 13,418,754,048 bytes before readiness.
VOICE09 used direct loading and reached the original LLM warmup, where
`TextIteratorStreamer` raised `_queue.Empty` under its unchanged 10 s queue wait.
The provider exited naturally with code 1; its sampled peak was 6,063,452,160
bytes. That lower observation in a failed startup is neither a bound on a
complete service nor an isolated causal comparison. The warmup producer lacked
an observed lifecycle, so a slow producer and an unobserved producer error
remain unresolved possibilities.

Neither episode reached postwarmup model attestation, browser cases or hand
construction. Both outer owned scopes closed ordinarily with exit 1; internal
provider cleanup recorded SIGTERM without SIGKILL, and all observed births
disappeared. Input and asset integrity held. The 150 s readiness deadline,
30 s case budget, model weights, FP32 policy and speech/physical gates were
preserved; successful CUDA speech service and complete voice execution remain open.

CUDA startup uses one absolute `time.monotonic` deadline, with at most 150
seconds remaining. A coordinating owner passes it through
`--startup-deadline-monotonic-s` before starting the provider scope; standalone
`serve` creates it before state verification. The serving child inherits the
same value. Imports, loading and both original dummy warmups consume that
allowance; neither a new warmup nor another chunk renews it. Warmup production
stops admitting work five seconds before the deadline, reserving that interval
for cancellation and join. Producer errors and uncertain closure fail startup.
Conversation streaming still uses its original ten-second queue wait, and
the voice trial's thirty-second deadline is unchanged.

After construction, the host verifies both actual warmup records, their
monotonic times, the inherited deadline and the actual conversational streamer
timeout before publishing model readiness. The records contain timing, counts
and error types, without generated text. Deadline checks reject late work;
they do not make native loading interruptible. An external owner still bounds
the process and readiness wait. The changed startup contract requires a fresh normal installation. The
CPU recipe and existing states are unchanged; previous failed trials remain
failed. The source-bound installation and VOICE10 observations below are separate
from the earlier startup results.

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

### Shared startup deadline: retained VOICE10 outcome

The [startup and VOICE10 record](evidence/robot-modularity/voice-startup-and-negative-20261004.json)
binds host `498930f`, provider `4306614` and a fresh normal installation.
Installation preserved all 234 assets and 153 dependency versions; exactly two
of the 116 provider runtime files changed. It constructed no models. In the
subsequent episode, both real CUDA warmups completed within the same original
150 s startup window, taking 10.633 s and 1.127 s. The postwarmup receipt observed
FP32 LLM tensors on CUDA, audio models on CPU and the unchanged ten-second
conversation streamer wait. This is one startup observation, not a latency or
repeatability guarantee.

**VOICE10 remains failed.** The first case's physical `index_flex` action and
post-ACK rest passed the frozen physics auditor, with 9,256 solves retained.
Its complete speech audit still rejects a missing browser response observation
for `capture.js`; the on-disk source hash does not replace that missing witness.
The second case retained 703 solves and no microphone input or dispatched
motion. Its reconnect returned HTTP 502 with a plain-text provider-unavailable
body while the provider's sole pipeline slot was still draining. The harness
parsed that error as JSON and raised `JSONDecodeError`. The trace and provider
log establish this closure boundary; they do not establish a model failure.
Playback was flushed and a software stop acknowledged, without second-case
physical-stop credit.

All four scopes closed and all 24 observed births disappeared. Owned SIGTERM
was recorded for the provider and second hand scope, with no forced kill; the
outer episode exited 1. The original 30 s case and 12 GiB provider RSS limits,
weights, prompts and physical gates were preserved. The first physical result
does not upgrade either the failed complete speech audit or the two-case result.
Earlier negatives and all original receipts remain intact.

### Two complete native speech cases — VOICE13

The [closed VOICE13 record](evidence/robot-modularity/voice-complete-two-case-20261004.json)
passes the unchanged full auditor in both bounded cases. Synthetic operator
speech entered the actual Chromium microphone API, the provider transcribed it,
and the model requested `index_flex`. The real tool route moved the native
simulated LEAP hand and verified approach plus a 0.2 s post-ACK rest window.
The actual tool result then produced a reply from the same input origin;
186,624 PCM samples per case reached the browser after rest and its queue drained.
The first case retained 9,642 solves and the second 9,433, for 19,075 total.

The second case also exercised operator stop after actual queued greeting audio,
playback flush, terminal authority revocation, reset with the observed generation,
provider-slot release and a new microphone-authorized conversation. The original
AudioWorklet request and resolved `addModule` promise were observed without an
extra fetch. These checks retain the original 30 s case limit and physical gates.
They do not turn a software stop ACK into independent physical-stop evidence.

This episode used service `3879453`, host `b7aaf64`, normally installed provider
`4306614` in state 05, and the unchanged hand model pin. The new installation
preserved all 116 provider runtime files, 153 dependency versions and 234 assets
from state 04. Its host binds the corrected absolute deadline comparison without
adding tolerance or renewing the deadline. Actual pool readiness occurred
34.716 s into the shared 150 s startup window. The LLM remained FP32 on CUDA and
ASR/TTS on CPU; the conversation streamer's ten-second wait stayed unchanged.
The 100 ms RSS watchdog observed a 6,448,181,248-byte group peak below its original
12 GiB budget. That sample is not a hard memory bound or performance guarantee.

All four scopes closed and all 25 observed births disappeared. Provider-owned
SIGTERM and child-group SIGTERM are explicit; exits were zero, with no forced
kill, remaining members or closure errors. **VOICE12 remains globally failed**
at its terminal-authority assertion; neither its auditor nor its evidence was
changed. VOICE13 applies the source correction in a separate new episode.

This is two cases in one run, not general conversation reliability. Input came
from a synthetic utterance, output was browser PCM scheduling/drain, and the
hand was simulated. Hardware microphone/speaker validation, hand hardware,
contact tasks, repeated coverage and the broader speech roadmap remain open.
