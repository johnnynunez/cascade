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
  deadline. A pending input's deadline is retained when its response arrives.
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
- Default barge-in invalidates pending speech/tool contexts and stops active or
  pending robot work. Initial idle speech does not latch a stop. Optional
  `speech_only` interruption is available only without motion authority.
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

## Provider contract and self-hosting

The target is the HF speech-to-speech GA WebSocket subset at revision
`411399d34555b2169823a6eaeb7f8ff192db89db`: `session.update`,
`input_audio_buffer.append`, input text, `response.create`, `response.cancel`,
output PCM events and function outputs. It does not claim complete OpenAI API
or WebRTC compatibility. The upstream handler includes `response_id` and
`output_index` on completed function arguments; the gateway binds both.
[Protocol](https://github.com/huggingface/speech-to-speech/blob/411399d34555b2169823a6eaeb7f8ff192db89db/src/speech_to_speech/api/openai_realtime/README.md),
[response handler](https://github.com/huggingface/speech-to-speech/blob/411399d34555b2169823a6eaeb7f8ff192db89db/src/speech_to_speech/api/openai_realtime/handlers/response.py).

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
  tests/test_conversation_protocol.py tests/test_conversation_frontend.py -q
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
