# Local speech-provider integration: 2 October 2026

Actual HF speech-to-speech inference reached CASCADE's conversation supervisor
over loopback WebSocket and returned generated PCM. The second recipe also
executed a readonly sensor-catalog request from speech. This is a measured
software integration with a synthetic sensor profile. It does not establish
reliable speech control: one of its three requested intents failed, despite
the model narrating success.

The [machine-readable receipt](../benchmark/results/conversation_native_20261002.json)
contains model and source hashes, all six case assessments, input/output hashes,
failed attempts and closure evidence. CASCADE was clean at
`6c793fad8ec81232e1695ddf716fa296987c4ca8`; the recorded runtime files remained
unchanged across both recipes. Raw logs, waveforms, wire events and isolated
environments are local artifacts under
`/home/johnny/Projects/demo/cascade-lab/CONVERSATION/provider-feasibility`.
They are identified by relative paths and SHA-256 in the receipt. Audio and
weights are not committed.

## What was connected

The native client used the real `ConversationSession`, `RealtimeWebSocket`,
`ConversationDomain`, composed `RobotRuntime` and `conversation_mock` profile.
It streamed its own synthetic speech as mono PCM16 little endian at 24 kHz,
in 40 ms chunks paced in real time. Each waveform had 0.5 seconds of preceding
silence and two seconds afterward. Concatenated wire audio matched those exact
bytes. Whisper consumed resampled 16 kHz audio. Output PCM matched the saved
24 kHz WAV files. No expected transcript or function call was injected into
the audio trials.

The provider was an owned subprocess on GPU1, bound only to loopback, with a
900-second watchdog, two Torch CPU threads and a Torch allocator ceiling below
12 GiB. TTS ran on CPU. The second recipe's one NVML observation was 4,308 MiB;
this is not a peak measurement. Each response had a 90-second observation
deadline, and tool intent age was bounded at 60 seconds. No motion tool or
external paid API was enabled. Receipt `ok` values refer to software operations;
all profiles and results retain `physical_acceptance=false`.

## Recipes and primary pins

Both recipes use HF speech-to-speech
[`411399d34555b2169823a6eaeb7f8ff192db89db`](https://github.com/huggingface/speech-to-speech/tree/411399d34555b2169823a6eaeb7f8ff192db89db)
with its local Transformers backend. The isolated provider environment contains
Torch/Torchaudio 2.11.0, Transformers 5.15.1, OpenAI protocol types 2.28.0 and
NLTK 3.10.3. The complete selected-environment freeze is hash-bound in the
receipt. Unused default backends were not installed; this is not validation of
every upstream optional dependency.

| Component | Recipe 1 | Recipe 2 |
|---|---|---|
| ASR | [Whisper tiny, `169d4a4`](https://huggingface.co/openai/whisper-tiny/tree/169d4a4341b33bc18d8881c4b69c2e104e1cc0af) | [Whisper base, `e37978b`](https://huggingface.co/openai/whisper-base/tree/e37978b90ca9030d5170a5c07aadb050351a65bb) |
| LLM | [SmolLM2 360M Instruct, `a10cc15`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/tree/a10cc1512eabd3dde888204e902eca88bddb4951) | [Qwen3 1.7B, `70d244c`](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e) |
| LLM generation | Greedy, maximum 48 tokens | Seed 42; sampling; temperature 0.7; model top-p 0.95/top-k 20; maximum 128 tokens |
| TTS and fixture voice | [MMS English, `c71de0f`](https://huggingface.co/facebook/mms-tts-eng/tree/c71de0fe7204c83f1c10820a7d696d0b450048ba) | Same model and fixture bytes |

Whisper, SmolLM2 and Qwen model cards declare Apache-2.0; MMS declares
CC-BY-NC-4.0. The MMS voice, generated fixtures and responses remain local
research artifacts. No personal voice recording or voice clone was used.

Qwen's [official model card](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/README.md)
describes tool use and recommends Qwen-Agent. This experiment instead exercises
HF speech-to-speech's own tool template/parser; upstream Qwen capability is not
evidence that this particular integration is reliable. Actual request prompts
disable thinking in the HF handler. The recipe's top-p differs from the card's
non-thinking recommendation and is recorded explicitly.

HF loads Silero through a moving `master` reference. The cached 29 Python/model
files were checked byte for byte against official
[`1e261b036686cd0017d500ee96acd1c4ba572a9d`](https://github.com/snakers4/silero-vad/tree/1e261b036686cd0017d500ee96acd1c4ba572a9d).
That audit binds this run's cache; a future default download is not implicitly
the same revision.

## Observed results

Recipe 1 retained three separate outcomes:

- Audio “Hello. Please say hello.” became “L, please say hello.” and returned
  “Hello.” with 0.512 seconds of generated audio.
- Audio “Please read the sensor.” became “Wee's read the censor.” The model
  emitted an invalid function name, rejected by the provider; no tool or audio
  result followed. This is a failed speech request.
- A separate **typed** explicit function request did execute
  `sensing.read_sensor(sensor_id="imu")` and return the synthetic IMU envelope.
  Its spoken follow-up was malformed and a later invalid function was rejected.
  This proves the readonly function path, not speech understanding.

Recipe 2 used the two original waveforms plus one new known synthetic phrase:

| Input phrase | Actual ASR | Actual tool/result | Output audio | Assessment |
|---|---|---|---|---|
| Hello. Please say hello. | Hello, please say hello. | No tool; greeting | 2.784 s | Requested greeting satisfied |
| Please read the sensor. | We is read the sensor. | Only `sensing.list_sensors`; no measurement read | 6.784 s | **Failed intent.** Model claimed data had been retrieved without a matching read |
| What sensors are available? | What sensors are available? | `sensing.list_sensors`, actual catalog containing synthetic IMU | 3.808 s | Requested catalog satisfied; spoken answer names IMU |

All three second-recipe sessions completed audio input, inference and output.
Two satisfied the requested intent. Three phrases are not an accuracy benchmark.
The model's unsupported sensor-reading narration is preserved in the receipt,
and must not be promoted to an operation or physical-success certificate.
Authoritative tool receipts remain separate from provider prose.

An onset audit checked the first recipe before replacing models. Full uncropped
CPU Whisper-tiny inference also misrecognized both phrases. The reported VAD
intervals and a diagnostic crop are retained, but that crop is not a tap of the
provider's internal STT tensor; native inference used CUDA float16 and the
diagnostic used CPU float32. Correct sample rates and exact wire bytes were
verified. These observations do not isolate VAD onset as the cause.

## Reproduction boundaries and retained failures

Keep the model server separate from the [lightweight gateway](CONVERSATION.md).
Select local pinned snapshots explicitly; the relevant native command is:

```bash
speech-to-speech serve --host 127.0.0.1 --port 18877 \
  --stt whisper --stt_model_name "$WHISPER_BASE_SNAPSHOT" --stt_device cuda \
  --stt_gen_max_new_tokens 64 \
  --llm_backend transformers --model_name "$QWEN_SNAPSHOT" --llm_device cuda \
  --llm_gen_max_new_tokens 128 --llm_gen_temperature 0.7 --llm_gen_do_sample True \
  --tts facebookMMS --facebook_mms_model_name "$MMS_SNAPSHOT" \
  --facebook_mms_device cpu --enable_live_transcription False \
  --smart_turn False --num_pipelines 1 --chat_size 6
```

These variables denote the exact local snapshots in the receipt. The retained
supervisor additionally selects the GPU UUID, sets seed/thread/allocator limits,
uses private caches and stops only its own process group. The invocation alone
does not establish those process limits.

Unmodified recipe 2 failed at startup because the upstream text
`TextIteratorStreamer` waits only one second. Its `queue.Empty` traceback and
exit 1 remain intact. The successful retry uses a **one-line local provider
patch**, changing that timeout from 1.0 to 10.0 seconds. Both file hashes and
`provider-streamer-timeout.patch` are recorded. It changes no CASCADE admission,
freshness, stop or intent deadline. An accidentally launched unpatched retry
after a patch-guard assertion was promptly stopped and retained separately.
Earlier missing-dependency/client-environment failures are retained as well.

Both successful provider processes exited 0 after requested shutdown, were
reaped, were independently absent afterward, and their loopback ports were
bindable. Every completed client reports no pending action and closed runtime
resources. The first and second providers ran for 323.763 and 123.605 seconds
respectively, including setup and idle time. Per-case wall times are not
realtime latency measurements on the shared GPU.

The referenced Reachy Space and its allocator were inspected read-only. The
Space was running as a static landing page; the allocator returned a session
allocation URL. No allocation POST, fabricated robot identity, public service
deployment or hosted inference test was performed. Browser audio hardware and
native-provider playback/backpressure remain separate validation work; the
native test drained received PCM without a speaker device.

An offline timestamp/length replay found a concrete frontend limitation at the
tested commit: its two-second scheduling limit would reject these provider
bursts (maximum implied backlogs 2.655, 6.434 and 3.627 seconds). That finding
is retained as `playback-timing.json`; inference success does not establish that
the then-current browser could play these responses. A bounded playback-queue
fix is a separate change and validation.

The subsequent playback fix retains a two-second scheduling horizon and adds a
15-second/512-object queue. Node executes the real browser scheduler against the
three timestamp/length traces: all samples drain, and controls verify overflow
rejection, flush and stale-context suppression. Generated sample bytes are used;
the fixture contains no MMS audio. This is CPU scheduling evidence, still not
a microphone/speaker-device test or another GPU inference run. See the
[playback receipt](../benchmark/results/conversation_playback_20261002.json).

## Native CPU provider and actual Chromium in one episode

A later isolated run bound CASCADE `4d8c350` to a pinned local Whisper-base /
Qwen3-1.7B / Kokoro-82M provider and Chromium 153. A generated `bm_fable` WAV
entered Chromium's fake microphone and actual AudioWorklet; the ASR returned
“Hello, please say hello back.” The model replied “Hello! How can I assist you
today?” Actual AudioContext playback consumed all 70,656 mono 24 kHz samples
(**2.944 s**), drained its queue, and the browser Stop latched the runtime.
No page errors occurred. Gateway/runtime and the provider closed; independent
checks found both provider/supervisor PIDs absent and its port bindable.

Total response time was **68.923 s**, including 61.580 s of CPU LLM inference.
This exceeds the unchanged maximum 60 s tool-intent deadline; no tool was requested
or executed in this greeting. It is evidence of the complete voice path, not
responsive interaction, CPU tool admission or physical microphone/speaker proof.
The provider log's 2.20 s audio field counts segmented input, not generated output;
source inspection confirms negotiated 24 kHz output. The previous failed
sensor-read intent is unchanged. No public deployment is claimed.

[Source/model/version/closure receipt](evidence/robot-modularity/conversation-native-chromium.json)
and [actual browser screenshot](evidence/robot-modularity/conversation-native-chromium.png).
Waveforms and authenticated browser trace remain local, indexed by their hashes.
