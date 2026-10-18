# Nemotron 3.5 Bolt as the default cascade brain (2026-10-18)

**Decision (owner, 2026-10-18): NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB is the default brain for both the agent
(OpenClaw) and cascade's perception. Qwen3.8-27B + DFlash2 stays selectable (`--brain qwen`,
`scripts/serve_qwen_vllm.sh`); every `--brain qwen` path is byte-identical to main before the switch (`91bac52`),
its vLLM 0.31.0 included (section 11).**

**Serving decision (this report): Bolt ships on llama.cpp v0.7.0** (`scripts/serve_nemotron_bolt_llamacpp.sh`: GGUF +
vision projector + LiLiCorr DFlash drafter, two slots). Every vLLM build tried for Bolt faulted the GPU (Xid 31)
beside Isaac Sim Newton under agent + perception load, the card's own nightly container included (section 9).
llama.cpp ran the full e2e pick through the OpenClaw Control UI twice and a 30-minute soak beside Isaac Newton with
0 Xid and 0 brain errors (sections 7, 10). The vLLM route stays as an opt-in script with the evidence in its header.

Versions shipped:

| Brain | Server | Status |
|---|---|---|
| **Bolt (default)** | llama.cpp v0.7.0 (`620d2225`), Bolt GGUF + mmproj + LiLiCorr GGUF drafter, 2 slots, weights read at start (`--load-mode none`) | e2e ×2 + 30-min soak beside Isaac Newton: 0 Xid (sections 7, 10) |
| Bolt, vLLM (opt-in) | the model card's container: vLLM nightly `77154ca6` (0.32.0rc1) by digest + b12x 1.5.1 by hash, no drafter | faulted beside Isaac; warns at every start (section 9) |
| Qwen (`--brain qwen`), vLLM | vLLM 0.31.0 (as before #375; 0.32.0 cu130 measured 3-4% slower) | section 11 |
| Qwen, llama.cpp | llama.cpp v0.7.0 (pinned in #375; before: `4695f001`) | not re-measured |

## 1. What the default switch changes

| Piece | Before (main) | Now |
|---|---|---|
| Brain server | `scripts/serve_qwen_vllm.sh` (Qwen3.8 NVFP4 + DFlash2) | `scripts/serve_nemotron_bolt_llamacpp.sh` (llama.cpp v0.7.0, pinned GGUFs); `serve_nemotron_bolt_vllm.sh` opt-in |
| Served id | `qwen3.8-27b` | `nemotron-3.5-bolt` |
| OpenShell sandbox model / label | `cascade/qwen3.8-27b`, "Qwen3.8-27B (cascade brain)" (hard-coded) | `cascade/nemotron-3.5-bolt`, "Nemotron 3.5 Bolt 74B-A9B (cascade brain)"; the label follows the served id |
| Perception default (`configs/demo.yaml`, `demo.BRAIN_MODEL`) | `qwen3.8-27b` | `nemotron-3.5-bolt` |
| `launch.sh --brain` | `auto\|keep\|cosmos\|cosmos-sglang\|qwen` | adds `nemotron-bolt`; `auto` resolves a local Cosmos, else the brain on :8080 (Bolt when it serves `nemotron-3.5-bolt`, else Qwen as before) |
| cascade agent profile | `local_qwen` | `local_nemotron_bolt` (dev runners default to it) |

## 2. Serving contracts

**llama.cpp (default), `scripts/serve_nemotron_bolt_llamacpp.sh`.** Same contract as `serve_qwen_llamacpp.sh`: 127.0.0.1,
PORT 8080, a 0600 key FILE handed to `llama-server --api-key-file` (argv carries the path, never the key),
`--setup-only` / `--check`. Pinned: llama.cpp v0.7.0 (`620d2225`: Nemotron 3.5 Bolt #30520, LiLiCorr speculative decoding
#30506), built for the GPU's CUDA architecture in `$REPO/.llama.cpp` with a build receipt; the model, projector, drafter
and licence by sha256 (`deploy/brev/profiles/nemotron-3.5-bolt-24gb.json`, 24.8 GB). Flags: the card's recipe (q8_0 KV,
flash attention, `--jinja`, thinking off), `--parallel 2` with 64k tokens per slot, `-ngl 999`, `--load-mode none`
(v0.7.0's `--no-mmap`: weights read once at start), `SPEC=dflash` (default: LiLiCorr, 7 drafted tokens) or `SPEC=none`.
27.3 GiB of GPU memory with the drafter (vLLM: 47.5 GiB at GPU_UTIL 0.45); ready in 26 s from disk, 5 s from the
page cache.

**vLLM (opt-in), `scripts/serve_nemotron_bolt_vllm.sh`.** Mirrors `serve_qwen_vllm.sh`: 127.0.0.1 by default, PORT 8080,
a 0600 key file read into `VLLM_API_KEY` and handed to docker by name (never argv), no network bind without a key,
`--setup-only` / `--dry-run`; logs a WARNING (opt-in, Xid beside Isaac) at every start.

- Image: `vllm/vllm-openai@sha256:a7deff2c…` (the card's tag `nightly-77154ca6…`, vLLM 0.32.0rc1) plus `b12x==1.5.1`
  installed with `--require-hashes` into a local image (`cascade/nemotron-bolt-vllm:77154ca6-b12x1.5.1`), so serving
  needs no network. The trial installed b12x at every start.
- Weights: `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB@c6cc2e8f` (and, with `SPEC=dflash`, the LiLiCorr drafter
  `@abd47806`), from the host's HF cache, mounted read-only, `HF_HUB_OFFLINE=1`. The card's
  `ultra_v3_reasoning_parser.py` is checked by sha256 (`14e03de5…`) before serving.
- Flags: as the trial (`v5-bolt/serve_bolt.sh`): 64k context, `--max-num-seqs 4`, GPU_UTIL 0.45, fp8 KV, b12x
  linear/MoE, FlashInfer attention + Mamba, qwen3_coder tool parser, ultra_v3 reasoning parser, thinking off by default.
- Changed from the trial: no drafter by default (`SPEC=none`; `SPEC=dflash` restores LiLiCorr ×7, section 4); no image
  cap (`--limit-mm-per-prompt {"video":0}`, section 7b); `--block-size 64 --prefix-match-unit 64` (section 4); Docker
  CDI GPU; the container runs as the calling user; MemAvailable watched during startup (killed below `MEM_FLOOR_GB`,
  default 3); a stopping unit stops the container.

## 3. Thinking off for every client

The card's ultra_v3 parser reads only the request's `chat_template_kwargs`: a non-streaming request without
`{"enable_thinking": false}` came back with `content: null` and its tool calls unparsed in `reasoning_content`
(trial issue 2). Clients of the brain and what they send:

| Client | Sends `enable_thinking: false` |
|---|---|
| OpenClaw in the OpenShell sandbox | yes, for Bolt: `agents.defaults.models["cascade/nemotron-3.5-bolt"].params` (render-config.mjs) |
| OpenClaw on the host (Spark kitchen profile) | yes, already on main for every model (launch.sh) |
| OpenClaw on the host (other profiles) | yes, for Bolt (launch.sh `brain_local`) |
| cascade perception (`BrainDetector`), `VLMGrounder` | yes, already on main |
| `demo_proof.py --probe-native` | yes, already on main |
| cascade's own agent (`--llm local_nemotron_bolt`) | yes: new `extra_body` profile field |
| kitchen / franka OpenClaw plugins, cloud runtime judge | yes, already on main |

## 4. Prefix caching on the vLLM route, and why it runs without the drafter

**Root cause of the trial's 0 % hit rate.** The served model has 11 KV-cache groups: the target's attention/Mamba
groups (block forced up to 4,496 tokens so an attention page holds a Mamba state) and the LiLiCorr drafter's 5
sliding-window layers (block 64). vLLM's hybrid manager hashes every gcd(4496, 64) = 16 tokens, but allows a hit
below the LCM block only if every prefix-cacheable group supports fine-grained lookup; `SlidingWindowManager` does not.
So hits could only land on multiples of **lcm = 17,984 tokens** (server log: `kv cache group sizes (4496 ×10, 64)`,
`kv lcm block sizes 17984`). The agent's prompts are 13.6k-18.2k tokens: no hit was ever possible.

**Fix 1.** `--block-size 64 --prefix-match-unit 64`: vLLM rounds the attention block up to a multiple of 64 that holds
a Mamba page (4,544), and every group, the drafter's included, uses it: `kv lcm block sizes 4544`. With the drafter,
hits land every 4,544 tokens.

**Fix 2.** Without the drafter the sliding-window group is gone and a follow-up call hits every 64 tokens up to the
previous prompt's tail (13.5k-16.6k of 15-18k tokens).

**Measured** on a realistic agent replay (`tools/replay_agent.py`): the trial's recorded OpenClaw session (proxy
bodies 41, 43, 44, 47, 49, 51: 13.6k → 18.2k prompt tokens, one camera image, 49 tool schemas, each call extending
the previous one exactly as OpenClaw sent it), replayed twice with fresh session ids, 32 output tokens,
temperature 0, one request at a time; hits from vLLM's `/metrics` counters. Raw rows: `results/*.jsonl`.

TTFT in seconds, replay 1 / 2, (prefix-hit tokens), the card's container beside Isaac Sim 6.2 Newton (kitchen loaded,
bridge `ping` engine newton):

| call (prompt tokens) | trial config (drafter, 17,984 block) | drafter, block 64 | no drafter, block 64 (vLLM route default) |
|---|---|---|---|
| 1 (13.6k) | 13.35 / 13.40 (0) | 36.29 cold JIT / 14.08 (0) | 18.78 / 16.86 (0) |
| 2 (15.0k) | 14.42 / 14.42 (0) | 16.16 / 10.58 (4,544) | 2.48 / 1.95 (13,568) |
| 3 (15.0k) | 14.95 / 14.95 (0) | 7.04 / 6.51 (9,088) | 3.45 / 2.83 (13,568) |
| 4 (16.5k) | 16.08 / 15.95 (0) | 10.56 / 7.63 (9,088) | 1.87 / 2.06 (15,040) |
| 5 (16.6k) | 16.44 / 16.35 (0) | 8.75 / 7.96 (9,088) | 1.33 / 1.56 (16,448) |
| 6 (18.2k) | 17.87 / 17.77 (0) | 10.06 / 9.52 (9,088) | 2.48 / 2.39 (16,576) |

(`before-isaac`, `after-isaac`, `ctr-nospec-isaac`; the last measured with Isaac's sim loop running, right after
run C of section 7.) Alone, trial config: 8.5-11.8 s on every call, 0 hits (`before-alone`).

**The trade.** The drafter triples decode (beside Isaac: 47 tok/s with LiLiCorr ×7 against 11-17 tok/s without, on
vLLM 0.32.0, `decode-*`); the agent's calls are mostly short (tool calls of 18-60 tokens; the e2e's text replies
59-190 tokens), and every follow-up call saves 5-13 s of prefill without it. The break-even is ~100 output tokens per
call; in run C (section 7) a turn's first robot tool came 2.5-6.3 s after the message, against 18-22 s in the trial.
`SPEC=dflash` keeps the drafter for long-answer use.

**Limits.** A new OpenClaw session never hits on its first call: the session id sits in the system prompt's Runtime
line, early in the prompt. Every first call of a chat prefills 13.6k tokens: 17-19 s beside Newton, 8 s alone.

## 5. demo_proof's readiness and world checks

**Readiness root cause.** OpenClaw 2026.9.3's agent system prompt has a "Silent Replies" rule: "Nothing to say:
entire reply exactly NO_REPLY" (`system-prompt-params`, `silentReplyPromptMode` "generic" for
`openclaw agent --message`; only a message-tool-only delivery or a session-stable conversation context turns it off,
neither is a config switch). OpenClaw then drops that token, so the payload is empty. The launcher asks "Reply with
exactly OK and nothing else. Do not call tools."; Bolt sometimes reads that as nothing to say.

**Fix.** For Bolt only (`cascade/nemotron-3.5-bolt`), the question says a visible reply is required and NO_REPLY or an
empty reply fails it; the judge (`brain_said_ok`) is unchanged and still refuses NO_REPLY. Every other brain keeps main's
question, so the Qwen check is not weakened.

**Measured** (`tools/readiness_probe.py`: the trial's real readiness request, proxy body 5, with OpenClaw's full system
prompt and 49 tools, streamed as OpenClaw sends it, server-default sampling):

| question | NO_REPLY / runs |
|---|---|
| main's, no kwargs (as in the trial) | trial: 1 / 1; here: 0 / 14 |
| main's, with `enable_thinking: false` (as OpenClaw now sends) | 1 / 14 |
| Bolt's | 0 / 14 |

**World check.** The robot proof's step 2 asks "What is the session status?" and requires the turn to execute
`world_state`. In the launch beside Newton (run B, 2026-10-18) Bolt answered from the tool's description without
calling it: no `toolSummary`, NOT READY. For Bolt only, steps 2 and 4 name the tool ("… Read it with the world_state
tool."); the judge (the executed tools) is unchanged, and `tests/test_demo_proof.py` proves a Bolt turn that skips
the tool still fails. The next two launches (run C) were READY.

## 6. The trial (copied from `~/Videos/cascade/cascade-demo-2026-10-18-newton-bolt.REPORT.md`)

Sim-only, Isaac Sim 6.2 NEWTON, OpenClaw in the OpenShell `sim` sandbox, Bolt the only brain (agent + perception):

- 3/3 orders done; placement **VERIFIED** (`confirmed`, channel physics, engine newton, 15 observed samples, 13 final,
  4.7 mm; Isaac truth 4.6 mm); reset restored all props; **0 tool-call errors** (7 tool calls + 7 text replies).
- Every Bolt LLM call took **17-25 s to its first token** (prompts of 13.6k-18.2k tokens); agent LLM first token per call
  **17.1, 17.8 / 19.4, 20.8 / 21.3, 25.2 s**; prefix cache hit rate **0 %**.
- Send → first robot tool: **18 / 20 / 22 s** for Bolt against **3 / 6 / 4 s** for Qwen.
- Turns: **46 / 186 / 74 s** against **39 / 154 / 39 s** for Qwen.
- Perception in the pick: brain names + boxes the cube **2.81 s**; SAM2.1-L fresh-frame mask / frame1 age (limit 5 s)
  **0.93 s / 0.475 s**; GraspGen-X 0.80 s.
- Memory: 53.0 GiB freed when Bolt stopped; EngineCore 47.5 GiB on the GPU; weights 24.09 GiB; fp8 KV 20.31 GiB
  (649,870 tokens) at util 0.45. Cold start 567 s (first-ever). LiLiCorr mean acceptance length 4.91.
- Perception probe: REFER "green square" returned the green cube's box (Bolt confused the two); the pick did not depend
  on it (the destination is a configured zone).

## 7. End to end beside Isaac Sim 6.2 NEWTON

### 7a. Run D: the shipped default (llama.cpp v0.7.0), 2026-10-18 17:36-18:01 CEST

Isaac Sim 6.2 **NEWTON** (bridge ping: `engine newton`, scene `paai-cocina-asier-v1`, `results/e2e/bridge-ping.json`),
GraspGen-X, `launch.sh --brain nemotron-bolt` with a private OpenShell instance, the brain from this branch's
`serve_nemotron_bolt_llamacpp.sh` (llama.cpp v0.7.0) serving agent and perception through a timing proxy. The messages
were typed into the **original OpenClaw Control UI** in Chromium with the **cascade camera extension** attached (panel
"Physical Agentic AI", live Worktop camera, model label "Nemotron 3.5 Bolt 74B-A9B (cascade brain)"); every robot action
was the agent's choice. The same four messages ran twice in fresh chats, once per drafter setting (the brain restarted
behind the proxy between them).

- Launch: `[launch] READY sim=isaac arm=isaac_kitchen_gpu cameras=isaac,isaac_side,isaac_proof`: readiness,
  world_state, a physics-confirmed pick (pink cube → green square, `confirmed`, physics, newton) and a reset.

| message | the agent did | turn, drafter on / off | send → first robot tool, on / off |
|---|---|---|---|
| "Reset the scene." | `reset_scene`; all props back (physics) | 74.0 / 72.5 s | 36.8 / 32.6 s (a new chat's first call) |
| "What do you see on the table?" | `describe_scene`; named the tray/box, green square, both cubes, lemon, orange, can, plate | 24.0 / 22.0 s | 5.4 / 4.0 s |
| "Put the green cube in the green square." | `pick_and_place(green cube, green square)`: 1 grasp, 1 place, `postcondition confirmed`, channel physics, engine newton, 15 observed samples; robot time 141.7 / 143.6 s | 162.0 / 163.5 s | 5.3 / 8.1 s |
| "Reset the scene." | `reset_scene`; all props back | 39.0 / 41.0 s | 2.8 / 2.9 s |

- Isaac truth (`results/e2e/truth-*.json`): green cube at (0.24, 0.01) → **(0.144, -0.266) / (0.142, -0.266), in the
  green square** → (0.24, 0.01) after each reset. The agent's replies matched the tool results
  (`results/e2e/openclaw-ui-transcript.txt`).
- Agent calls: 8 per run, 0 errors; perception calls 9 / 7, 0 errors (`results/e2e/brain-calls.jsonl`).
- Memory: lowest MemAvailable 56 GB during the e2e (Isaac + GraspGen-X + the robot server + llama.cpp).
- 0 Xid. The 30-minute soak then ran on the default (section 10).

### 7b. Run C: the vLLM route (the card's nightly container), 2026-10-18 15:43-16:21 CEST

Isaac Sim 6.2 **NEWTON** (bridge ping: `engine newton`, scene `paai-cocina-asier-v1`),
GraspGen-X, `launch.sh` with the OpenShell `sim` sandbox, the brain from this branch's `serve_nemotron_bolt_vllm.sh`
defaults (container, no drafter, then the default) serving agent and perception. The three messages were typed into the **original
OpenClaw Control UI** in Chromium with the **cascade camera extension** attached (panel "Physical Agentic AI", live
Worktop camera, model label "Nemotron 3.5 Bolt 74B-A9B (cascade brain)"); every robot action was the agent's choice.

- Launch: `[launch] READY sim=isaac arm=isaac_kitchen_gpu`: readiness, world_state, a physics-confirmed pick and a reset.
- Turns (a reset first, to start from the spawn layout):

| message | the agent did | turn | send → first robot tool |
|---|---|---|---|
| "Reset the scene." | `reset_scene`; all 5 props within 0.02 m (physics) | 60 s | 18.4 s (a new chat's first call) |
| "What do you see on the table?" | `describe_scene`; named the green square, box, both cubes, orange, lemon, can, plate | 25 s | 2.6 s |
| "Put the green cube in the green square." | `pick_and_place(green cube, green square)`: 1 grasp, 1 place, `postcondition confirmed`, channel physics, engine newton, 14 final samples; robot time 138 s | 154 s | 6.3 s |
| "Reset the scene." | `reset_scene`; all props back | 36 s | 2.5 s |

- Isaac truth (`results/e2e-vllm-runC/truth-*.json`): green cube at (0.24, 0.01) → **(0.139, -0.272), in the green square** →
  (0.24, 0.01) after the reset.
- Agent calls: 8; first-token latency 17.6 s (the new chat's first call), then 1.7-4.1 s; brain errors 0.
- 0 Xid during the e2e.

**Found and fixed on the way: one image per prompt broke the agent.** In the first attempt (run C, turn 2) the agent
answered "What do you see?" with `get_observation` + `describe_scene`. OpenClaw puts a tool's images in the
conversation and keeps them (the model is declared `input: ["text", "image"]`), so the next request carried two
camera frames; the card's `--limit-mm-per-prompt {"video":0,"image":1}` made vLLM refuse it (HTTP 400) and every
later call of that chat ("LLM request failed: provider rejected the request schema or tool payload"). The image cap is
now vLLM's default (999), as the Qwen brain has; the memory profile is unchanged (`profiled with 1 image items of the
maximum feature size`; KV 26.36 → 26.78 GiB) because one maximum-size image already fills the encoder budget.

**The soak after it faulted.** `soak_load.sh` (the agent replay, the decode probe, then mixed replays: two agent
sessions + perception, two camera inventories and a REFER) beside Isaac, right after this e2e: at 16:20:43, during the
6th mixed replay, the kernel logged `Xid 31 … VLLM::EngineCor … FAULT_PDE ACCESS_TYPE_VIRT_WRITE` and vLLM died
(`Triton Error [CUDA]: an illegal memory access`, `EngineDeadError`; `results/xid-kernel.txt`). This is the fourth
fault and the first on the card's image (section 9).

## 8. Performance: llama.cpp v0.7.0 vs vLLM (nightly container, 0.32.0)

Same tools for every row: `replay_agent.py` (section 4: the trial's OpenClaw session, 13.6k → 18.2k tokens, one image,
49 tools, twice, one call at a time), `decode.sh` (one 256-token completion of a short prompt, three times, temperature
0), `replay_mixed.py` (the agent replay with perception requests in flight: two camera inventories and a REFER).
TTFT seconds, replay 1 / 2.

| server | where | 1st agent call (13.6k, no hit) | follow-up calls | decode tok/s |
|---|---|---|---|---|
| **llama.cpp v0.7.0, LiLiCorr (default)** | beside Isaac | 25.9 / 25.9 | 1.0-3.5 | 41.2 |
| llama.cpp v0.7.0, no drafter | beside Isaac | 23.5 / 23.7 | 1.2-3.4 | 17.6 |
| vLLM nightly container, no drafter | beside Isaac | 17.8 / 14.4; 18.8 / 16.9 live | 1.2-2.8; 1.3-3.5 live | 11.2 live |
| vLLM 0.32.0 cu132, no drafter | alone | 8.0 / 8.2 | 0.6-1.3 | 28.8 |
| vLLM 0.32.0 cu130, no drafter | alone | 8.1 / 7.8 | 0.6-1.3 | 29.3 |
| vLLM 0.32.0 cu132, no drafter | beside Isaac | 13.0 / 12.9 | 1.1-2.2 | 16.9 |
| vLLM 0.32.0 cu130, no drafter | beside Isaac, live | 17.8 / 17.2 | 1.5-4.7 | 11.3 |
| vLLM 0.32.0 cu132, LiLiCorr ×7 | beside Isaac | 18.0 / 13.2 | 6.5-13.0 | 47.2 |

With perception in flight (mixed replay beside Isaac): llama.cpp with the drafter 30.0 / 28.2 s first call, follow-ups
1.8-6.5 s, camera inventories 10-42 s, REFER 7.1-7.4 s; without it 25.8 / 25.8 s, 1.5-3.9 s, 28-56 s, 8.0 s; vLLM
follow-ups 1.2-5.7 s. "Live" = Isaac's sim loop and the OpenClaw gateway running after an e2e.

llama.cpp's first call of a chat is the slowest (13.6k tokens of prefill: 24-26 s beside Isaac against 13-19 s on
vLLM); its follow-ups match vLLM's because each slot keeps its prompt. The e2e turns (section 7a) are what an attendee
sees: 22-24 s for a description, 162-164 s for a pick (robot time ~142 s), 2.8-8.1 s from a message to the robot tool.

## 9. GPU faults (Xid 31) of Bolt on vLLM beside Isaac, and the decision

**What happened.** Bolt on vLLM killed its engine with a GPU MMU fault four times, each beside Isaac Newton under agent +
perception load; three on vLLM 0.32.0 from a uv venv (b12x 1.5.1, FlashInfer 0.7.0.post1, torch 2.13) and one on the
model card's own nightly container:

| time (CEST) | server | load | kernel log |
|---|---|---|---|
| 12:39:05 | vLLM 0.32.0, torch cu132 | mixed replay beside Isaac | `Xid 31, pid=3176695, name=VLLM::EngineCor … faulted @ 0xffb4_3fff4000. FAULT_PDE ACCESS_TYPE_VIRT_WRITE` |
| 13:49:00 | vLLM 0.32.0, torch cu132 | e2e (OpenClaw UI) beside Isaac; core dumped | `Xid 31, pid=3483718 … faulted @ 0xe92b_5fff4000. FAULT_PDE ACCESS_TYPE_VIRT_WRITE` |
| 15:37:53 | vLLM 0.32.0, torch cu130 | e2e (OpenClaw UI) beside Isaac | `Xid 31, pid=353919 … faulted @ 0xf018_dfffc000. FAULT_PDE ACCESS_TYPE_VIRT_WRITE` |
| 16:20:43 | **the card's nightly container** (`77154ca6` + b12x 1.5.1, no drafter) | soak beside Isaac, 6th mixed replay, right after a passing e2e (run C) | `Xid 31, pid=578934 … faulted @ 0xf9cf_ffff4000. FAULT_PDE ACCESS_TYPE_VIRT_WRITE` |

vLLM reported `CUDA driver error: an illegal memory access` (the nightly: in b12x's Triton MoE route packing) and
`EngineDeadError`. CUDA reports such errors asynchronously, so the kernel named in each traceback (Inductor Triton,
Mamba `causal_conv1d`, b12x route packing) is not necessarily the faulting one.

**Host hard-resets.** The host reset with no shutdown or kernel log twice: at 13:56, 7 minutes after the second fault,
and at ~17:07, 26 minutes into a soak of vLLM 0.32.0 cu130 with CUDA graphs off (`--enforce-eager`, no drafter) beside
Isaac that had passed its e2e (green cube placed, Isaac truth); that boot logged no Xid after 16:20 and MemAvailable
was 33.9 GB ten seconds before. The CPU was busy (a pytest run of this branch beside the soak and other agents' work,
load average 20-25). A link between the resets and vLLM is not proven; every
Bolt-on-vLLM-beside-Isaac run is treated as a risk to the host, which also drives the real robot.

**What the faults share.** Each scheduler dump shows one step mixing decodes with a chunk of an agent prompt that
carries a camera image (0.32.0: 1 + 2,047 fresh tokens; 1 + 192 tokens after a 16,192-token prefix hit; 1 + 1 + 2,046
tokens of a 13,583-token prompt; the nightly: 3 running requests, a 14,951-token prompt with an image). Each faulting
page sits 16-48 KiB below a 512 MiB-aligned address (0x…4000_0000, 0x…6000_0000, 0x…e000_0000, 0x…0000_0000) with no
page-directory entry: a write just before the start of a large buffer, into unmapped address space.

**What was ruled out.**

- Torch build: faults on cu132 (uv's `--torch-backend=auto` pick on this driver) and cu130 (the build the card's image
  and vLLM 0.31.0 use).
- vLLM release vs nightly: both fault.
- Load alone: Bolt ALONE survived 4 stress runs, 55 min and 1,217 requests with 0 errors and 0 Xid, on both torch
  builds (`tools/stress_crash.py`: 1-2 perception threads every 0.3 s + 1-2 agent threads replaying the recorded
  session, fresh and prefix-hit, temperature 0 and OpenClaw's default sampling with 256 tokens; `results/stress-*`),
  plus the 8 single-shape repro steps of `repro_crash.py`. The fault needs Isaac's GPU contention.

**Not isolated.** No bisection; the common parts of every faulting route are vLLM's Nemotron-H / Mamba code and b12x
1.5.1's IQ2 kernels. A non-b12x MoE backend was not tried: the rule became "no Bolt on vLLM beside Isaac".

**Decision.** No vLLM configuration met the bar (a 30-minute soak including a full e2e with 0 Xid). Bolt ships on
llama.cpp v0.7.0, which met it (section 10). `serve_nemotron_bolt_vllm.sh` stays as an opt-in route: its header lists
these faults and every start logs a warning. Revisit with a vLLM release that contains the card's nightly commits AND a
soak beside Isaac.

## 10. llama.cpp v0.7.0: drafter on or off, and the soak

Both settings ran the same replays beside Isaac (section 8) and the same four e2e messages (section 7a):

| | LiLiCorr drafter (`SPEC=dflash`) | no drafter (`SPEC=none`) |
|---|---|---|
| e2e turns (reset / describe / pick / reset) | 74.0 / 24.0 / 162.0 / 39.0 s (sum 299.0) | 72.5 / 22.0 / 163.5 / 41.0 s (sum 299.0) |
| send → first robot tool | 36.8 / 5.4 / 5.3 / 2.8 s | 32.6 / 4.0 / 8.1 / 2.9 s |
| agent replay: 1st call / follow-ups | 25.9 s / 1.0-3.5 s | 23.5 s / 1.2-3.4 s |
| mixed replay: follow-ups, inventories, REFER | 1.8-6.5 s, 10-42 s, 7.1-7.4 s | 1.5-3.9 s, 28-56 s, 8.0 s |
| decode alone | 41.2 tok/s (draft acceptance 0.53) | 17.6 tok/s |

The agent turns tie (299.0 s each): the agent's calls are short (18-120 output tokens) and prefill-bound. The drafter
keeps its other gains (2.3× decode, camera inventories done in about two thirds of the time) at no cost in prefix hits,
since llama.cpp keeps each slot's prompt; it stays on, the model card's recipe. `SPEC=none` turns it off.

**Soak (default, beside Isaac, after both e2e runs, the launched stack still up), 18:02-18:33 CEST.** `soak_load.sh`:
the agent replay, the decode probe, then mixed replays (the agent session with two camera inventories and a REFER in
flight) back to back for 30 min, while the robot server's own watcher kept sending its inventories to the same brain.
**0 Xid, 0 errors, brain healthy at the end** (`results/xid-kernel.txt`, `results/ll070-dflash-soak-*`):

| | requests | result |
|---|---|---|
| agent replay, one call at a time (2 sessions) | 12 | 1st call 36.5 / 38.6 s, follow-ups 2.6-6.1 s |
| decode probe | 3 | 13.7-32.9 tok/s (median 17.9; the watcher's inventories shared the GPU), draft acceptance 0.57 |
| mixed replays (11) | 66 agent + 56 perception | 0 errors; agent 1st call 33-37 s, follow-ups 1.8-35.2 s (median 15.1); inventories 13-61 s; REFER 4.0-17.3 s (median 9.4) |
| robot server watcher (through the proxy) | 32 | 0 errors |

This load is heavier than an attendee's: two slots served the replay's agent, its two inventories and REFER, and the
watcher at once, so follow-ups queued behind inventories (median 15 s). In the e2e (real load) follow-ups took 2-8 s.
The pick's REFER stayed under its 20 s limit throughout (max 17.3 s).

## 11. Qwen3.8-27B + DFlash2 on vLLM 0.31.0 vs 0.32.0

Measured alone (no Isaac), one server at a time, 2026-10-18 18:39-18:49 CEST, same tools as section 8 with
`--model qwen3.8-27b` (`results/qwen-*`): 0.31.0 = main's `serve_qwen_vllm.sh` before #375 (`91bac52`), 0.32.0 = #375's
(torch cu130, `vllm[b12x]`). TTFT seconds, replay 1 / 2.

| | vLLM 0.31.0 | vLLM 0.32.0 cu130 |
|---|---|---|
| agent 1st call (12.5k tokens; replay 2 hits 10.8k) | 5.84 / 1.23 | 6.01 / 1.27 |
| follow-up calls | 0.93-1.84 | 0.94-1.88 |
| decode tok/s (three runs) | 34.3 / 34.2 / 34.1 | 33.0 / 33.1 / 33.0 |
| DFlash2 tokens accepted / drafted (decode probe) | 197 / 413 | 197 / 413 |
| engine start: weights, then profile + warm-up | 37 s, 142 s | 41 s, 186 s |
| errors, Xid | 0, 0 | 0, 0 |

0.32.0 is 3.5% slower to decode, 1-11% slower to the first token on 11 of 12 calls and 49 s slower to start; the
rule was to keep it only if it is not worse. **Qwen stays on vLLM 0.31.0**: `scripts/serve_qwen_vllm.sh` and
`tests/test_qwen_vllm.py` are restored to `91bac52`.

## 12. Known limits

- A new chat's first agent call prefills 13.6k tokens: 24-26 s beside Newton on llama.cpp (vLLM: 13-19 s); follow-ups
  take 1-5 s.
- Two slots: while the agent and a background inventory both run, a REFER waits for a slot (7-8 s in the mixed replay;
  the pick's localization limit is 20 s).
- The rig needs the Bolt GGUFs (24.8 GB) and the llama.cpp v0.7.0 build before the brain unit starts:
  `scripts/serve_nemotron_bolt_llamacpp.sh --setup-only`, then restart the brain unit (docs/OPENSHELL.md step 0).
- Bolt on vLLM is opt-in and not for use beside Isaac Sim (section 9).

## 13. Method, tools, hygiene

- Worktree `feat/brain-nemotron-bolt` from `origin/main` 91bac52; GPU work in `systemd-run --user` units, one brain at a
  time, MemAvailable watched; Isaac always NEWTON (`newton_prebundle.py`: CONSISTENT before every run). Run D used its own
  ports and OpenShell instance (bridge 8631, GraspGen-X 5576, brain 18495, proxy 18496), a kernel-log watcher that would
  stop every GPU unit at the first Xid, and no CPU-heavy work while the GPU stack ran.
- Never committed: frames, keys, chat URLs, Chromium profiles, request bodies.
- `tools/replay_agent.py`, `tools/readiness_probe.py`, `tools/stress_crash.py`, `tools/replay_mixed.py` read the trial's
  proxy bodies from the scratch dir (not committed).
