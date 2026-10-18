# Nemotron 3.5 Bolt as the default cascade brain (2026-10-18)

**Decision (owner, 2026-10-18): NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB is the default brain for both the agent
(OpenClaw) and cascade's perception. Qwen3.8-27B + DFlash2 stays selectable (`--brain qwen`,
`scripts/serve_qwen_vllm.sh`); every `--brain qwen` path is byte-identical to main (`91bac52`).**

Status of this report: updated as each commit of the PR lands. Sections marked *pending* are not measured yet.

## 1. What the default switch changes

| Piece | Before (main) | Now |
|---|---|---|
| Brain server | `scripts/serve_qwen_vllm.sh` (Qwen3.8 NVFP4 + DFlash2) | `scripts/serve_nemotron_bolt_vllm.sh` (the model card's vLLM container, pinned) |
| Served id | `qwen3.8-27b` | `nemotron-3.5-bolt` |
| OpenShell sandbox model / label | `cascade/qwen3.8-27b`, "Qwen3.8-27B (cascade brain)" (hard-coded) | `cascade/nemotron-3.5-bolt`, "Nemotron 3.5 Bolt 74B-A9B (cascade brain)"; the label follows the served id |
| Perception default (`configs/demo.yaml`, `demo.BRAIN_MODEL`) | `qwen3.8-27b` | `nemotron-3.5-bolt` |
| `launch.sh --brain` | `auto\|keep\|cosmos\|cosmos-sglang\|qwen` | adds `nemotron-bolt`; `auto` resolves a local Cosmos, else the brain on :8080 (Bolt when it serves `nemotron-3.5-bolt`, else Qwen as before) |
| cascade agent profile | `local_qwen` | `local_nemotron_bolt` (dev runners default to it) |

## 2. Serving contract (`scripts/serve_nemotron_bolt_vllm.sh`)

Mirrors `serve_qwen_vllm.sh`: 127.0.0.1 by default, PORT 8080, a 0600 key file read into `VLLM_API_KEY` and handed to
docker by name (never argv), no network bind without a key, `--setup-only` / `--dry-run`.

- Image: `vllm/vllm-openai@sha256:a7deff2c…` (the card's tag `nightly-77154ca6…`, vLLM 0.32.0rc1) plus `b12x==1.5.1`
  installed with `--require-hashes` into a local image (`cascade/nemotron-bolt-vllm:77154ca6-b12x1.5.1`), so serving
  needs no network. The trial installed b12x at every start.
- Weights: `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB@c6cc2e8f` and the LiLiCorr drafter `@abd47806`, from the
  host's HF cache, mounted read-only, `HF_HUB_OFFLINE=1`. The card's `ultra_v3_reasoning_parser.py` is checked by sha256
  (`14e03de5…`) before serving.
- Flags: as the trial (`v5-bolt/serve_bolt.sh`): 64k context, `--max-num-seqs 4`, GPU_UTIL 0.45, fp8 KV, b12x
  linear/MoE, FlashInfer attention + Mamba, `--limit-mm-per-prompt {"video":0,"image":1}`, qwen3_coder tool parser,
  ultra_v3 reasoning parser, thinking off by default, LiLiCorr dflash ×7.
- New: `--block-size 64 --prefix-match-unit 64` (section 4), Docker CDI GPU, the container runs as the calling user,
  MemAvailable watched during startup (killed below `MEM_FLOOR_GB`, default 3), a stopping unit stops the container.

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

## 4. Prefix caching

**Root cause of the trial's 0 % hit rate.** The served model has 11 KV-cache groups: the target's attention/Mamba
groups (block forced up to 4,496 tokens so an attention page holds a Mamba state) and the LiLiCorr drafter's 5
sliding-window layers (block 64). vLLM 0.32's hybrid manager hashes every gcd(4496, 64) = 16 tokens, but allows a hit
below the LCM block only if every prefix-cacheable group supports fine-grained lookup; `SlidingWindowManager` does not.
So hits could only land on multiples of **lcm = 17,984 tokens** (server log: `kv cache group sizes (4496 ×10, 64)`,
`kv lcm block sizes 17984`). The agent's prompts are 13.6k-18.2k tokens: no hit was ever possible.

**Fix.** `--block-size 64 --prefix-match-unit 64`: vLLM rounds the attention block up to a multiple of 64 that holds a
Mamba page (4,544), and every group, the drafter's included, uses it: `kv cache group sizes (4544 ×11)`,
`kv lcm block sizes 4544`. Hits now land every 4,544 tokens.

**Measured** on a realistic agent replay (`tools/replay_agent.py`): the trial's recorded OpenClaw session (proxy
bodies 41, 43, 44, 47, 49, 51: 13.6k → 18.2k prompt tokens, one camera image, 49 tool schemas, each call extending
the previous one exactly as OpenClaw sent it), replayed twice with fresh session ids, 32 output tokens,
temperature 0, one request at a time; hits from vLLM's `/metrics` counters. Raw rows: `results/*.jsonl`.

TTFT, seconds, beside Isaac Sim 6.2 Newton (the kitchen scene loaded, bridge `ping` engine newton):

| call (prompt tokens) | before: replay 1 / 2 (hit) | after: replay 1 / 2 (hit) |
|---|---|---|
| 1 (13.6k) | 13.35 / 13.40 (0) | 36.29 cold JIT / 14.08 (0) |
| 2 (15.0k) | 14.42 / 14.42 (0) | 16.16 / 10.58 (4,544) |
| 3 (15.0k) | 14.95 / 14.95 (0) | 7.04 / 6.51 (9,088) |
| 4 (16.5k) | 16.08 / 15.95 (0) | 10.56 / 7.63 (9,088) |
| 5 (16.6k) | 16.44 / 16.35 (0) | 8.75 / 7.96 (9,088) |
| 6 (18.2k) | 17.87 / 17.77 (0) | 10.06 / 9.52 (9,088) |

Alone (no Isaac), before: 8.5-11.8 s on every call, 0 hits (`results/before-alone.jsonl`). After, alone: *pending*.

**Limits.** A new OpenClaw session never hits on its first call: the session id sits in the system prompt's Runtime
line, early in the prompt (replay 2, call 1: 0 hits). Within a session the hit is the last 4,544-token boundary the
previous call reached, so 5-9k tokens are still prefilled per call. Without the drafter the sliding-window group is
gone and vLLM may hit at 64-token granularity: *pending measurement*.

## 5. demo_proof's readiness check

**Root cause.** OpenClaw 2026.9.3's agent system prompt has a "Silent Replies" rule: "Nothing to say: entire reply
exactly NO_REPLY" (`system-prompt-params`, `silentReplyPromptMode` "generic" for `openclaw agent --message`; only a
message-tool-only delivery or a session-stable conversation context turns it off, neither is a config switch). OpenClaw
then drops that token, so the payload is empty. The launcher asks "Reply with exactly OK and nothing else. Do not call
tools."; Bolt sometimes reads that as nothing to say.

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

*Further runs pending.*

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

## 7. Known limits

- Agent latency: even with prefix hits, Bolt's per-call TTFT beside Newton is 6.5-16 s against Qwen's ~3 s; see
  section 4 for what is left.
- One image per prompt (`--limit-mm-per-prompt`), no video.
- The rig needs the Bolt weights (24 GB: target + drafter) and the pinned image on disk before the brain unit starts:
  `scripts/serve_nemotron_bolt_vllm.sh --setup-only`, then restart the brain unit (`cascade-brain-vllm`).

## 8. Method, tools, hygiene

- Worktree `feat/brain-nemotron-bolt` from `origin/main` 91bac52; GPU work in `systemd-run --user` units, one brain at a
  time, MemAvailable watched (minimum logged in `results/`).
- Never committed: frames, keys, chat URLs, Chromium profiles, request bodies.
- `tools/replay_agent.py`, `tools/readiness_probe.py` read the trial's proxy bodies from the scratch dir (not committed).
