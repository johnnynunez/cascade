# NVFP4 KV cache (vLLM 0.32.0) on CASCADE's two vLLM brains (2026-10-19)

**Result.** vLLM 0.32.0 serves an NVFP4 KV cache on SM12x. It loads, decodes through XQA, and scores the same as
the BF16/FP8 caches on GSM8K, a 54k-token passkey test and a tool-choice probe. Both vLLM brains now serve it by
default (Johnny's choice); `KV_DTYPE` restores the previous cache. The measured costs:

- **Qwen3.8-27B (hybrid, Gated DeltaNet):** `scripts/serve_qwen_vllm.sh` moves to vLLM 0.32.0 for it (it was on
  0.31.0 with BF16). The KV token count is about the same as BF16 (190-179k vs 201-210k). Prefix reuse is
  coarser, though: an agent follow-up call at ~27k tokens took 0.85 s instead of 0.37 s, and a new session's first
  call 0.61 s instead of 0.32 s (section "Prefix reuse"). A 54k-token prefill is 32% slower; decode is within
  run-to-run spread of 0.32.0 BF16. `KV_DTYPE=auto` restores BF16 (on 0.32.0).
- **Nemotron 3.5 Bolt, vLLM route (hybrid, Mamba-2):** 28% more KV tokens, though the FP8 cache already held 26× a
  64k request. Long-prompt prefill is 20% slower and decode 1.4% slower. `KV_DTYPE=fp8` restores FP8.

**Why the capacity does not grow on Qwen.** With prefix caching, vLLM's hybrid allocator sizes every page to hold
one Mamba state (`Setting attention block size to N tokens to ensure that attention page size is >= mamba page
size`). A smaller attention cache does not shrink that page. Instead, the attention block grows (832 → 2912 tokens)
and the drafter's sliding-window layers become a separate 64-token group. Prefix hits can then only land on the
least common multiple of the block sizes (832 → 5824 tokens), and the agent's follow-up calls reuse less of their
prefix. The Bolt route keeps 64-token hits with `--prefix-match-unit 64` because it runs without the drafter.

## Setup

| what | value |
|---|---|
| host | x86 workstation, 2× RTX PRO 6000 Blackwell Workstation (SM120, 96 GB), driver 615.71.09; one GPU per server |
| Qwen | `Inferact/Qwen3.8-27B-NVFP4@6128240e` + DFlash2 (7 tokens); `GPU_UTIL=0.55`, CTX 65536, 4 seqs |
| Qwen engines | vLLM 0.31.0 (`scripts/serve_qwen_vllm.sh` before this change) and vLLM 0.32.0, torch 2.13.0+cu130, FlashInfer 0.7.0.post1 (`tools/serve_qwen_vllm_032_kv.sh`, whose body is the shipped `scripts/serve_qwen_vllm.sh`: the 0.32.0 recipe from `8926d148` + `KV_DTYPE`; NVFP4 adds `--attention-backend FLASHINFER`) |
| Bolt | `scripts/serve_nemotron_bolt_vllm.sh` (the card's nightly `77154ca6` + b12x 1.5.1, which contains vLLM #46963 and #60452), no drafter, `GPU_UTIL=0.45`, prefix match 64; the bench pinned the container to one GPU (`--device nvidia.com/gpu=N`) |
| probes | `tools/bench_one.sh`: server log, `decode_probe.py` (from `brain-nemotron-bolt-2026-10-18/tools`, 3 runs, short prompt), `tools/brain_quality.py` (GSM8K test, first 200; 9 passkeys in a 53.7k-token context), `tools/agent_probe.py` (the brain as CASCADE's built-in agent, MuJoCo SO-101, hidden physics checks, 4 tasks × 2); `tools/bench_tools.sh` + `tools/tool_probe.py` (30 operator requests, CASCADE's 43 tools, first tool call) |

All runs: temperature 0, thinking off. Raw rows are in `results/` (`*.jsonl`, plus `server-*.kvlines.txt` with each
server's engine, KV and backend lines).

## Qwen3.8-27B + DFlash2

| | A: 0.31.0 BF16 (before) | B: 0.32.0 BF16 | C: 0.32.0 NVFP4 (default now) | D: 0.32.0 FP8 |
|---|---|---|---|---|
| KV cache memory | 20.38 GiB | 20.97 GiB | 20.06 GiB | 20.06 GiB |
| GPU KV cache size (tokens) | 195,693 | 201,332 | **178,658** | 155,146 |
| attention block / prefix lcm (tokens) | 832 / 832 | 832 / 832 | 2912 / **5824** | 1648 / 6592 |
| decode, short prompt (tok/s) | **174.0** | 143.5 | 162.0 | 149.1 |
| DFlash2 accepted / drafted | 194 / 476 | 177 / 560 | 192 / 490 | 180 / 532 |
| 53.7k-token passkey: found, s/request | 9/9, **5.88** | 9/9, 5.94 | 9/9, 7.82 | 9/9, 6.55 |
| GSM8K (200) | 189 | 192 | 191 | 191 |
| tool probe (30) | - | 29 | 30 | - |
| agent probe, physics check passed (8) | 7 | 5 | 7 (one claimed failure) | 7 |

Decode uses FlashInfer XQA on NVFP4 (`decode_backend=xqa, kv_cache_dtype=torch.uint8, arch=sm120`). The agent probe
(n = 8, two rollouts per task) is too small to rank the configurations. It only shows that every cache drives the
robot.

## Prefix reuse (Qwen, agent-shaped calls)

`tools/prefix_probe.py` (via `tools/bench_prefix.sh`): CASCADE's system prompt and 43 tools, then ~26k tokens of
observations. Each session makes four calls, and each call adds an assistant turn and an observation (~950 tokens);
a second session repeats this with a new nonce. Columns: first call; median TTFT of the other seven calls; the most
prefix-cache tokens any call hit. All on vLLM 0.32.0.

| | KV tokens | groups / lcm | 1st call | other calls (median) | max hit tokens |
|---|---|---|---|---|---|
| P1: BF16 | 209,867 | 832 ×15 / 832 | 3.06 s | **0.37 s** | 26,624 |
| P2: NVFP4 (**the default now**) | 190,764 | 2912 ×14 + 64 / 5824 | 3.34 s | 0.85 s | 23,296 |
| P3: NVFP4, drafter skipped (`--kv-cache-dtype-skip-layers sliding_window`) | 408,804 | 2912 ×14 + 816 / 148,512 | 3.46 s | 3.71 s | **0** |
| P5: NVFP4 + `--block-size 64 --prefix-match-unit 64` | 438,056 | 2944 ×15 / 2944 | 3.41 s | 0.97 s | 23,552 |
| P6: P3 + `--block-size 16 --prefix-match-unit 16` | 436,482 | 2912 ×14 + 816 / 148,512 | 3.41 s | 3.65 s | 0 |

With the drafter's layers kept in BF16 (P3, P6), its group is 816 tokens, the lcm becomes 148,512 and no call
hits. That setup with a 64-token match unit (P4) is refused at start (`Invalid prefix_match_unit=64 ... block sizes
must be divisible`, 816 is not). P5 doubles the KV tokens but hits only on whole 2944-token
blocks, so it does not beat P2's reuse; the shipped default is plain NVFP4 (P2). Decode (3 runs, tok/s): P1 166,
P2 162, P3 173, P5 157, P6 163, against 143.5 for B above. That is the run-to-run spread: two servers shared the
host.

## Nemotron 3.5 Bolt, vLLM route

| | E: FP8 (before) | F: NVFP4 (default now) |
|---|---|---|
| GPU KV cache size (tokens), concurrency of 64k | 1,737,761, 26.5× | **2,230,954, 34.0×** |
| kv cache groups / lcm (tokens) | 4224 ×6 / 4224 | 7552 ×6 / 7552 |
| decode, short prompt (tok/s) | **184.0** | 181.5 |
| 53.7k-token passkey: found, s/request | 9/9, **6.61** | 9/9, 7.94 |
| GSM8K (200) | 193 | 193 |
| tool probe (30) | 29 | 28 (one `place_at` request answered with `move_relative`) |

## Not measured

- The DGX Spark (GB10, SM121, unified memory). There, decode reads weights and KV at 273 GB/s. A smaller cache helps
  decode only when the KV bytes are a sizeable share of a step: for Qwen at 18k tokens, about 8% of the bytes in
  BF16. This could shift the decode column; the capacity and prefix-block rows do not depend on the GPU.
- Beside Isaac Sim, and the Bolt route's Xid 31 faults (`brain-nemotron-bolt-2026-10-18`, section 9).
- The prefix-reuse cost on the Spark. There, prefill runs several times slower than here (`brain-nemotron-bolt-2026-10-18`,
  section 11: Qwen's first agent call took 5.84 s at 12.5k tokens), so the tokens a follow-up call no longer reuses
  (up to ~5.8k) cost seconds rather than the ~0.5 s measured here.

## Side finding

Bolt as CASCADE's **built-in** agent (`configs/llm/local_nemotron_bolt.yaml`, the agent probe) called no tool in 16 of
16 MuJoCo rollouts with either cache. It wrote its reasoning as text and the orchestrator stopped after 3 turns
without a call. The same server answered 30/30 single-turn tool requests (tool probe), and OpenClaw drives it fine
(`brain-nemotron-bolt-2026-10-18`), so the gap is in the built-in agent's multi-turn requests. It has nothing to do
with the KV cache.
