# NVFP4 KV cache (vLLM 0.32.0) on CASCADE's two vLLM brains (2026-10-19)

**Result.** vLLM 0.32.0 serves an NVFP4 KV cache on SM12x. It loads, decodes through XQA, and scores the same on
GSM8K, a 54k-token passkey test and a tool-choice probe. For our two brains it gives nothing they need, and costs
prefill time:

- **Qwen3.8-27B (hybrid, Gated DeltaNet):** fewer KV tokens than BF16 (−11%). Prefix-cache blocks are 7× coarser,
  long-prompt prefill is 32% slower, and decode is 7% slower than the vLLM 0.31.0 we serve. The default stays
  vLLM 0.31.0 with a BF16 cache.
- **Nemotron 3.5 Bolt, vLLM route (hybrid, Mamba-2):** 28% more KV tokens, but the FP8 cache already holds 26× a
  64k request. Long-prompt prefill is 20% slower and decode 1.4% slower. FP8 stays the default; `KV_DTYPE=nvfp4`
  is now a supported knob of `scripts/serve_nemotron_bolt_vllm.sh`.

**Why the capacity does not grow on Qwen.** With prefix caching, vLLM's hybrid allocator sizes every page to hold
one Mamba state (`Setting attention block size to N tokens to ensure that attention page size is >= mamba page
size`). A smaller attention cache does not shrink that page. Instead, the attention block grows (832 → 2912 tokens)
and the drafter's sliding-window layers become a separate 64-token group. Prefix hits can then only land on the
least common multiple of the block sizes (832 → 5824 tokens), and the agent's 13–18k-token follow-up calls lose most
of their cached prefix. The Bolt route keeps 64-token hits with `--prefix-match-unit 64`.

## Setup

| what | value |
|---|---|
| host | x86 workstation, 2× RTX PRO 6000 Blackwell Workstation (SM120, 96 GB), driver 615.71.09; one GPU per server |
| Qwen | `Inferact/Qwen3.8-27B-NVFP4@6128240e` + DFlash2 (7 tokens); `GPU_UTIL=0.55`, CTX 65536, 4 seqs |
| Qwen engines | vLLM 0.31.0 (`scripts/serve_qwen_vllm.sh`, main) and vLLM 0.32.0, torch 2.13.0+cu130, FlashInfer 0.7.0.post1 (`tools/serve_qwen_vllm_032_kv.sh`: main's 0.32.0 recipe from `8926d148` + `KV_DTYPE`; NVFP4 adds `--attention-backend FLASHINFER`) |
| Bolt | `scripts/serve_nemotron_bolt_vllm.sh` (the card's nightly `77154ca6` + b12x 1.5.1, which contains vLLM #46963 and #60452), no drafter, `GPU_UTIL=0.45`, prefix match 64; the bench pinned the container to one GPU (`--device nvidia.com/gpu=N`) |
| probes | `tools/bench_one.sh`: server log, `decode_probe.py` (from `brain-nemotron-bolt-2026-10-18/tools`, 3 runs, short prompt), `tools/brain_quality.py` (GSM8K test, first 200; 9 passkeys in a 53.7k-token context), `tools/agent_probe.py` (the brain as CASCADE's built-in agent, MuJoCo SO-101, hidden physics checks, 4 tasks × 2); `tools/bench_tools.sh` + `tools/tool_probe.py` (30 operator requests, CASCADE's 43 tools, first tool call) |

All runs: temperature 0, thinking off. Raw rows are in `results/` (`*.jsonl`, plus `server-*.kvlines.txt` with each
server's engine, KV and backend lines).

## Qwen3.8-27B + DFlash2

| | A: 0.31.0 BF16 (main) | B: 0.32.0 BF16 | C: 0.32.0 NVFP4 | D: 0.32.0 FP8 |
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

## Nemotron 3.5 Bolt, vLLM route

| | E: FP8 (shipped) | F: NVFP4 |
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

## Side finding

Bolt as CASCADE's **built-in** agent (`configs/llm/local_nemotron_bolt.yaml`, the agent probe) called no tool in 16 of
16 MuJoCo rollouts with either cache. It wrote its reasoning as text and the orchestrator stopped after 3 turns
without a call. The same server answered 30/30 single-turn tool requests (tool probe), and OpenClaw drives it fine
(`brain-nemotron-bolt-2026-10-18`), so the gap is in the built-in agent's multi-turn requests. It has nothing to do
with the KV cache.
