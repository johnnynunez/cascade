# Cosmos3-Edge on vLLM 0.32.0 with an NVFP4 KV cache (2026-10-19)

**Result.** `scripts/serve_cosmos_vllm.sh` moves Cosmos3-Edge (HF reasoner-only export, `--model-impl
transformers`) from vLLM 0.29.0 / transformers 5.17.0 to vLLM 0.32.0 / torch 2.13.0+cu130 / transformers 5.19.0. At a
BF16 KV cache the new stack answers exactly like the old one: the same 22 tool replies, the same image answers, the
same passkey recall (A vs B below). NVFP4 KV works with the Transformers backend on SM120 (`Using nvfp4 data type to
store kv cache`, FlashInfer XQA decode) and is the default now (Johnny's standing choice); `KV_DTYPE=auto` restores
BF16. Its measured costs:

- **Accuracy on long prompts.** The built-in agent's requests (12.2k prompt tokens: system prompt plus 42 tools) fell
  from 9/14 to 4/14 correct tool calls; several NVFP4 replies are garbled XML. FP8 does about as badly (5/14). Short
  OpenClaw-shaped requests (~1.2k tokens, 7 tools) stay 8/8 with every cache, and a passkey at 16k tokens comes back
  3/3 with every cache.
- **Speed.** Decode ~14% slower than BF16 (median 113 vs 132 tok/s, eager mode), a 16k-token prefill 12% slower
  (0.37 vs 0.33 s TTFT), a 30k-token one 13% slower (0.96 vs 0.85 s).
- **Gain.** 3.6× the KV tokens at the same `GPU_FRAC` 0.20: 440,672 instead of 123,936 (13.4 instead of 3.8 requests
  of 32k). The sidecar serves at most 2 sequences of 32k, which BF16 already holds.

| | A: 0.29.0 BF16 (shipped) | B: 0.32.0 BF16 | **C: 0.32.0 NVFP4 (default now)** | B2: 0.32.0 BF16 on FLASHINFER | D: 0.32.0 FP8 |
|---|---|---|---|---|---|
| attention backend | FLASH_ATTN | FLASH_ATTN | FLASHINFER (XQA decode) | FLASHINFER (XQA) | FLASHINFER (XQA) |
| GPU KV cache size (tokens) | 124,032 | 123,936 | **440,672** | 123,936 | 247,872 |
| tool probe, correct / 22 | 17 | 17 | **12** | 16 | 13 |
| … built-in agent, 12.2k-token prompt (14) | 9 | 9 | **4** | 8 | 5 |
| … OpenClaw-shaped, 1.2k-token prompt (8) | 8 | 8 | 8 | 8 | 8 |
| replies with a native `tool_calls` (22) | 20 | 20 | 17 | 20 | 16 |
| image questions, scored correct / 4 | 2 | 2 (same answers as A) | 2 | 2 | 2 |
| decode tok/s, median of 8 (range) | 141 (122-156) | 132 (122-155) | **113 (100-127)** | 147 (135-170) | 144 (131-166) |
| TTFT, 16.0k-token prompt (s, median of 3) | 0.33 | 0.33 | **0.37** | 0.32 | 0.32 |
| passkey at 16.0k tokens | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| TTFT, 30.5k-token prompt (s) | 0.85 | 0.85 | 0.96 | 0.79 | 0.80 |
| `tests/test_cosmos_live.py` (text, readiness, stream, red/blue image tool calls) | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |

`nvfp4_4over6` (E) was refused at start: `Selected backend AttentionBackendEnum.FLASHINFER is not valid for this
configuration. Reason: ['kv_cache_dtype not supported']`.

## Setup

| what | value |
|---|---|
| host | x86 workstation, RTX PRO 6000 Blackwell Workstation (SM120, 96 GB), all servers on the same GPU, one at a time |
| model | `nvidia/Cosmos3-Edge@a9d944e2`, HF reasoner-only export (`build_export`), `--model-impl transformers --enforce-eager`, CTX 32768, `GPU_FRAC` 0.20, 2 seqs |
| A | `scripts/cosmos_serving.py` of main `073293c7` (vLLM 0.29.0, transformers 5.17.0, torch 2.13.0), built by its own `--setup-only` into a scratch venv |
| B, C, D | this branch's script, `KV_DTYPE=auto`, `nvfp4` (adds `--attention-backend FLASHINFER`), `fp8`; one venv built by its `--setup-only` (vLLM 0.32.0, torch 2.13.0+cu130, FlashInfer 0.7.0.post1 + `vllm download-kernels`) |
| B2, E | `tools/serve_variant.py`: B's exact serve command plus `--attention-backend FLASHINFER` (B2) or `--kv-cache-dtype nvfp4_4over6 --attention-backend FLASHINFER` (E) |
| probes | `tools/bench_one.sh` → `tools/cosmos_probe.py` (tools, image, decode, ttft, ttft-log) and `tests/test_cosmos_live.py`; temperature 0, thinking off |

Tool probe: 14 operator requests through CASCADE's own Cosmos client (`configs/llm/local_cosmos.yaml`, FR3 + D435i: 42
tools sent `strict`, one call per turn, the built-in agent's system prompt) and 8 OpenClaw-shaped requests (the
attendee catalog as `cascade__*` tools, non-strict, `tool_choice=auto`). A case passes when the reply carries a native
tool call to an accepted tool whose arguments pass the client's validator (built-in) or the tool's JSON schema
(OpenClaw). Raw rows: `results/*.jsonl`; engine/KV lines per server: `results/server-*.kvlines.txt`. Round 2 is the
table; `results/round1/` is the first A/B/C round (same tool and image replies), `results/decode-5runs.jsonl` a
decode-only round with nothing else on the GPUs.

## Where NVFP4 loses the built-in agent

Of the 14 built-in requests, C fails 5 that A and B pass and passes none they fail (`results/tools.jsonl`): "Where is
the blue bowl?" (prose with invented coordinates), "Open the gripper." (an invented `value` argument), "Move the arm a
little to the left." (`arm` filled with repeated `<parameter=1` fragments), "Find the scissors…" (prose), "You are
done…" (`task_done` as text). FP8 fails 4 of the same kind; BF16 on FlashInfer (B2) only the last one, so the backend
switch alone costs at most one case. The structural-tag grammar is active (xgrammar compiles the 42 schemas), but it
cannot pick the right tool or stop prose. Prefix caching is unaffected: this dense model has one KV group, 16-token
blocks with every dtype.

## Compatibility patches on 0.32.0

Every patch is re-derived for the new wheels, gated on the exact vLLM/transformers versions and the file's sha256,
proven by running all smokes in a fresh interpreter that imports the candidate bytes (`_Overlay`), written by an
atomic inode swap, then re-proven from the installed files (`results/smokes-pristine-0.32.0.json`).

| smoke | 0.29.0 + 5.17.0 (main) | 0.32.0 + 5.19.0 as installed | now |
|---|---|---|---|
| RoPE without videos | patched (transformers) | passes for `None` and the documented `(0, 3)` grid | **dropped**; the 1-D `torch.empty(0)` case still fails and was removed: no caller produces it (`results/dropped-smoke-cases.json`) |
| video token counts | patched (transformers) | passes; fixed upstream in transformers 5.18 (huggingface/transformers#48900) | **dropped**; the `do_resize=False` case still counts 4 tokens for 1 (every Qwen2-VL-family counter ignores it) and was removed: CASCADE always resizes |
| model config (`hidden_act` AttributeError) | patched `ModelConfig.__post_init__` | fails | **kept**: delete `cosmos3_edge` from `_CONFIG_REGISTRY` (backport of vllm-project/vllm#60823) |
| image processor (vendored row-major processor) | patched `get_hf_processor` | fails | **kept**: delete `Cosmos3EdgeProcessor` from `_CLASS_TO_MODULE` / `__all__` (vllm#60823) |
| tool grammar (bare `<function=` calls) | patched `get_structural_tag` (strict rewrite + served-name check) | fails at `--tool-strict-level parameter` | **kept**: `--tool-strict-level parameter` replaces the strict rewrite; backport of vllm-project/vllm#62368 (byte-identical result) |

Each kept patch fixes only its own smoke (each alone overlaid on the pristine venv). vllm#60823 is approved, vllm#62368
is open; a patch whose smoke passes on a later release reports `upstream-ok` and writes nothing.

## Chat template (unchanged; Hub PR prepared)

`serving_chat_template` still renders tool schemas as JSON instead of XML `<parameter>` blocks. Measured without any
grammar (`--tool-strict-level auto`, 12 requests, `tools/template_ab.sh`): the Hub template gets 1/12 valid calls
(the model copies the schema XML, e.g. `<function>pick_and_place</function>\n<parameters>\n<parameter>\n<name>object`),
the JSON template 6/12 (`results/template_ab.jsonl`; raw completions in `results/template_raw_completions.json`).

## Not measured

- The DGX Spark (GB10, SM121, unified memory) and Jetson Thor (SM110): neither the install (cu130 aarch64 wheels,
  `download-kernels`, the FlashInfer JIT of the fp4 prefill and XQA kernels, ~14 s here on the first NVFP4 start) nor
  any number above. Decode there is bandwidth-bound and may move differently.
- Beside Isaac Sim, OpenClaw end to end, video inputs over HTTP.
- More than 14 built-in requests; the 9 → 4 drop is one direction only (5 cases lost, none gained), but n is small.
- Calibrated KV scales: the checkpoint ships none (`it may cause accuracy drop without a proper scaling factor`).

## Re-run

1. `VENV=<dir> EXPORT_DIR=<dir> scripts/serve_cosmos_vllm.sh --setup-only` (and main's script into another `VENV`
   for A).
2. One configuration: `PORT=8182 PROBE_PYTHON=<CASCADE venv python> VENV=… EXPORT_DIR=… KV_DTYPE=nvfp4
   tools/bench_one.sh C <out> python3 scripts/cosmos_serving.py` (~2 min); variants:
   `tools/bench_one.sh B2 <out> <venv>/bin/python tools/serve_variant.py scripts/cosmos_serving.py --attention-backend FLASHINFER`.
   `PROBE_ARGS="--probes decode --decode-runs 5" LIVE=0` gives the decode-only round.
3. Patch smokes on a pristine 0.32.0 venv: `tools/smoke_evidence.py scripts/cosmos_serving.py <site-packages> <export>
   <venv python> out.json`; dropped cases: `<venv python> tools/dropped_cases.py <export>`.
4. Template A/B: `CATALOG=demo/kitchen/openclaw-plugin/attendee-tools.json tools/template_ab.sh <venv> <export>
   <derived template> <out>`; raw completions: `tools/raw_example.py` against any of those servers.
