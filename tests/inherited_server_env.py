"""Environment variables a brain server acts on, for the serve-script env-hygiene tests.

A serve script passes every server setting it means as a flag. Anything the server would also take from the
caller's environment must therefore never reach it: an inherited value could open an endpoint, add an accepted
key or move the bind behind the script's back (#405 review, Minors 1 and 2).

llama.cpp v0.7.0 (620d2225):
- common/arg.cpp: every server flag reads `LLAMA_ARG_<FLAG>` (and `LLAMA_ARG_NO_<FLAG>`) before argv; a flag on argv
  wins for that one flag. `--api-key` reads `LLAMA_API_KEY`, and keys ACCUMULATE: an inherited key or key file is
  accepted beside the key file the script passes.
- tools/server/server-http.cpp: `AIP_MODE=PREDICTION` (Vertex AI) replaces `--port` with `AIP_HTTP_PORT` (argv cannot
  override it) and adds `POST $AIP_PREDICT_ROUTE`, which dispatches to ANY registered route by name (/slots, /props).
- tools/server/server-context.cpp: `LLAMA_SERVER_SLOTS_DEBUG` puts every slot's prompt text in GET /slots.

Not the environment, the same effect: before env and argv, llama-server applies `/etc/llama.cpp/config.ini`, then
`${XDG_CONFIG_HOME:-$HOME/.config}/llama.cpp/config.ini` (common/arg.cpp `common_params_apply_system_config`,
common/common.cpp `fs_get_config_directory`). Each takes ANY server option, no allowlist (common/preset.cpp):
HOSTILE_CONFIG_INI below. The user file follows an inherited XDG_CONFIG_HOME or HOME.

vLLM 0.32.0 (entrypoints/): `VLLM_SERVER_DEV_MODE` adds the dev and RL routes (/collective_rpc, /sleep,
/update_weights, ...; outside /v1, so keyless), `VLLM_ALLOW_RUNTIME_LORA_UPDATING` adds /v1/load_lora_adapter and
/v1/unload_lora_adapter (inside the sandbox's /v1/**), `VLLM_ENABLE_COHERE_API` adds /cohere/v2/chat, and
`VLLM_USE_RUST_FRONTEND` + `VLLM_RUST_FRONTEND_PATH` replace the whole HTTP server with another binary.
`VLLM_ENABLE_RESPONSES_API_STORE` (vllm/envs.py) keeps every stored /v1/responses request and answer in memory until
the server exits: a caller inside /v1/** grows the brain's memory without bound.

SGLang 0.5.22 (sglang/srt/): `SGLANG_GRPC_PORT` starts a native gRPC listener beside HTTP that the HTTP key does not
cover, `SGLANG_RUST_SERVER` swaps in the embedded Rust HTTP server, `SGLANG_OLLAMA_*_ROUTE` and `AIP_PREDICT_ROUTE`
move the Ollama routes and /vertex_generate anywhere (/v1/** too), `DUMPER_SERVER_PORT` starts a keyless control server
on 0.0.0.0 (`reuse`: /dumper/{method}) and the other `DUMPER_*` dump tensors or load code,
`SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES` lets x-override-* headers set request fields, `EXA_API_KEY` adds a web
search tool behind /v1/responses, `SGLANG_EXTERNAL_MODEL_PACKAGE` imports a package that replaces the whole model
registry (srt/models/registry.py), `SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE` one of multimodal processors
(managers/tokenizer_manager.py) and `SGLANG_EXTERNAL_MM_MODEL_ARCH` names that package's architecture, and
`SGLANG_LOGGING_CONFIG_PATH` replaces the logging setup with a dictConfig file (utils/common.py configure_logger): its
`()` factories import code, and an INFO handler writes the start line, the key included, wherever it says.
`SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE` replaces the torch.distributed init method (distributed/bootstrap.py
`_resolve_dist_init_method`): `env://` (MASTER_ADDR / MASTER_PORT) or `tcp://HOST:PORT` puts the rendezvous store on a
port the caller picks, and HOST may be another machine. torch's TCPStore listens on every interface whatever host it is
given (torch 2.14: `[::]`), and SGLang skips its own port checks under the override (server_args.py PortArgs).
"""
from __future__ import annotations

from pathlib import Path

#: Inherited by a llama.cpp serve script; none may reach llama-server. The values are what an attacker or a stray
#: shell would set: each one alone opens something on a keyless or WARM_START=0 start.
HOSTILE_LLAMA_ENV = {
    "LLAMA_ARG_SLOT_SAVE_PATH": "/tmp",          # POST /slots/N?action=save|restore|erase
    "LLAMA_ARG_ENDPOINT_PROPS": "1",             # POST /props (change global properties)
    "LLAMA_ARG_ENDPOINT_SLOTS": "1",
    "LLAMA_ARG_ENDPOINT_METRICS": "1",           # GET /metrics
    "LLAMA_ARG_HOST": "0.0.0.0",                 # a bind beyond loopback (argv's --host wins today)
    "LLAMA_ARG_API_PREFIX": "/v1",               # /slots served as /v1/slots, inside the sandbox's /v1/**
    "LLAMA_ARG_API_KEY_FILE": "/tmp/other.key",  # a second accepted key file
    "LLAMA_ARG_NO_WEBUI": "1",                   # the LLAMA_ARG_NO_<FLAG> spelling
    "LLAMA_API_KEY": "inherited-extra-key",      # a second accepted key
    "AIP_MODE": "PREDICTION",                    # Vertex AI mode: the port below, the predict route
    "AIP_HTTP_PORT": "8090",
    "AIP_PREDICT_ROUTE": "/v1/predict",          # reaches /slots by name, from inside /v1/**
    "AIP_HEALTH_ROUTE": "/v1/alive",
}

#: Inherited by serve_qwen_vllm.sh and cosmos_serving.py; none may reach `vllm serve`.
HOSTILE_VLLM_ENV = {
    "VLLM_SERVER_DEV_MODE": "1",
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1",
    "VLLM_ENABLE_COHERE_API": "1",
    "VLLM_USE_RUST_FRONTEND": "1",
    "VLLM_RUST_FRONTEND_PATH": "/tmp/other-frontend",
    "VLLM_ENABLE_RESPONSES_API_STORE": "1",
}

#: Inherited by serve_cosmos_sglang.sh; none may reach SGLang 0.5.22 (sglang/srt/environ.py, entrypoints/http_server.py,
#: arg_groups/fields/serving.py, debug_utils/dumper.py; read from the 0.5.22 wheel).
HOSTILE_SGLANG_ENV = {
    "SGLANG_GRPC_PORT": "50051",                         # a native gRPC listener beside HTTP, outside the HTTP key
    "SGLANG_RUST_SERVER": "1",                           # the embedded Rust HTTP server instead of FastAPI
    "SGLANG_OLLAMA_ROOT_ROUTE": "/v1/ollama",            # the Ollama routes moved anywhere, /v1/** included
    "SGLANG_OLLAMA_CHAT_ROUTE": "/v1/ollama/chat",
    "SGLANG_OLLAMA_GENERATE_ROUTE": "/v1/ollama/generate",
    "SGLANG_OLLAMA_TAGS_ROUTE": "/v1/ollama/tags",
    "SGLANG_OLLAMA_SHOW_ROUTE": "/v1/ollama/show",
    "AIP_PREDICT_ROUTE": "/v1/predict",                  # /vertex_generate (raw /generate parameters) inside /v1/**
    "DUMPER_SERVER_PORT": "40000",                       # a keyless dumper control server on 0.0.0.0
    "DUMPER_ENABLE": "1",                                # tensor dumps to disk
    "SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES": "1",       # x-override-* headers set request fields
    "EXA_API_KEY": "inherited-exa-key",                  # a web search tool behind /v1/responses
    "SGLANG_EXTERNAL_MODEL_PACKAGE": "caller_models",    # imported: replaces the whole model registry
    "SGLANG_EXTERNAL_MM_MODEL_ARCH": "CallerArch",       # that package's multimodal architecture
    "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE": "caller_mm",  # imported: multimodal processors
    "SGLANG_LOGGING_CONFIG_PATH": "/tmp/logging.json",   # a dictConfig: `()` factories, INFO (the key) to any file
    "SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE": "tcp://192.0.2.7:29500",  # the rendezvous store: any port, any host
}

#: Inherited SGLang-looking names the server keeps: tuning of features it never starts, and look-alikes.
BENIGN_SGLANG_ENV = {
    "SGLANG_GRPC_WORKER_THREADS": "4",   # only used with a gRPC listener, which never starts
    "SGLANG_OLLAMA": "kept",             # not SGLANG_OLLAMA_*
    "XDUMPER_SERVER_PORT": "kept",       # not DUMPER_*
    "SGLANG_TORCH_PROFILER_DIR": "/tmp/profiles",
    "SGLANG_EXTERNAL": "kept",           # not SGLANG_EXTERNAL_*
    "SGLANG_LOG_MS": "1",                # milliseconds in the log's timestamps
    "MASTER_PORT": "29500",              # torch's env:// init reads it; only the scrubbed override selects env://
}

#: A llama.cpp config.ini as an attacker would write it: every tool (exec_shell_command, write_file) at /tools, and
#: every route (/slots, /props, /tools) moved under /v1, inside the OpenShell sandbox's /v1/**.
HOSTILE_CONFIG_INI = "tools = all\napi-prefix = /v1\nslot-save-path = /tmp\n"

#: Inherited values a server may keep: unrelated settings, the server's own logging knobs, and names that only look
#: like the scrubbed ones. A scrub that drops these is too wide (an operator's CUDA or HF settings would vanish).
BENIGN_ENV = {
    "CASCADE_BENIGN_PROBE": "kept",
    "HF_TOKEN": "hf-token-kept",
    "LLAMA_TRACE": "0",                 # logs only
    "LLAMA_ARGS": "kept",               # not LLAMA_ARG_*
    "MY_LLAMA_ARG_HOST": "kept",
    "AIPMODE": "kept",                  # not AIP_*
    "VLLM_LOGGING_LEVEL": "INFO",        # logging only
    "VLLM_SERVER_DEV_MODEX": "kept",
}


def leaked(server_env: dict, hostile: dict) -> dict:
    """The hostile names that reached the server, with their values."""
    return {k: server_env[k] for k in hostile if k in server_env}


def dropped(server_env: dict, benign: dict = BENIGN_ENV) -> dict:
    """Benign names the server did not get with their inherited value."""
    return {k: v for k, v in benign.items() if server_env.get(k) != v}


def llama_config_files(server_env: dict, system: Path) -> list[Path]:
    """The config.ini files llama-server v0.7.0 applies when started with `server_env`, in its order: the system
    file (`system` stands for /etc/llama.cpp/config.ini), then the user's, found as llama.cpp finds it."""
    config_home = server_env.get("XDG_CONFIG_HOME") or str(Path(server_env["HOME"]) / ".config")
    return [p for p in (system, Path(config_home) / "llama.cpp" / "config.ini") if p.exists()]
