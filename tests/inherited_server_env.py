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

vLLM 0.32.0 (entrypoints/): `VLLM_SERVER_DEV_MODE` adds the dev and RL routes (/collective_rpc, /sleep,
/update_weights, ...; outside /v1, so keyless), `VLLM_ALLOW_RUNTIME_LORA_UPDATING` adds /v1/load_lora_adapter and
/v1/unload_lora_adapter (inside the sandbox's /v1/**), `VLLM_ENABLE_COHERE_API` adds /cohere/v2/chat, and
`VLLM_USE_RUST_FRONTEND` + `VLLM_RUST_FRONTEND_PATH` replace the whole HTTP server with another binary.
"""
from __future__ import annotations

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
}

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
