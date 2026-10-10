# B71 — extra-hop latency of `cascade.robot-runtime/1`, plaintext vs TLS (10 Oct 2026)

`latency.json` is the unedited output of

    python scripts/bench_robot_endpoint.py --calls 500 --warmup 20 --out latency.json

run from the B71 working tree (on top of main `38f6d08`; the code merged with this
evidence) with `CUDA_VISIBLE_DEVICES=-1`, on one host (x86-64, 128 hardware
threads, OpenSSL 3.5.7, Python 3.12.13) while other test suites ran (load average
32.5 at start, 30.1 at the end, recorded in the file).

What it measures: per call, the client's real path (`RemoteRobotRuntime`) to a
`cascade-robot-service` child on loopback, in clear and over TLS (throwaway
private CA, RSA-2048 leaf, TLSv1.3 negotiated), against the same robot profile
(`microduck_conversation_mock`) built and called in-process. Two operations:
the generation read (`GET /v1/state`) and one read-only tool
(`locomotion.get_base_state`). The three modes are interleaved call by call.
Every call opens a new connection (HTTP/1.0), so each TLS call includes a full
handshake. All 500 executes succeeded in every mode; every sample is in the file
(`samples_ms`, call order).

p50 / p95 in ms: in-process 0.003 / 0.004 (state), 0.79 / 1.12 (tool);
plaintext 0.53 / 0.67, 1.51 / 1.85; TLS 1.57 / 1.89, 2.37 / 2.75.

Not measured: two hosts, a real network, load generated on purpose, other key
types or session resumption. The numbers depend on the host and its load; the
tests never assert on them (`tests/test_robot_endpoint_tls.py` only checks that
the documentation quotes this file).
