# B15a — GraspGen-X server warms its model before it binds (9 October 2026)

**Question.** A fresh learned GraspGen-X server answered its first `infer_object`
of a 5 cm cube in **15.53 s** and every later one in **0.09 s** (coordinator's
cold-client measurement on this rig, 9 Oct 2026). CASCADE's client waits
`grasp.graspgenx.timeout_ms` = 8000 ms, so the first pick of a session fell back
to the analytic OBB planner (B36 signature 4, first seen in the
[B35 NemoClaw run](../b35-nemoclaw-openshell-20261008/REPORT.md): 15.5 s).

**Change.** `scripts/graspgenx_server.py` runs one synthetic inference (a 5 cm
resting cube, the reBot sweep volume `demo.yaml` makes the client send) through
its own `_dispatch` between model load and `serve_forever()`, so the port opens
only on a warm model. `health` reports `warmed_up`, `warmup_s`, `warmup_error`.

## Live run (coordinator, GPU, 01:15 9 Oct 2026)

Rig: x86, NVIDIA RTX PRO 6000 Blackwell Workstation Edition (capability 12.0),
torch 2.7.0+cu128, GraspGen-X release checkpoints (gen epoch 736, dis epoch
1056), port 18532. Files: [`server-log.txt`](server-log.txt) (server stdout/stderr),
[`client.txt`](client.txt) (health reply + three `infer_object` calls).

| step | time / value (from the files) |
| --- | --- |
| first log line | 01:15:45.372 |
| model loaded | 01:15:48.132 |
| warm-up inference (server log "Time taken for inference") | **15.39 s** — the cold first inference, now paid before binding |
| `[graspgenx] warm-up inference 15.42 s before binding 127.0.0.1:18532` | 01:16:03.55 |
| ZMQ server listening | 01:16:03.550 (18.2 s after the first log line, 15.4 s after model load) |
| `health` | `warmed_up: true`, `warmup_s: 15.417`, `device: cuda:0` |
| client calls 1–3, wall | **0.16 / 0.14 / 0.13 s** (server-side inference 0.098 / 0.108 / 0.092 s), 316 grasps each (100 diffusion + 216 OBB) |

So the first request a client can make is warm: 0.16 s against the 8 s timeout.

## Not covered

- The live run used the first version of the change, in which a failed warm-up
  refused to serve. The committed version fails open (logs a `WARNING: warm-up
  inference failed ... binding ... COLD` line and serves as before); that
  branch and the `--no-warmup` opt-out are covered by CPU tests with a stubbed
  upstream server (`tests/test_graspgenx_warmup.py`), not by a GPU run. The
  success path is the same code in both versions.
- No pick was run: this measures inference latency, not grasp quality or the
  B36 pick-reliability signatures 1–3.
- GB10 / DGX Spark (aarch64) not measured.
- The client script that produced `client.txt` was not preserved; it called
  `health` and then `infer_object` three times with a 5 cm cube.
