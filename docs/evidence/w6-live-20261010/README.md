# Wave-6 live validations on Isaac Sim (10 Oct 2026): B46 read-only lane, B63 env forwarding, B44 judge turn

Three items that landed CPU-tested on 9 Oct owed one live Isaac run each. All three were run on the same rig, on
10 Oct 2026 between 05:24 and 06:13 CEST, from main 86373d7.

**Rig.**
- Isaac Sim 6.2 (internal build), PhysX, on GPU 0 (PCI 21:00) only. Each run got a fresh stage:
  `scripts/isaac_launch.py … scripts/isaac_bridge.py --engine physx` on the bare reBot RS scene
  (`assets/usd/RS-rebot-dev-arm/RS-rebot-dev-arm.usda`).
- Cameras `isaac,isaac_side`, arm `isaac`, occupancy off.
- GraspGen-X ran as a warm sidecar (`scripts/serve_graspgenx.sh`, warm-up 15.29 s before it bound).
- Private ports: bridge 47000, GraspGen-X 47001, VLA stub 47010. No shared port was dialled, except the local
  Qwen judge at `127.0.0.1:8080`, which was used as a client only.
- Before every Isaac start, `guard.sh` checked that no user benchmark was running and that nothing foreign was on
  GPU 0. It never fired.
- Task: `pick_and_place` the pink cube to the drop zone, as in the B36 pick-reliability series.

**Files.**
- `b46_lane_ab.json`, `b63_vla_env.json`, `b44_judge_turn.json`: per-run records, copied from the raw probe /
  driver JSON and the server traces.
- `manifest.json`: sha256 of every raw record they came from. The raw records are kept in the item directory and
  are not committed.
- The scripts that ran, verbatim: `bridge.sh`, `guard.sh`, `b46_*.sh`, `live_lane_probe.py`, `launcher_turn.py`,
  `b63_*.sh`, `vla_stub_logged.py`, `vla_chunks.json`, `judge_turn.sh`, `summarize_b46.py` and
  `build_evidence.py`.
- `tests/test_w6_live_evidence.py` recomputes every number below from the JSON and checks that the docs quote
  them.

## B46: read-only MCP lane during a live pick (#285)

**Commands.** `b46_series.sh` ran `b46_run.sh <lane> <tag> [stop_after]` in the order on1, off1, on2, off2, on3,
off3, on4, off4, then stop1. For each run:
1. `bridge.sh` brought up a fresh stage.
2. `live_lane_probe.py --cameras isaac,isaac_side --arm isaac --object "pink cube" --destination "drop zone"
   --lane 0|1` spawned the real `python -m cascade.apps.mcp_server` over stdio.
3. The probe started the pick, then sent four reads 1 s into the motion: `world_state`, `robot_knowledge`,
   `verify_last_action` and `camera_snapshot`.

This is the B46 probe; this copy only adds the server's cwd and log and a raw-frame dump.

| | lane on (on1–on4) | lane off (off1–off4) |
| --- | --- | --- |
| reads answered before the pick returned | 16 / 16 | 0 / 16 |
| marked `served_during_motion: pick_and_place` | 16 / 16 | 0 / 16 |
| `world_state` / `robot_knowledge` / `verify_last_action` latency | 0.001–0.003 s | 65.418–164.703 s |
| `camera_snapshot` latency | 0.024–0.038 s | 65.468–164.778 s |
| pick confirmed on the physics channel | 2 / 4 (on1, on4) | 1 / 4 (off3) |

- **Lane off.** Every read waited for the whole pick: its latency is the pick duration minus the 1 s read delay.
- **Pick failures (5 of 8).** All are the open B36 signatures, seen in both arms:
  - on2, on3 and off1: attempt 1 ended `did not settle at grasp lift pose` with the cube lifted. Every retry
    then refused `already holding 'pink cube'`.
  - off2 and off4: the cube was placed 11.4 cm and 12.0 cm from the drop-zone centre.
  - The lane's reads were all answered within 0.04 s of being sent, about 50 s before any lift.
- **Not marked after the motion.** A `world_state` sent after the motion returned was not marked in any run.
- **Stop (stop1, lane on, `--stop-after 3`).**
  - `emergency_stop` answered `stopped: true` 0.032 s after it was sent, before the pick returned.
  - The pick ended at 3.27 s: `simulation motion stopped (e-stop latched; not retrying)`, with the cube 0.0 cm from
    where it started.
  - The lane reads of that run also answered during the motion.
- **Smoke run.** It ran before the series (lane on, not counted above) and gave the same lane behaviour. Its pick
  failed with the B36 lift signature.

**Verdict: passed.** With the lane on, reads answer during a live pick in milliseconds and are marked. Pick outcomes
are no worse with the lane than without it (2/4 vs 1/4), and the stop still preempts.

## B63: a launcher-registered MCP entry carries `CASCADE_GRASP_EXECUTOR=vla` to a live server (#284)

**Commands.** `launcher_turn.py --tag <t> [--vla-port 47010] [--proof] [--fresh-memory]`, driven by `b63_run.sh`
(series 1) and `b63_run2.sh` (series 2). It does the following:
1. Extracts `scripts/launch.sh`'s registration heredoc **verbatim**, with the same regex
   `tests/test_mcp_env_forwarding.py` uses.
2. Runs it the way launch.sh does (`"$PY" - <args>` for `--sim isaac --occupancy none --headless`). The launch
   owner comes from `process_owner.py init`, and the profile is `hermes-live-w6`. The registering shell exports the
   private ports and `CUDA_VISIBLE_DEVICES=<GPU 0>`, plus `CASCADE_GRASP_EXECUTOR=vla CASCADE_VLA_PORT=47010` for
   the vla arm.
3. Starts the server from the entry: `command` plus `args` in `cwd`, with **only** the entry's `env`. The only
   additions are `PATH`, `HOME` and `PYTHONPATH=<wt>/src`, which stands in for a checkout's own editable `.venv`.
4. Sends the proof turn's tool calls over stdio.

The VLA stub replays `vla_chunks.json`: hold open at `home_q`, then close. vla1 used `scripts/serve_vla_stub.py`.
vla2 used the same `ScriptedPolicy` / `StubPolicyServer` classes with a request log added.

| run | entry env | server log / `world_state.backends` | pick |
| --- | --- | --- | --- |
| vla1, vla2 | `CASCADE_GRASP_EXECUTOR=vla`, `CASCADE_VLA_PORT=47010` | `[cascade] grasp executor: vla -> 127.0.0.1:47010 (policy=scripted-stub, chunks=2)`, `vla_policy=yes`; `grasp_executor: vla (…)` | `grasp failed after 8 attempts … air grasp: gripper closed fully, object not held`; cube 0.0 cm from start |
| an1–an5 | neither variable | no executor line, no `vla_policy` cell, no `grasp_executor` | analytic GraspGen-X pipeline: an5 confirmed (2.6 cm from the drop-zone centre); an1–an4 retried `already holding 'pink cube'` with the cube held at the home pose (the B36 lift signature's end state) |

- **What the stub logged (vla2).** It received **17** policy requests for the 8 grasp episodes: 3 chunks in the
  first episode, then 2 in each later one, because the stub's script position carries over and later episodes
  start on the close chunk.
- Every request carried the prompt `pick up the pink cube`, a 224×224×3 uint8 image, and the **measured** joints
  (near `home_q`) plus the jaw fraction.
- So the chunks ran on the Isaac arm through the harness, and the analytic jaw check refused the empty close.

**Verdict: passed.**
- A server started with only the env that launch.sh's registration wrote selected and ran the VLA executor against
  the live bridge.
- Without the pair, the same registration ran the analytic executor.

This was not driven through an actual MCP host (OpenClaw), because no gateway or brain was started. The
registration code, the entry and the server are the real ones, reproduced faithfully.

## B44: the launcher judge pass on a live proof turn (#274)

**The turn (an5).**
- `launcher_turn.py --proof` sent the calls `demo_proof.run_proof` has the brain make, directly over stdio:
  `world_state` → `pick_and_place {pink cube, drop zone}` → `reset_scene` → `world_state`.
- It then wrote the receipt with `demo_proof.py`'s own validators (`_bound_world`, `validate_pick_trace`,
  `validate_reset_trace`, `_write_receipt`).
- an5 was `verified: true`. Receipts an1–an4 were `verified: false`, for the reason given in the B63 table.

**The pass.** `judge_turn.sh` runs launch.sh's judge block verbatim (the lines between `# >>> judge pass` and
`# <<< judge pass`) with `JUDGE=vlm` and `CASCADE_JUDGE_CONFIG` set to the local Qwen config (shape in
`b44_judge_turn.json`; any key works).

**Results.**
- Banner: `judge: advisory judge=vlm:Qwen/Qwen3.8-27B mode=incremental scored=1/1 agreement=100% tp=1 tn=0 fp=0
  fn=0 final_progress=1.00 wrist=front-repeat:1 (physics verdict unchanged; …/run-summary.json)`, after 7.73 s.
- The judge scored the pick hop +1.0 (`<score>+100%</score>`), and physics said `confirmed`.
- `run-summary.json` confusion: tp=1 tn=0 fp=0 fn=0 (n_scored 1).
- `proof.json` sha256 `3ac715a48fe3df7e57e3e847d1c082bb87ab8e164c2f7906c257d469bed69c44` was the same before and
  after the pass, for both the receipt and its evidence copy.

**Verdict: passed, for one turn.**

## Not claimed

- **No MCP host and no brain.** No OpenClaw gateway and no LLM brain were involved: B63 and B44 drove the
  registered server's tools directly. The full `./run.sh --sim isaac --judge vlm` launch, with OpenClaw and the
  Qwen brain, has still not been run.
- **Small samples.** 4 + 4 interleaved B46 runs and one stop run are a small sample. "No worse" means 2/4 vs 1/4
  confirmed picks, not a measured equality of pick rates.
- **The stub is not a policy.** The VLA stub replays two scripted chunks and never looks at the image. No real
  policy ran, and no VLA grasp succeeded or was expected to.
- **One judged step.** B44 judged a single confirmed step. A refuted proof turn never reaches the judge, because
  the receipt is unverified and the pass skips it. Nothing here adds to the B44-live matrix over 72 recorded picks.
- **No Newton, kitchen or real-arm runs.**

## Finding for the coordinator (not fixed here)

On this rig the analytic pick hit the open B36 lift signature in 8 of 14 picks today: 4 of the 9 B46-probe picks,
including the smoke run and excluding stop1, and 4 of the 5 launcher-style turns.
- **The signature.** Attempt 1 ends `did not settle at grasp lift pose` with the cube held. Every retry then
  refuses `already holding 'pink cube'`, and the cube ends held at the home pose, 34.1 cm from the target.
- **an1–an4.** They had no grasp-evidence directory, so their first-attempt message was not recorded. They show the
  same retries and the same end state, and their scene reset then failed with `did not settle at home`.

That is the ROADMAP's open B36 item, and it is now the main reason a proof turn is unverified (an1–an4). The lane,
the env forwarding and the judge are not involved.
