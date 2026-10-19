# Cosmos3-Edge as the OpenClaw agent (2026-10-19)

**Summary.** On the OpenShell/OpenClaw route Cosmos3-Edge (2.4B reasoner, vLLM 0.32.0, NVFP4 KV) answered OpenClaw's own
captured requests with a native tool call 1 time in 55. The `cosmos` agent profile (`deploy/openshell/render-config.mjs`
+ `deploy/openshell/cosmos-plugin/`) gets 44/55 (NVFP4) and 43/55 (BF16) right first tool calls on the same requests:
thinking off, OpenClaw's JSON metadata envelopes removed, a short system prompt, a small tool set, temperature 0, one
call per reply. After a restart of the vLLM server the same bytes were answered differently (NVFP4 KV and the prefix
cache: see below), and the six first-turn requests of the live run fell to 9/18 (warm cache) and 12/18 (cold); with
`place_on_object`, `place_at` and `localize_object` removed from the profile they are 18/18 in both states. One tool runs
per request; a bounded multi-step loop was implemented and measured, and is not the default (it made unrequested robot
calls). The report turn after the tool reads the request and the results as one message: no JSON copies or junk.

| | first-turn requests right | replies with junk or a JSON copy | unrequested tool calls |
|---|---|---|---|
| OpenClaw as captured (49 tools, envelopes) | 1/55 NVFP4, 0/55 BF16 | n/a | n/a |
| profile, 13 tools (live2) | 44/55, 43/55; live2's six after a restart: 9/18 warm, 12/18 cold | 6/54 | 0 |
| **profile, 10 tools, one tool per request (shipped)** | live2's six after a restart: **18/18 warm, 18/18 cold** | **0/40** | **0** |
| bounded loop, 3 rounds, 13 tools | (same first turn) | 0/54 | 3/54 replays + 8/8 on the live capture; failed the readiness proof |
| bounded loop, 3 rounds, 10 tools | (same first turn) | 0/40 | 16 calls in 40 replays (move_home, reset_scene, pick_and_place) |

## Method

1. Branch `feat/cosmos-vllm-032` (PR #397); server `KV_DTYPE=nvfp4 scripts/serve_cosmos_vllm.sh` (vLLM 0.32.0,
   `--model-impl transformers`, `qwen3_coder` tool parser, FLASHINFER, `--max-num-seqs 2`) on an RTX PRO 6000 (SM120),
   behind a logging pass-through proxy that captured every request OpenClaw 2026.9.3 sent (`requests-profile/`).
2. First turn (`tools/replay_profile.py`, results `first-turn-v*.jsonl`): the captured OpenClaw request, varied one
   factor at a time (prompt, envelopes, tool set, temperature, `parallel_tool_calls`, frequency penalty), 14 operator
   prompts, 5 repetitions per prompt and KV dtype (BF16 on a second server). Measured before the machine was rebooted.
3. First turn after the restart (`tools/replay_firstturn.py`, `first-turn-live2-replay.jsonl`): the six first-turn
   requests of the live2 run, byte-identical (the plugin's first turn renders them unchanged), 3 repetitions, once as
   sent (warm prefix cache) and once with a fresh vLLM `cache_salt` per request (no prefix reuse), then with tool sets
   and prompt wordings varied; plus 8 more operator prompts.
4. After the tool (`tools/replay_loop.mjs`, `loop-*.jsonl`): the REAL plugin module (`wrapStreamFn`) drives the model
   through the whole request; the first call is pinned to the right one, every later call and every reply is the model's.
   Tool results are real robot-server results (live kitchen trace, mock-rig dumps, captured world_state) relabelled for
   the call's arguments; one case (`S_dropzone_live`) uses the live run's pick result verbatim. 20 requests: 9 single-step,
   7 multi-step, 3 failures, 1 live capture; 2-3 repetitions.
5. Live (Isaac Sim 6.2.0.0 kitchen, PhysX, reBot `isaac_kitchen_gpu`, GraspGen-X, Cosmos3-Edge also as the perception
   detector, grounding and judge): `scripts/launch.sh --sim isaac --brain cosmos ... --graspgenx external --judge vlm`,
   six operator prompts through the chat UI from a reset scene; scored from tool results and physics postconditions.

## Root causes (OpenClaw route)

- **Thinking on by default.** The export reasons unless told not to; vLLM's `qwen3` parser put the whole reply into
  `reasoning_content` and OpenClaw reported "Agent couldn't generate a response". `chat_template_kwargs:
  {enable_thinking: false}` (render-config `BRAINS`).
- **JSON envelopes.** OpenClaw 2026.9.3 adds ```json metadata to the system prompt (`### Message Context`) and to each
  user turn (`Conversation info: ⟦openclaw:ctx⟧`), with no config switch. The 2.4B model imitates them and answers with
  a ```json block instead of a tool call. The plugin strips them from what the model reads.
- **49 tools and a ~10k-character prompt.** Too many look-alike tools: `list_objects` drew pick, reset and pointing
  requests; `place_on_object` drew "put X in Y" before any pick; `localize_object` drew "point at X". Ten tools and a
  three-sentence prompt.
- **Whitespace loop on argument-less calls.** Greedy decoding of `<function=cascade__move_home>` continued with
  "\n   " until `maxTokens`; frequency penalty 0.5 ends it.
- **Multi-turn.** Asked again with tools after a result, the model never ended its turn (it re-placed a cube a verified
  pick had already placed, then retried a failure eight times), copied results back as JSON, and with two or more tool
  rounds in chat form answered with junk ("座", "ザ"). Hence one tool per request and a report turn without tools that
  reads the request and the results as one message.

## Server restarts change answers (NVFP4 KV + prefix cache)

The same request bytes (`0151` before the reboot, `0260` after: identical SHA-1 of messages and parameters) were answered
`pick_and_place(green cube, green square)` 2/2 before and `place_on_object(green cube)` 5/5 after. On the restarted
server the six live2 requests scored 9/18 as sent and 12/18 with a fresh `cache_salt` — a different set failing in each
state — so the greedy choice depends on whether the shared prefix (system prompt + tools) comes from the quantized
prefix cache. Several prompts sit at a decision boundary. The 44/55 above is one server's state; the table below is
the shipped tool set's margin across both cache states.

| live2 first-turn request | 13 tools, warm | 13 tools, cold | **10 tools, warm** | **10 tools, cold** |
|---|---|---|---|---|
| What can you see on the table? | 3/3 | 0/3 (JSON text) | 3/3 | 3/3 |
| Could you put the green cube in the green square? | 0/3 (place_on_object) | 3/3 | 3/3 | 3/3 |
| Please put the orange in the open box. | 3/3 | 3/3 | 3/3 | 3/3 |
| Point at the lemon. | 3/3 | 3/3 | 3/3 | 3/3 |
| Please put the pink cube in the green square, then put the lemon in the open box. (step 1) | 0/3 (JSON text) | 0/3 (place_on_object) | 3/3 | 3/3 |
| Let's start over. | 0/3 (move_home) | 3/3 | 3/3 | 3/3 |
| **total** | 9/18 | 12/18 | **18/18** | **18/18** |

Other prompts with the 10 tools (warm / cold, 3 each): the adjusted two-step request (step 1), "drop zone", "Go home.",
"Open the gripper.", "Stop!" 3/3 / 3/3; "What is the current world state?" 2/3 / 0/3 (describe_scene instead);
"Where is the tomato can?" goes to describe_scene (its positions), since `localize_object` is not offered.

Evaluated, not adopted (6 live2 requests + 7 more, warm and cold): stating "to put an object somewhere, call
pick_and_place; to start over, call reset_scene" (B: 12/22 and 10/22) or listing the main tools in the prompt (D: 21/22
and 19/22, but `point_at` arguments came back quoted, `"\"lemon\""`); frequency penalty 0.2 or 0 (same quoting).

## After the tool: one tool, or a bounded loop

| policy (plugin) | single-step right, no extra call | multi-step finished | step 2 right | extra calls | repeated calls | junk / JSON-copy replies |
|---|---|---|---|---|---|---|
| WIP: one tool, chat-form reply, 13 tools | 33/33 | 0/21 | 0/21 | 0 | 0 | 6/54 |
| one tool, one-message report, 13 tools | 39/39 | 0/21 | 0/21 | 0 | 0 | 0/60 |
| **one tool, one-message report, 10 tools (shipped)** | **26/26** | 0/14 | 0/14 | **0** | 0 | **0/40** |
| loop ≤3 rounds, decide-then-report, 13 tools | 33/33 | 6/21 | 9/21 | 3 | 0 | 0/54 |
| … on the live pick result (`S_dropzone_live`) | 0/2 (+ `place_on_object`) | | | | | |
| loop ≤3 rounds, 10 tools | 20/26 | 1/14 | 3/14 | 16 | 0 | 0/40 |

The loop (`maxToolRounds` 2-3 in the plugin config; default 1): after a successful result the model is asked, with tools
and a continuation prompt, whether the request still needs a step; a new call runs, no call or an exact repeat ends the
loop, and so do a failed result and the round limit; the report turn then answers. It finished "point at the lemon,
then go home" and "put the pink cube in the green square, then the lemon in the open box" on replays, but on the live
run's own pick result it called `place_on_object` 8 of 8 times (the launcher's readiness proof failed on it), and with
the 10 tools it added `move_home`, `reset_scene` and a `pick_and_place` of an unnamed "pink object" to single-step
requests. More explicit continuation wordings started more second steps and also invented `place_at` coordinates.

On three dumped two- and three-result report requests the chat form answered junk 3/3 and the one-message form 0/3
(`tools/report_variants.py`). Unfinished steps: for "put the pink cube in the open box, then the lemon in the green
square" the WIP chat-form reply claimed the lemon was placed (3/3); the one-message report says it was not done. With
13 tools the report still claimed an unfinished step for "Open the gripper, then go home" ("returned home", 3/3) and
the three-step request (3/3); with the shipped 10 tools no multi-step report claimed an unfinished step (0/14), some
announce it as next ("Next, I will put the lemon ...").

## Live run (Isaac kitchen, 2026-10-19)

Six operator prompts through the chat UI, one chat turn each, from a reset scene; scored from tool results and the
physics postconditions in the robot server's trace. live2: before the reboot, 13 tools, the WIP plugin. live4: after
the restart, 10 tools, the first point_at fix only. live5: the pushed tree (10 tools, both point_at fixes, main
merged), with an operator `reset_scene` before T5 because T3 leaves the orange in the open box.

| | prompt | live2 (13 tools) | live4 (10 tools) | **live5 (shipped)** |
|---|---|---|---|---|
| T1 | What can you see on the table? | describe_scene; reply misses the green cube (partial) | describe_scene; reply names the mat, box, cubes, orange, can | describe_scene; reply vague: "a wooden floor with various objects" (partial) |
| T2 | Could you put the green cube in the green square? | pick_and_place, placed (physics-confirmed) | pick_and_place, placed (confirmed) | **pick_and_place, placed (confirmed)** |
| T3 | Please put the orange in the open box. | pick_and_place, placed (confirmed) | pick_and_place, placed (confirmed) | **pick_and_place, placed (confirmed; 2 grasp attempts, 318 s)** |
| T4 | Point at the lemon. | point_at(lemon), failed: CUDA tensor to numpy | point_at(lemon), failed: CUDA tensor (remembered neighbours) | **point_at(lemon), pointed at [0.323, 0.002] (lemon spawn [0.32, 0.005])** |
| T5 | live2: pink cube → green square, then lemon → open box; live4/5: pink cube → open box, then lemon → green square | step 1 called; place did not settle (square held the green cube) | step 1 called; place did not settle (box held the orange) | **step 1: pink cube placed in the open box (confirmed); step 2 not attempted (one tool per request)** |
| T6 | Let's start over. | reset_scene, verified | reset_scene, verified | **reset_scene, every prop within 2 cm of its spawn (physics)** |
| | right first tool | 6/6 | 6/6 | **6/6** |

live3 (13 tools on the restarted server) called `place_on_object(green cube)` for T2 (refused: not holding anything)
and was stopped; that request is byte-identical to live2's (see above). Readiness proof after the restart: with 13 tools
NOT READY in 3 of 3 launches (the pick turn chose `material: "fragile"` and air-grasped twice; once a queued OpenClaw
turn reached the model as "Continue the current task ..." and it called describe_scene), and with the loop on, once
more (`place_on_object` after the pick); with the 10 tools READY in 2 of 2 (`pick_and_place(pink cube, drop zone)`,
rigid grip, placed and confirmed).

## Limits

- One RTX PRO 6000 workstation (x86). Not run on DGX Spark (GB10), Jetson Thor or the booth hardware.
- Replay results stand in for the robot: they are real robot-server results relabelled, not new executions.
- The first-turn numbers depend on the vLLM server's cache state; repetitions within one state are near-deterministic
  (temperature 0), so "3/3" measures one state, not a probability.
- The readiness proof's pick prompt ("Please put the pink cube in the drop zone.") was answered with
  `material: "fragile"` after the restart (13 tools, 3 of 3 launches); the fragile grip air-grasped the pink cube in 2
  of 3. Not investigated here.

## Re-run

Server and proxy (heavyrun units), then from `~/.hermes/cache/scratch/cosmos-demo` (paths in the tools):
- `python3 tools/replay_firstturn.py http://127.0.0.1:8083/v1 3 asis,salt,A+noplace+noloc,A+salt+noplace+noloc --extra`
  (~5 min; appends `firstturn.jsonl`).
- `TOOLSET=10 node tools/replay_loop.mjs 8083 deploy/openshell/cosmos-plugin/index.mjs out.jsonl 2 plugin` (one tool,
  ~2 min) and the same with `ROUNDS=3` (the loop); `python3 tools/summarize_loop.py out.jsonl --detail`.
- `python3 tools/replay_profile.py out.jsonl 5 <variant>` for the first-turn matrix (needs the captured OpenClaw
  request, not committed: it carries the session's full system prompt).
