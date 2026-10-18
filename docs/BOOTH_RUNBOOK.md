# Booth runbook — hands-on expo session

**Next action:** do the [pre-conference checklist](#1-pre-conference-checklist-do-at-home-not-at-the-booth)
at home (about 2 h, most of it the offline downloads and the rehearsal).

Scope: the physical tabletop rig. Other booths:

- DGX Spark simulated kitchen: [presenter card](#9-simulated-kitchen-on-dgx-spark-presenter-card)
  and [Spark setup](DGX_SPARK_SETUP.md). It requires real GraspGen-X inference and never uses
  the analytic fallback below.
- PAAI (Physical Agentic AI) at Build a Claw on Brev: [staff guide](#10-paai-on-brev-staff-guide)
  and [Brev deployment](BREV.md). Simulated arm, OpenClaw on the booth computer, a public page
  with cameras only; the staging, throw and hardware-stop procedures here do not apply.

Generic/local profile fallbacks and resets must never bypass required GraspGen-X, a latched
stop or a retained payload episode. Capability status: [ROADMAP](ROADMAP.md).

Design constraints:

1. **15 minutes per group** of 3–5 attendees + 1 host, reset included (~20 people/hour/pod;
   pods scale linearly). Hard gates keep one overrun from cascading.
2. **Attendees type, the robot moves**: a real agent chat (Claude Code / OpenClaw on the MCP
   server), no custom UI, no voice (expo halls defeat ASR).
3. **First visible result within ~3 minutes**; slower beats must show progress (dashboard,
   narration feed, tool calls).
4. **One surprise beat and one proof of the core message**: persistent memory, learned grasps
   and autonomous recovery are more than a natural-language remote control.
5. **No reliable venue internet**: local Qwen is the fallback brain; everything else is on disk.

## 1. Pre-conference checklist (do at home, not at the booth)

1. **Booth tuning is one env var.** `CASCADE_BOOTH=1` deep-merges `configs/booth.yaml` over
   `configs/demo.yaml` in every entry point (demo CLI, MCP server, dashboard runner). Without it
   dev values stay; the regen command below bakes it into `.mcp.json`.

   | key | dev default | `CASCADE_BOOTH=1` | why |
   |---|---|---|---|
   | `grasp.persist_seconds` | 120 | **45** | one unlucky pick can't eat 2 of 15 minutes |
   | `grasp.max_pick_attempts` | 8 | **3** | 3 visible attempts tell the recovery story |
   | `grasp.graspgenx.timeout_ms` | 8000 | **2000** | a dead grasp server costs 2 s of dead air per pick, not 8 |
   | `memory.horizon_s` | 15.0 | **60** | the narration feed survives a talk-track pause |
   | `perception_loop.belief_fallback_age_s` | 3.0 | **20** | "point at where it was" accepts the remembered belief after the cup covered it >3 s |
   | `stream.mode` | lazy | **eager** | the big screen is live before doors open and stays up between visitors |

2. **Regenerate `.mcp.json` for the real rig** from the repo root (the checked-in file pins the
   Isaac sim stack; the rig has no bare `python`, so call the shared booth venv `../.demo` directly):

   ```bash
   ../.demo/bin/python scripts/setup_agents.py --host claude --write \
       --camera l515 --arm rebot_rs \
       --hide-tools reset_stop --env CASCADE_BOOTH=1
   ```

   - No `--detect-classes`: the brain names whatever is on the table and "pick the X" uses the
     attendee's words. A vocabulary makes perception a closed set and costs one brain request
     per class per refresh.
   - `--hide-tools reset_stop` makes clearing an e-stop staff-only (otherwise the attendee's
     model calls `reset_stop` and resumes motion). `emergency_stop` is never hideable.
   - Offline env (`YOLO_OFFLINE` / `ULTRALYTICS_OFFLINE`) is emitted by default: online
     ultralytics (SAM) looks assets up on GitHub and can stall the watcher for seconds per tick.

3. **Pre-download everything**: Qwen GGUF + mmproj (`scripts/serve_qwen_llamacpp.sh` clones,
   builds, downloads), GraspGen-X checkpoints, SAM weights (`models/sam2.1_l.pt`). Verify with
   the network cable pulled.

4. **Dry-run the offline brain**, from the repo root with the same venv:
   `../.demo/bin/python scripts/booth_rehearsal.py --llm local_qwen`
   runs the session prompts through the real orchestrator on the mock stack (booth tuning on,
   learned memory isolated to the run dir) and reports tool calls, steps and outcome per prompt.
   Re-run with the final cheat-card nouns (`--prompts card.txt`) before shipping the rig. (First
   run, Qwen3.6-27B: 6/6 clean tool calls, 4/6 tasks; both failures were the mock arm's designed
   air-grasp.)

5. **Rehearse the scripted failure** (§5, the disc beat): pick a prop (a flat foam disc) that
   reliably *fails once then succeeds* after the grasp memory's z-nudge. Tune the prop, not the
   script. The nudge is millimetres: the receipt is `preview_grasp`'s `memory_prior` block.

Onsite hardware bring-up (CAN bus, gripper travel, hand-eye calibration, table plane,
`pytest -m hardware`): the [onsite checklist](ROADMAP.md#onsite-bring-up-checklist).

## 2. Day-of bring-up

```bash
scripts/serve_graspgenx.sh          # learned grasps (terminal 1)
scripts/serve_qwen_llamacpp.sh      # offline brain   (terminal 2)
scripts/booth_up.sh                 # pre-flight: red/green every silent killer
```

1. `serve_graspgenx.sh` opens :5556 only after one warm-up inference (the first CUDA inference
   takes ~15 s, past the client's 8 s timeout): wait for its `warm-up inference ... before
   binding` line. `WARNING: warm-up inference failed` means it bound cold: restart it before
   doors open.
2. `booth_up.sh` checks the offline flags, the SAM weights (missing = boxes without masks), the
   perception brain and GraspGen-X (optional profiles report the analytic OBB fallback and
   retry the learned server after a 5 s cooldown; the Spark profile errors instead). It
   snapshots the morning grasp-memory baseline and prints the dashboard URL for the big screen:
   use that URL, not `live_view_url`'s LAN probe, on an air-gapped LAN.
3. Object names come from the Qwen brain at `CASCADE_PERCEPTION_URL` (default
   `http://127.0.0.1:8080/v1`, the llama.cpp brain above; the rig's OpenShell instance uses
   :8081). Measured on Qwen3.8-27B NVFP4 (vLLM); the llama.cpp Q4 GGUF serves the same model,
   unmeasured.
4. Open the attendee terminal: `cd <repo> && claude`. Perception pre-warms at once; motors stay
   unpowered until the first motion command.

**Big screen** = the dashboard (`http://<booth-ip>:8090/`): camera tiles with live boxes, the
narration feed (green actions, red failures), the world-model table (remembered-but-unseen
objects dim), the **grasp-memory panel**, a **via:** chip (reflex/experience/llm/mcp-host) and
the **keyframes** link (`/keyframes`, per-skill before/after evidence). Staff phone keeps the
dashboard open **for the STOP button** (§6).

## 3. Staffing & throughput

One **host** runs the script and owns all safety actions; attendees rotate on one keyboard (the
MCP session is a single chat). A second staffer floats for technical conversations. Expect ~20
attendees/hour/pod; the reset (§7) runs during the Q&A tail.

## 4. Props & cheat card

1. 5–7 chunky, matte props the brain names reliably (rehearse "pick the <noun>"), 3 vetted
   spares. Taped zones keep staging inside the reachable workspace (x 0.15–0.45 m, |y| ≤ 0.25 m).
2. One opaque cup (object permanence), one flat disc (scripted failure), one catch bin where the
   throw was rehearsed to land (doubles as the prop-return bin).
3. A laminated **cheat card**: nouns are the rehearsed prop names, and phrasings match the tier-1
   reflex grammar verbatim ("wave", "point at the \<x\>", "hand me the \<x\>", "sort the objects
   by color", "go home", "open the gripper"), so the card is also the LLM-outage fallback (§8):
   those exact strings run LLM-free in the dashboard chat box and the demo CLI.

## 5. The 15-minute script

Gates G1–G4 are on the host's phone timer and are **outcome-independent**: when a gate fires,
move on, win or lose (exit lines in §8). One `pick_and_place` per group, ever.

| clock | beat |
|---|---|
| 0:00–0:45 | **Seat & stage.** Attendees place 2–3 props on the zones; boxes pop on the big screen. Host: "It's already watching, and it remembers every grasp it botched since 9 a.m." |
| 0:45–1:45 | **First result** (never cut): *"Wave hello to the group, then tell us what's on the table."* No perception dependency. |
| 1:45–2:30 | **BEFORE receipt** (no motion): *"Before you touch anything — preview a grasp on the disc. Have you tried this object before?"* `preview_grasp` returns `memory_prior` (times seen, success rate, what to avoid); the first group of the day gets none ("you're the control group"). **G1 @ 2:30**: LLM-stall ladder (§8) if no tool call yet. |
| 2:30–4:30 | **The failure, live.** *"OK — pick up the disc."* It fails once (red line). Host: "Don't fix anything." *"Try again."*: the memory prior is applied and it holds. *"Put it back down."* **G2 @ 5:00.** |
| 4:30–5:15 | **AFTER receipt** (no motion; stretch or compress to land on time): *"Preview a grasp on the disc again."* Seen-count up, success rate updated, geometry shifted. |
| 5:15–7:45 | **Sabotage.** *"Pick up the pink cube and put it in the box."* The **host** slides the cube 5 cm as the jaws descend; 3 attempts / 45 s cap; narration streams the recovery. **G3 @ 8:00.** |
| 7:45–9:15 | **Object permanence.** Host covers the green cube with the cup. *"Where is the green cube?"* answers from memory (position, age) while its row dims. *"Point at where it was."*; host lifts the cup at the fingertip. Needs the `belief_fallback_age_s` tuning (§1). |
| 9:15–10:45 | **The throw.** *"Grab the banana and throw it into the bin!"* Every waypoint is still vetted by the harness at 50 Hz. **G4**, only if ≥60 s ahead: *"Hand me the pink cube"* holds until *"open the gripper"*. |
| 10:45–11:30 | **Close** (never cut): *"Wave goodbye, then go home."* "Go home" is `move_to_rest`: folded rest pose, verified, motors off (mandatory between groups). "Initial pose" is the raised `move_home`. |
| 11:30–15:00 | **Q&A + reset** (§7): the grasp-memory panel and `~/.cascade/GRASP_MEMORY.md` (refreshed by the reset script), and the keyframes link. Walk the group out at 14:30. |

## 6. Safety rules (host-only, non-negotiable)

1. **Hardware e-stop in the host's hand during every motion.** Software stops are conveniences,
   not the primary.
2. **Software freeze paths (all work mid-motion):** the dashboard STOP button (any browser on
   the LAN); **Esc in Claude Code** during a motion tool (the cancellation freezes the arm); an
   `emergency_stop` frame from a host that pipelines requests (answered out-of-band); first
   Ctrl+C on the server (latches the e-stop, no free-fall). A stop during startup is applied
   when startup finishes.
3. **Clearing a stop is staff-only**: `reset_stop` is hidden via `CASCADE_HIDE_TOOLS`. Staff
   assesses, restages, then clears from a rig terminal with `pkill -USR1 -f
   cascade.apps.mcp_server` (SIGUSR1 is the staff reset channel; a server restart also works
   but loses the warm state).
4. **Never kill or disconnect with the arm loaded or raised**: exit disables torque and the arm
   falls. Park first (*"go home"* = `move_to_rest`; wait for `safe_to_power_off: true`).
5. Attendees never reach into the workspace while the arm is powered; the host stages.

## 7. Reset between groups (target: 60 s of hands, run during Q&A)

1. Say **"reset the scene"** in the chat (a reflex, works with the brain down; also the
   `reset_scene` tool). It moves the arm to its raised home pose (`move_home`, not the park),
   restores SIM props, opens closed empty jaws, clears the world model and task-memory frames
   and takes one fresh observation. In SIM a held object is released at home. On the real rig
   (the reBot) it refuses before any motion while the jaws hold something, also an object
   they stalled on when a STOP interrupted the grasp: place it first (`place_at`; while the
   approach column is kept after a STOP at the object, `move_home` first, see §8.3). Jaws that
   closed on air after such a STOP hold nothing: the reset runs and opens them at home. It
   judges them like the grasp's own check: the rigid close on air stops at 85 % of travel
   (15 % open, 13.5 mm), and jaws up to 3 % wider (16.2 mm) also count as air when the plan
   sized the object at 19.8 mm or more. So an object about 13.5-16.2 mm thick that the plan
   sized that wide is opened at home and drops: after a STOP on such a thin object, take it
   out by hand before the reset. After such a STOP it also refuses when the jaws cannot be
   read. While the approach column is kept, `move_home` first (`open_gripper` first gives the
   column up and the arm stays frozen at the object); then, at home (about 0.44 m up), check
   the jaws and take the object out by hand before `open_gripper` (opening drops it from
   there), then `reset_scene`. Otherwise: take the object out by hand, then `open_gripper`,
   then `reset_scene`. Real props do not teleport (`props_reset: []`): put them back on the tape.
2. Run `scripts/booth_reset.sh` (keeps the learned brain). It prints the physical checklist
   (park → props to tape → `/clear` the chat) and verifies every prop shows LIVE in the world
   model before you seat anyone: a "remembered" ghost row at minute 0 is a mystery failure at
   minute 5.
3. Start every group in a **new chat session** (`/clear` or a new `--session-id`); a stale
   session shows the previous group's `task_memory`.

**Memory policy:** grasp memory and tier-2 habits persist all day (improving from morning to
afternoon *is* the demo). `--wipe-brain` is for a cold-start day; `--restore-brain` rolls back
to the morning baseline after a bad streak (restart the server to load the restored file).

## 8. Fallback ladders

1. **Grasp fails (disc beat):** failure is the script. First-try success: "it already learned
   this disc" (show `memory_prior`). Retry fails too: run the AFTER preview anyway and move on
   at the gate.
2. **Grasp fails (sabotage):** capped at 3 attempts / 45 s; exit at G3 ("that failure just went
   into the diary"). **Grasping flaky all day:** degraded throw: *"open the gripper"* → host
   hand-places the prop → *"close the gripper, then throw it in the bin"*.
3. **Tool call cancelled mid-motion** (host timeout or Esc): the arm freezes and the e-stop
   LATCHES. Staff `reset_stop`, then `reset the scene`. **Stopped with the gripper at the object
   (reBot):** the first motion call after the stop must be `move_home` or `reset the scene`, within
   5 min: it raises the gripper straight up its approach column first. `open the gripper`,
   *"go home"* (`move_to_rest`, §6.4) or a relative move first gives that column up and the arm
   stays frozen (jog it by hand). Object still in the jaws: `move_home` (within those 5 min the
   reset refuses and says so), then `place_at`, then `reset the scene`. If `move_home` reports
   `holding_unknown` (the jaws could not be read), the arm is at home height (about 0.44 m):
   check the gripper by hand there and take the object out before `open the gripper` (opening
   drops it from that height), then `reset the scene`. A refusal that said `move_home` first,
   followed by a `move_home` after the 5 min or after the arm was jogged out of the column, gets
   the old home move: frozen while the gripper is still at the object (jog it by hand). The
   launcher registers the server with a 300 s budget so a persistent pick (60–120 s) is never
   cut by the host's 60 s default; picks dying at exactly 60 s mean a stale registration:
   `./run.sh <mode>` re-registers.
4. **LLM stalls (G1 ladder, one-way):** cloud dead → pre-warmed local Qwen session (same MCP
   server). Local model loops → the **dashboard chat box** (MCP mode runs the tier-1 reflex
   grammar with zero LLM: every cheat-card motion beat survives; narrate the preview/where-is
   beats from the grasp-memory panel; never type in both chats). Dashboard unreachable → the demo
   CLI REPL is the equivalent fallback (same LLM-free strings). Total loss → host-driven
   scripted pick plus architecture talk; `scripts/record_demo.py` output as the attract loop.
5. **Detections missing:** off-card noun → "the words are on the card"; one prop boxless →
   nudge it 2 cm or swap the spare; *all* boxes gone → wrong CWD / offline-stall signature:
   run only perception-free beats (wave, degraded throw, handover, home), close early, restart
   via `booth_up.sh`. **Dead air:** the narration feed is the talk track.

## 9. Simulated kitchen on DGX Spark (presenter card)

To explain the demo: [simple overview](assets/paai_demo_simple_light.png) for visitors,
[full pipeline](assets/paai_demo_pipeline_light.png) for technical questions
(dark: [simple](assets/paai_demo_simple_dark.png), [pipeline](assets/paai_demo_pipeline_dark.png)).

1. Start: `python3 scripts/desktop.py launch --repo "$PWD"` from the installed checkout (Spark profile, Qwen brain).
   Wait for **READY**, then check the cameras and the `grasp_planner=graspgenx (learned 6-DoF)` banner.
2. Demo: one request at a time, e.g. "What can you see on the table?", then
   "Could you put the green cube in the green square?", then "Let's start over."
3. Restore objects: **Start over** in the Spark chat, **Reset** in PAAI OpenClaw, or "Reset the scene."
   Wait for the running order's result first.
4. After a crash with Qwen still up: `./run.sh isaac --engine physx --graspgenx local --no-robot-turn --no-judge`
   (STARTED / UNVERIFIED, no READY). Do not start a second model on an occupied port 8080.
5. Stop: `./run.sh down`.

Move props in Isaac only while the arm is idle and physics is playing.
If a reset fails or a placement is unverified, check the cameras and the reported error before any further movement.

## 10. PAAI on Brev (staff guide)

The staff guide is [`deploy/brev/staff.html`](../deploy/brev/staff.html), served at `/staff/`. The private
`/guide/`, `/openclaw/` and `/cameras/` links work only over the booth's Tailscale network; the public attendee
link shows cameras only. After each order click **Reset**, wait for **Scene reset** and **Live** in all three
cameras, then click **+** for the next attendee.
