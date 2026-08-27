# Booth runbook — WRC 2026 (Beijing)

A hands-on booth session for this rig, designed around hard constraints:

- **15 minutes end-to-end per group** of 3–5 attendees + 1 staff host,
  *including* the reset for the next group. That is throughput math: ~20
  people/hour/pod; multiple pods scale linearly. Overrunning one slot
  cascades into every slot behind it, so the script below has hard gates.
- **Attendees type, the robot moves.** The interface is a real agent chat
  (Claude Code / OpenClaw driving the MCP server), not a custom UI and not
  voice — expo halls defeat ASR, and a real terminal is the point: they are
  iterating on the actual application, not pressing our buttons.
- **First visible result within ~3 minutes** of sitting down. Anything
  slower must show evidence of progress (dashboard streams, narration feed,
  tool calls landing in the terminal) while the host narrates.
- **At least one surprise beat** and **one moment that proves the core
  message**: a long-running agent with persistent memory, learned grasps and
  autonomous recovery is categorically more than a natural-language remote
  control.
- Assume **no reliable venue internet**: local Qwen is the fallback brain,
  everything else must already be on disk.

Everything in this document is grounded in the current code; commands are
copy-pasteable on the rig.

---

## 1. Pre-conference checklist (do at home, not at the booth)

1. **Booth tuning is one env var:** `CASCADE_BOOTH=1` deep-merges
   `configs/booth.yaml` over `configs/demo.yaml` at load time, in every
   entry point (demo CLI, MCP server, dashboard runner). Dev runs without
   the env keep dev values; the regen command below bakes it into
   `.mcp.json`. The overrides and why:

   | key | dev default | `CASCADE_BOOTH=1` | why |
   |---|---|---|---|
   | `grasp.persist_seconds` | 120 | **45** | caps the retry loop: one unlucky pick can't eat 2 of 15 minutes |
   | `grasp.max_pick_attempts` | 8 | **3** | 3 visible attempts tell the recovery story; 8 tell a sad one |
   | `grasp.graspgenx.timeout_ms` | 8000 | **2000** | if the grasp server dies mid-day, each pick gains 2 s of dead air, not 8 |
   | `memory.horizon_s` | 15.0 | **60** | the narration feed must survive a talk-track pause instead of self-erasing |
   | `perception_loop.belief_fallback_age_s` | 3.0 | **20** | the object-permanence beat ("point at where it was") needs the remembered belief accepted after the cup has covered it for >3 s |

2. **Regenerate `.mcp.json` for the real rig** (the checked-in file pins the
   Isaac sim stack). One command now carries everything (there is no bare
   `python` on the rig — use the shared venv):

   ```bash
   source ../.demo/bin/activate
   python scripts/setup_agents.py --host claude --write \
       --camera l515 --arm rebot_rs \
       --detect-classes "pink cube,green cube,banana,disc,box,bowl,cup" \
       --hide-tools reset_stop --env CASCADE_BOOTH=1
   ```

   - `--detect-classes` must name **exactly what attendees will ask for**:
     beliefs are keyed by detector labels, so "banana" cannot resolve a
     belief labeled "fruit". Match the physical prop bin, word for word,
     and put the same nouns on the cheat card (§4).
   - `--hide-tools reset_stop` makes clearing an e-stop staff-only: without
     it, an attendee's model will helpfully call `reset_stop` and resume
     motion after staff froze the arm. (`emergency_stop` is never hideable.)
   - Offline env (`YOLO_OFFLINE`/`ULTRALYTICS_OFFLINE`) is now emitted by
     default — online ultralytics phones GitHub on class re-embeds and
     stalls the perception watcher for seconds per tick.

3. **Pre-download everything that fetches on first run:** the Qwen GGUF +
   mmproj (`scripts/serve_qwen_llamacpp.sh` clones/builds/downloads),
   GraspGen-X checkpoints, and detector weights. Then verify with the
   network cable pulled.

4. **Dry-run the offline brain:** `scripts/booth_rehearsal.py --llm
   local_qwen` runs the session prompts through the real orchestrator on
   the mock stack (booth tuning on, learned memory isolated to the run
   dir) and reports tool calls, steps and outcome per prompt. First run
   (2026-07-20, Qwen3.6-27B via llama.cpp): 6/6 prompts produced clean
   tool calls, 4/6 tasks succeeded — the two failures are the mock arm's
   designed air-grasp, which the model handled with sensible retries and
   an honest failure report. Re-run with the final cheat-card nouns
   (`--prompts card.txt`) before shipping the rig.

5. **Rehearse the scripted failure** (§5, beat 3): pick a prop (a flat foam
   disc works) that reliably *fails once then succeeds* after the grasp
   memory's z-nudge. This beat carries the whole session; tune the prop, not
   the script. Note the learned z-nudge is physically subtle (mm) — the
   *receipt* of learning is `preview_grasp`'s `memory_prior` block in the
   chat, not the naked eye.

6. **Onsite bring-up** stays as in `docs/ROADMAP.md` (CAN bus, gripper
   travel, hand-eye calibration, table plane, `pytest -m hardware`).

## 2. Day-of bring-up

```bash
scripts/serve_graspgenx.sh          # learned grasps (terminal 1)
scripts/serve_qwen_llamacpp.sh      # offline brain   (terminal 2)
scripts/booth_up.sh                 # pre-flight: red/green every silent killer
```

`booth_up.sh` exists because the worst failure modes here are **silent**:
GraspGen-X down degrades to the analytic OBB planner without an error
("booth rule"), a wrong launch directory makes every detection vanish
(YOLOE's text encoder resolves relative to the CWD), and online ultralytics
stalls the watcher. The script checks all of it, snapshots the morning
grasp-memory baseline, and prints the dashboard URL for the big screen —
use *that* URL, not `live_view_url`'s best-effort LAN probe, on an
air-gapped booth LAN.

Then open the attendee terminal: `cd <repo> && claude`. Perception
pre-warms immediately; motors stay unpowered until the first motion command.

**Big screen** = the dashboard (`http://<booth-ip>:8090/`): N camera tiles
with live detection boxes, the robot-narration feed (green actions, red
failures), the world-model object table where remembered-but-unseen objects
visibly dim, a **grasp-memory panel** (the day's learned priors, live), a
**via:** chip showing which tier served the last command
(reflex/experience/llm/mcp-host), and a **keyframes** link
(`/keyframes`) — the per-skill before/after evidence trail, auto-refreshing,
for the Q&A beat. Staff phone keeps the dashboard open **for the STOP
button** (§6).

## 3. Staffing & throughput

One **host** runs the script and owns all safety actions; attendees rotate
on one keyboard (turn-taking is the arbitration — the MCP session is a
single chat). If a second staffer is available, they float behind the group
for the slower technical conversations so the host never has to choose
between depth and the clock. Expect ~20 attendees/hour/pod and size the pod
count to expected traffic; the reset (§7) is designed to run *during* the
Q&A tail so the slot boundary is clean.

## 4. Props & cheat card

- 5–7 chunky, matte props that match `CASCADE_DETECT_CLASSES` exactly; 3 vetted
  spares in the catch bin. Taped outline zones keep attendee staging inside
  the reachable workspace (x 0.15–0.45 m, |y| ≤ 0.25 m).
- One opaque cup (host's pocket — the object-permanence beat), one flat
  disc (the scripted-failure prop), one catch bin placed where the throw
  beat was rehearsed to land (it doubles as the prop-return bin: the laugh
  pays the reset cost).
- A laminated **cheat card** of known-good prompts. Two constraints, both
  load-bearing: nouns exactly match the detector vocabulary, and phrasings
  also match the tier-1 reflex grammar verbatim ("wave", "point at the
  \<x\>", "hand me the \<x\>", "sort the objects by color", "go home",
  "open the gripper") — that makes the card double as the deepest LLM-outage
  fallback, because those exact strings run LLM-free in the demo CLI (§8).

## 5. The 15-minute script

Gates G1–G4 are on the host's phone timer and are **outcome-independent**:
when the gate fires you move on, win or lose — every beat has a scripted
exit line for the lose case (§8). One `pick_and_place` per group, ever.

| clock | beat |
|---|---|
| 0:00–0:45 | **Seat & stage.** Attendees place 2–3 props on the taped zones; detection boxes pop on the big screen as they land. Host: "It's already watching — motors wake on your first command. It's also been learning all day: it remembers every grasp it's botched since 9 a.m., including the ones the last group caused." |
| 0:45–1:45 | **First result** (never cut). Attendee 1 types: *"Wave hello to the group, then tell us what's on the table."* Wave has zero perception dependency — first visible motion lands well inside 3 minutes. |
| 1:45–2:30 | **The BEFORE receipt** (zero motion). Attendee 2: *"Before you touch anything — preview a grasp on the disc. Have you tried this object before?"* `preview_grasp` returns a `memory_prior` (times seen, success rate, what to avoid). Host reads it aloud. First group of the day: there is **no `memory_prior` block at all** for a never-attempted object — the host line becomes "no prior of any kind: you're the control group; the 4 p.m. robot will be better *because of you*." **G1 @ 2:30** — LLM-stall ladder (§8) if no tool call has landed yet. |
| 2:30–4:30 | **The failure, live.** Attendee 2: *"OK — pick up the disc."* The rehearsed prop fails once (red "grasp FAILED" line on the big screen). Host: "Don't fix anything. Watch what it does with that failure." *"Try again."* — narration shows the grasp-memory prior being applied; it holds. *"Put it back down."* **G2 @ 5:00.** |
| 4:30–5:15 | **The AFTER receipt** (zero motion — this is the shock absorber; stretch or compress it to land on time). *"Preview a grasp on the disc again."* Seen-count up, success rate updated, grasp geometry shifted. Host: "Before and after are two screens apart in your own scrollback. A stateless controller makes the identical mistake forever." |
| 5:15–7:45 | **Sabotage.** Attendee 3: *"Pick up the pink cube and put it in the box."* As the jaws descend, the **host** slides the cube 5 cm. Booth config caps this at 3 attempts / 45 s; narration streams the recovery ("attempt 2/3 — re-homing, re-scanning, planning a fresh grasp") with no one typing anything. **G3 @ 8:00.** |
| 7:45–9:15 | **Object permanence.** Host covers the green cube with the cup, theatrically. Attendee 4: *"Where is the green cube?"* — the answer comes from memory, with position and age, while the object's row dims on the big screen. *"Point at where it was."* — the arm points at the cup; host lifts it at the fingertip; the row re-brightens. (Requires the `belief_fallback_age_s` booth tuning, §1.) |
| 9:15–10:45 | **The throw.** The group picks who types: *"Grab the banana and throw it into the bin!"* Jaws pop open mid-arc. Host: "Every waypoint of that throw was still vetted by the safety harness at 50 hertz." **G4:** only if ≥60 s ahead — *"Hand me the pink cube"*: handover presents and holds until *"open the gripper"*. "Two-message protocol — it never lets go until you say so." |
| 10:45–11:30 | **Close** (never cut). *"Wave goodbye, then go home."* `move_home` parks the arm safely — mandatory posture between groups. Group photo on the wave. |
| 11:30–15:00 | **Q&A + reset in parallel** (§7). Talking points: the robot's "diary" (the dashboard's grasp-memory panel, plus `~/.cascade/GRASP_MEMORY.md` refreshed by the reset script — "your disc failure is in here now; the next group inherits it"), and the **keyframes** link on the big screen — this session's before/after evidence trail ("the black-box recorder of the last ten minutes", no file browser needed). Walk the group out at 14:30. |

## 6. Safety rules (host-only, non-negotiable)

- **Hardware e-stop in the host's hand during every motion.** Software
  layers below are conveniences, not the primary.
- **Software freeze paths, all of which now work mid-motion:** the
  dashboard STOP button (any browser on the LAN — wired up in MCP mode as
  of 2026-07-20); **Esc in Claude Code** during a motion tool (the
  cancellation notification freezes the arm rather than orphaning the
  motion — this is the practical in-chat path, since Claude Code won't send
  a new tool call while one is running); an `emergency_stop` frame from any
  host that pipelines requests (the server answers it out-of-band, ahead of
  the running call); first Ctrl+C on the server (latches the e-stop, no
  free-fall). A stop that lands while the runtime is still starting is
  remembered and applied the moment startup finishes.
- **Clearing a stop is staff-only** (`reset_stop` is hidden from the
  attendee session via `CASCADE_HIDE_TOOLS`, and the model then cannot call
  it): staff assesses, restages, then clears from a rig terminal with
  `pkill -USR1 -f cascade.apps.mcp_server` (SIGUSR1 is the staff reset
  channel; restarting the server also works but costs the warm state).
- **Never kill or disconnect with the arm loaded or raised** — exit
  disables torque and the arm falls. Park first (*"go home"*), always.
- Attendees never reach into the workspace while the arm is powered; the
  host does all staging inside the taped zones.

## 7. Reset between groups (target: 60 s of hands, run during Q&A)

```bash
scripts/booth_reset.sh              # keep the learned brain (default)
```

Prints the physical checklist (park → props to tape → `/clear` the chat)
and then verifies the **next** group's perception: every prop must show
LIVE in the dashboard's world model before you seat anyone — a
"remembered" ghost row at minute 0 becomes a mystery failure at minute 5.

**Memory policy is deliberate:** grasp memory and tier-2 habits persist all
day, because the robot measurably improving from morning to afternoon *is*
the demo. `--wipe-brain` exists for a cold-start day; `--restore-brain`
rolls back to the morning baseline if a bad streak contaminates the priors
(the running server keeps its in-memory copy — restart it to load the
restored file).

## 8. Fallback ladders

**Grasp fails (disc beat):** failure *is* the script. If it succeeds first
try: "it already learned this disc from an earlier group — here's the prior
it applied" (show `memory_prior`). If the retry also fails: run the AFTER
preview anyway — the failure was still recorded, "the next group inherits
the correction" — and move on at the gate.

**Grasp fails (sabotage beat):** capped at 3 attempts/45 s; exit at G3
regardless: "perfect — that failure just went into the diary; it'll be on
the big screen in three minutes."

**Grasping flaky all day** (host decides by the first slot): degraded
throw — *"open the gripper"* → host hand-places the prop → *"close the
gripper, then throw it in the bin"*. Zero perception dependency, laugh
intact.

**LLM stalls (G1 ladder, one-way — never switch back mid-session):**
1. Cloud dead → pre-warmed session on local Qwen (same MCP server, same
   prompts).
2. Local model mis-formats/loops → the **dashboard chat box** (same MCP
   server, no restart): in MCP mode it runs the tier-1 reflex grammar with
   zero LLM, so every cheat-card phrasing executes at rule-match latency —
   the **motion beats** all survive; the memory-receipt beats
   (preview/where-is) have no reflex mapping, so the host narrates those
   from the grasp-memory panel instead. This is why the card's wording is
   frozen (§4). Don't type in both chats at once — the fallback replaces
   the host, never runs beside it. (The demo CLI REPL remains an
   equivalent option if the dashboard is unreachable.)
3. Total loss → host-driven scripted pick + architecture talk over the
   dashboard; `scripts/record_demo.py` output as the attract loop.

**Detections missing:** off-vocabulary noun → "it only knows ten words
today — they're on the card"; prop present but boxless → nudge it 2 cm
(forces a fresh detection) or swap the vetted spare; *all* boxes gone →
wrong CWD / offline-stall signature: run only perception-free beats (wave,
degraded throw, handover, home), close the slot early, restart via
`booth_up.sh` during an extended reset.

**Dead air of any kind:** the narration feed is the talk track. "Green
lines are its actions, red are failures — that feed is its inner monologue.
Nobody has typed anything since the failure; everything you're watching is
recovery."
