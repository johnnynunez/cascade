# Agentic upgrades (2026-07-31)

Six papers, one thesis: **the frozen policy is rarely the bottleneck — the
loop around it is.** Pigey names the measurable version of this the
*orchestration gap* (12.8% → 53.3% on LIBERO-PRO with identical weights;
16.7% → 97.3% on a real Franka). VIA reaches 96.7% on LIBERO-Goal with an
off-the-shelf agent and no robot fine-tuning at all. ASPIRE's largest single
ablation gain (14% → 62%) comes from per-primitive evidence, not a better
policy. Harness-VLA beats baselines by 38.6 points by learning the *operating
range* of primitives it never retrains.

This document maps each idea onto what shipped here, and — as importantly —
what was deliberately **not** built.

## What each source contributed

| Source | Idea taken | Landed as |
|---|---|---|
| **Pigey** (2607.21725) | Closed-loop orchestrator that *tracks and verifies outcomes from observation* and recovers | `agent/effects.py` — postconditions; `sim/truth.py` — independent truth channel |
| **Agentic-VLA** (2605.22896) | Decompose into checkable sub-goals; use progress as signal; VLM critic on failure | `agent/milestones.py` — symbolic+visual milestone verification, stall detection |
| **Harness-VLA / RPent** (2607.08448) | Learn the *operating range* + failure model of a fixed primitive library instead of growing it | `memory/envelope.py` — per-primitive envelopes, failure taxonomy |
| **ASPIRE** (NVIDIA GEAR) | Diagnose traces → repair → distil validated fixes into a retrievable skill library | `agent/aspire.py` + `scripts/learn_from_runs.py` |
| **VIA** (2607.11119) | Give the agent an *interface it can read*, not raw pixels | `perception/visual_interface.py` — numbered marks, metric grid, reachability overlay |
| **Waddle Labs** | Agents that write/test/improve behaviour on real hardware | The outer loop: `learn_from_runs.py` as a between-session job |

## The four mechanisms

### 1. Effect verification (Pigey) — the important one

Before this change every skill **self-reported**. `grasp_object` returned `ok`
when the jaw stopped short of fully closing (`air_grasp_frac`) — a *proxy* for
holding something that cannot distinguish a grasped cube from a jammed finger,
and says nothing about whether the object left the table.

Now each motion primitive has a postcondition checked against a channel the
actuator does not own:

```
grasp_object  -> the object's own pose must RISE >= 1.5 cm
pick_and_place-> the object must be measurably displaced / near the target
push_object   -> displacement >= 30% of the requested distance
place_on_object -> held object centred over the target and above it
```

Verdicts are `confirmed` / `refuted` / `unverified`. **A refuted postcondition
downgrades a self-reported success** (`annotate_result`), and the orchestrator
escalates it loudly instead of building on a false premise. `unverified` is a
first-class outcome: the demo's real failure mode was never "the robot lied",
it was "the robot did not know".

In sim the channel is physics truth read through the bridge; on the real rig it
degrades to the belief store automatically — same code path.

Verified live: `pink cube moved 30.8 cm to (0.178, -0.157)`, `channel: physics`.

### 2. Milestone progress (Agentic-VLA)

The orchestrator already decomposed tasks and already shipped a `VERIFY_USER`
prompt — but nothing ever called it, so milestones were decoration. They are now
checked in two tiers:

1. **symbolic** — resolved against the belief store (`"the cube is in the bin"`
   is decidable from 3D beliefs, zero tokens, no hallucination surface);
2. **visual** — the VLM with `VERIFY_USER`, rate-limited, only when the
   symbolic tier abstains.

Unverifiable milestones report `UNKNOWN` rather than guessing, and a claimed
success carrying unverified milestones is annotated in the final summary. Three
motions with no milestone advance triggers a strategy change instead of a
blind retry.

### 3. Operating envelopes (Harness-VLA)

The repo learned grasp geometry per object profile (`GraspOutcomeMemory`). That
pattern now generalises to **every** primitive: each `(skill, numeric arg)` pair
accumulates the range where calls actually succeeded, plus a normalised failure
taxonomy (`geometry:link_below_table`, `kinematics:ik_unreachable`,
`contact:air_grasp`, …).

This turns the B601-RS's brutal IK envelope from tribal knowledge in a human's
head into data the agent reads at task start. Cold start is silent by design
(`MIN_SUPPORT`), and the check is advisory — the harness remains the only
authority that refuses motion ("booth rule").

### 4. Readable interface (VIA)

`annotated_view` renders what the agent actually needs: numbered badges per
object (so it says "object 2", not a label string it hopes matches), a 5 cm
base-frame grid, the TCP, and the top-down IK band shaded green. Projection
verified against live frames — badges land on the cubes.

## The outer loop

`scripts/learn_from_runs.py` runs **between** sessions (cron / Hermes job):

```bash
python scripts/learn_from_runs.py --report
```

It folds every `runs/*/trace.jsonl` into the envelope model and distils
validated repairs (a failure followed by the same primitive succeeding) into
`skills_library/*.md`, deduped by `(skill, signature)`. The orchestrator
retrieves guard-matched entries at task start.

Learning deliberately does **not** happen mid-demo: it would change behaviour
under the audience's feet and burn booth seconds. Each morning's robot is
better than last night's, with no retrained weights.

## Cosmos3-Edge

`configs/llm/cosmos3_edge.yaml` + `agent/cosmos3.py` + `scripts/serve_cosmos3_edge.sh`.

Why it fits: trained for Physical AI reasoning, natively multimodal (no mmproj),
accepts video at ~4 fps, 256K context, and BF16 ≈ 8.6 GB — it coexists with
Isaac Sim on one RTX PRO 6000 where Qwen3-VL Q4_K_M (~18 GB + projector) is a
squeeze.

**The catch that justifies a dedicated backend:** Cosmos3-Edge does not emit
OpenAI-style JSON `tool_calls`. It emits XML:

```xml
<tool_call><function=grasp_object><parameter=label>
pink cube
</parameter></function></tool_call>
```

Pointing the existing `openai_compat` profile at it "works" and silently never
calls a single tool — every turn parses as prose. `Cosmos3EdgeClient` parses
both shapes and coerces parameters against the JSON schema in `TOOL_SPECS`, so
`distance_m` arrives as a float rather than `"0.08"`. Use `type: cosmos3`.

## Deliberately not built

- **ASPIRE's evolutionary program search.** Parallel program variants need a
  reset-able environment and a long compute budget; a 15-minute booth slot has
  neither. The diagnose→distil→retrieve loop is the part that compounds.
- **VIA's browser 3D interface.** The annotation idea transfers; a second
  rendering stack does not earn its keep next to the existing dashboard.
- **Agentic-VLA's GRPO online adaptation.** That needs a VLA policy to adapt;
  the repo has no policy weights yet. Tracked in the ROADMAP behind the VLA
  executor.
- **Hard-blocking on learned envelopes.** Advisory only. A learned prior must
  never veto the safety harness or stall a live demo.

## Files

```
src/wrc_demo/agent/effects.py             Pigey postconditions
src/wrc_demo/agent/milestones.py          Agentic-VLA milestone verification
src/wrc_demo/agent/aspire.py              ASPIRE diagnose / distil / retrieve
src/wrc_demo/agent/cosmos3.py             Cosmos3-Edge client (XML tool calls)
src/wrc_demo/memory/envelope.py           Harness-VLA operating envelopes
src/wrc_demo/perception/visual_interface.py  VIA annotated view
src/wrc_demo/sim/truth.py                 physics-truth verification channel
src/wrc_demo/apps/live_control.py         on-demand live-view lifecycle
scripts/learn_from_runs.py                the outer loop
scripts/serve_cosmos3_edge.sh             vLLM-Omni serving
configs/llm/cosmos3_edge.yaml             Cosmos3-Edge profile
tests/test_agentic_upgrades.py            29 tests
tests/test_live_view.py                   20 tests
```

## The UI: headless-first, cameras on demand

The chat client (Hermes / OpenClaw / any MCP host) is the interface. The
browser dashboard is a **diagnostic surface you attach**, not the product —
so nothing binds a port at startup.

Why this is not just tidiness: every open MJPEG stream re-encodes JPEGs at the
stream rate whether or not a human is looking, and a bound port on a booth LAN
is an attack surface nobody asked for. Perception is unaffected — the
CameraRig keeps pumping and the WorldWatcher keeps beliefs warm — so the agent
answers "what do you see?" instantly with no dashboard at all.

**Modes** (`stream.mode`, overridden by `WRC_STREAM`):

| mode | behaviour |
|---|---|
| `lazy` *(default)* | binds on first `open_live_view`; auto-closes after `idle_timeout_s` (900 s) with nobody watching |
| `eager` | binds at startup — pinned in `configs/booth.yaml`, because the big screen must be live before doors open |
| `off` | never binds. `WRC_STREAM=0` maps here and stays a hard kill switch |

**Skills** (all in `TOOL_SPECS`, so chat and MCP both see them):

- `analyze_scene` — the headless answer to "what do you see?": per-camera
  detections + confidence, depth **quality** (source, min/median/max, valid
  fraction), scene description, numbered object key. No image tokens.
- `open_live_view` / `close_live_view` / `live_view_status` — attach and
  detach the browser UI; `live_view_url` opens it implicitly.

**The dashboard**, when open, has all cameras together with per-tile view
switches, plus the chat that drives the same robot:

- **rgb** — detections + HUD (what the *detector* sees)
- **depth** — colormap + range stats (what the *geometry* sees)
- **agent** — VIA marks, metric grid, reachable IK band (what the *agent*
  reasons on)

The depth view earns its place because `depth_source` degrades silently
(`sensor -> mono -> plane -> none`) and a wrong grasp z is usually a depth
problem the RGB view physically cannot show. Verified live: `depth: sensor,
min 0.599, median 0.792, max 1.282 m (83% valid)`.

Switching a tile swaps the `<img>` src, which tears down the previous MJPEG
socket — only one stream per tile ever encodes, so 3 cameras x 3 views stays
affordable. An open MJPEG socket also refreshes the idle timer, so a viewer
who is watching but not clicking never gets reaped.


## Pitfalls discovered while building this (all cost a real failure)

1. **`LazyArm.__getattr__` materializes the arm.** Probing `arm.client` to set
   up verification powered up the motors as a side effect — exactly what
   LazyArm exists to prevent. Never probe an unmaterialized LazyArm.
2. **Verification must be read-only.** Forcing a detector pass before checking a
   postcondition overwrote the belief `place_at` had just authored, so the
   object appeared not to have moved. `fresh=False` by default.
3. **PhysX does not write live poses back to USD.** Reading a prop with
   `UsdGeom.Xformable.ComputeLocalToWorldTransform` returns the *authored*
   spawn transform (both cubes read `(0,0,0)` on the live rig) — dynamic bodies
   must be read via `RigidPrim`.
4. **Skill arg names differ.** `pick_and_place` takes `object`; grasp/push take
   `label`. Checking only `label` silently degraded every verification to
   `unverified`.
5. **A missing `TOOL_SPECS` entry fails no test.** The skill simply never
   becomes visible to the LLM/MCP (documented in CLAUDE.md; still true).
