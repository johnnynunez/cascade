# Brain-driven perception: Qwen3.8 identifies, SAM segments, YOLOE goes

Date: 2026-10-15 · Status: design approved in chat by the owner; spec for review
Branch: `feat/brain-perception` · Evidence: `docs/evidence/perception-bakeoff-20261015/REPORT.md`

## 1. Why

YOLOE-11s (prompt-free, the production detector) misidentifies most real booth objects.
A head-to-head test on 47 real reBot keyframes (36 scene + 11 wrist camera, 157 identifiable
ground-truth objects) measured:

| Task | YOLOE-11s today | Qwen3.8-27B (+SAM masks) |
|---|---|---|
| Inventory: objects correctly identified | 25 % | **83 %** |
| Inventory: precision (reported objects that are real and right) | 12 % | **81 %** |
| Inventory: wrong-label rate on found objects | 67 % | **6 %** |
| Refer ("find the X"): named object found | 40 % (53 % empty) | **94 %** |
| Latency per scene camera frame | 0.01 s | 5.4 s inventory, 1.9 s refer |

Other candidates measured and rejected as the primary path: SAM 3 / SAM 3.1 text prompts
(refer 73-88 %), LLMDet-L (92 % refer, needs transformers < 5), YOLOE-26 (61 %), and
union fusions of Qwen with a detector (no gain, lower precision). SAM 2.1-L, SAM 3 and
SAM 3.1 produce equivalent masks from Qwen's boxes.

The owner's requirement: object identity comes from the brain backend, and that backend is
**Qwen3.8-27B** (the local vLLM brain on :8081). Astra/Codex is not used for perception.

## 2. Scope

In:
1. A `brain` detector backend: Qwen3.8-27B inventory + refer prompts, SAM 2.1 masks from its boxes.
2. Robot server wiring so it reaches the same brain (URL, served model name, key file).
3. A WorldWatcher that keeps the safety heartbeat and occupancy refresh independent of
   the (now slow) detector, and keeps beliefs fresh between brain calls.
4. Remove YOLOE entirely (class, configs, weights, installer assets, launcher checks, docs, tests).

Out (follow-up PR, same design): UniPose9D 9D pose replacing the PCA box behind
`pose.backend: unipose9d`, validated with an operator-approved RGB-D capture first.
Out: DINOv3 exemplar re-identification (weights gated, not measured yet).

## 3. Architecture

```
camera frame ──► WorldWatcher tick (every frame, cheap)
                   ├─ depth + occupancy refresh            (unchanged)
                   ├─ harness.heartbeat()                  (MOVED: before/independent of detection)
                   └─ track step: SAM box prompts from the last brain boxes → masks → fusion
                                                   ▲
brain worker (own thread, per fusing camera) ─────┘ every refresh_s or on scene change:
   Qwen3.8 inventory prompt → [{label, bbox_2d 0-1000}] → SAM masks → beliefs

localize_object / pick ("find the X")  ──► Qwen3.8 refer prompt → boxes → SAM masks → ObjectFix
```

### 3.1 `perception/brain_detector.py` — `BrainDetector(Detector)`

Replaces `vlm_detector.VLMDetector` (absorbs it; `type: vlm` stays an alias for one release).

- `detect(frame, classes=None)`: inventory prompt (the bake-off's P_BOXES wording: list every
  manipulable object on the table, one entry per instance, exclude robot/people/hand-held/
  screens/furniture/table; JSON `[{"label","bbox_2d":[x1,y1,x2,y2]}]`, 0-1000 coordinates).
- `detect(frame, classes=[q, ...])`: one refer prompt per query ("Locate every instance of
  \"{q}\" … or [] if it is not visible"); labels are the query strings, so beliefs keyed by
  the agent's own words keep working.
- Parser: strip `<think>…</think>` and code fences, then take the FIRST JSON list that
  parses (`json.JSONDecoder.raw_decode` at each `[`). Today's `_extract_json_array` takes the
  LAST array, and Qwen3.8 sometimes appends a second `[]` after the answer: the bake-off saw a
  full 12-object reply parsed as empty. Clamp and order boxes as today.
- Masks: `BoxSegmenter.masks(frame, boxes)` (§3.2); a detection without a mask keeps the
  existing bbox fallback.
- Transport: OpenAI-compatible client, `api_key` read from `api_key_file` (never argv/config
  text), `timeout_s` (default 20), `temperature 0`, `max_tokens 1500`, thinking off (the
  serve script's default chat-template kwargs). Any failure → `[]` + one rate-limited warning,
  never an exception (booth rule, same as VLMDetector).
- No global lock around the HTTP call: brain requests run concurrently; only the SAM step
  takes the GPU segmenter lock. This keeps a foreground refer from queueing behind a 5 s
  background inventory (today `LockedDetector` would serialize them and the 5 s image-age
  limit in `runtime._localize` would trip).

### 3.2 `perception/segmenter.py` — `BoxSegmenter`

- SAM 2.1-L via ultralytics (`sam2.1_l.pt`, Apache-2.0, no gated download), box prompts for all
  boxes of a frame in one call, `retina`-sized full-frame bool masks. Shares the model instance
  with `PointSegmenter` when both are configured (one GPU copy).
- `segmenter.model` config key; SAM 3.1 stays a measured, optional alternative (gated weights,
  native code needed) and is not wired in this PR.

### 3.3 WorldWatcher changes (`perception/world.py`)

1. Heartbeat: `harness.heartbeat()` moves ahead of `detector.detect` and fires for every new
   processed frame (depth/occupancy path). Semantics stay "perception pipeline is alive";
   a dead camera still stops the heartbeat and the watchdog still halts motion.
2. Brain refresh: per fusing camera a worker calls `detect(frame, classes=detect_classes or None)`
   when `now - last >= refresh_s` (default 4 s) or the frame changed (grayscale 64x36 mean
   abs diff > `scene_change` threshold, default 6.0) — never more than one in flight per camera.
3. Track step on every tick: SAM box prompts seeded with the latest boxes (a moved object's box
   is re-derived from its new mask, dilated 10 %), giving masks at tick rate (~0.3 s on GB10)
   so `belief_fallback_age_s` and fusion freshness keep working between brain passes.
   Tracks whose mask vanishes for `track_drop_frames` (default 3) are dropped until the brain
   sees them again. Fusion stays paused during motion exactly as today.
4. `set_overlay` keeps showing the current tracked detections.

### 3.4 Wiring: robot server ↔ brain

- Config (`configs/demo.yaml`):
  ```yaml
  detector:
    type: brain
    base_url: ${CASCADE_PERCEPTION_URL:-http://127.0.0.1:8081/v1}
    model: ${CASCADE_PERCEPTION_MODEL:-qwen3.8-27b}
    api_key_file: ${CASCADE_PERCEPTION_KEY_FILE:-~/.cascade/nemoclaw/llama.key}
    refresh_s: 4.0
  segmenter: {model: sam2.1_l.pt}
  ```
  The config loader learns `${VAR:-default}` and `~` expansion for these keys only (today it
  expands `${repo}`/`${assets}`).
- `apps/mcp_env.FORWARDED` gains the three `CASCADE_PERCEPTION_*` variables (golden JSON test
  updated). `scripts/openshell_rig.serve_mcp_env` sets them from its own `brain_port`,
  `DEFAULT_MODEL` and `CASCADE_BRAIN_KEY_FILE`, so the default instance needs no new flags.
  `launch.sh --brain qwen` does the same.
- Mock camera profiles keep `detector: {type: mock}`: offline runs, CI and the minimal-install
  job never import openai/ultralytics.

### 3.5 Removing YOLOE

- Delete `OpenVocabDetector` and YOLOE-only helpers from `perception/detector.py` (keep the
  `Detector` ABC and `MockDetector`); `type: yoloe` in a config raises a clear error naming
  `type: brain`.
- Drop `models/yoloe-11s-seg*.pt` (LFS) and `mobileclip_blt.ts` from `install_support.MODEL_ASSETS`,
  `launch.sh` weight checks, `CASCADE_DETECTOR_MODEL`, `CASCADE_DETECT_CLASSES` stays (now the
  refer vocabulary for the watcher), the `cwd=models/` rule (kept harmless, comment updated),
  `YOLO_OFFLINE` flags stay for SAM.
- Kitchen/Isaac profiles that pinned YOLOE on cpu switch to `type: brain` (Isaac sim) or `mock`.
- `ultralytics` stays in the `perception` extra (SAM); the CLIP fork pin is removed.
- Docs: README, ARCHITECTURE, QUICKSTART, BOOTH_RUNBOOK lines that name YOLOE; AGENTS.md
  changes are reported to the owner (protected file), not edited by agents.

## 4. Failure behaviour

| Situation | Behaviour |
|---|---|
| Brain down / timeout / garbled | `[]`; watcher keeps heartbeat + tracking of known objects; refer returns empty → existing belief fallback / VLM grounder path; agent sees "not found" |
| Brain slow (> refresh_s) | at most one request in flight per camera; ticks keep tracking |
| SAM unavailable | detections keep bbox masks (today's maskless path); Isaac observed-finger gate refuses as today (it requires a mask) |
| Refer while inventory running | independent HTTP request, no shared lock; SAM step serialized (~50 ms) |
| Wrong label from the brain | 6 % measured; beliefs keep aliases; agent can re-ask with refer |

## 5. Testing

- Parser: real recorded Qwen3.8 replies from the bake-off as fixtures (fenced JSON, trailing
  `[]`, thinking block, reversed boxes, empty answer).
- BrainDetector against an in-process fake OpenAI server: inventory, refer, timeout → `[]`,
  key read from file and never logged.
- WorldWatcher: heartbeat continues while a fake detector blocks 10 s; one in-flight request per
  camera; track step keeps beliefs younger than `belief_fallback_age_s`; scene-change trigger.
- Wiring: mcp_env golden env, openshell_rig env (default instance values pinned), config
  expansion of `${VAR:-default}` and `~`.
- Removal: tests that pinned YOLOE (`test_detector_*`, `test_spark_install` MODEL_ASSETS,
  launcher/openshell cwd tests, legacy registrars) updated to the new contract; full-suite
  failure-ID diff against origin/main per the repo's baseline procedure.
- Accuracy regression (offline, not in CI): the bake-off harness and scores are archived under
  `docs/evidence/perception-bakeoff-20261015/` (numbers, prompts, scorer; no images with people).
- Live acceptance (operator, demo stopped → `up`): `world_state` lists the table objects with
  correct names; "pick the paper cup" localizes via refer; watchdog never trips while idle.

## 6. Risks

- Brain load: inventory every 4 s per camera adds prefill (~900 visual tokens) to the agent's
  own brain traffic. Measured single-stream; mitigated by `refresh_s` and scene-change gating,
  and by the agent's own turns pausing nothing (separate requests).
- Small/dark objects: the dark brown spool was missed by every method; no regression vs YOLOE.
- Latency of "find the X": 1.9 s + SAM vs 0.8 s today, but today it fails 60 % of the time.
