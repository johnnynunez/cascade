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
3. An inventory cache + SAM tracking step inside the detector, and a watcher heartbeat
   that no longer waits for detection, so the slow brain never trips the watchdog or the 5 s age limit.
4. Remove YOLOE entirely (class, configs, weights, installer assets, launcher checks, docs, tests).

Out (follow-up PR, same design): UniPose9D 9D pose replacing the PCA box behind
`pose.backend: unipose9d`, validated with an operator-approved RGB-D capture first.
Out: DINOv3 exemplar re-identification (weights gated, not measured yet).

## 3. Architecture

```
watcher tick (heartbeat first) ──► detect_nowait(frame)   never calls the brain on its own thread
                                     ├─ same scene as the cached entry  → SAM box prompts on this frame (~0.3 s)
                                     ├─ entry older than idle_refresh_s → same, + background brain refresh
                                     └─ no entry / scene changed        → [] now, background refresh (objects next tick)
                                     (at most one BACKGROUND brain request at a time per detector, all cameras;
                                      synchronous calls are not capped; waiting cameras served oldest-first)
describe_scene / _reobserve ─────► detect(frame, None) or  same cache; older than refresh_s → + background refresh;
  (+ localization's vocabulary)    detect_vocabulary(frame, detect_classes)   no entry / scene changed → Qwen3.8 now (~5 s)
localize / pick "the X" ─────────► detect(frame, ["X"])     Qwen3.8 refer (~2 s, uncached) → SAM
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

### 3.3 Inventory cache + tracking (inside `BrainDetector`) and the heartbeat

`detect(frame, classes=None)` is called synchronously by the skill runtime
(`describe_scene`, `_reobserve`, `_localize`; the watcher uses `detect_nowait`, item 4a), all under the 5 s analysed-image age limit
(`skills/runtime._analyzed_frame_age_s`). A 5.4 s inventory per call would trip it. So:

1. Inventory cache per camera, keyed by `(H, W, round(fx), round(cx))` (scene and wrist differ;
   `Frame` carries no camera name). Each entry keeps the brain's boxes, the 64x36 grayscale
   thumbnail of the frame they came from, and its time.
2. `detect(frame, None)` with a cache entry that is younger than `max_cache_age_s` (default 30)
   and whose thumbnail differs from this frame by mean abs diff <= `scene_change` (default 6.0):
   **track step** — SAM box prompts with the cached boxes on THIS frame (~0.3 s on GB10),
   returned at once. A box whose mask comes back empty is dropped from the result.
   Exception (Ruling 35): an entry past `max_cache_age_s` whose scene is unchanged keeps the track step
   while that key's background refresh is in flight or waiting for the slot, instead of a second,
   synchronous brain call (`max_cache_age_s` stays 30).
3. Otherwise, or when the entry is older than `refresh_s` (default 4 s): the stale case
   (scene changed, no entry) runs the brain synchronously for `detect`; the merely-aged case returns the track
   step and starts a background refresh. At most ONE background request at a time per detector, across all
   cameras and keys (synchronous calls are not capped). A key that asks while it runs waits for the slot,
   from its first ask, with its latest frame; a free slot goes to the key that has waited longest (ties:
   first asked), whichever caller finds it free (Ruling 34). A waiting key leaves the queue when its
   refresh starts, when an answer for it is stored, or on close(); one backing off keeps its place.
4. `detect(frame, classes=[q, ...])` (refer: an agent's own words) never uses the cache: brain refer prompt,
   synchronous.
4a. `detect_nowait(frame, classes=None)` (the watcher's call) never calls the brain on the caller's thread:
   cached track step when the scene matches, else `[]` plus a background refresh at once. On an unchanged
   scene it refreshes only once the entry is older than `idle_refresh_s` (default 30 s; Ruling 31). Cache
   keys are `(camera key, None | tuple(classes))`, so a watcher vocabulary gets its own refer-based entry.
4c. `detect_vocabulary(frame, detect_classes)` (Ruling 30): the runtime's default-vocabulary passes
   (`describe_scene`, `_reobserve`, localization's vocabulary round, through
   `perception.detector.detect_vocabulary`) read the `(camera, tuple(classes))` entry the watcher keeps, under
   the rules of items 2-3 and 4b; `LockedDetector` forwards it, other detectors get `detect()`.
4b. Failure backoff: after any failed brain call for a key (synchronous or background), no new call for that key
   for `refresh_s`; during the backoff the fallback is the track step of a same-scene entry at any age, else `[]`.
5. Heartbeat (`perception/world.py`): `harness.heartbeat()` moves ahead of the detector call,
   still once per new processed frame (`frame_id`/capture marker dedupe unchanged), so a slow
   or failing brain cannot starve the watchdog. A dead camera still stops it.
6. Everything else in the watcher is unchanged: link-mask sampling before inference, overlay,
   fusion pause during motion.

### 3.4 Wiring: robot server ↔ brain

- Config (`configs/demo.yaml`):
  ```yaml
  detector:
    type: brain
    base_url: ${CASCADE_PERCEPTION_URL:-http://127.0.0.1:8080/v1}
    model: ${CASCADE_PERCEPTION_MODEL:-qwen3.8-27b}
    api_key_file: ${CASCADE_PERCEPTION_KEY_FILE:-~/.cascade/nemoclaw/llama.key}
    refresh_s: 4.0
  segmenter: {model: ${repo}/models/sam2.1_l.pt}
  ```
  Default port :8080 = `scripts/serve_qwen_*.sh`'s PORT; the rig's OpenShell instance sets
  `CASCADE_PERCEPTION_URL` to its own brain (:8081). SAM by absolute path: any working
  directory finds it (Rulings 24/25).
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
| Brain slow (> refresh_s) | watcher never waits (detect_nowait); at most one background request at a time per detector (synchronous calls are not capped), waiting cameras served oldest-first; ticks keep tracking |
| Concurrent-safe detector | `LockedDetector` skips its global lock (`concurrent_safe = True`), so a refer never queues behind an inventory |
| SAM unavailable | detections keep bbox masks (today's maskless path); Isaac observed-finger gate refuses as today (it requires a mask) |
| Refer while inventory running | independent HTTP request, no shared lock; SAM step serialized (~50 ms) |
| Wrong label from the brain | 6 % measured; beliefs keep aliases; agent can re-ask with refer |

## 5. Testing

- Parser: real recorded Qwen3.8 replies from the bake-off as fixtures (fenced JSON, trailing
  `[]`, thinking block, reversed boxes, empty answer).
- BrainDetector against an in-process fake OpenAI server: inventory, refer, timeout → `[]`,
  key read from file and never logged.
- WorldWatcher: heartbeat fires while a fake detector blocks 10 s.
- BrainDetector cache: track step served from cache on an unchanged scene (no brain call);
  scene change or no entry → synchronous brain; aged entry → cache result + exactly one
  background refresh; scene and wrist intrinsics get separate entries; refer bypasses the cache.
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

- Brain load: each inventory adds prefill (~900 visual tokens) to the agent's own brain traffic,
  and a configured vocabulary costs one refer per class per refresh. The watcher refreshes an
  unchanged scene only every `idle_refresh_s` (30 s) and a changed scene at once, with at most one
  BACKGROUND brain request at a time per detector (synchronous calls are not capped); waiting
  cameras are served oldest-first, so a camera whose scene changes every tick cannot starve the
  others. `describe_scene`/localization refresh in the background after `refresh_s` (4 s) when they
  run. Measured single-stream; the agent's own turns pause nothing (separate requests).
- Cadence vs cache age: with `idle_refresh_s` = `max_cache_age_s` (both 30 s), a `describe_scene`
  that lands while the watcher's idle refresh is in flight (entry 30 s to 30 s + one brain pass old)
  finds the entry past `max_cache_age_s`. On an unchanged scene it tracks the cached entry while that
  refresh is in flight or waiting (Ruling 35); without a pending refresh (watcher off) or on a
  changed scene it still asks the brain synchronously (~5 s; one slow-perception retry).
- Small/dark objects: the dark brown spool was missed by every method; no regression vs YOLOE.
- Latency of "find the X": 1.9 s + SAM vs 0.8 s today, but today it fails 60 % of the time.
