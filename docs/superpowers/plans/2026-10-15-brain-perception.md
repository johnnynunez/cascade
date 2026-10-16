# Brain-driven perception Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace YOLOE with Qwen3.8-27B (the brain) for object identity and SAM 2.1 for masks, without slowing the watchdog or tripping the 5 s image-age limit.

**Architecture:** A new `BrainDetector` behind the existing `Detector` interface asks the brain for
boxes (inventory or refer prompt), turns them into masks with a box-prompted SAM, and caches each
camera's inventory so repeated `detect(frame, None)` calls are a 0.3 s SAM track step. The watcher
heartbeat moves ahead of detection. YOLOE code, weights, installer assets and docs are removed.

**Tech Stack:** Python 3.12, openai client (lazy import), ultralytics SAM 2.1-L, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-10-15-brain-perception-design.md`

## Global Constraints

- Work only in `/home/spark/Projects/cascade-wt/brain-perception` (branch `feat/brain-perception`); the main checkout serves the live robot.
- Tests: `PYTHONPATH=$PWD/src /home/spark/Projects/cascade/.venv/bin/python -m pytest <tests> -q -p no:cacheprovider`. Never open cameras, CAN, the brain or GPU from tests.
- `openai` and `ultralytics` are imported lazily, only when `type: brain` / a SAM model is used: the CI `minimal-install` job asserts they are absent and runs a full mock episode.
- Mock camera profiles keep `detector: {type: mock}`.
- Failure shape: brain down/timeout/garbled → `[]` + one rate-limited warning, never an exception.
- The api key is read from `api_key_file` and never appears in argv, config text, logs or exceptions.
- Coordinates from the brain are 0-1000 normalized `bbox_2d: [x1, y1, x2, y2]`.
- Defaults: `base_url http://127.0.0.1:8081/v1`, `model qwen3.8-27b`, `api_key_file ~/.cascade/nemoclaw/llama.key`, `timeout_s 20`, `max_tokens 1500`, `temperature 0`, `refresh_s 4.0`, `max_cache_age_s 30`, `scene_change 6.0`, segmenter `sam2.1_l.pt`.
- No hardcoded `cuda` in `configs/` (pinned by `tests/test_device_and_profiles.py`).
- `AGENTS.md` is not edited; the needed changes are listed in the PR body.
- Lint: `/home/spark/.local/bin/uvx ruff check <changed .py files> --select F,E9` clean.

## Review Focus

- Brain returns a valid answer followed by a second `[]` → the answer is used (bake-off s16 lost 12 objects to last-array parsing). Test in Task 1.
- Brain reply with boxes outside 0-1000 or reversed corners → clamped/reordered, zero-area boxes dropped. Test in Task 1.
- The watcher keeps the heartbeat while the brain hangs for the full `timeout_s`. Test in Task 4.
- A foreground refer (`detect(frame, ["paper cup"])`) issued while a background inventory refresh is in flight returns without waiting for it. Test in Task 3.
- Wrist and scene frames never share a cache entry (different intrinsics), and an aged entry spawns exactly one refresh even if `detect` is called 10 times. Test in Task 3.

---

### Task 1: Prompts and reply parser

**Files:**
- Create: `src/cascade/perception/brain_detector.py` (module constants + `parse_boxes`; the class arrives in Task 3)
- Test: `tests/test_brain_detector_parse.py`
- Fixtures: `tests/fixtures/brain_replies/*.txt` — copy real replies from `~/Projects/perception-bakeoff/out/pred/qwen_boxes/{s05,s16,w03}.json` (`extra.raw`) plus a hand-written thinking-block case.

**Interfaces:**
- Produces: `INVENTORY_PROMPT: str`, `REFER_PROMPT: str` (format key `{q}`), `parse_boxes(text: str, width: int, height: int) -> list[tuple[str, np.ndarray]]` (label, float32 xyxy pixels).

- [ ] **Step 1: Write the failing tests**

```python
def test_trailing_empty_array_keeps_answer():      # s16 fixture: 12 boxes then "[]"
    assert len(parse_boxes(S16, 1280, 720)) == 12
def test_thinking_and_fences_stripped():           # "<think>[{...example}]</think>```json [..] ```"
    assert [l for l, _ in parse_boxes(T, 1000, 1000)] == ["paper cup"]
def test_scale_clamp_and_order():
    (l, b), = parse_boxes('[{"label":"cup","bbox_2d":[600,500,400,1200]}]', 1280, 720)
    np.testing.assert_allclose(b, [512, 360, 768, 720])
def test_zero_area_and_bad_entries_dropped():
    assert parse_boxes('[{"label":"a","bbox_2d":[5,5,5,9]},{"bbox_2d":[1,2,3,4]},"x"]', 100, 100) == []
def test_garbage_is_empty():
    assert parse_boxes("I cannot see anything.", 640, 480) == []
```

- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError: cascade.perception.brain_detector`)
- [ ] **Step 3: Implement.** Strip `<think>…</think>` (also unterminated) and ``` fences; scan each `[` with `json.JSONDecoder().raw_decode` and take the FIRST list whose items are dicts with `bbox_2d`; a missing/blank label is dropped. Prompt texts are exactly `P_BOXES` and `P_REFER` from `~/Projects/perception-bakeoff/tools/run_candidates.py` (the measured wording).
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(perception): brain box prompts and reply parser`

### Task 2: Box-prompted SAM segmenter

**Files:**
- Modify: `src/cascade/perception/segmenter.py` (add `BoxSegmenter`; `PointSegmenter` unchanged)
- Test: `tests/test_box_segmenter.py`

**Interfaces:**
- Produces: `BoxSegmenter(model_path: str = "sam2.1_l.pt", device: str = "auto")` with `masks(frame: Frame, boxes: np.ndarray) -> list[np.ndarray | None]` (one full-frame bool mask per (N,4) box, `None` where SAM returns nothing or an empty mask). `_load()` is lazy and shares the ultralytics model cache keyed by path with `PointSegmenter`.

- [ ] **Step 1: Failing tests** with a fake ultralytics model injected as `seg._model` (records the `bboxes=` it was called with, returns masks): one call per frame for all boxes; `N == 0` makes no call; an all-False mask maps to `None`; masks have shape `frame.rgb.shape[:2]`, dtype bool.
- [ ] **Step 2: Run, expect FAIL**
- [ ] **Step 3: Implement** — `model(frame.rgb, bboxes=boxes.tolist(), verbose=False)`; device via `cascade.device.resolve_device` (never a literal `cuda`); same CUDA-tensor check as `PointSegmenter`.
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(perception): box-prompted SAM segmenter`

### Task 3: `BrainDetector` (transport, refer, inventory cache, background refresh)

**Files:**
- Modify: `src/cascade/perception/brain_detector.py`
- Modify: `src/cascade/perception/vlm_detector.py` → keep only a deprecation shim: `VLMDetector = BrainDetector` subclass whose defaults are cosmos3-edge/:8082 and `segmenter=None`; delete its parser (Task 1 replaces it).
- Test: `tests/test_brain_detector.py`; update `tests/test_vlm_detector.py` to the shim.

**Interfaces:**
- Consumes: Task 1 `parse_boxes`, prompts; Task 2 `BoxSegmenter.masks`.
- Produces:
```python
class BrainDetector(Detector):
    def __init__(self, *, base_url: str, model: str, api_key_file: str | None = None,
                 timeout_s: float = 20.0, max_tokens: int = 1500, segmenter: BoxSegmenter | None = None,
                 refresh_s: float = 4.0, max_cache_age_s: float = 30.0, scene_change: float = 6.0,
                 client=None, clock=time.monotonic): ...
    def detect(self, frame: Frame, classes: list[str] | None = None) -> list[Detection]: ...
    def close(self) -> None: ...   # joins the refresh threads (thread_join pattern)
```
`Detection.conf` is 1.0 for brain boxes. `client` is injectable (tests); otherwise `openai.OpenAI(base_url, api_key=<file contents or "EMPTY">, timeout=timeout_s)` built lazily.

- [ ] **Step 1: Failing tests** (fake client object with `chat.completions.create` returning canned replies; fake segmenter; injected clock):
  - refer: `detect(f, ["paper cup"])` sends `REFER_PROMPT.format(q="paper cup")` with the JPEG data URL, labels equal the query, masks come from the segmenter.
  - inventory cold: first `detect(f, None)` calls the brain once (synchronously) and returns its boxes.
  - cache hit: second call on an identical frame within `refresh_s` makes **no** brain call and calls the segmenter with the cached boxes.
  - scene change: a frame whose 64x36 gray thumbnail differs by > 6.0 mean-abs triggers a synchronous brain call.
  - aged: after `refresh_s`, 10 rapid calls return cached-track results and start exactly **one** background request (fake client blocks on an Event).
  - refer during in-flight refresh: returns while the background request is still blocked.
  - separate cameras: frames with different `K[0,0]` keep separate entries.
  - failures: client raises `APITimeoutError`/`RuntimeError` → `[]`, warning logged at most once per 30 s, never raises; reply garbage → `[]`.
  - key hygiene: key read from `tmp_path/"k"`; the key string is absent from every log record and from `repr(detector)`.
  - `classes=[]` behaves as `None`.
- [ ] **Step 2: Run, expect FAIL**
- [ ] **Step 3: Implement.** Cache key `(H, W, round(K[0,0]), round(K[0,2]))`; entry = boxes, labels, thumbnail, t. One `threading.Lock` for the cache dict only — never held across the HTTP call. SAM calls go through the segmenter's own lock. Mask-less detections keep `mask=None` (today's bbox fallback). A track-step box whose mask is `None` is dropped only when a segmenter is configured.
- [ ] **Step 4: Run, expect PASS** (also `tests/test_vlm_detector.py`)
- [ ] **Step 5: Commit** `feat(perception): BrainDetector with inventory cache and refer`

### Task 4: Watcher heartbeat ahead of detection

**Files:**
- Modify: `src/cascade/perception/world.py:362-373` (`_tick`)
- Test: `tests/test_world_heartbeat_slow_detector.py`

- [ ] **Step 1: Failing test:** WorldWatcher on the mock camera with a detector whose `detect` blocks on an Event for 10 s and a fake harness counting `heartbeat()`; after 0.5 s of new frames the count is ≥ 1 while `detect` is still blocked. Second test: a duplicate `frame_id` produces no heartbeat (dedupe unchanged).
- [ ] **Step 2: Run, expect FAIL** (count 0)
- [ ] **Step 3: Implement:** move the `self._harness.heartbeat()` block to just after the `link.sample()` block, before `self._detector.detect(...)`; update its comment (alive = new frame processed, independent of semantic inference).
- [ ] **Step 4: Run** the new test plus `tests/test_shutdown_threads.py tests/test_perception_delivery.py`, expect PASS
- [ ] **Step 5: Commit** `fix(perception): heartbeat no longer waits for detection`

### Task 5: Config + factory: `type: brain`

**Files:**
- Modify: `src/cascade/config.py` (`_resolve_paths`: also `${NAME:-default}` env expansion and a leading `~`)
- Modify: `src/cascade/apps/demo.py` (`_make_detector`, and where the segmenter is built)
- Modify: `configs/demo.yaml` (detector + segmenter blocks per spec §3.4), `configs/cameras/isaac_side.yaml` and any profile pinning YOLOE
- Test: `tests/test_config_env_expansion.py`, `tests/test_make_detector_brain.py`

**Interfaces:**
- Produces: `_make_detector(cfg)` returns `BrainDetector` for `type: brain` and for `type: vlm` (alias, with the old cosmos defaults); `type: yoloe` raises `ValueError` whose message contains `type: brain`.

- [ ] **Step 1: Failing tests:** `${CASCADE_PERCEPTION_URL:-http://x/v1}` → env value when set, default when unset; `~/.cascade/k` → expanded home; strings without placeholders unchanged; `_make_detector` on demo.yaml yields `BrainDetector` with model `qwen3.8-27b`, url `http://127.0.0.1:8081/v1`, a `BoxSegmenter("sam2.1_l.pt")`, and imports neither openai nor ultralytics (`sys.modules` check) until first `detect`; the mock camera profile still yields `MockDetector`; `type: yoloe` → `ValueError` naming `type: brain`.
- [ ] **Step 2: Run, expect FAIL**
- [ ] **Step 3: Implement.** `api_key_file` passes through `os.path.expanduser`; a missing file → key `"EMPTY"` + one warning (vLLM without `--api-key` accepts it).
- [ ] **Step 4: Run** new tests + `tests/test_device_and_profiles.py`, expect PASS
- [ ] **Step 5: Commit** `feat(config): brain detector profile and env expansion`

### Task 6: Wire the robot server to the brain

**Files:**
- Modify: `src/cascade/apps/mcp_env.py` (`FORWARDED` += `CASCADE_PERCEPTION_URL`, `CASCADE_PERCEPTION_MODEL`, `CASCADE_PERCEPTION_KEY_FILE`)
- Modify: `scripts/openshell_rig.py` (`serve_mcp_env`: set the three from `brain_port`, `DEFAULT_MODEL`, the brain key file path — `setdefault`, so an operator override wins)
- Modify: `scripts/launch.sh` (`--brain qwen` exports the same three)
- Test: `tests/test_mcp_env_forwarding.py` golden, `tests/test_openshell_rig.py`, `tests/test_launcher_frontdoor.py`

- [ ] **Step 1: Failing tests:** golden FORWARDED list includes the three names; default OpenShell instance env has `CASCADE_PERCEPTION_URL=http://127.0.0.1:8081/v1`, `CASCADE_PERCEPTION_MODEL=qwen3.8-27b`, `CASCADE_PERCEPTION_KEY_FILE=<home>/.cascade/nemoclaw/llama.key` and still none of the brain key's contents; the `sim` instance points at its own brain port; launch.sh `--brain qwen` dry path exports them (uname shim `Linux`).
- [ ] **Step 2: Run, expect FAIL**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `feat(openshell): forward the brain endpoint to perception`

### Task 7: Remove YOLOE

**Files (non-comment references found by `grep -rniE "yoloe|mobileclip|OpenVocabDetector"`):**
- `src/cascade/perception/detector.py` (delete `OpenVocabDetector`, `_rule_*` only if unused by `MockDetector`), `perception/__init__.py`, `apps/viewer.py` (use `BrainDetector` from config), `perception/world.py`, `perception/workspace.py`, `perception/vlm_ground.py`, `perception/thread_join.py`, `skills/runtime.py`, `memory/beliefs.py` (comments)
- `scripts/install_support.py` (`MODEL_ASSETS`), `scripts/install.sh`, `scripts/launch.sh`, `scripts/booth_up.sh`, `scripts/openclaw_demo.sh`, `scripts/hermes_demo.sh`, `scripts/record_demo.py`, `scripts/nemoclaw_mcp.py`, `scripts/dashboard_runner.py`, `scripts/setup_env.sh`, `scripts/setup_agents.py`
- `deploy/runtime/runtime.py`, `deploy/brev/bundle_assets.json`, `pyproject.toml` (CLIP fork pin if present), `models/yoloe-*.pt` + `models/mobileclip_blt.ts` (`git rm`, LFS)
- Tests pinning YOLOE (22 files): `test_colour_identity_beliefs, test_shutdown_threads, test_colour_identity_floor, test_openshell_rig, test_fusion_self_mask, test_perception_delivery, test_detector_query_scope, test_launcher_frontdoor, test_detector_preparation, test_setup_agents, test_legacy_mcp_registrars, test_depth_grasp_geometry, test_vab_rgb_detection, test_detector_mask_geometry, test_fusion_size_gate, test_vlm_detector, test_launch_delivery, test_colors, test_detector_after_camera_open, test_link_self_mask, test_mcp_env_forwarding, test_spark_install`

- [ ] **Step 1:** For each test file: if it tests YOLOE internals (`test_detector_preparation`, `test_detector_query_scope`, `test_detector_after_camera_open`, `test_detector_mask_geometry`) delete the YOLOE-only cases and keep any contract that now belongs to `BrainDetector` (move it into `tests/test_brain_detector.py`); if it only used `OpenVocabDetector` as a stand-in, switch to `MockDetector` or a fake; goldens (`MODEL_ASSETS`, bundle assets, launcher weight checks) are updated to the new lists. Record each decision in the commit body.
- [ ] **Step 2:** Remove the source references above; `grep -rniE "yoloe|mobileclip|OpenVocabDetector" src scripts configs deploy tests pyproject.toml` returns only the `type: yoloe` error message and its test.
- [ ] **Step 3: Run** the 22 files + Task 1-6 tests, expect PASS; ruff clean on changed files.
- [ ] **Step 4: Commit** `refactor(perception)!: remove YOLOE`

### Task 8: Docs, evidence, full-suite diff

**Files:**
- Create: `docs/evidence/perception-bakeoff-20261015/REPORT.md` (tables from `~/Projects/perception-bakeoff/out/RESULTS_20261015.md` + SAM 3.1 / Cosmos rows, method, prompts, limits), `score.py` and `run_candidates.py` copies (no frames: they show people)
- Modify: `README.md`, `docs/ARCHITECTURE.md`, `docs/QUICKSTART.md`, `docs/BOOTH_RUNBOOK.md`, `docs/DETECTOR_MODEL_REUSE.md` (mark superseded) — every YOLOE mention → brain perception
- [ ] **Step 1:** Write docs; `grep -rni yoloe README.md docs/*.md` only in historical/evidence context.
- [ ] **Step 2:** Full-suite failure-ID diff vs a detached `origin/main` worktree (4 shards, `-rfE --tb=no`, same interpreter/env, per the cascade-feature-worktree skill). Expected: no new failing IDs.
- [ ] **Step 3: Commit** `docs: brain-driven perception and bake-off evidence`
- [ ] **Step 4:** Push, `gh pr create` (body: summary, bake-off table, AGENTS.md changes needed, live acceptance pending operator), wait for every CI check green on the exact head.

### Task 9 (operator-gated): live acceptance

Only with the owner's go-ahead (robot, cameras, brain): after merge and an authorized deploy, `world_state` lists the table objects with correct names, "pick the paper cup" localizes via refer, and the watchdog never trips while idle for 5 min. Not part of the PR gate.
