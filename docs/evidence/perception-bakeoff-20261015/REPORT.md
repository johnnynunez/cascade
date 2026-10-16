# Perception bake-off, 2026-10-15

Decision: the local **Qwen3.8-27B** brain names and boxes every object, **SAM 2.1-L** masks the boxes,
and "find the X" is a Qwen refer prompt. It replaces YOLOE-11s prompt-free (the former production detector).

| | YOLOE-11s prompt-free | Qwen3.8-27B boxes + SAM 2.1-L |
|---|---|---|
| Objects correctly identified (id_lenient) | 0.25 | **0.83** |
| Reported objects that are real and right (precision) | 0.12 | **0.81** |
| Wrong label on found objects | 0.67 | **0.06** |
| Refer: named object found (hit) | 0.40 | **0.94** |
| Median latency | 0.01 s inventory, 0.77 s refer | 5.37 s inventory, 1.87 s refer |

## Method

1. Frames: 47 real reBot keyframes (36 scene camera, 11 wrist camera) from booth runs on Oct 10-13.
2. Ground truth: 182 objects labelled by vision labeler subagents, checked on overlays. 25 objects the
   labelers could not identify are excluded, leaving **157** (`CERTAIN_ONLY=1`, the scorer's default).
3. Match: box IoU >= 0.3 AND the GT point inside the predicted mask (box when there is no mask).
   Inventory ignores predictions centred off the table polygon.
4. Labels: judged against per-object word lists (`specific`, acceptable `generic`, known `wrong`).
5. Brain settings: temperature 0, max_tokens 1500, thinking off unless the row says otherwise. Local models
   ran one request at a time on one DGX Spark (GB10); Qwen3.8-27B NVFP4 on vLLM.

Metrics (from `tools/score.py`):
- `loc_recall`: GT objects matched by a prediction. `id_strict`: matched and named specifically
  ("lobster"). `id_lenient`: specific or an acceptable category ("stuffed toy").
- `precision`: correct (lenient) predictions / all predictions on the table. `wrong_label`: matched
  predictions with a wrong name.
- Refer (query = the GT name): `hit` = a returned instance matches; `top1` = the best-scored instance is
  right; `empty` = nothing returned although the object is visible.
- Latency: median per frame (inventory) or per query (refer). Pipelines that reuse an earlier brain call
  include that call's time.

## Prompts

Inventory (`P_BOXES` in `tools/run_candidates.py`; the brain detector uses this wording):

> You are the perception module of a tabletop robot arm. List every manipulable object resting on the table
> or inside a container on the table, one entry per instance. Exclude the robot arm and its gripper, people
> and anything in their hands, screens, monitors, laptops, keyboards, furniture, walls and the table itself.
> Give each a short specific noun phrase (e.g. 'red plush lobster', 'green filament spool', 'paper cup',
> 'gray plastic basket') and a tight bounding box. Answer ONLY with JSON: [{"label": "...", "bbox_2d":
> [x1, y1, x2, y2]}] with coordinates normalized to 0-1000. Answer [] if there is none.

Refer (`P_REFER`):

> Locate every instance of "{q}" in the image. Answer ONLY with JSON: [{"label": "{q}", "bbox_2d":
> [x1, y1, x2, y2]}] with coordinates normalized to 0-1000, or [] if it is not visible.

`P_NAMES` (object kinds only, the input of the `*_qn` rows) and the cloud prompts (absolute pixel
coordinates, image size stated) are in `tools/run_candidates.py` and `tools/run_astra.py` / `tools/run_cli_vlm.py`.

## Pipelines

- `<brain>_boxes`: the brain lists objects with boxes. `_sam2` / `_sam31` / `_sam3n`: the same boxes as
  box prompts to SAM 2.1-L / SAM 3.1 / SAM 3 (Meta's repo) for masks.
- `qwen_names`: Qwen names object kinds without boxes (scores 0 by design; input of `*_qn`).
- `*_qn`: Qwen's names as text prompts to LLMDet-L (+ SAM 2.1-L masks), SAM 3 (transformers), SAM 3.1 /
  SAM 3 (Meta's repo) or YOLOE-26-X. `sam3_an`: SAM 3 prompted with Astra's labels.
- `fuse_*`: Qwen boxes + SAM 2.1-L masks, plus LLMDet / SAM 3 instances that overlap no Qwen instance
  (mask IoU < 0.5); `_hi` keeps LLMDet scores >= 0.5 only.
- `yoloe11s_pf`: the former production detector as configured. `yoloe26_pf`: YOLOE-26-X prompt-free.
- Brains: `qwen` = Qwen3.8-27B (local). `nemo` = Nemotron 3.5 Super VL NVFP4 (local, thinking off).
  `cosmos` = Cosmos3-Edge (local, thinking off); `cosmosT` = Cosmos3-Edge with thinking on (max_tokens 4000).
  `astra` = GPT-6-Astra via the Codex CLI and `opus` = Claude Opus 5.5 via `claude -p`: **cloud references
  only**, never a perception backend.

## Inventory (157 GT objects, 47 frames)

| pipeline | loc_recall | id_strict | id_lenient | precision | wrong_label | median s |
|---|---|---|---|---|---|---|
| astra_boxes (cloud reference) | 0.92 | 0.83 | 0.84 | 0.61 | 0.09 | 20.10 |
| astra_boxes_sam2 (cloud reference) | 0.92 | 0.83 | 0.84 | 0.61 | 0.09 | 21.07 |
| fuse_qwen_llmdet | 0.90 | 0.79 | 0.83 | 0.72 | 0.07 | 5.48 |
| fuse_qwen_llmdet_hi | 0.89 | 0.79 | 0.83 | 0.78 | 0.06 | 5.48 |
| fuse_qwen_sam3 | 0.90 | 0.79 | 0.83 | 0.66 | 0.07 | 5.56 |
| opus_boxes (cloud reference) | 0.90 | 0.79 | 0.83 | 0.67 | 0.08 | 9.51 |
| opus_boxes_sam2 (cloud reference) | 0.90 | 0.79 | 0.83 | 0.67 | 0.08 | 10.12 |
| qwen_boxes | 0.89 | 0.79 | 0.83 | 0.81 | 0.06 | 5.12 |
| **qwen_boxes_sam2 (shipped)** | **0.89** | **0.79** | **0.83** | **0.81** | **0.06** | **5.37** |
| qwen_boxes_sam31 | 0.89 | 0.79 | 0.83 | 0.81 | 0.06 | 5.62 |
| qwen_boxes_sam3n | 0.89 | 0.79 | 0.83 | 0.81 | 0.06 | 5.63 |
| nemo_boxes (thinking off) | 0.83 | 0.75 | 0.78 | 0.72 | 0.07 | 5.26 |
| llmdet_qn | 0.83 | 0.68 | 0.74 | 0.69 | 0.11 | 5.23 |
| sam3_an | 0.76 | 0.71 | 0.73 | 0.65 | 0.03 | 28.55 |
| sam3_qn | 0.76 | 0.63 | 0.69 | 0.62 | 0.08 | 5.30 |
| sam31_qn | 0.71 | 0.61 | 0.66 | 0.62 | 0.07 | 2.13 |
| sam3n_qn | 0.70 | 0.61 | 0.66 | 0.65 | 0.06 | 2.13 |
| cosmosT_boxes (thinking on) | 0.46 | 0.40 | 0.45 | 0.89 | 0.04 | 6.78 |
| yoloe26_qn | 0.46 | 0.35 | 0.41 | 0.70 | 0.12 | 2.30 |
| cosmos_boxes (thinking off) | 0.33 | 0.32 | 0.33 | 0.93 | 0.00 | 0.90 |
| yoloe26_pf | 0.85 | 0.24 | 0.26 | 0.12 | 0.69 | 0.03 |
| **yoloe11s_pf (former production)** | 0.76 | 0.23 | 0.25 | 0.12 | 0.67 | 0.01 |
| qwen_names (no boxes by design) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.63 |

## Refer (157 queries; query = GT name)

| pipeline | hit | top1 | empty | median s |
|---|---|---|---|---|
| ref_opus (cloud reference) | 0.98 | 0.98 | 0.01 | 4.67 |
| ref_astra (cloud reference) | 0.96 | 0.96 | 0.01 | 6.74 |
| **ref_qwen (shipped)** | **0.94** | **0.94** | **0.01** | **1.87** |
| ref_llmdet | 0.92 | 0.85 | 0.01 | 1.30 |
| ref_sam3 | 0.88 | 0.87 | 0.12 | 0.77 |
| ref_nemo (thinking off) | 0.85 | 0.85 | 0.12 | 1.56 |
| ref_cosmos (thinking off) | 0.82 | 0.82 | 0.15 | 0.85 |
| ref_sam31 | 0.73 | 0.73 | 0.27 | 0.31 |
| ref_cosmosT (thinking on; **partial: 105 of 157 queries**) | 0.71 | 0.71 | 0.15 | 3.68 |
| ref_sam3n | 0.69 | 0.69 | 0.31 | 0.32 |
| ref_yoloe26 | 0.61 | 0.56 | 0.32 | 0.21 |
| **ref_yoloe11s (former production)** | 0.40 | 0.38 | 0.53 | 0.77 |

Nemotron with thinking on (`nemoT`) finished after this table was scored (refer stopped at 49 of 182
queries): not reported.

## Why this pipeline

1. Qwen boxes + SAM 2.1-L ties the best local identification (0.83) and has the best precision (0.81) of
   the pipelines that find most objects (Cosmos is more precise but identifies a third of them).
2. SAM 2.1-L, SAM 3.1 and SAM 3 give the same scores from Qwen's boxes. SAM 2.1-L loads through
   ultralytics, is Apache-2.0 and needs no gated download.
3. Fusing a detector into Qwen adds no identification (0.83) and lowers precision (0.66-0.78).
4. LLMDet refer (0.92) needs transformers < 5 (5.19 returns wrong boxes); Nemotron and Cosmos score lower.
5. The cloud references are at most 0.01 better at inventory and 0.02-0.04 at refer, less precise
   (0.61-0.67), and not local.

## Failure modes seen

- The dark brown spool of thread is missed by every method.
- Qwen names the red lobster keychain's lanyard "red cable".
- YOLOE prompt-free labels booth objects from its fixed vocabulary: "rag doll", "electric chair".
- Qwen sometimes appends a second `[]` after its answer: parse the FIRST JSON list (the brain detector does).

## Limits

- Labels come from vision subagents with overlay checks, not human annotators.
- One booth, one table, 47 frames.
- 2D only: no 9D pose, real-depth or grasp test yet.
- Latency is single-stream; the live brain also serves the agent's own turns.

## Re-run

The bake-off workspace (`~/Projects/perception-bakeoff`: `frames/`, `gt/`, `weights/`, `out/pred/`) is
local to the Spark and stays there: the frames show people. Only the scripts are archived here, lint-cleaned
(unused imports removed; no behaviour change).

1. Put `tools/` next to `frames/` (JPEGs + `manifest.json`), `gt/<id>.json` and `weights/`
   (torch 2.14.0+cu130, ultralytics 8.4.177, transformers 5.19.0; LLMDet in a transformers 4.x venv).
2. Qwen: `BRAIN_URL=http://127.0.0.1:8081/v1 BRAIN_MODEL=qwen3.8-27b python tools/run_candidates.py qwen_boxes`,
   then `qwen_boxes_sam2` (reuses the replies) and `ref_qwen`. The key is read from
   `~/.cascade/nemoclaw/llama.key`. Other brains: `BRAIN_TAG`, `BRAIN_EXTRA`, `BRAIN_MAXTOK`.
3. SAM 3.1 / SAM 3 (Meta's repo, own venv): `python tools/run_sam31.py <sam3.1|sam3> <qboxes|qn|ref>`.
4. Cloud references: `python tools/run_astra.py`, `python tools/run_cli_vlm.py <astra|opus> <boxes|ref>`.
5. `python tools/score.py` writes `out/scores.md` and `out/scores.json`.

Not archived: `fuse.py` (the `fuse_*` rows, no model calls), the YOLOE-11s runner (it imports the removed
`OpenVocabDetector`), the chain and server scripts.
