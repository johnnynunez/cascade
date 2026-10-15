"""Cloud VLMs through their CLIs as the perception brain (reference only, never production).

  python3 tools/run_cli_vlm.py <astra|opus> <boxes|ref> [workers]
    astra boxes -> out/pred/astra_boxes   (already run by run_astra.py; cached files are skipped)
    astra ref   -> out/pred/ref_astra
    opus  boxes -> out/pred/opus_boxes
    opus  ref   -> out/pred/ref_opus
Same prompts for both (absolute pixel coordinates, the image size stated). Astra = GPT-6-Astra via
`codex exec -i` (ChatGPT login, reasoning effort low); Opus = claude-opus-5-5 via `claude -p` with
only the Read tool, reading the frame from a scratch copy. No GPU, no robot.
"""
import concurrent.futures as cf
import json
import pathlib
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import FRAMES, PRED, frame_ids, load_gt, save_pred  # noqa: E402
from run_astra import EMPTY, PROMPT  # noqa: E402
from run_candidates import parse_json  # noqa: E402

CLAUDE_DIR = pathlib.Path.home() / ".hermes/cache/scratch/claude-empty"
REFER = ("Locate every instance of \"{q}\" in the image. The image is {w}x{h} pixels. Answer ONLY with JSON: "
         "[{{\"label\": \"{q}\", \"bbox\": [x1, y1, x2, y2]}}] in absolute pixel coordinates, or [] if it is not "
         "visible.")


def ask(backend, fid, prompt):
    img = FRAMES / f"{fid}.jpg"
    t0 = time.time()
    if backend == "astra":
        r = subprocess.run(["codex", "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check", "-s",
                            "read-only", "-C", str(EMPTY), "-m", "gpt-6-astra", "-c", "model_reasoning_effort=low",
                            "--color", "never", "-i", str(img), "-"],
                           input=prompt, capture_output=True, text=True, timeout=300)
        raw = r.stdout
        text = raw.split("tokens used")[-1] if "tokens used" in raw else raw
        meta = {"rc": r.returncode}
    else:
        r = subprocess.run(["claude", "-p", "--model", "claude-opus-5-5", "--allowedTools", "Read",
                            "--output-format", "json",
                            f"Read the image ./{fid}.jpg first. Then answer this, about that image:\n\n{prompt}"],
                           cwd=CLAUDE_DIR, capture_output=True, text=True, timeout=300)
        try:
            d = json.loads(r.stdout)
            raw = text = d.get("result", "") or ""
            meta = {"rc": r.returncode, "cost_usd": d.get("total_cost_usd"), "turns": d.get("num_turns"),
                    "is_error": d.get("is_error")}
        except json.JSONDecodeError:
            raw, text, meta = r.stdout + r.stderr[-2000:], "", {"rc": r.returncode}
    dt = time.time() - t0
    preds = []
    for o in parse_json(text):
        if isinstance(o, dict) and isinstance(o.get("bbox"), list) and len(o["bbox"]) == 4:
            try:
                preds.append({"label": str(o.get("label", "object")), "score": 1.0,
                              "bbox": [float(v) for v in o["bbox"]], "mask": None})
            except (TypeError, ValueError):
                pass
    # "raw_text", not "raw": run_candidates' *_sam2 reuse parses extra.raw as 0-1000 Qwen boxes.
    return preds, dt, {"raw_text": raw[-6000:], **meta}


def job(backend, task, fid, obj=None):
    from PIL import Image
    w, h = Image.open(FRAMES / f"{fid}.jpg").size
    if task == "boxes":
        out = f"{backend}_boxes"
        if (PRED / out / f"{fid}.json").exists():
            return f"{fid} cached"
        preds, dt, extra = ask(backend, fid, PROMPT.format(w=w, h=h))
        save_pred(out, fid, preds, dt, extra=extra)
        return f"{fid} {len(preds)} objects {dt:.1f}s {extra.get('cost_usd')}"
    q = obj["name"]
    import hashlib
    if (PRED / f"ref_{backend}" / f"{fid}__{hashlib.md5(q.encode()).hexdigest()[:10]}.json").exists():
        return f"{fid} '{q}' cached"
    preds, dt, extra = ask(backend, fid, REFER.format(q=q, w=w, h=h))
    save_pred(f"ref_{backend}", fid, preds, dt, task="refer", query=q, extra=extra)
    return f"{fid} '{q}' {len(preds)} {dt:.1f}s"


def main(backend, task, workers):
    CLAUDE_DIR.mkdir(parents=True, exist_ok=True)
    EMPTY.mkdir(parents=True, exist_ok=True)
    for fid in frame_ids():
        if not (CLAUDE_DIR / f"{fid}.jpg").exists():
            shutil.copy(FRAMES / f"{fid}.jpg", CLAUDE_DIR / f"{fid}.jpg")
    jobs = [(fid, None) for fid in frame_ids()] if task == "boxes" else \
        [(fid, o) for fid in frame_ids() for o in (load_gt(fid) or {}).get("objects", [])]
    with cf.ThreadPoolExecutor(workers) as ex:
        for msg in ex.map(lambda a: job(backend, task, *a), jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
