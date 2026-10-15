"""GPT-6-Astra (through the Codex CLI, like cascade's codex_astra profile) as the perception brain.

  python3 tools/run_astra.py [workers]      -> out/pred/astra_boxes/<id>.json (raw reply kept)
Absolute pixel coordinates (GPT-family models localize best in pixels). No GPU, no robot.
"""
import concurrent.futures as cf
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import FRAMES, PRED, frame_ids, save_pred  # noqa: E402
from run_candidates import EXCLUDE, parse_json  # noqa: E402

EMPTY = pathlib.Path.home() / ".hermes/cache/scratch/codex-empty"
PROMPT = ("You are the perception module of a tabletop robot arm. List every manipulable object resting on the "
          "table or inside a container on the table, one entry per instance. " + EXCLUDE + " Give each a short "
          "specific noun phrase (e.g. 'red plush lobster', 'green filament spool', 'paper cup', 'gray plastic "
          "basket') and a tight bounding box. The image is {w}x{h} pixels. Answer ONLY with JSON: "
          "[{{\"label\": \"...\", \"bbox\": [x1, y1, x2, y2]}}] in absolute pixel coordinates. Answer [] if there is none.")


def call(fid):
    out = PRED / "astra_boxes" / f"{fid}.json"
    if out.exists():
        return fid, "cached"
    from PIL import Image
    w, h = Image.open(FRAMES / f"{fid}.jpg").size
    t0 = time.time()
    r = subprocess.run(["codex", "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check", "-s",
                        "read-only", "-C", str(EMPTY), "-m", "gpt-6-astra", "-c", "model_reasoning_effort=low",
                        "--color", "never", "-i", str(FRAMES / f"{fid}.jpg"), "-"],
                       input=PROMPT.format(w=w, h=h), capture_output=True, text=True, timeout=300)
    dt = time.time() - t0
    raw = r.stdout
    tail = raw.split("tokens used")[-1] if "tokens used" in raw else raw
    preds = []
    for o in parse_json(tail):
        if isinstance(o, dict) and isinstance(o.get("bbox"), list) and len(o["bbox"]) == 4:
            try:
                preds.append({"label": str(o.get("label", "object")), "score": 1.0,
                              "bbox": [float(v) for v in o["bbox"]], "mask": None})
            except (TypeError, ValueError):
                pass
    save_pred("astra_boxes", fid, preds, dt, extra={"raw": raw[-6000:], "rc": r.returncode})
    return fid, f"{len(preds)} objects {dt:.1f}s rc={r.returncode}"


if __name__ == "__main__":
    EMPTY.mkdir(parents=True, exist_ok=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    with cf.ThreadPoolExecutor(n) as ex:
        for fid, msg in ex.map(call, frame_ids()):
            print(fid, msg, flush=True)
