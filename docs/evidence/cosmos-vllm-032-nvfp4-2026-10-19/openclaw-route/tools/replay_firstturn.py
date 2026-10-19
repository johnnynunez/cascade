#!/usr/bin/env python3
"""Replay the six live2 first-turn requests (requests-profile/) against a
Cosmos3-Edge server and score the first tool call.

usage: replay_firstturn.py <base_url> <reps> [variant,...]
variants: asis (byte-identical), salt (fresh vLLM cache_salt per request: no
prefix-cache reuse), plus any of the prompt variants named in PROMPTS.
Prints per-request counts and a total; appends rows to firstturn.jsonl."""
import copy
import json
import sys
import uuid
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from replay_matrix_lib import send  # noqa: E402

D = Path(__file__).parent
CASES = {
    "0148": ("cascade__describe_scene", {}),
    "0151": ("cascade__pick_and_place", {"object": "green cube", "destination": "green square"}),
    "0156": ("cascade__pick_and_place", {"object": "orange", "destination": "open box"}),
    "0161": ("cascade__point_at", {"label": "lemon"}),
    "0164": ("cascade__pick_and_place", {"object": "pink cube", "destination": "green square"}),
    "0169": ("cascade__reset_scene", {}),
}
EXTRA = {
    "T5new": ("Please put the pink cube in the open box, then put the lemon in the green square.",
              ("cascade__pick_and_place", {"object": "pink cube", "destination": "open box"})),
    "dropzone": ("Please put the pink cube in the drop zone.",
                 ("cascade__pick_and_place", {"object": "pink cube", "destination": "green square"})),
    "home": ("Go home.", ("cascade__move_home", {})),
    "open": ("Open the gripper.", ("cascade__open_gripper", {})),
    "ws": ("What is the current world state?", ("cascade__world_state", {})),
    "where": ("Where is the tomato can?", ("cascade__localize_object", {"label": "tomato can"})),
    "stop": ("Stop!", ("cascade__emergency_stop", {})),
}
PROMPTS = json.loads((D / "firstturn_prompts.json").read_text()) if (D / "firstturn_prompts.json").exists() else {}


def score(want, out):
    if not out.get("calls"):
        return False
    c = out["calls"][0]
    if c["name"] != want[0]:
        return False
    try:
        args = json.loads(c["arguments"] or "{}")
    except ValueError:
        return False
    return all(str(args.get(k, "")).strip().lower() in ((v, "drop zone") if v == "green square" and want is EXTRA["dropzone"][1] else (v,))
               for k, v in want[1].items())


def build(n, variant):
    if n in CASES:
        req = copy.deepcopy(json.loads((D / f"requests-profile/{n}.json").read_text())["request"])
        want = CASES[n]
    else:
        req = copy.deepcopy(json.loads((D / "requests-profile/0151.json").read_text())["request"])
        req["messages"][1]["content"] = EXTRA[n][0]
        want = EXTRA[n][1]
    req["stream"] = True
    for tok in variant.split("+"):
        if tok == "salt":
            req["cache_salt"] = uuid.uuid4().hex
        elif tok == "noplace":
            req["tools"] = [t for t in req["tools"] if t["function"]["name"] not in ("cascade__place_on_object", "cascade__place_at")]
        elif tok == "noloc":
            req["tools"] = [t for t in req["tools"] if t["function"]["name"] != "cascade__localize_object"]
        elif tok.startswith("fp") and tok[2:].replace(".", "").isdigit():
            req["frequency_penalty"] = float(tok[2:])
        elif tok in PROMPTS:
            req["messages"][0]["content"] = PROMPTS[tok]
    return req, want


base, reps = sys.argv[1], int(sys.argv[2])
variants = (sys.argv[3] if len(sys.argv) > 3 else "asis").split(",")
names = list(CASES) + (list(EXTRA) if "--extra" in sys.argv else [])
out_rows = D / "firstturn.jsonl"
for variant in variants:
    total, ok = 0, 0
    for n in names:
        c = Counter()
        good = 0
        for _ in range(reps):
            req, want = build(n, variant)
            out = send(base, req)
            s = score(want, out)
            good += s
            key = (out["calls"][0]["name"].replace("cascade__", "") + out["calls"][0]["arguments"]) if out["calls"] \
                else "TEXT:" + out["text"][:40]
            key = key[:90]
            c[key] += 1
            with out_rows.open("a") as f:
                f.write(json.dumps({"variant": variant, "req": n, "ok": s, "out": key, "base": base}) + "\n")
        total += reps
        ok += good
        print(f"{variant:14} {n:9} {good}/{reps}  {dict(c)}", flush=True)
    print(f"{variant:14} TOTAL {ok}/{total}", flush=True)
