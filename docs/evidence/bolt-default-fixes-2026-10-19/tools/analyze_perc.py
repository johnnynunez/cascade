"""Offline parse of the saved raw Bolt replies with cascade's PRODUCTION parsers (throwaway, PR perf/bolt-default-fixes).

  PYTHONPATH=<worktree>/src .venv/bin/python analyze_perc.py <worktree> SOURCE:PARSER:TAG ...

SOURCE  cmp       = the 2026-10-18 brain comparison's raw Bolt replies (raw/perception_boltll_{inventory,refer}.jsonl)
        <variant> = this PR's raw replies (raw/perc_<variant>.jsonl from perc_run.py)
PARSER  main      = main's BrainDetector: a cut reply (finish_reason length) or one without a box list is garbled -> []
        bounded   = this branch's parse_guarded_boxes for a guarded brain (cut -> complete boxes before the cut; a reply
                    that hit a bound loses its repetition loop; exact repeats and, in inventories, entries naming a
                    category the prompt excludes dropped); garbled as on the branch
        bounded<N>= as bounded, after keeping only the first N entries of the reply (what a schema maxItems of N gives:
                    greedy decoding is identical up to the bound)
Writes out/pred/<TAG>_boxes/ (inventory, boxes only, latency of the raw call) and ref_<TAG>/ (refer) for score_cmp.py,
plus raw/loops_<TAG>.json: per frame the finish reason, completion tokens, the longest same-label run in the raw reply.
"""
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path("/home/spark/Projects/perception-bakeoff")
OUT = ROOT / "out/bolt-default-fixes-2026-10-18"
sys.path.insert(0, str(ROOT / "tools"))
from common import save_pred  # noqa: E402

WT = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(WT / "src"))
import cascade.perception.brain_detector as bd  # noqa: E402

assert pathlib.Path(bd.__file__).resolve().is_relative_to(WT.resolve()), bd.__file__
W, H = 1280, 720


def rows(source):
    if source == "cmp":
        base = ROOT / "out/brain-comparison-2026-10-18/raw/perception_boltll_"
        out = []
        for task in ("inventory", "refer"):
            for line in (base.parent / f"perception_boltll_{task}.jsonl").read_text().splitlines():
                r = json.loads(line)
                out.append({"task": task, "frame": r["frame"], "query": r.get("query"), "raw": r["raw"],
                            "finish": r["finish"], "completion_tokens": r.get("completion_tokens"),
                            "latency_s": r["latency_s"]})
        return out
    return [json.loads(line) for line in (OUT / f"raw/perc_{source}.jsonl").read_text().splitlines()]


def first_entries(text, n):
    """The reply cut after its first n complete entries and closed (a schema maxItems of n)."""
    body = bd._reply_body(text)
    items = bd._answer_list(body) or bd._complete_items(body)
    return json.dumps(items[:n]), len(items) > n


def longest_run(text):
    labels = [" ".join(m.lower().split()) for m in re.findall(r'"label"\s*:\s*"([^"]*)"', text)]
    best = cur = 0
    for i, lab in enumerate(labels):
        cur = cur + 1 if i and lab == labels[i - 1] else 1
        best = max(best, cur)
    return best, len(labels)


def parse(parser, r):
    text, cut = r["raw"], r["finish"] == "length"
    if parser == "main":
        return bd.parse_boxes(text, W, H) if not cut and bd._is_answer(text) else []
    n = int(parser[len("bounded"):] or bd.GUARDED_MAX_BOXES)
    if n != bd.GUARDED_MAX_BOXES:
        text, more = first_entries(text, n)
        cut = cut and not more  # the schema's bound ends the list before the token cap
    old, bd.GUARDED_MAX_BOXES = bd.GUARDED_MAX_BOXES, n
    try:
        found = bd.parse_guarded_boxes(text, W, H, cut=cut, inventory=r["task"] == "inventory")
    finally:
        bd.GUARDED_MAX_BOXES = old
    if (cut and not found) or (not cut and not bd._is_answer(text)):
        return []
    return found


def main(specs):
    for spec in specs:
        source, parser, tag = spec.split(":")
        loops = {}
        for r in rows(source):
            preds = [{"label": label, "score": 1.0, "bbox": [float(v) for v in box], "mask": None}
                     for label, box in parse(parser, r)]
            extra = {"finish": r["finish"], "completion_tokens": r.get("completion_tokens"), "source": source,
                     "parser": parser, "raw": r["raw"]}
            if r["task"] == "inventory":
                save_pred(f"{tag}_boxes", r["frame"], preds, r["latency_s"], extra=extra)
                run, n = longest_run(r["raw"])
                loops[r["frame"]] = {"finish": r["finish"], "completion_tokens": r.get("completion_tokens"),
                                     "entries": n, "longest_same_label_run": run, "kept": len(preds),
                                     "latency_s": round(r["latency_s"], 2)}
            else:
                save_pred(f"ref_{tag}", r["frame"], preds, r["latency_s"], task="refer", query=r["query"], extra=extra)
        (OUT / f"raw/loops_{tag}.json").write_text(json.dumps(loops, indent=1))
        cut = sum(v["finish"] == "length" for v in loops.values())
        runs = sum(v["longest_same_label_run"] >= bd._LOOP_ENTRIES for v in loops.values())
        empty = sum(v["kept"] == 0 for v in loops.values())
        print(f"{tag}: frames {len(loops)}, cut at max_tokens {cut}, replies with a same-label run >= {bd._LOOP_ENTRIES}: "
              f"{runs}, frames with no box {empty}, max completion tokens "
              f"{max((v['completion_tokens'] or 0) for v in loops.values())}")


if __name__ == "__main__":
    main(sys.argv[2:])
