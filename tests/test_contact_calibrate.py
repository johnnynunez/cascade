"""scripts/contact_calibrate.py: thresholds from measure-mode reports (offline).

It reads trace.jsonl rows (context.contact_monitor) and plain report lines,
takes per-joint worst values over every motion, and suggests a lag_s and
per-joint track_tol_rad that are margin x what was seen -- never below the
settle tolerance + 0.02 rad -- and lists every motion that would have
tripped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import contact_calibrate as cc


def report(err, lag=(0.08,) * 6, ratio=(0.9,) * 6, would=None, motion="move_joints"):
    return {"motion": motion, "mode": "measure", "lag_s": 0.25, "samples": [10] * 6,
            "max_track_err_rad": list(err), "max_raw_err_rad": [e + 0.02 for e in err],
            "max_effective_lag_s": list(lag), "min_advance_ratio": list(ratio),
            "stall_ratio": 0.2, "max_feedback_gap_s": 0.12, "read_failures": 0,
            "reads": {"per_cycle": 1.5}, "would_trip": would, "would_trips": 1 if would else 0,
            "tripped": None}


def test_reads_trace_rows_and_report_lines(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text("\n".join([
        json.dumps({"skill": "move_home", "context": {"contact_monitor": [report([0.01] * 6)]}}),
        json.dumps({"skill": "list_objects", "context": {"arm": "default"}}),
        "not json",
    ]))
    log = tmp_path / "rig.jsonl"
    log.write_text(json.dumps(report([0.02] * 6)) + "\n")
    assert len(cc.load_reports([trace, log])) == 2


def test_suggests_margin_times_worst_per_joint_floored_at_settle():
    s = cc.suggest([report([0.010, 0.030, 0.012, 0.004, 0.0, 0.002]),
                    report([0.013, 0.020, 0.040, 0.004, 0.0, 0.002],
                           lag=(0.09, 0.12, 0.10, 0.06, 0.05, 0.05))])
    assert s["max_track_err_rad"] == [0.013, 0.030, 0.040, 0.004, 0.0, 0.002]
    assert s["suggested"]["track_tol_rad"] == [0.07, 0.09, 0.12, 0.07, 0.07, 0.07]
    assert s["suggested"]["lag_s"] == 0.2              # 1.5 x 0.12, up to 0.05 steps
    assert s["min_advance_ratio"] == [0.9] * 6


def test_lag_is_clamped_and_would_trips_are_listed():
    s = cc.suggest([report([0.01] * 6, lag=(0.5,) * 6, would="contact detected at joint 2: ...")])
    assert s["suggested"]["lag_s"] == 0.4
    assert s["would_trip"] == ["move_joints: contact detected at joint 2: ... (1 sample(s))"]
    text = cc.render(s)
    assert "WOULD_TRIP" in text and "must be explained" in text


def test_a_stall_ratio_close_to_the_threshold_is_flagged():
    text = cc.render(cc.suggest([report([0.01] * 6, ratio=(0.9, 0.3, 0.9, 0.9, 0.9, 0.9))]))
    assert "joint(s) [2] came within 2x of stall_ratio" in text


def test_nothing_monitored_is_an_error(tmp_path, capsys):
    empty = tmp_path / "trace.jsonl"
    empty.write_text(json.dumps({"skill": "x", "context": {}}) + "\n")
    assert cc.main([str(empty)]) == 1
    assert "nothing to calibrate" in capsys.readouterr().out
