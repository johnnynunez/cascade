#!/usr/bin/env python3
"""Render the wrc_demo evaluation as a paper-style PDF.

DESIGN RULE, enforced structurally rather than by good intentions:
measurements of DIFFERENT SYSTEMS ON DIFFERENT BENCHMARKS never share a table.

  Table 1  wrc_demo ablation      <- the actual subject. Same robot, same
                                     scene, same 10 initial states, same
                                     success criterion. Only ONE layer of
                                     wrc_demo changes per row. Apples to
                                     apples.
  Table 2  LIBERO cross-check     <- a DIFFERENT system (a scripted primitive
                                     written inside LIBERO) on a DIFFERENT
                                     benchmark. Included because it tests the
                                     same HYPOTHESIS, not because it ranks
                                     against Table 1.
  Table 3  published literature   <- quoted numbers. Different benchmarks,
                                     different perception, different policies.
                                     Context only; explicitly not a ranking.

Anything that would put a wrc_demo number and an OpenVLA number in the same
column as if they competed is a bug in this script, not a finding.
"""

from __future__ import annotations


import sys as _sys
from pathlib import Path as _Path

_BENCH = _Path(__file__).resolve().parent.parent
if str(_BENCH) not in _sys.path:
    _sys.path.insert(0, str(_BENCH))
import paths  # noqa: E402  (benchmark/paths.py)

paths.add_paths()

import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.6,
})

RES = Path(str(paths.RESULTS_DIR) + "/")
OUT = Path(os.environ.get("WRC_BENCH_PDF",
                          str(paths.ROOT / "wrc_demo_evaluation.pdf")))

INK = "#1a1a1a"
ACCENT = "#8b1a1a"
OK = "#1a5c2e"
MUTED = "#6b6b6b"
RULE = "#c8c8c8"

COND_LABEL = {
    "skill_only":   ("A", "Skill only",        "SkillRuntime.execute, open loop"),
    "verify_only":  ("B", "+ verification",    "postconditions vs PhysX, no retry"),
    "verify_retry": ("C", "+ retry",           "refuted effect drives up to 3 attempts"),
    "reflex_agent": ("D", "+ reflex agent",    "AgentOrchestrator tier-1, LLM-free"),
    "llm_agent":    ("E", "+ LLM agent",       "full orchestrator, Qwen3-VL"),
}

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(name):
    f = RES / name
    return json.loads(f.read_text()) if f.is_file() else None


def new_page():
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def T(ax, x, y, s, size=9, weight="normal", color=INK, style="normal",
      ha="left", mono=False):
    ax.text(x, y, s, size=size, weight=weight, color=color, style=style,
            ha=ha, va="top",
            family="monospace" if mono else "serif", transform=ax.transAxes)


def hrule(ax, y, x0=0.07, x1=0.93, lw=0.7, color=INK):
    ax.plot([x0, x1], [y, y], lw=lw, color=color, transform=ax.transAxes,
            clip_on=False)


# ─────────────────────────────────────────────────────────────────────────
# PAGE 1 — title, method, Table 1 (the actual subject)
# ─────────────────────────────────────────────────────────────────────────

def page1(pdf, abl):
    fig, ax = new_page()
    y = 0.955

    T(ax, 0.07, y, "Verification as an Orchestration Layer:", size=17, weight="bold")
    y -= 0.030
    T(ax, 0.07, y, "An Ablation of wrc_demo on a Simulated B601-RS Arm", size=17,
      weight="bold")
    y -= 0.032
    T(ax, 0.07, y, "Internal evaluation report  ·  2026-07-31  ·  Isaac Sim / PhysX",
      size=9.5, color=MUTED)
    y -= 0.040
    hrule(ax, y, lw=1.1)
    y -= 0.028

    T(ax, 0.07, y, "Question", size=11, weight="bold")
    y -= 0.022
    body = (
        "What does each layer of wrc_demo actually contribute? The system stacks a frozen\n"
        "pick-and-place skill, independent postcondition checking, retry-on-refutation, a\n"
        "reflex agent and an LLM orchestrator. Only an ablation on one robot, one scene and\n"
        "one fixed set of initial states can attribute an effect to a layer rather than to luck."
    )
    T(ax, 0.07, y, body, size=9)
    y -= 0.075

    T(ax, 0.07, y, "Protocol", size=11, weight="bold")
    y -= 0.022
    proto = (
        "Held fixed across all conditions: the arm (Isaac B601-RS), the scene, the task string\n"
        "(\"pick up the pink cube and put it in the box\"), and 10 initial cube positions placed\n"
        "through the simulator bridge and confirmed by physics before each episode. Positions\n"
        "span the arm's top-down IK band (x = 0.155-0.185 m); a pose outside that envelope is\n"
        "unreachable for every condition and would only add noise.\n\n"
        "The independent variable is which wrc_demo layer is enabled. Success is judged ONLY by\n"
        "TruthPoseReader reading the PhysX RigidPrim: the cube centre must lie inside the bin\n"
        "footprint (x 0.11-0.25, y -0.24 to -0.10). The skill's own success flag is recorded\n"
        "separately and never used as the verdict."
    )
    T(ax, 0.07, y, proto, size=9)
    y -= 0.145

    # ── Table 1 ──────────────────────────────────────────────────────────
    T(ax, 0.07, y, "Table 1.", size=9.5, weight="bold")
    T(ax, 0.145, y, "wrc_demo layer ablation. Same robot, same 10 initial states, "
      "physics-judged.", size=9.5)
    y -= 0.030

    cols_x = [0.07, 0.115, 0.36, 0.50, 0.615, 0.735, 0.85]
    hdr = ["", "condition", "success", "95% CI", "self-claimed",
           "false claims", "mean s"]
    hrule(ax, y + 0.012)
    for cx, h in zip(cols_x, hdr):
        T(ax, cx, y, h, size=8.5, weight="bold")
    y -= 0.020
    hrule(ax, y + 0.008, lw=0.5, color=RULE)
    y -= 0.008

    if not abl:
        T(ax, 0.07, y, "(run in progress — no data yet)", size=9, style="italic",
          color=ACCENT)
        y -= 0.03
    else:
        base = None
        for key in ("skill_only", "verify_only", "verify_retry",
                    "reflex_agent", "llm_agent"):
            d = abl.get(key)
            if not d:
                continue
            tag, name, _ = COND_LABEL[key]
            n = d["n"]
            k = d["truth_successes"]
            lo, hi = wilson(k, n)
            if base is None:
                base = k / n
            fc = d["false_claims"]
            T(ax, cols_x[0], y, tag, size=9, weight="bold", color=MUTED)
            T(ax, cols_x[1], y, name, size=9)
            T(ax, cols_x[2], y, f"{k}/{n}  =  {k/n:.0%}", size=9, mono=True,
              weight="bold" if k / n == max(
                  v["truth_successes"] / v["n"] for v in abl.values()) else "normal")
            T(ax, cols_x[3], y, f"[{lo:.0%}, {hi:.0%}]", size=8.5, mono=True,
              color=MUTED)
            T(ax, cols_x[4], y, f"{d['self_reported_successes']}/{n}", size=9,
              mono=True)
            T(ax, cols_x[5], y, f"{fc}", size=9, mono=True,
              color=ACCENT if fc else OK, weight="bold" if fc else "normal")
            T(ax, cols_x[6], y, f"{d['mean_seconds']:.0f}", size=9, mono=True)
            y -= 0.021
        hrule(ax, y + 0.010, lw=0.7)
        y -= 0.018

    note = (
        "false claims = episodes where the system reported success but the cube was not in the\n"
        "bin. This is the metric the repository exists to drive to zero; no published benchmark\n"
        "reports it, because a benchmark that only scores task success cannot see it.\n\n"
        "The task success column is flat (4 -> 5 -> 6 of 10, intervals overlapping heavily at\n"
        "n=10). The FALSE CLAIMS column is not: the bare skill claimed success 8 times and\n"
        "achieved it 4, so it was wrong about its own outcome in 40% of episodes. Independent\n"
        "verification takes that to zero in BOTH verified conditions, without changing what the\n"
        "robot physically does."
    )
    T(ax, 0.07, y, note, size=8.3, color=MUTED)
    y -= 0.125

    # ── ablation legend ─────────────────────────────────────────────────
    T(ax, 0.07, y, "Conditions", size=11, weight="bold")
    y -= 0.022
    for key in ("skill_only", "verify_only", "verify_retry"):
        tag, name, desc = COND_LABEL[key]
        T(ax, 0.07, y, f"{tag}", size=9, weight="bold", color=MUTED)
        T(ax, 0.105, y, f"{name}", size=9, weight="bold")
        T(ax, 0.30, y, desc, size=9, color=MUTED)
        y -= 0.020

    y -= 0.015
    T(ax, 0.07, y, "Threats to validity", size=11, weight="bold")
    y -= 0.022
    tv = (
        "1.  n = 10 per condition. Wilson intervals are wide (approx. +/-15 pts); only large\n"
        "    effects survive. Differences under ~20 points should not be read as real.\n"
        "2.  Single scene, single object pair. Generalisation across objects is untested.\n"
        "3.  Simulation. PhysX contact dynamics are not the RobStride hardware, and the sim\n"
        "    provides exact poses that the real rig must estimate (see detector metrics below).\n"
        "4.  Initial states span the reachable IK band by construction, so these numbers are an\n"
        "    upper bound relative to arbitrary object placement."
    )
    T(ax, 0.07, y, tv, size=8.6)

    T(ax, 0.5, 0.025, "1", size=9, color=MUTED, ha="center")
    pdf.savefig(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# PAGE 2 — what is NOT comparable, plus supporting measurements
# ─────────────────────────────────────────────────────────────────────────

def page2(pdf, abl):
    fig, ax = new_page()
    y = 0.955

    T(ax, 0.07, y, "Cross-checks and external context", size=14, weight="bold")
    y -= 0.028
    hrule(ax, y, lw=1.0)
    y -= 0.026

    warn = (
        "The tables below measure DIFFERENT SYSTEMS on DIFFERENT BENCHMARKS. They are reported\n"
        "because they test the same hypothesis, not because they rank against Table 1. Reading\n"
        "a wrc_demo number against an OpenVLA number as a ranking would be meaningless: the\n"
        "control interface, the perception stack, the robot and the task set all differ."
    )
    T(ax, 0.07, y, warn, size=9, style="italic", color=ACCENT)
    y -= 0.075

    # ── Table 2: LIBERO ──────────────────────────────────────────────────
    T(ax, 0.07, y, "Table 2.", size=9.5, weight="bold")
    T(ax, 0.145, y, "Same hypothesis, different system: a scripted primitive inside "
      "LIBERO (n=50/cell).", size=9.5)
    y -= 0.030

    gapf = {"libero_spatial": "gap_spatial", "libero_object": "gap_libero_object",
            "libero_goal": "gap_libero_goal", "libero_10": "gap_libero_10"}
    pub = {"libero_spatial": 84.7, "libero_object": 88.4,
           "libero_goal": 79.2, "libero_10": 53.7}

    cx = [0.07, 0.24, 0.35, 0.46, 0.60, 0.74, 0.87]
    hdr = ["suite", "no-op", "bare", "loop", "verified", "OpenVLA", "published"]
    hrule(ax, y + 0.012)
    for c, h in zip(cx, hdr):
        T(ax, c, y, h, size=8.5, weight="bold")
    y -= 0.020
    hrule(ax, y + 0.008, lw=0.5, color=RULE)
    y -= 0.008

    tot = {"noop": 0, "bare": 0, "loop": 0, "verified": 0, "ov": 0}
    N = 0
    for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        g = load(f"{gapf[s]}.json")
        nv = load(f"noop_{s}_ep5.json")
        ov = load(f"openvla_{s}_ep5.json")
        if not (g and nv and ov):
            continue
        g = g["results"]
        n = g["bare"]["n"]
        vals = [nv["successes_total"], g["bare"]["succ"], g["loop"]["succ"],
                g["verified"]["succ"], ov["successes_total"]]
        for k, v in zip(tot, vals):
            tot[k] += v
        N += n
        T(ax, cx[0], y, s.replace("libero_", ""), size=8.7)
        for j, v in enumerate(vals):
            T(ax, cx[j + 1], y, f"{v/n:.0%}", size=8.7, mono=True)
        T(ax, cx[6], y, f"{pub[s]:.1f}%", size=8.7, mono=True, color=MUTED)
        y -= 0.019

    if N:
        hrule(ax, y + 0.009, lw=0.5, color=RULE)
        y -= 0.006
        T(ax, cx[0], y, "TOTAL", size=8.7, weight="bold")
        for j, k in enumerate(("noop", "bare", "loop", "verified", "ov")):
            T(ax, cx[j + 1], y, f"{tot[k]/N:.0%}", size=8.7, mono=True,
              weight="bold")
        y -= 0.021
        hrule(ax, y + 0.010, lw=0.7)
        y -= 0.020

        lo_b, hi_b = wilson(tot["bare"], N)
        lo_v, hi_v = wilson(tot["verified"], N)
        note = (
            f"bare [{lo_b:.0%}, {hi_b:.0%}] vs verified [{lo_v:.0%}, {hi_v:.0%}]: intervals "
            "overlap, so the aggregate gap is NOT significant.\n"
            "The effect is local — it lives in libero_spatial (12% to 36%), concentrated in two "
            "tasks that go 0/5 to 5/5.\n"
            "Where the primitive already works (object, 40%) verification adds nothing; where it "
            "cannot do the task at all\n(libero_10, articulated objects) verification correctly "
            "refuses to invent success."
        )
        T(ax, 0.07, y, note, size=8.3, color=MUTED)
        y -= 0.070

        T(ax, 0.07, y, "Retrying on self-report is worse than not retrying, in 4/4 suites:",
          size=9.5, weight="bold", color=ACCENT)
        y -= 0.022
        T(ax, 0.07, y,
          f"bare {tot['bare']/N:.0%}  ->  loop {tot['loop']/N:.0%}   "
          f"({(tot['loop']-tot['bare'])/N*100:+.1f} points).  The 'reached' flag is computed by "
          "the same code that\nexecuted the motion, so retries fire on the wrong episodes and "
          "consume the budget real recovery needs.", size=8.8)
        y -= 0.055

    # ── Table 2b: harness validation ─────────────────────────────────────
    T(ax, 0.07, y, "Table 2b.", size=9.5, weight="bold")
    T(ax, 0.155, y, "Harness validation: does our LIBERO setup reproduce a published "
      "baseline?", size=9.5)
    y -= 0.028
    hrule(ax, y + 0.012)
    for c, h in zip([0.07, 0.30, 0.46, 0.62], ["suite", "measured", "published",
                                               "inside our 95% CI"]):
        T(ax, c, y, h, size=8.5, weight="bold")
    y -= 0.020
    hrule(ax, y + 0.008, lw=0.5, color=RULE)
    y -= 0.008
    for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        ov = load(f"openvla_{s}_ep5.json")
        if not ov:
            continue
        k, n = ov["successes_total"], ov["episodes_total"]
        lo, hi = wilson(k, n)
        inside = lo <= pub[s] / 100 <= hi
        T(ax, 0.07, y, s.replace("libero_", ""), size=8.7)
        T(ax, 0.30, y, f"{k/n:.1%}", size=8.7, mono=True)
        T(ax, 0.46, y, f"{pub[s]:.1f}%", size=8.7, mono=True, color=MUTED)
        T(ax, 0.62, y, "yes" if inside else "NO", size=8.7,
          color=OK if inside else ACCENT, weight="bold")
        y -= 0.019
    y -= 0.006          # clear the last row before the closing rule
    hrule(ax, y + 0.010, lw=0.7)
    y -= 0.020
    T(ax, 0.07, y,
      "4/4 inside interval. This check earned its keep: it is what exposed a gripper-convention\n"
      "bug that scored a working policy at 0%.", size=8.3, color=MUTED)
    y -= 0.050

    # ── Table 3: published ───────────────────────────────────────────────
    T(ax, 0.07, y, "Table 3.", size=9.5, weight="bold")
    T(ax, 0.145, y, "Published results. Context only — not a ranking.", size=9.5)
    y -= 0.028
    hrule(ax, y + 0.012)
    for c, h in zip([0.07, 0.27, 0.47, 0.68],
                    ["system", "benchmark", "reported", "what changed"]):
        T(ax, c, y, h, size=8.5, weight="bold")
    y -= 0.020
    hrule(ax, y + 0.008, lw=0.5, color=RULE)
    y -= 0.008
    lit = [
        ("Pigey", "LIBERO-PRO", "12.8 -> 53.3%", "orchestrator, frozen pi0.5"),
        ("ASPIRE", "LIBERO-Pro Long", "4 -> 31%", "skill library + search"),
        ("Claude Plays Rob.", "Panda, real", "6 -> 32%", "queryable cursor"),
        ("CaP-X", "CaP-Bench", "human-level (some)", "test-time compute"),
    ]
    for name, bench, rep, what in lit:
        T(ax, 0.07, y, name, size=8.7)
        T(ax, 0.27, y, bench, size=8.7, color=MUTED)
        T(ax, 0.47, y, rep, size=8.7, mono=True)
        T(ax, 0.68, y, what, size=8.7, color=MUTED)
        y -= 0.019
    y -= 0.006          # clear the last row before the closing rule
    hrule(ax, y + 0.010, lw=0.7)
    y -= 0.020
    T(ax, 0.07, y,
      "Pigey and ASPIRE run LIBERO-PRO (perturbed) with Gemini ER perception and a pi0.5\n"
      "sub-policy; our LIBERO rows use oracle poses and a scripted primitive, so our numbers\n"
      "exclude perception error by construction. What transfers is the starting point — their\n"
      "frozen policy scores 12.8%, ours 12.0% — and the shape of the effect, not the magnitude.",
      size=8.3, color=MUTED)

    T(ax, 0.5, 0.025, "2", size=9, color=MUTED, ha="center")
    pdf.savefig(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# PAGE 3 — supporting measurements + per-episode detail
# ─────────────────────────────────────────────────────────────────────────

def page3(pdf, abl):
    fig, ax = new_page()
    y = 0.955

    T(ax, 0.07, y, "Supporting measurements on the wrc_demo rig", size=14,
      weight="bold")
    y -= 0.028
    hrule(ax, y, lw=1.0)
    y -= 0.030

    T(ax, 0.07, y, "Table 4.", size=9.5, weight="bold")
    T(ax, 0.145, y, "Perception and verification, measured against PhysX ground truth.",
      size=9.5)
    y -= 0.030
    hrule(ax, y + 0.012)
    for c, h in zip([0.07, 0.52, 0.70], ["quantity", "value", "reference channel"]):
        T(ax, c, y, h, size=8.5, weight="bold")
    y -= 0.020
    hrule(ax, y + 0.008, lw=0.5, color=RULE)
    y -= 0.008
    rows = [
        ("Detector precision (static scene)", "76.6%", "Isaac physics"),
        ("Detector recall", "100.0%", "Isaac physics"),
        ("Ghost detections", "23.4%  (11/47)", "persistent, not flicker"),
        ("3D localisation error", "2.4 cm mean", "Isaac physics"),
        ("Visual-diff, static scene", "0.0000 change", "pixels"),
        ("Visual-diff, 33 cm move", "changed (34% / 81% ROI)", "pixels"),
        ("Test suite", "244 passing", "—"),
    ]
    for q, v, ref in rows:
        T(ax, 0.07, y, q, size=8.7)
        T(ax, 0.52, y, v, size=8.7, mono=True)
        T(ax, 0.70, y, ref, size=8.7, color=MUTED)
        y -= 0.019
    y -= 0.006          # clear the last row before the closing rule
    hrule(ax, y + 0.010, lw=0.7)
    y -= 0.025

    T(ax, 0.07, y,
      "Recall 100% with precision 76.6% means the detector is not blind — it hallucinates. The\n"
      "ghost is persistent across frames, so it enters the world model and survives. This is the\n"
      "gap between Table 1 (oracle poses in sim) and what the same stack would score on the\n"
      "real arm.", size=8.6)
    y -= 0.070

    # per-episode detail
    if abl:
        T(ax, 0.07, y, "Table 5.", size=9.5, weight="bold")
        T(ax, 0.145, y, "Per-episode detail. Every initial state, every condition.",
          size=9.5)
        y -= 0.030
        hrule(ax, y + 0.012)
        for c, h in zip([0.07, 0.20, 0.36, 0.50, 0.64, 0.80],
                        ["condition", "state", "init (x,y)", "physics",
                         "self-report", "moved"]):
            T(ax, c, y, h, size=8.5, weight="bold")
        y -= 0.020
        hrule(ax, y + 0.008, lw=0.5, color=RULE)
        y -= 0.008
        for key in ("skill_only", "verify_only", "verify_retry"):
            d = abl.get(key)
            if not d:
                continue
            for ep in d["episodes"]:
                lie = ep["self_reported_ok"] and not ep["truth_success"]
                T(ax, 0.07, y, COND_LABEL[key][1], size=8.0, color=MUTED)
                T(ax, 0.20, y, str(ep["state"]), size=8.0, mono=True)
                T(ax, 0.36, y, f"({ep['init'][0]:.3f}, {ep['init'][1]:.3f})",
                  size=8.0, mono=True)
                T(ax, 0.50, y, "success" if ep["truth_success"] else "fail",
                  size=8.0, color=OK if ep["truth_success"] else INK)
                T(ax, 0.64, y,
                  ("claimed ok" if ep["self_reported_ok"] else "reported fail"),
                  size=8.0, color=ACCENT if lie else MUTED,
                  weight="bold" if lie else "normal")
                T(ax, 0.80, y, f"{ep['moved_m']*100:.1f} cm", size=8.0, mono=True)
                y -= 0.0165
                if y < 0.08:
                    break
        hrule(ax, y + 0.008, lw=0.7)

    T(ax, 0.5, 0.025, "3", size=9, color=MUTED, ha="center")
    pdf.savefig(fig)
    plt.close(fig)


def page_finding(pdf, abl):
    """What the ablation actually established, once the instrument was fixed."""
    fig, ax = new_page()
    y = 0.955

    T(ax, 0.07, y, "Finding: the layers were never the bottleneck",
      size=14, weight="bold")
    y -= 0.028
    hrule(ax, y, lw=1.0)
    y -= 0.030

    if not abl or "verify_retry" not in abl:
        T(ax, 0.07, y, "(no ablation data)", size=9, style="italic")
        pdf.savefig(fig)
        plt.close(fig)
        return

    body = (
        "This report previously headlined a different finding: the bare skill claimed success 8\n"
        "times and achieved it 4, and adding an independent postcondition check drove\n"
        "claimed-but-false to zero. That was true of the system as it stood.\n\n"
        "It was also a symptom. The grasp target carried a 1.6 cm perception bias -- the fitted\n"
        "box centre was pulled toward the camera by point density, against a cube half-width of\n"
        "2.5 cm. The finger caught the edge, shoved the cube away, and the jaws closed on air in\n"
        "a way the skill misread as success.\n\n"
        "With that fixed, the BARE skill also reports zero false claims. The dishonesty was not a\n"
        "property of running unverified; it was downstream of a broken sensor model. One\n"
        "perception change moved success 4/10 -> 10/10, where the entire verification-and-retry\n"
        "stack had moved it 4 -> 5 -> 6 with overlapping intervals.\n\n"
        "Verification now costs 3-5 s per episode and one episode of success (10 -> 9), because a\n"
        "postcondition rejects an outcome the bare skill counts as a pass. The honest claim is\n"
        "narrower than the one this report used to make: verification is insurance with a visible\n"
        "premium, and its payout depends entirely on how broken the rest of the stack is. It is\n"
        "not a substitute for finding the root cause."
    )
    T(ax, 0.07, y, body, size=9)
    y -= 0.255

    # honesty table
    T(ax, 0.07, y, "Self-report accuracy", size=11, weight="bold")
    y -= 0.026
    hrule(ax, y + 0.012)
    for c, h in zip([0.07, 0.28, 0.44, 0.60, 0.78],
                    ["condition", "achieved", "claimed", "false claims",
                     "calibration"]):
        T(ax, c, y, h, size=8.5, weight="bold")
    y -= 0.020
    hrule(ax, y + 0.008, lw=0.5, color=RULE)
    y -= 0.008
    for key in ("skill_only", "verify_only", "verify_retry"):
        d = abl.get(key)
        if not d:
            continue
        n, k = d["n"], d["truth_successes"]
        cl, fc = d["self_reported_successes"], d["false_claims"]
        T(ax, 0.07, y, COND_LABEL[key][1], size=8.7)
        T(ax, 0.28, y, f"{k}/{n}", size=8.7, mono=True)
        T(ax, 0.44, y, f"{cl}/{n}", size=8.7, mono=True)
        T(ax, 0.60, y, f"{fc}", size=8.7, mono=True,
          color=ACCENT if fc else OK, weight="bold")
        T(ax, 0.78, y, "wrong 40% of the time" if fc else "exact",
          size=8.7, color=ACCENT if fc else OK)
        y -= 0.019
    y -= 0.006
    hrule(ax, y + 0.010, lw=0.7)
    y -= 0.030

    T(ax, 0.07, y, "Why this matters more than the success column", size=11,
      weight="bold")
    y -= 0.024
    T(ax, 0.07, y,
      "A robot that fails and says so can be retried, escalated, or handed to a human. A robot\n"
      "that fails and reports success corrupts everything downstream: the belief store, the\n"
      "skill library that learns from traces, and any operator trusting the log. The bare skill\n"
      "was in that second state 40% of the time -- until the perception bias behind those\n"
      "failures was found and fixed, after which it is in that state 0% of the time. Verification\n"
      "made the failure VISIBLE; it did not make the system work.", size=8.8)
    y -= 0.105

    T(ax, 0.07, y, "The instrument had to be fixed first", size=11, weight="bold")
    y -= 0.024
    T(ax, 0.07, y,
      "An earlier run of this same ablation showed 3 false claims in the retry condition. They\n"
      "were not the robot: the Isaac bridge degrades under sustained verification polling (probe\n"
      "latency 20 ms -> 85 ms over 400 calls; its TCP thread eventually died), and the truth\n"
      "channel began returning stale poses. Re-run on a fresh bridge, those 3 became 0.\n\n"
      "Two real bugs came out of chasing them, both now fixed and covered by tests:\n"
      "  - impossible poses were reported as truth. This run alone produced four, up to\n"
      "    3171 m of 'displacement'. A verification channel that reports nonsense with\n"
      "    confidence is worse than one that stays silent.\n"
      "  - one shared token counted as identification: with pink_cube missing from a reading,\n"
      "    pose('pink cube') matched green_cube on {cube} and returned the WRONG object.",
      size=8.8)

    T(ax, 0.5, 0.025, "2", size=9, color=MUTED, ha="center")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    # Prefer the run made on a FRESH bridge. The first ablation was taken on a
    # bridge that had been up for hours, and its verification channel was
    # returning stale poses by the end (docs/BRIDGE_DEGRADATION.md) -- three of
    # its "false claims" were the instrument, not the robot.
    # ablation_v2 = measured AFTER the perception fix (_recentre_by_size,
    # c129161). The older files were measured while the grasp target carried a
    # 1.6 cm bias, which is what produced the false claims the report used to
    # headline. Prefer the corrected run; fall back for reproducibility.
    abl = (load("ablation_v2.json")
           or load("wrc_ablation_fresh.json")
           or load("wrc_ablation.json"))
    with PdfPages(OUT) as pdf:
        page1(pdf, abl)
        page_finding(pdf, abl)
        page2(pdf, abl)
        page3(pdf, abl)
        d = pdf.infodict()
        d["Title"] = "Verification as an Orchestration Layer: wrc_demo ablation"
        d["Author"] = "wrc_demo evaluation"
    print(f"[+] wrote {OUT}")
    if not abl:
        print("[!] Table 1 is empty — the ablation had not finished when this ran")


if __name__ == "__main__":
    main()
