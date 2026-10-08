"""Write docs/evidence/h2-turn-ramp-live-20261008/REPORT.md from turns.json / summary.json / run_meta.json."""
import json
from pathlib import Path

EV = Path('/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/h2-turn-ramp/cascade/docs/evidence/h2-turn-ramp-live-20261008')
rows = json.loads((EV / 'turns.json').read_text())
summ = json.loads((EV / 'summary.json').read_text())
meta = json.loads((EV / 'run_meta.json').read_text())
ind = [r for r in rows if r['independent']]


def fmt(v, nd=3):
    return '—' if v is None else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


order = ['ramp on', 'ramp off', 'MCP, ramp on']
cell_lines = ['| arm | angle (rad) | n | confirmed | refuted | timeout (3 s command) | settle heading drift (rad, limit 0.05) |',
              '| --- | --- | --- | --- | --- | --- | --- |']
for arm in order:
    for ang in (0.6, 0.8, 1.0):
        c = summ['cells'].get(f'{arm} {ang:.1f}')
        if not c:
            continue
        drift = ', '.join(fmt(x) for x in c['settle_drift_rad'])
        cell_lines.append(f"| {arm} | ±{ang:.1f} | {c['n']} | **{c['confirmed']}** | {c['refuted']} | {c['timeout']} | {drift} |")

turn_lines = ['| run | turn | arm | status | reason / error | within 0.05 rad at (s, verifier samples) | rate at stop | settle drift (rad) | settle max ω (rad/s) | verifier path (m) |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
for r in rows:
    why = r['error'] or r['reason'] or ''
    turn_lines.append(f"| {r['run']} | {r['tag']} | {r['arm']} | {r['status']}{'' if r['independent'] else ' (excluded)'} | {why[:60]} | "
                      f"{fmt(r['goal_reached_after_s'], 2)} | {fmt(r['commanded_rate_at_stop_rad_s'], 2)} | {fmt(r['settle_drift_rad'])} | "
                      f"{fmt(r['settle_max_angular_speed_rad_s'])} | {fmt(r['verifier_path_length_m'])} |")

tmo = [r for r in ind if r['error'] and 'deadline' in r['error']]
tmo_lines = [f"- run {r['run']} `{r['tag']}`: dip to {fmt(r['stall_yaw_rate_rad_s'], 2)} rad/s at {r['stall_window_s']:.2f}–{r['stall_window_s'] + 0.25:.2f} s "
             f"while commanded {fmt(r['commanded_rate_during_stall_rad_s'], 2)} rad/s; yaw within 0.05 rad of the target {('at ' + fmt(r['goal_reached_after_s'], 2) + ' s') if r['goal_reached_after_s'] is not None else 'never within the command'}"
             for r in tmo]
st = summ['stall_then_goal_in_command']
full = st.get('commanded >= 0.45', {'stalled_turns': 0, 'goal_reached_in_command': 0})
low = st.get('commanded < 0.45 (ramp already lowering)', {'stalled_turns': 0, 'goal_reached_in_command': 0})
one = [r for r in ind if abs(r['angle_rad']) == 1.0 and r['verifier_path_length_m'] is not None]
noramp_one = [r for r in ind if abs(r['angle_rad']) == 1.0 and r['arm'] == 'ramp off']
m1, m2 = meta['b29-ramp-live1'], meta['b29-ramp-live2']
rts = sorted(v for m in (m1, m2) for v in (m['rt_factor'] or {}).values())

text = f"""# B29 — live A/B of the revision-3 turn goal ramp on the H2 owner (8 October 2026)

The owner episode the revision-3 section of [`docs/HUMANOID_H2.md`](../../HUMANOID_H2.md) left open. Diagnostic,
not physical admission. Measured by Hermes on the x86 test rig, GPU 0 (never a Spark number). Raw rows stay in
`cascade-lab/`; [`manifest.json`](manifest.json) lists path, bytes and SHA-256 of all of them. `turns.json` (every
turn), `yaw_rate_profiles.json` (measured yaw rate per 0.25 s + every ramp update), `summary.json` and the harness
(`harness/`) are copied here; the tables below are generated from them (`harness/make_live_evidence.py`,
`harness/make_live_report.py`).

## Setup

- Source: this PR's head `60fe7f1` (on `origin/main` c5012e7). Owner `scripts/isaac_h2_bridge.py`, PhysX, the internal
  Isaac Sim 6.2 build, pinned to GPU 0 by UUID, with **`--velocity-scaling`**; `hello.capabilities` =
  `{m1['hello_capabilities']}` (both runs).
- Private base pinned to each owner's identity (`bases/h2_episode.yaml` → `extends: h2_velocity_candidate`, so it
  inherits `goal_ramp: {{decel_rad_s2: 0.4, min_rate_rad_s: 0.15, rate_step_rad_s: 0.025}}`). The ramp-off arm is the
  same file plus `turn_control: {{goal_ramp: null}}`; before any motion the harness checked that the two resolved
  bases differ **only** in `turn_control.goal_ramp` (checked in both runs, `run_meta.json` → `arms_check`).
- Verifier and SafeBase limits unchanged: rotation tolerance 0.05 rad, settle heading drift 0.05 rad, stop angular
  speed 0.20 rad/s, settle window 0.4 s, `max_lateral_drift_m` 0.20 m, the 3 s command.
- Design: run 1 = ramp / no-ramp / ramp blocks (±0.8, ±1.0 rad; ±0.8) + 2 MCP turns (`python -m cascade.apps.mcp_server`,
  `CASCADE_ROBOT=h2`, ramp on); run 2 = ABAB blocks (±0.8, ±1.0, ±0.6 rad each) + 4 MCP turns, with `reset_stop`
  before any turn whose predecessor ended latched. Excluded as not independent: {', '.join('run ' + e.replace(':', ' `') + '`' for e in summ['excluded'])}
  (refused by the previous turn's latched stop). **{len(ind)} independent turns.**
- Real-time factor per block {rts[0]:.3f}–{rts[-1]:.3f}; neither owner receipt shows a fall.

## Result

{chr(10).join(cell_lines)}

- **0.8 rad, the case revision 3 was built for:** ramp on **8/8** confirmed on the runtime path versus **1/6** with the
  ramp off (all five refutations are settle heading drift 0.053–0.084 rad > 0.05). Through MCP (ramp on) 4/6.
- **1.0 rad:** ramp on **1/5** (three timeouts at the unchanged 3 s command, one refuted) versus **5/6** ramp off.
- **0.6 rad:** 3/4 in both arms.

## Mechanism (from the verifier's own samples)

1. **Settle.** Every "did not settle" refutation is heading drift in the 0.4 s settle window above 0.05 rad (with
   ω above 0.20 rad/s in some). With the ramp the yaw rate at the goal is lower (the commanded rate is at the
   0.15 rad/s floor at every ramped stop) and the 0.8 rad drifts stay 0.028–0.050 rad, one at 0.0499: the margin is thin.
2. **Timeouts (all four ramp-on, all left turns):**
{chr(10).join('   ' + t for t in tmo_lines)}
   (Times here are from the verifier's samples, measured from its first sample; SafeBase measures from its first
   post-admission state with its own `turn_tolerance_rad` and had not reached it when the 3 s command ended.)
   The three +1.0 rad turns dipped at full command, recovered, then ran out of the 3 s command during the ramp's
   deceleration; without the ramp all six 1.0 rad turns came within 0.05 rad of the target at
   {min(r['goal_reached_after_s'] for r in noramp_one):.2f}–{max(r['goal_reached_after_s'] for r in noramp_one):.2f} s. The +0.6 rad timeout dipped after the ramp
   had lowered the command to 0.29 rad/s and stayed at 0.02–0.06 rad/s until the deadline.
3. **A mid-turn yaw-rate dip is a policy property, present in both arms.** {summ['stall_count']} of the {len(ind)} turns have a
   0.25 s window below 0.2 rad/s between 0.75 and 2.0 s that ends ≥ 0.25 s before the goal. With the command still
   ≥ 0.45 rad/s, {full['goal_reached_in_command']} of {full['stalled_turns']} such turns reached the goal within the command (the misses are the three
   ramp-on +1.0 timeouts); with the ramp already lowering the command (0.26–0.37 rad/s), {low['goal_reached_in_command']} of {low['stalled_turns']} did.
4. **Translation at 1.0 rad, both arms.** The verifier's path length over a 1.0 rad turn is
   {min(r['verifier_path_length_m'] for r in one):.3f}–{max(r['verifier_path_length_m'] for r in one):.3f} m against `max_lateral_drift_m` 0.20 m; the two −1.0 rad refutations (one per arm) are
   "unrequested translation during turn" at 0.225 and 0.227 m.

## Verdict

The ramp removes the 0.8 rad settle refutation on the runtime path and costs the 1.0 rad turn its time budget. It
is not admitted and the candidate stays a candidate; no limit, budget or ramp value was changed after the
measurement. Candidate next steps, each to be designed and measured on its own: a ramp that accounts for the
command time left (no deceleration the 3 s command cannot hold) or holds the commanded rate while the measured
yaw rate is in a dip; and the 1.0 rad translation margin, which is a gait property independent of the ramp.

## Every turn

{chr(10).join(turn_lines)}

## Not claimed

No physical admission, no hardware, no Newton run, no Spark number. Cell sizes are 2–8 turns: the 0.8 rad
contrast (8/8 vs 1/6) is large, the others are not established beyond these counts. The left/right asymmetry of
the timeouts (four of four left turns) is observed, not explained.
"""
(EV / 'REPORT.md').write_text(text)
print(text[:600])
print('...', len(text), 'chars')
