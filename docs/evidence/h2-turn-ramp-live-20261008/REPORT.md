# B29 — live A/B of the revision-3 turn goal ramp on the H2 owner (8 October 2026)

The owner episode the revision-3 section of [`docs/HUMANOID_H2.md`](../../HUMANOID_H2.md) left open. Diagnostic,
not physical admission. Measured by Hermes on the x86 test rig, GPU 0 (never a Spark number). Raw rows stay in
`cascade-lab/`; [`manifest.json`](manifest.json) lists path, bytes and SHA-256 of all of them. `turns.json` (every
turn), `yaw_rate_profiles.json` (measured yaw rate per 0.25 s + every ramp update), `summary.json` and the harness
(`harness/`) are copied here; the tables below are generated from them (`harness/make_live_evidence.py`,
`harness/make_live_report.py`).

## Setup

- Source: this PR's head `60fe7f1` (on `origin/main` c5012e7). Owner `scripts/isaac_h2_bridge.py`, PhysX, the internal
  Isaac Sim 6.2 build, pinned to GPU 0 by UUID, with **`--velocity-scaling`**; `hello.capabilities` =
  `['state', 'velocity', 'stop', 'reset_stop', 'velocity_scaling']` (both runs).
- Private base pinned to each owner's identity (`bases/h2_episode.yaml` → `extends: h2_velocity_candidate`, so it
  inherits `goal_ramp: {decel_rad_s2: 0.4, min_rate_rad_s: 0.15, rate_step_rad_s: 0.025}`). The ramp-off arm is the
  same file plus `turn_control: {goal_ramp: null}`; before any motion the harness checked that the two resolved
  bases differ **only** in `turn_control.goal_ramp` (checked in both runs, `run_meta.json` → `arms_check`).
- Verifier and SafeBase limits unchanged: rotation tolerance 0.05 rad, settle heading drift 0.05 rad, stop angular
  speed 0.20 rad/s, settle window 0.4 s, `max_lateral_drift_m` 0.20 m, the 3 s command.
- Design: run 1 = ramp / no-ramp / ramp blocks (±0.8, ±1.0 rad; ±0.8) + 2 MCP turns (`python -m cascade.apps.mcp_server`,
  `CASCADE_ROBOT=h2`, ramp on); run 2 = ABAB blocks (±0.8, ±1.0, ±0.6 rad each) + 4 MCP turns, with `reset_stop`
  before any turn whose predecessor ended latched. Excluded as not independent: run 1 `R1-4--1.0`
  (refused by the previous turn's latched stop). **39 independent turns.**
- Real-time factor per block 0.454–0.478; neither owner receipt shows a fall.

## Result

| arm | angle (rad) | n | confirmed | refuted | timeout (3 s command) | settle heading drift (rad, limit 0.05) |
| --- | --- | --- | --- | --- | --- | --- |
| ramp on | ±0.6 | 4 | **3** | 0 | 1 | 0.040, 0.022, —, 0.037 |
| ramp on | ±0.8 | 8 | **8** | 0 | 0 | 0.034, 0.029, 0.046, 0.050, 0.032, 0.030, 0.044, 0.028 |
| ramp on | ±1.0 | 5 | **1** | 1 | 3 | —, —, 0.029, —, 0.044 |
| ramp off | ±0.6 | 4 | **3** | 1 | 0 | 0.025, 0.035, 0.031, 0.077 |
| ramp off | ±0.8 | 6 | **1** | 5 | 0 | 0.067, 0.073, 0.075, 0.023, 0.084, 0.053 |
| ramp off | ±1.0 | 6 | **5** | 1 | 0 | 0.046, 0.045, 0.047, 0.032, 0.033, 0.044 |
| MCP, ramp on | ±0.8 | 6 | **4** | 2 | 0 | 0.081, 0.040, 0.025, 0.038, 0.045, 0.069 |

- **0.8 rad, the case revision 3 was built for:** ramp on **8/8** confirmed on the runtime path versus **1/6** with the
  ramp off (all five refutations are settle heading drift 0.053–0.084 rad > 0.05). Through MCP (ramp on) 4/6.
- **1.0 rad:** ramp on **1/5** (three timeouts at the unchanged 3 s command, one refuted) versus **5/6** ramp off.
- **0.6 rad:** 3/4 in both arms.

## Mechanism (from the verifier's own samples)

1. **Settle.** Every "did not settle" refutation is heading drift in the 0.4 s settle window above 0.05 rad (with
   ω above 0.20 rad/s in some). With the ramp the yaw rate at the goal is lower (the commanded rate is at the
   0.15 rad/s floor at every ramped stop) and the 0.8 rad drifts stay 0.028–0.050 rad, one at 0.0499: the margin is thin.
2. **Timeouts (all four ramp-on, all left turns):**
   - run 1 `R1-3-+1.0`: dip to 0.12 rad/s at 1.25–1.50 s while commanded 0.50 rad/s; yaw within 0.05 rad of the target at 2.93 s
   - run 2 `R1-3-+1.0`: dip to 0.09 rad/s at 1.50–1.75 s while commanded 0.50 rad/s; yaw within 0.05 rad of the target never within the command
   - run 2 `R2-3-+1.0`: dip to 0.08 rad/s at 1.50–1.75 s while commanded 0.50 rad/s; yaw within 0.05 rad of the target never within the command
   - run 2 `R2-5-+0.6`: dip to 0.03 rad/s at 1.75–2.00 s while commanded 0.29 rad/s; yaw within 0.05 rad of the target never within the command
   (Times here are from the verifier's samples, measured from its first sample; SafeBase measures from its first
   post-admission state with its own `turn_tolerance_rad` and had not reached it when the 3 s command ended.)
   The three +1.0 rad turns dipped at full command, recovered, then ran out of the 3 s command during the ramp's
   deceleration; without the ramp all six 1.0 rad turns came within 0.05 rad of the target at
   2.21–2.43 s. The +0.6 rad timeout dipped after the ramp
   had lowered the command to 0.29 rad/s and stayed at 0.02–0.06 rad/s until the deadline.
3. **A mid-turn yaw-rate dip is a policy property, present in both arms.** 24 of the 39 turns have a
   0.25 s window below 0.2 rad/s between 0.75 and 2.0 s that ends ≥ 0.25 s before the goal. With the command still
   ≥ 0.45 rad/s, 17 of 20 such turns reached the goal within the command (the misses are the three
   ramp-on +1.0 timeouts); with the ramp already lowering the command (0.26–0.37 rad/s), 3 of 4 did.
4. **Translation at 1.0 rad, both arms.** The verifier's path length over a 1.0 rad turn is
   0.166–0.227 m against `max_lateral_drift_m` 0.20 m; the two −1.0 rad refutations (one per arm) are
   "unrequested translation during turn" at 0.225 and 0.227 m.

## Verdict

The ramp removes the 0.8 rad settle refutation on the runtime path and costs the 1.0 rad turn its time budget. It
is not admitted and the candidate stays a candidate; no limit, budget or ramp value was changed after the
measurement. Candidate next steps, each to be designed and measured on its own: a ramp that accounts for the
command time left (no deceleration the 3 s command cannot hold) or holds the commanded rate while the measured
yaw rate is in a dip; and the 1.0 rad translation margin, which is a gait property independent of the ramp.

## Every turn

| run | turn | arm | status | reason / error | within 0.05 rad at (s, verifier samples) | rate at stop | settle drift (rad) | settle max ω (rad/s) | verifier path (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | M-1-+0.8 | MCP, ramp on | refuted | did not settle: active controller, residual velocity or pose | 2.19 | 0.15 | 0.081 | 0.248 | 0.165 |
| 1 | M-2--0.8 | MCP, ramp on | confirmed | independent measured yaw matches requested angle | 2.15 | 0.15 | 0.040 | 0.102 | 0.169 |
| 1 | N1-1-+0.8 | ramp off | refuted | did not settle: active controller, residual velocity or pose | 1.90 | — | 0.067 | 0.187 | 0.142 |
| 1 | N1-2--0.8 | ramp off | refuted | did not settle: active controller, residual velocity or pose | 1.83 | — | 0.073 | 0.245 | 0.161 |
| 1 | N1-3-+1.0 | ramp off | confirmed | independent measured yaw matches requested angle | 2.23 | — | 0.046 | 0.096 | 0.188 |
| 1 | N1-4--1.0 | ramp off | confirmed | independent measured yaw matches requested angle | 2.21 | — | 0.045 | 0.136 | 0.194 |
| 1 | R1-1-+0.8 | ramp on | confirmed | independent measured yaw matches requested angle | 1.92 | 0.15 | 0.034 | 0.032 | 0.159 |
| 1 | R1-2--0.8 | ramp on | confirmed | independent measured yaw matches requested angle | 2.54 | 0.15 | 0.029 | 0.068 | 0.173 |
| 1 | R1-3-+1.0 | ramp on | unverified | measured yaw did not reach target before simulation deadline | 2.93 | — | — | — | — |
| 1 | R1-4--1.0 | ramp on | unverified (excluded) | stop latched or control operation active | — | — | — | — | — |
| 1 | R2-1-+0.8 | ramp on | confirmed | independent measured yaw matches requested angle | 2.25 | 0.15 | 0.046 | 0.080 | 0.168 |
| 1 | R2-2--0.8 | ramp on | confirmed | independent measured yaw matches requested angle | 2.06 | 0.15 | 0.050 | 0.123 | 0.166 |
| 2 | M-1-+0.8 | MCP, ramp on | confirmed | independent measured yaw matches requested angle | 2.46 | 0.15 | 0.025 | 0.044 | 0.162 |
| 2 | M-2--0.8 | MCP, ramp on | confirmed | independent measured yaw matches requested angle | 2.50 | 0.15 | 0.038 | 0.026 | 0.175 |
| 2 | M-3-+0.8 | MCP, ramp on | confirmed | independent measured yaw matches requested angle | 2.29 | 0.15 | 0.045 | 0.146 | 0.151 |
| 2 | M-4--0.8 | MCP, ramp on | refuted | did not settle: active controller, residual velocity or pose | 2.09 | 0.15 | 0.069 | 0.218 | 0.166 |
| 2 | N1-1-+0.8 | ramp off | refuted | did not settle: active controller, residual velocity or pose | 1.96 | — | 0.075 | 0.212 | 0.143 |
| 2 | N1-2--0.8 | ramp off | confirmed | independent measured yaw matches requested angle | 1.98 | — | 0.023 | 0.044 | 0.146 |
| 2 | N1-3-+1.0 | ramp off | confirmed | independent measured yaw matches requested angle | 2.23 | — | 0.047 | 0.105 | 0.177 |
| 2 | N1-4--1.0 | ramp off | confirmed | independent measured yaw matches requested angle | 2.43 | — | 0.032 | 0.064 | 0.166 |
| 2 | N1-5-+0.6 | ramp off | confirmed | independent measured yaw matches requested angle | 1.58 | — | 0.025 | 0.054 | 0.099 |
| 2 | N1-6--0.6 | ramp off | confirmed | independent measured yaw matches requested angle | 2.15 | — | 0.035 | 0.088 | 0.150 |
| 2 | N2-1-+0.8 | ramp off | refuted | did not settle: active controller, residual velocity or pose | 2.48 | — | 0.084 | 0.278 | 0.169 |
| 2 | N2-2--0.8 | ramp off | refuted | did not settle: active controller, residual velocity or pose | 1.86 | — | 0.053 | 0.148 | 0.158 |
| 2 | N2-3-+1.0 | ramp off | confirmed | independent measured yaw matches requested angle | 2.23 | — | 0.033 | 0.089 | 0.187 |
| 2 | N2-4--1.0 | ramp off | refuted | unrequested translation during turn | 2.33 | — | 0.044 | 0.109 | 0.225 |
| 2 | N2-5-+0.6 | ramp off | confirmed | independent measured yaw matches requested angle | 1.55 | — | 0.031 | 0.036 | 0.078 |
| 2 | N2-6--0.6 | ramp off | refuted | did not settle: active controller, residual velocity or pose | 1.43 | — | 0.077 | 0.240 | 0.100 |
| 2 | R1-1-+0.8 | ramp on | confirmed | independent measured yaw matches requested angle | 1.88 | 0.15 | 0.032 | 0.036 | 0.152 |
| 2 | R1-2--0.8 | ramp on | confirmed | independent measured yaw matches requested angle | 2.54 | 0.15 | 0.030 | 0.034 | 0.190 |
| 2 | R1-3-+1.0 | ramp on | unverified | measured yaw did not reach target before simulation deadline | — | — | — | — | — |
| 2 | R1-4--1.0 | ramp on | refuted | unrequested translation during turn | 2.90 | 0.15 | 0.029 | 0.042 | 0.227 |
| 2 | R1-5-+0.6 | ramp on | confirmed | independent measured yaw matches requested angle | 1.81 | 0.15 | 0.040 | 0.115 | 0.121 |
| 2 | R1-6--0.6 | ramp on | confirmed | independent measured yaw matches requested angle | 1.81 | 0.15 | 0.022 | 0.040 | 0.126 |
| 2 | R2-1-+0.8 | ramp on | confirmed | independent measured yaw matches requested angle | 2.19 | 0.15 | 0.044 | 0.120 | 0.182 |
| 2 | R2-2--0.8 | ramp on | confirmed | independent measured yaw matches requested angle | 2.52 | 0.15 | 0.028 | 0.027 | 0.171 |
| 2 | R2-3-+1.0 | ramp on | unverified | measured yaw did not reach target before simulation deadline | — | — | — | — | — |
| 2 | R2-4--1.0 | ramp on | confirmed | independent measured yaw matches requested angle | 2.40 | 0.15 | 0.044 | 0.127 | 0.186 |
| 2 | R2-5-+0.6 | ramp on | unverified | measured yaw did not reach target before simulation deadline | — | — | — | — | — |
| 2 | R2-6--0.6 | ramp on | confirmed | independent measured yaw matches requested angle | 2.19 | 0.15 | 0.037 | 0.026 | 0.141 |

## Not claimed

No physical admission, no hardware, no Newton run, no Spark number. Cell sizes are 2–8 turns: the 0.8 rad
contrast (8/8 vs 1/6) is large, the others are not established beyond these counts. The left/right asymmetry of
the timeouts (four of four left turns) is observed, not explained.
