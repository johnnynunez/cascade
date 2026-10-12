| Run | Scenario | Robot | Verdict at run | Current verifier | Agent claim | Clearance m | LLM calls | Sim s | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| backstage1 | backstage_obstacles | duck_a | FAIL | FAIL | failure | 0.315 | 23 | 20.1 | first agentic run: plumbing check, agent too timid (no waypoint tools yet) |
| stage1 | stage_edge | duck_a | FAIL | PASS | failure | 0.094 | 12 | 20.4 | verifier changed AFTER this run (point -> edge band); old verdict kept |
| backstage3 | backstage_obstacles | duck_a | FAIL | FAIL | failure | 0.006 contact? | 26 | 49.3 | go_to_waypoint added; budget spent on bearing_to/scan, backtrack |
| stage2 | stage_edge | duck_a | PASS | PASS | failure | 0.247 | 11 | 19.5 | fresh run under the edge-band verifier |
| backstage4 | backstage_obstacles | duck_a | PASS | FAIL | failure | 0.009 contact? | 27 | 39.2 | reached, but body origin 9 mm from the FlightCase box: task says no contact (no_contact not yet in verifier) |
| multiroom1 | multi_room_fetch | duck_a | FAIL | FAIL | failure | 0.067 | 89 | 135.2 | no detector stand-in yet: agents cannot identify the boxes |
|  |  | duck_b | PASS | PASS | failure | 0.021 contact? |  |  |  |
|  |  | duck_c | FAIL | FAIL | failure | 0.191 |  |  |  |
| multiroom2 | multi_room_fetch | duck_a | FAIL | FAIL | success | 0.026 contact? | 102 | 196.5 | detector stand-in; centre-radius verifier at run time |
|  |  | duck_b | PASS | PASS | failure | 0.098 |  |  |  |
|  |  | duck_c | PASS | PASS | success | 0.166 |  |  |  |
| gym1 | creative_gym | duck_a | FAIL | FAIL | failure | - | 23 | 43.7 | extra scenario: onboard_hero fell on the ridge, get_up recovered, stuck tilted |
| backstage5 | backstage_obstacles | duck_a | FAIL | FAIL | failure | 0.044 | 41 | 68.1 | no_contact verifier from here on; budget exhausted |
| multiroom3 | multi_room_fetch | duck_a | FAIL | FAIL | failure | 0.257 | 45 | 64.8 | planner with 0.25 m inflation closed the 0.5 m doorway: agents gave up |
|  |  | duck_b | PASS | PASS | failure | 0.202 |  |  |  |
|  |  | duck_c | FAIL | PASS | success | 0.261 |  |  |  |
| backstage6 | backstage_obstacles | duck_a | FAIL | FAIL | failure | 0.085 | 41 | 62.4 | agent requested 0.25-0.30 m inflation: no path through 0.36-0.39 m gaps |
| multiroom4 | multi_room_fetch | duck_a | PASS | PASS | success | 0.066 | 63 | 149.7 | planner 0.18 m; centre-radius verifier at run time |
|  |  | duck_b | PASS | PASS | success | 0.068 |  |  |  |
|  |  | duck_c | FAIL | FAIL | success | 0.261 |  |  |  |
| backstage7 | backstage_obstacles | duck_a | FAIL | FAIL | failure | 0.137 | 27 | 49.0 | stopped by me mid-run (old planner superseded) |
| backstage8 | backstage_obstacles | duck_a | FAIL | FAIL | failure | 0.136 | 11 | 23.4 | gave up after two no-path plans at 0.30/0.25 m (before automatic retry) |
| multiroom5 | multi_room_fetch | duck_a | PASS | PASS | success | 0.114 | 83 | 190.7 | centre-radius verifier at run time; 3/3 under the corrected base-to-surface verifier |
|  |  | duck_b | PASS | PASS | success | 0.000 contact? |  |  |  |
|  |  | duck_c | FAIL | PASS | success | 0.261 |  |  |  |
| backstage9 | backstage_obstacles | duck_a | PASS | PASS | failure | 0.056 | 21 | 51.0 | planner auto-retry + base-to-surface prompt |
| backstage10 | backstage_obstacles | duck_a | FAIL | FAIL | failure | 0.000 contact? | 25 | 76.1 | parallel attempt with the same code as backstage9 |
| multiroom6 | multi_room_fetch | duck_a | PASS | PASS | success | 0.011 contact? | 70 | 178.3 | fresh run under the near_object (base-to-surface) verifier |
|  |  | duck_b | PASS | PASS | success | 0.000 contact? |  |  |  |
|  |  | duck_c | PASS | PASS | success | 0.252 |  |  |  |
