# Native visitor turn budget

The native visitor and the normal pick acceptance proof share a 360-second
whole-turn deadline, followed by at most 30 seconds for the CLI to return.
`scripts/native_turn_budget.py` owns these constants. The turn includes model
inference, tool calls and the final response. It reserves 60 seconds for the host
around the existing 300-second MCP call limit; multiple tools still share one
turn. The proof's existing optional timeout scale remains local to
the proof. The attendee path does not read or apply that scale.

The MCP call limit remains 300 seconds. Search, motion, settling, physics,
mapper and safety limits are unchanged. An actual turn expiry still cancels
the in-flight tool, and MCP cancellation of motion still latches e-stop. A
missing CLI result, CLI watchdog expiry or an aborted agent envelope leaves
the visitor's durable uncertain-order latch in place. This change adds no
retry, recovery, automatic home command or stop reset.

## Host time is separate from tool time

On `28061a6`, final-MAIN-03 passed its two-object proof and the first four
campaign cases. The final tomato-can case reached confirmed placement and
retreated, but its return home was cancelled. The gateway recorded
`timeoutMs=299999`; the MCP server recorded cancellation during motion and
latched e-stop. The requested reset then refused without resetting props.
The aggregate campaign remains failed.
The [diagnostic receipt](../benchmark/results/spark-native-final-main03-timeout.json)
preserves allowlisted gateway and MCP lines, original snapshot hashes and the
tool/reset timeline without publishing full logs or authentication settings.

The tool started about 21.062 seconds after the visitor order began. Its trace
lasted 283.244 seconds including completion after cancellation: the whole-turn
deadline expired before the independent 300-second tool limit. Equal limits
left no allowance for inference, routing or the final response. The 360-second
turn adds a bounded host reserve; it does not extend any individual tool's
deadline or clear a stop. Successful physical campaign and restart acceptance
still require new evidence.

Deterministic tests cover a tool timeline plus host overhead beyond 300 seconds,
completion at 359 seconds and cancellation at 360 and 361 seconds. Expiry calls
the real MCP cancellation handler and safety latch with a nonphysical arm
double. The CLI watchdog remains separately bounded at 390 seconds. A test
executes the launcher's real MCP registration code and checks that the per-call
limit remains 300 seconds. These are software checks, not a physical replay.

## Earlier visitor/proof mismatch

On source `12534a4`, native07's two normal proof cases passed. The subsequent
five-object visitor campaign stopped on its first case because the visitor
hardcoded a shorter 240-second agent deadline. The green cube was grasped and
placed, with post-placement containment, release and settling confirmed. The
deadline cancelled the tool during its return home. The subsequent requested
reset was rejected because e-stop was latched and home had not been reached;
the campaign issued no second case.

The source-bound [diagnostic receipt](../benchmark/results/spark-native-five07-timeout.json)
contains only allowlisted gateway timeout and MCP cancellation lines, hashes
of the original log snapshots, and the tool timeline. It includes no complete
gateway log, authentication configuration or personal OpenClaw state. The
gateway records `timeoutMs=239999`; MCP records client cancellation followed
by its out-of-band emergency-stop latch.

The earlier regression modeled completed turns at 249 and 299 seconds and expiry
at 300 and 301 seconds. The expiry path calls the real MCP cancellation handler
and the real safety latch with a nonphysical arm double. A separate test keeps
the CLI watchdog at 330 seconds and checked the durable uncertain state. Those
are deterministic boundary tests, not a real-time OpenClaw or physical replay.
The existing stdio cancellation test checks JSON-RPC cancellation routing.

The correction is merged. Subsequent native tools exceeded 240 seconds without
the old premature cancellation; complete campaign and restart results remain
separate, source-bound stages in [project status](PROJECT_STATUS_20261001.md).
The original native07 proof passes and the failed campaign remain separate
evidence; this change does not upgrade the failed campaign to a pass.
