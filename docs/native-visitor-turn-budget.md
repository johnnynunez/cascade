# Native visitor turn budget

The native visitor and the normal pick acceptance proof share a 300-second
whole-turn deadline, followed by at most 30 seconds for the CLI to return.
`scripts/native_turn_budget.py` owns these constants. The turn includes model
inference, tool calls and the final response; it does not guarantee 300 seconds
for every tool. The proof's existing optional timeout scale remains local to
the proof. The attendee path does not read or apply that scale.

The MCP call limit remains 300 seconds. Search, motion, settling, physics,
mapper and safety limits are unchanged. An actual turn expiry still cancels
the in-flight tool, and MCP cancellation of motion still latches e-stop. A
missing CLI result, CLI watchdog expiry or an aborted agent envelope leaves
the visitor's durable uncertain-order latch in place. This change adds no
retry, recovery, automatic home command or stop reset.

## Why the paths must agree

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

The new regression models completed turns at 249 and 299 seconds and expiry
at 300 and 301 seconds. The expiry path calls the real MCP cancellation handler
and the real safety latch with a nonphysical arm double. A separate test keeps
the CLI watchdog at 330 seconds and checks the durable uncertain state. These
are deterministic boundary tests, not a real-time OpenClaw or physical replay.
The existing stdio cancellation test checks JSON-RPC cancellation routing.

The correction is merged. Subsequent native tools exceeded 240 seconds without
the old premature cancellation; complete campaign and restart results remain
separate, source-bound stages in [project status](PROJECT_STATUS_20261001.md).
The original native07 proof passes and the failed campaign remain separate
evidence; this change does not upgrade the failed campaign to a pass.
