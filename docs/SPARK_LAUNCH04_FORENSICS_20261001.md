# Spark launch04: green task and cameras pass; orange grasp fails

The normal desktop launch at `a19b65c59f13d468d990e1dc49d626c143674cee`
used the reviewed 30 Hz target executor, preserved legacy safety sampling,
disabled headless viewport updates and explicit camera cadence six at 1280×720.
Memory and the existing motion, settling, camera-age and tool budgets were
retained. This launch ended **NOT READY**.

The green cube passed the complete native and saved physical case: pickup,
bilateral-contact lift, placement, release, return home, reset and all three
cameras. Its native action took 263.120 seconds and reset took 29.119 seconds.
The placement verdict was confirmed through physics, including full collider
containment and actual jaw release. Unlike launch03, camera freshness passed.
Changing the camera cadence and selected grasp prevents attributing this
wall duration to the executor alone.

Orange then failed its only grasp attempt: the jaws closed without retaining
it. The grasp telemetry took 125.272 seconds; the complete failed action,
including its existing recovery handling, took 148.145 seconds. No second
attempt was executed. The launch aborted before a completed orange case or
its physical reset, and normal owned-service cleanup finished. The green
result does not establish five-object or restart acceptance.

The original orange postcondition reports a destination error against the
box grouping prim at `[0, 0]`, rather than the configured cavity center
`[0.3, -0.14]`. That distance is not valid destination evidence. It is retained
unchanged; the independent grasp failure still rejects the action.

The [retention receipt](../benchmark/results/spark_native_launch04_forensics_20261001.json)
binds 85 saved files to the exact owner, MCP process, trace and original proof.
The strict offline gate accepts the green case and rejects the incomplete
launch. The archive includes the two grasp attempts, physical witnesses,
native envelopes, memory and preparation records; no OpenClaw private
configuration or session archive is included. No subsequent attempt is implied.
