# Isaac motion clock

Isaac arm trajectories and settling use authoritative simulation seconds. Wall
clock slowdowns must not compress a multi-second joint profile into a fraction
of a physics second. This change has offline regression coverage; physical
acceptance is still pending. It does not establish collision-free grasps.

The bridge reads q, dq and `physics_clock` together on the simulation thread.
Version 1 names the robot prim, engine, clock (`SimulationManager` for PhysX or
`newton_stage` for Newton), bridge-process epoch, simulation time, physics step
and configured physics timestep. The driver binds the endpoint locally, checks
wire joint dimensions before sign conversion, and the executor rejects missing,
nonfinite, regressive or inconsistent clocks and source/robot/epoch changes.
There is no wall-clock fallback. `IsaacArm.validate_simulation_clock()` provides
the same validation in one read-only call for diagnostics before scene reset.

Each nominal min-jerk waypoint is safety-approved with its interval measured in
physics seconds. Delayed samples authorize at most one waypoint; elapsed time
is discarded rather than caught up. After each command acknowledgement the
executor reads a fresh physical sample and waits a full interval from that
sample before issuing the next command. Transport delays therefore cannot
credit the next interval. These conservative intervals can make the profile
longer than its requested minimum physical duration.

Isaac defaults to `motion_rate_hz: 30` nominal actuator targets. An explicit
`stream_to(..., rate_hz=...)` takes precedence; omitted/None uses the profile.
Hardware keeps its existing 50 Hz default. The shared nominal profile retains
every exact legacy 50 Hz safety sample AND its original complete edge/dt,
adds each actual target, and checks each real command edge and the virtual
subedges with their own physical dt. Legacy edges intersecting a command's
interval are checked before entering them, even if they end just beyond its
target. This preserves refusals from tolerances that apply per whole edge.
These checks use the ordinary approval/escape rules and issue no virtual
actuator commands. Planned NV routes use the same union during initial planning
and feedback revalidation. This preserves sampled coverage, not a continuous
collision proof. Larger command steps change tracking dynamics and require
new physical acceptance; no duration, velocity limit or clearance is relaxed.

NV route preflight retains its existing bounded feedback rebind (1 mrad),
callback order and thread. The start callback runs before streaming. During
physics pauses, the ordinary approval callback revalidates the pending or last
nominal edge to check cancellation without issuing targets. Revalidating the
edge preserves existing escape rules for an arm initially at a boundary.

Settling requires finite exact-DOF q and dq, position inside the existing
configured target tolerance, and at least three fresh physical samples within
1 mrad of a fixed position anchor for 0.1 simulation seconds. Duplicate steps
cannot prove stability and an unsampled gap longer than the window resets the
anchor. Raw PhysX dq magnitude is not a settling threshold: retained-contact
measurements showed nonzero solver velocities with effectively fixed position.

`motion_wall_timeout_s` (120 seconds) independently bounds each stream/settle
operation; `motion_rpc_timeout_s` (1 second) caps each RPC to the remaining
budget. Transport lock, send and all response reads share one absolute deadline.
After timeout or malformed transport data the connection closes so a late reply
cannot be interpreted as the next request's response. Synchronous safety
callbacks stay on the caller thread and are checked against the deadline before
and after execution; their own implementations must remain bounded. A callback
that overruns cannot authorize a subsequent target. A timed-out command may
already have reached the bridge; reconnecting never retries it automatically.

Existing grasp durations, target tolerances, MCP budgets, geometry and GGX
selection are unchanged. At low real-time factors, correct physical pacing can
exceed the wall budget. Rendering performance and end-to-end launch budgets
require separate measured validation; increasing timeouts is not part of this
change.

The skill's `_wait_gripper_closed` still uses its existing 0.5-second wall-clock
stall window outside this executor. Successful joint settling does not establish
physical gripper closure or object retention.
