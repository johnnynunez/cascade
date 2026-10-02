# MicroDuck native SIGTERM closure — 2026-10-02

The native replay closed its owned Kit process after SIGTERM without SIGKILL.
The child exited **143**; Isaac's `python.sh` translated that to launcher exit
**1**. The owned scope became inactive/dead. Source hashes and the protected
learned envelope remained unchanged. [Compact receipt and artifact hashes](evidence/robot-modularity/microduck-native-sigterm.json).

This replay used source `9240770`, containing the persistent signal fix
`401fad0` and inner SDK-call barriers `0607aa6`, plus an explicit rough-e/BAM
research recipe. It retained 807 consecutive solved records, 202 ONNX commits,
41 native force probes with zero reported conversion difference, and 41
hash-bound actual rendered frames. Those observations establish the replay
context; they do not admit locomotion.

The final persisted receipt explicitly says `before_sdk_shutdown`, signal 15,
and no teardown errors. `SimulationApp.close` exits the native process before
Python returns, so there is no returned-close or later teardown-file claim.
The independent supervisor's child-exit record and inactive owned scope supply
the process-closure evidence.

The supervisor recorded 801 completed rows through solve 803 before locating
and signalling the birth-bound owned PID. The final trace ends at solve 809.
The recording did not include the exact pidfd-send timestamp or per-solve wall
timestamps; those six additional solves cannot be classified as post-delivery
work. File modification timestamps are retained as such, not relabelled as
signal delivery times. Software regressions separately test that a consumed
signal prevents subsequent camera updates, model initialization and solver
submissions once the persistent checkpoint observes it.

The earlier campaign that required SIGKILL remains a failed lifecycle record.
This successful closure does not revise any failed motion or rest verdict.
