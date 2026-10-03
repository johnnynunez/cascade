# Ground post-bootstrap guard — 2026-10-03

Separate follow-up from frozen `106608f8d71754dfa5da90ecda6efa77df107e12`.
Root identified an appearance gap between initial authoring and completion of
the parent's camera/bootstrap sequence. The earlier checkpoint is preserved.

The benchmark now retains the authored Ground/material snapshot, checks it
after camera creation and again after bootstrap, and writes separate observed
and authored hashes. Complete native invariance is unchanged. This does not
alter the SDK, physics owner, render cadence or verification budgets.

External evidence: sibling `RGBD_GROUND_POSTBOOTSTRAP/causal-{before,after}`.
One identical CPU probe uses actual OpenUSD composition with a software parent
that mutates shader, ST, descriptor, Ground width, ancestor transform or an ST
time sample during camera creation or after export. The old entrypoint returned
for all twelve mutations (12 failing controls, 2 clean controls passing). The
candidate rejects all twelve and preserves both clean cases (14 controls pass).
No Kit, solver, GPU or native episode was used. Source/protected hashes are saved
before and after each probe. These are causal lifecycle/appearance controls,
not proof of actual RTX texture behavior.
