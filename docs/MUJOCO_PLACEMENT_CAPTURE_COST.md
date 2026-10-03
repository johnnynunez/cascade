# Placement observation cost checkpoint

The original Stage2 episode at07080fad failed: the red footprint crossed the
region's maximum x boundary by3.515mm. Withdrawal/home completed, the physics
verifier refuted the placement, and blue was never attempted. The process
closed naturally after174.61s; source/assets/protected stores were unchanged.
The unchanged original task's6cm criterion was not reached or relaxed. Saved
native proof SHA256:7da833ffe21f0443d571ef6a5ea36253ccdb26f332d63aac10cd1d14cea275c5.

An isolated, no-solve measurement serialized the same67,993,908-byte model five
times. All identities matched the episode; median hashing time was61.801ms.
The old capture repeated this complete fingerprint three times per batch.
Multiplying the isolated median by940 observed batches and three hashes gives
174.28s; this is a cost estimate, not an in-episode profiler attribution.

This change keeps one complete compiled-model fingerprint per capture under
the world lock. The two scratch FK reads share only that admission. Public
geometry and verification reads still perform their own full checks, and no
cache survives between calls. The model has one writer under the same lock;
there is no callback, live forward or model write between validation and use.
Clocks, raw forces, contacts, journal capacity and verifier predicates are
unchanged. This does not provide concurrent unsynchronized model mutation
support or relax the model-identity contract.

Six new CPU controls replay a saved row against the exact07080fa capture AST,
confirm identical journal/counters and three-to-one hash calls, preserve public
checks, and reject model mutation at the next admission. These use recorded
contact doubles and native scratch FK, without another solver step. Together
with region contracts:72passed in9.80s, inputs/stores unchanged. No new physical
episode or elapsed-time improvement has been measured on this candidate.
