# Factory process boundary: CPU transport candidate

`sim/factory_process_wire.py` and `sim/factory_process_protocol.py` provide the
transport used by the private CPU adapter described below. Factory launchers
and default composition still use their existing owner, journal and raw queue.
No native profile selects this candidate, and its software tests do not admit a
physical solve or transfer an earlier model identity.

The wire codec accepts exact primitive types with explicit byte, node, depth
and text limits. It preserves mapping order, tuple/list types, signed zero and
finite float64 bits. It does not load pickle, import a named class or execute an
object reducer. Frames have a version, kind, monotonically increasing sequence,
payload length and digest. Lengths are checked before payload allocation. Every
send, receive, lock wait and decode consumes the caller's original absolute
monotonic deadline. A partial, malformed or late exchange makes that channel
unusable. Separate inherited Unix socketpairs provide bulk, admission and
priority paths; they do not share a send lock or feeder queue.

`RawOutbox` retains bytes and preallocated scalar outcome/acknowledgment slots.
`transact` commits RAW before calling acceptance once, then records whether the
call returned or raised. An acceptance outcome records that historical fact;
it does not bypass a subsequent fault. A slot cannot be reused until archive,
outcome and reader consumers have all acknowledged it. A captured prefix ends
at the rows present when it was requested. Full byte/count capacity and missing
outcome at EOF are faults, never overwrites or implicit acceptance.

`FencedDecoder` obtains nonce-bound child status before and after decoding. The
child serializes its status reply with sticky fault publication. The second
reply is the child-side delivery linearization point: a later child fault is
observed by the next read. A separate host fault lock prevents a sibling reader
whose lock wait timed out from allowing an already active reader to return.
The final local delivery check uses that same lock. Original freshness checks
run after decoding and again after the second reply; capture times are never
renewed. Here `boot_id` means a fresh 128-bit process-incarnation nonce encoded
as 32 hex digits, not the Linux boot UUID or a replacement for owned PID birth
verification.

CPU tests use fresh exec children, socketpairs and a synthetic backend with the
actual fastening stop guard. They cover blocked bulk with a separate stop ACK,
remote fault during host decode, a controlled child stall rejected by the
existing 200 ms observation gate, malformed/partial frames, concurrent local
reader failure, RAW/outcome ordering, wrap and acknowledgment leases. They
leave GC enabled with its original thresholds and establish no physics or
native latency result.

`factory_process_snapshot.py` defines the complete current private readback
schema with a model-bound layout. It encodes full contact columns once and
reconstructs known solve, upload and shoulder-witness records. Raw float32
channels, metadata order and disabled-buffer markers survive unchanged. The
schema has no age or task-success gate: a well-formed row that the controller
later rejects remains archivable. For the retained 2048-capacity model its
conservative payload bound is 440294 bytes, including a shoulder witness for
every candidate. CPU tests cover full capacity and physical vetoes after
decoding; a bounded external sample also round-trips the first and final
rejected records of the preserved native episode.

`factory_process_history.py` supplies the private CPU adapter. `_LatestJournal`
retains only the child's current row and captured continuity metadata, while
the unchanged controller performs every physical check. Cursor reads on that
private journal explicitly refuse. `_AcceptedHistory` keeps the host's finite
recent-history ring. Archive, ring and reader references share one payload
charge in `_ByteLeases`; evicting a ring reference does not free bytes still
pinned by a reader or archive. Explicit release is required, including on
errors and closed batches; no destructor or cyclic-GC timing returns capacity.
One store also caps live leases at 98304 and batches at 8, including empty
snapshots and active iterator references. Exhaustion is sticky even after
cleanup. Partial snapshots return every acquired lease and their batch slot;
closing a paused batch drops its whole reference tuple, retaining only the
single charged active payload. These metadata counts are separate from the
once-per-payload byte charge and must be included in the 32 MiB bookkeeping
budget before native admission.
The adapter routes startup/task/rest cursor reads to this host history and
requires the actual acceptance outcome before publication.

Freshness follows each existing call site. A current observation or newly
received cursor batch retains the original observation checks and deadline.
Analysis of previously admitted ThreadHistory is historical evidence: it does
not gain a newly fabricated capture time, and it does not acquire a new
per-sample `check_solve(now)` gate that the original analysis never had. That
history cannot authorize a current command. Sticky fault checks still bracket
decoding in both cases, and terminal analysis keeps its original wall budget.

`factory_process_threading.py` adds the host archive for those already
admitted `ThreadSample` records. Its primitive schema retains all eleven
fields, vector order and float bits. It allows at most 32768 rows and 64 MiB,
with an individual 2048-byte cap. The full bound includes three 512-byte
identities and all numeric fields; it does not use observed average lengths.
At most eight finite range views and eight concurrent decodes may exist.
Views capture their original range without copying bytes or following later
appends. Close or a fault during decode prevents delivery; capacity remains
charged until the active decode and all views have closed. Neither codec nor
archive judges threading geometry. The unchanged verifier consumes the
reconstructed records, including finite records that it refutes. The private
task adapter explicitly closes views and archive; default seating still uses
its original implementation. Terminal verification takes a finite view rather
than iterating the growing archive. The adapter bounds the decoded working set
by serializing domain operations and decoding one row at a time; these private
stores retain no cache of reconstructed records.

Native integration still requires canonical model/source admission, an owned
process launcher, measured memory headroom and a separately reviewed retained
corpus replay. The CPU adapter binds its model/epoch/layout handshake but does
not construct or authenticate a native producer. Primitive decoding alone is
not a Factory observation validator. Transport faults reach the actual child
controller stop path; an acknowledgment alone never establishes physical rest.

Before a native plan can be frozen, frame bounds must account for all 2048
declared contact candidates, complete shoulder witnesses, every native channel
and the fixed metadata schema. The 32768 raw and 20000 history row capacities
are count maxima, not permission for an unbounded byte backlog. An explicit
shared byte cap must fit the mandatory observation windows plus the SDK, host,
in-flight IPC copies and decoded working set inside the same 24 GiB scope.
Shared immutable byte references count once per process; actual cross-process
copies count separately. Overflow remains a terminal refusal. Observed average
contact counts cannot justify this bound.

Process separation can remove host verifier/JSON/archive GC and GIL stalls from
the solver interpreter. It does not remove child SDK, observer, codec or local
journal pauses, nor GPU synchronization, scheduling or memory pressure. The
previous native age refusals and all original limits remain unchanged.

# CPU adapter integration (candidate, no native entry point)

`factory_process_adapter.py` now connects the private protocol to
`FactorySolveOwner` through an explicit `_process_transport` argument. The
default owner still selects `SolveJournal` and its existing raw queue. The
optional child selects the latest-only journal, keeps the same admission queue,
write guard, solve loop and physical checks, and encodes each full RAW snapshot
before calling `accept_solve` once. Its separate sender publishes the observed
acceptance result. A sticky transport fault prevents the next owner cycle;
priority stop still uses the actual controller guard and never claims physical
rest from an acknowledgment.

Three inherited Unix socketpairs separate RAW/outcomes, admission, and
stop/fault fences. Normal admission carries the deadline established before
the host's generation query; receiving a request does not restart its one-second
budget. Observation decoding uses the original action, rest or readiness
deadline, bounded by the already fixed session deadline. No decoder substitutes
host receipt time for capture time or applies a new current-age test to stored
threading history. The original domain's `check_solve` calls remain in place.

The host stores accepted packets in the charged ring and invokes the original
`FasteningDomain` algorithms through `_ProcessDomain`. A batch freezes its extent,
decodes without a cache between two authoritative fences, and releases the active
lease before returning a detached row. The domain serializes observation
operations, retains only current/previous solved rows, and explicitly closes
finite `ThreadArchive` views after the unchanged threading verifier. Domain
completion and closure release all observation leases; no destructor or GC event
reclaims budget.

`_FileArchive` creates exclusive `solves.jsonl` and `outcomes.jsonl` files with an
explicit disk cap. RAW acknowledgment follows a complete write and `fsync`.
Acceptance outcome acknowledgment follows its separate persisted record. The
third acknowledgment means that the host holds its own immutable bytes; child
archive acknowledgment alone cannot release a slot. A partial write remains
retained and prevents a complete receipt. Slow persistence can exhaust the child
outbox and fail the episode; it is not a throughput guarantee.

Closure preserves the original two-second owner join interval and the original
external session deadline for archive draining. Guard, owner, thread or channel
close failures remain negative while the other owned cleanups are attempted.
The pre-drain owner receipt is retained literally, including its pending-record
count. Separate stream closure establishes whether those remaining records were
fully acknowledged afterward.
An incomplete RAW outcome at EOF also remains a sticky owner error; it cannot
replace an earlier error or skip resolution of pending admission futures.

CPU subprocess tests exercise real socketpairs, owner/controller and domain code
over explicitly synthetic rows. They cover readiness, reset, turn, stop/rest,
stale rejection with RAW retained, exclusive persistence, and stop while archive
IO is blocked. These are software contract tests, not robot motion evidence.
The aggregate memory table remains a planning reservation: a native launcher,
model/source admission, measured SDK/host headroom, bounded disk admission and a
new retained-corpus replay are still required. No native profile or CLI selects
this adapter, and no prior native result transfers to it.
