# Factory process boundary: CPU transport candidate

`sim/factory_process_wire.py` and `sim/factory_process_protocol.py` provide an
optional transport substrate. The production Factory owner, launchers, model
identity, physical checks and public APIs still use their existing paths. These
modules do not start a simulator, select an actuator or admit a physical solve.

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

`factory_process_history.py` is an unwired CPU candidate. `_LatestJournal`
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
The future adapter must route startup/task/rest cursor reads to this host
history and validate the actual acceptance outcome before publication.

Freshness follows each existing call site. A current observation or newly
received cursor batch retains the original observation checks and deadline.
Analysis of previously admitted ThreadHistory is historical evidence: it does
not gain a newly fabricated capture time, and it does not acquire a new
per-sample `check_solve(now)` gate that the original analysis never had. That
history cannot authorize a current command. Sticky fault checks still bracket
decoding in both cases, and terminal analysis keeps its original wall budget.

`factory_process_threading.py` adds an unwired host archive for those already
admitted `ThreadSample` records. Its primitive schema retains all eleven
fields, vector order and float bits. It allows at most 32768 rows and 64 MiB,
with an individual 2048-byte cap. The full bound includes three 512-byte
identities and all numeric fields; it does not use observed average lengths.
At most eight finite range views and eight concurrent decodes may exist.
Views capture their original range without copying bytes or following later
appends. Close or a fault during decode prevents delivery; capacity remains
charged until the active decode and all views have closed. Neither codec nor
archive judges threading geometry. The unchanged verifier consumes the
reconstructed records, including finite records that it refutes. The future
task adapter must explicitly close views and archive; default seating still
uses its original implementation. Terminal verification must take a finite
view explicitly rather than iterate the growing archive itself. The adapter
also remains responsible for the bounded public decoded working set; these
private stores retain no cache of reconstructed records.

This is not yet a native transport adapter. That adapter still requires a
canonical model/source handshake, original
request/stop semantics, aggregate memory accounting, owned process lifecycle,
and a separately reviewed retained-corpus replay. Primitive decoding alone is
not a Factory observation validator. Its terminal faults must reach the actual
child controller stop path; a locally fabricated ACK is never sufficient.

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
