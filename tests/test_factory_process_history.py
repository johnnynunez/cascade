"""Private child latest-only and host byte leases; no SDK or native solve."""
from dataclasses import replace

import pytest
from test_fastening_runtime import binding, limits, row

from cascade.control.fastening import FasteningController, FasteningFault, SolveJournal
from cascade.sim.factory_process_history import (
    _AcceptedHistory,
    _ByteLeases,
    _LatestJournal,
)


@pytest.mark.parametrize("failure", ["epoch", "step", "time", "binding", "type"])
def test_child_publish_matches_original_identity_clock_refusals(failure):
    legacy, latest = SolveJournal(binding(), _compact=True), _LatestJournal(binding())
    for journal in (legacy, latest):
        journal.publish(row(1))
    changes = {"epoch": {"epoch": "other"}, "step": {"step": 3},
               "time": {"simulation_time_s": .03}, "binding": {"binding_sha256": "b"*64}}
    value = object() if failure == "type" else replace(row(2), **changes[failure])
    errors = []
    for journal in (legacy, latest):
        with pytest.raises(FasteningFault) as caught:
            journal.publish(value)
        errors.append(str(caught.value))
        with pytest.raises(FasteningFault, match="producer stream"):
            journal.read()
    assert errors[0] == errors[1]


def test_latest_journal_preserves_controller_admission_and_stop_without_local_history():
    journal = _LatestJournal(binding())
    controller = FasteningController(binding(), limits(), journal, synthetic=True,
                                     close_owner=lambda: {"ok": True}, clock=lambda: 10.)
    controller.accept_solve(row(1))
    reset = controller.reset_stop()
    controller.accept_solve(row(2, generation=reset["generation"]))
    permit = controller.request_turn(expected_generation=reset["generation"])
    assert permit.admission_step == 2 and permit.generation == reset["generation"] + 1
    assert controller.stop()["generation"] == permit.generation + 1
    with pytest.raises(FasteningFault, match="host reader"):
        journal.read(0)
    assert journal.read() == (controller._previous,)


def test_child_latest_metadata_is_not_mutated_through_an_exposed_row():
    journal = _LatestJournal(binding())
    exposed = row(1)
    journal.publish(exposed)
    object.__setattr__(exposed, "step", 900)
    object.__setattr__(exposed, "epoch", "changed")
    object.__setattr__(exposed, "simulation_time_s", 5.)
    journal.publish(row(2))
    assert journal.read()[0].step == 2


def test_child_unchanged_controller_still_latches_original_age_fault():
    journal = _LatestJournal(binding())
    controller = FasteningController(binding(), limits(), journal, synthetic=True,
                                     close_owner=lambda: {"ok": True}, clock=lambda: 10.)
    controller.accept_solve(row(1))
    generation = controller.generation
    with pytest.raises(FasteningFault, match="stale"):
        controller.accept_solve(row(2, captured=9.7))
    assert controller.generation > generation
    with pytest.raises(FasteningFault, match="stale"):
        journal.read()


def store(*, byte_capacity=64):
    return _ByteLeases(capacity=16, byte_capacity=byte_capacity, frame_bytes=4)


def add(history, pool, sequence):
    archive = pool.register(sequence, bytes([sequence]) * 4)
    history.publish(archive, row(sequence))
    return archive


def test_one_charge_shared_by_archive_ring_and_reader_then_release():
    pool = store()
    history = _AcceptedHistory(binding(), pool, capacity=3)
    archives = [add(history, pool, i) for i in range(1, 4)]
    with history.snapshot(0) as batch:
        assert list(batch) == [bytes([i])*4 for i in range(1, 4)]
        assert pool.retained_bytes == 12  # three roles, one immutable copy
        for archive in archives:
            archive.close()
        history.close()
        assert pool.retained_bytes == 12  # reader still pins all three
    assert pool.retained_bytes == 0
    pool.require_drained()


def test_old_batch_pins_evicted_ring_and_capacity_cannot_spend_its_bytes_twice():
    pool = store(byte_capacity=16)
    history = _AcceptedHistory(binding(), pool, capacity=3)
    for i in range(1, 4):
        add(history, pool, i).close()
    pinned = history.snapshot(0)
    add(history, pool, 4).close()
    assert pool.retained_bytes == 16
    with pytest.raises(FasteningFault, match="capacity"):
        pool.register(5, b"five")
    assert pool.retained_bytes == 16  # failed register changes neither content nor charge
    history.close()
    assert pool.retained_bytes == 12
    pinned.close()
    assert pool.retained_bytes == 0
    with pytest.raises(FasteningFault, match="capacity"):
        pool.require_drained()  # successful cleanup cannot erase the fault


def test_releasing_old_snapshot_permits_reuse_without_losing_archive_copy():
    pool = store(byte_capacity=16)
    history = _AcceptedHistory(binding(), pool, capacity=3)
    archives = [add(history, pool, i) for i in range(1, 4)]
    pinned = history.snapshot(0)
    for archive in archives:
        archive.close()
    add(history, pool, 4).close()
    pinned.close()
    assert pool.retained_bytes == 12
    add(history, pool, 5).close()
    with history.snapshot(2) as current:
        assert list(current) == [bytes([i])*4 for i in (3, 4, 5)]
    with pytest.raises(FasteningFault, match="lost solves"):
        history.snapshot(1)
    history.close()
    pool.require_drained()


def test_active_iteration_keeps_charge_if_batch_closes_during_decode():
    pool = store()
    history = _AcceptedHistory(binding(), pool, capacity=3)
    add(history, pool, 1).close()
    batch = history.snapshot(0)
    iterator = iter(batch)
    assert next(iterator) == b"\1"*4
    batch.close()
    history.close()
    assert pool.retained_bytes == 4
    iterator.close()
    pool.require_drained()


def test_paused_iterator_does_not_pin_closed_batch_lease_metadata():
    import sys

    pool = store()
    history = _AcceptedHistory(binding(), pool, capacity=3)
    for step in range(1, 4):
        add(history, pool, step).close()
    batch = history.snapshot(0)
    saved = batch._leases[1]
    iterator = iter(batch)
    assert next(iterator) == b"\1"*4
    batch.close()
    history.close()
    # Only this local test variable and getrefcount's temporary refer to the
    # closed second lease. The paused iterator keeps its single active lease.
    assert sys.getrefcount(saved) == 2
    assert pool.retained_metadata == {"entries": 1, "leases": 1, "batches": 0}
    iterator.close()
    pool.require_drained()


def test_publishing_same_packet_as_a_second_solve_is_sticky_failure():
    pool = store()
    history = _AcceptedHistory(binding(), pool, capacity=3)
    archive = add(history, pool, 1)
    with pytest.raises(FasteningFault, match="identity/epoch/clock"):
        history.publish(archive, row(2))
    with pytest.raises(FasteningFault):
        history.snapshot(0)
    archive.close()
    history.close()


def test_captured_host_metadata_survives_mutation_of_original_row():
    pool = store()
    history = _AcceptedHistory(binding(), pool, capacity=3)
    archive = pool.register(1, b"row1")
    value = row(1)
    history.publish(archive, value)
    object.__setattr__(value, "simulation_time_s", 9.)
    archive.close()
    add(history, pool, 2).close()
    history.close()
    pool.require_drained()


def test_retain_exhausts_metadata_without_spending_payload_bytes_again():
    pool = _ByteLeases(capacity=16, byte_capacity=64, frame_bytes=4,
                       lease_capacity=3)
    first = pool.register(1, b"row1")
    leases = [first, first.retain(), first.retain()]
    with pytest.raises(FasteningFault, match="lease metadata capacity"):
        first.retain()
    assert pool.retained_bytes == 4
    assert pool.retained_metadata == {"entries": 1, "leases": 3, "batches": 0}
    for lease in leases:
        lease.close()
    assert pool.retained_bytes == 0
    assert pool.retained_metadata == {"entries": 0, "leases": 0, "batches": 0}
    with pytest.raises(FasteningFault, match="lease metadata capacity"):
        pool.register(2, b"row2")  # Cleanup does not heal the original fault.


def test_repeated_full_and_empty_snapshots_share_one_global_batch_cap():
    pool = _ByteLeases(capacity=16, byte_capacity=64, frame_bytes=4,
                       batch_capacity=2)
    history = _AcceptedHistory(binding(), pool, capacity=3)
    add(history, pool, 1).close()
    full, empty = history.snapshot(0), history.snapshot(1)
    assert pool.retained_metadata == {"entries": 1, "leases": 2, "batches": 2}
    with pytest.raises(FasteningFault, match="batch metadata capacity"):
        history.snapshot(1)
    assert pool.retained_bytes == 4
    history.close()
    full.close()
    empty.close()
    empty.close()  # Batch close is idempotent, never decrements twice.
    assert pool.retained_metadata == {"entries": 0, "leases": 0, "batches": 0}
    with pytest.raises(FasteningFault, match="batch metadata capacity"):
        pool.require_drained()


def test_partial_snapshot_releases_its_leases_and_batch_slot_on_limit_failure():
    pool = _ByteLeases(capacity=16, byte_capacity=64, frame_bytes=4,
                       lease_capacity=5)
    history = _AcceptedHistory(binding(), pool, capacity=3)
    for step in range(1, 4):
        add(history, pool, step).close()
    with pytest.raises(FasteningFault, match="lease metadata capacity"):
        history.snapshot(0)  # Third retain refuses after two successful ones.
    assert pool.retained_bytes == 12
    assert pool.retained_metadata == {"entries": 3, "leases": 3, "batches": 0}
    history.close()
    assert pool.retained_metadata == {"entries": 0, "leases": 0, "batches": 0}
    with pytest.raises(FasteningFault, match="lease metadata capacity"):
        pool.require_drained()


def test_batch_close_releases_other_leases_and_slot_after_a_release_error():
    pool = store()
    history = _AcceptedHistory(binding(), pool, capacity=3)
    for step in range(1, 4):
        add(history, pool, step).close()
    batch = history.snapshot(0)
    # A private ownership violation must not strand the other two leases.
    batch._leases[0].close()
    with pytest.raises(FasteningFault, match="already released"):
        batch.close()
    assert pool.retained_metadata == {"entries": 3, "leases": 3, "batches": 0}
    history.close()
    pool.require_drained()


def test_failed_ring_append_releases_unpublished_lease_and_latches_fault():
    from collections import deque

    class RefusingRing(deque):
        def append(self, value):
            raise MemoryError("injected ring allocation failure")

    pool = store()
    history = _AcceptedHistory(binding(), pool, capacity=3)
    history._ring = RefusingRing()
    archive = pool.register(1, b"row1")
    with pytest.raises(MemoryError, match="ring allocation"):
        history.publish(archive, row(1))
    assert pool.retained_metadata == {"entries": 1, "leases": 1, "batches": 0}
    assert history._metadata is None
    archive.close()
    history.close()
    assert pool.retained_metadata == {"entries": 0, "leases": 0, "batches": 0}
    with pytest.raises(FasteningFault, match="publication failed"):
        pool.require_drained()
