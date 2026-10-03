"""Opt-in diagnostic target submissions; never actuator or timing authority.

Queue acceptance, first successful setter return, and repeated setter returns
are distinct. The bounded ledger may lose evidence, never invent application.
All inputs are existing values; this module makes no SDK calls or file writes.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import threading
import time
import uuid

import numpy as np


def vector(value, dtype):
    a = np.asarray(value, dtype=dtype)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError("finite one-dimensional target required")
    return {"dtype": a.dtype.str, "shape": list(a.shape), "values": a.tolist(),
            "sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest()}


class TargetReceipts:
    def __init__(self, robot_id, joint_names, epoch, *, capacity=4096):
        self.robot_id, self.joint_names, self.epoch = robot_id, list(joint_names), epoch
        self.instance = uuid.uuid4().hex
        self.capacity = capacity
        self.sequence = 0
        self.records = OrderedDict()
        self.last_written = None
        self.boundary = None
        self.superseded = self.evicted = self.error_count = 0
        self.first_error = None
        self.stopped = False
        self._lock = threading.Lock()

    def _call(self, fn):
        with self._lock:
            try:
                return copy.deepcopy(fn())
            except Exception as exc:
                self.error_count += 1
                if self.first_error is None:
                    self.first_error = f"{type(exc).__name__}: {exc}"[:400]
                return None

    def queued(self, q, command_id=None):
        def record():
            self.sequence += 1
            previous = self.records.get(self.sequence - 1)
            if previous is not None and previous['first_write'] is None:
                # Its target may already be snapshotted/in a setter call. This
                # flags uncertain evidence ordering, never proof of no write.
                previous['superseded_before_setter_receipt'] = True
                self.superseded += 1
            if command_id is not None and (not isinstance(command_id, str) or not 1 <= len(command_id) <= 128):
                raise ValueError("invalid diagnostic command id")
            item = {'version': 1, 'instance': self.instance, 'robot_id': self.robot_id,
                    'epoch': self.epoch, 'sequence': self.sequence, 'command_id': command_id,
                    'status': 'queued', 'accepted_monotonic_s': time.monotonic(),
                    'requested_asset_q': vector(q, '<f8'), 'first_write': None,
                    'last_write': None, 'write_count': 0, 'setter_failures': 0,
                    'submission_failures': 0,
                    'superseded_before_setter_receipt': False}
            self.records[self.sequence] = item
            if len(self.records) > self.capacity:
                self.records.popitem(last=False)
                self.evicted += 1
            self.stopped = False
            return item
        return self._call(record)

    def update_started(self):
        # A failed update/history capture must not reuse the previous boundary.
        def record():
            self.boundary = None
        self._call(record)

    def completed_update(self, clock):
        # This is the already-read PhysX frame-history clock, not a new read.
        def record():
            if clock.get('epoch') != self.epoch:
                raise ValueError('completed-update epoch mismatch')
            self.boundary = copy.deepcopy(clock)
        self._call(record)

    def copy_target(self, target):
        # Snapshot the borrowed argument before the SDK can mutate/reuse it.
        return self._call(lambda: np.array(target, dtype='<f4', copy=True))

    def setter_returned(self, sequence, target, arm_indices):
        def record():
            item = self.records.get(sequence)
            if item is None or item['epoch'] != self.epoch:
                raise ValueError('setter submission has no retained command in this epoch')
            values = np.asarray(target, dtype='<f4')
            event = {'status': 'setter_returned', 'producer_monotonic_s': time.monotonic(),
                     'joint_names': self.joint_names,
                     'arm_target': vector(values[list(arm_indices)], '<f4'),
                     'full_target': vector(values, '<f4'),
                     'prior_completed_update': copy.deepcopy(self.boundary),
                     'clock_scope': 'last completed update before target setter; no subsequent solve claimed'}
            if item['first_write'] is None:
                item['first_write'] = copy.deepcopy(event)
            item['last_write'] = event
            item['write_count'] += 1
            item['status'] = 'setter_returned'
            self.last_written = sequence
        self._call(record)

    def setter_failed(self, sequence, error, *, stage='set'):
        def record():
            item = self.records.get(sequence)
            if item is not None:
                item['submission_failures'] += 1
                item['setter_failures'] += int(stage == 'set')
                item['last_setter_error'] = str(error)[:400]
                item['last_failure_stage'] = stage
        self._call(record)

    def stop(self):
        def record():
            self.stopped = True
        self._call(record)

    def invalidate(self, epoch):
        def record():
            self.epoch = epoch
            self.records.clear()
            self.last_written = self.boundary = None
        self._call(record)

    def snapshot(self):
        def record():
            latest = self.records.get(self.sequence)
            written = self.records.get(self.last_written)
            return {'version': 1, 'instance': self.instance, 'robot_id': self.robot_id,
                    'epoch': self.epoch, 'last_queued': latest, 'last_written': written,
                    'stopped_observed': self.stopped,
                    'coverage': {'retained_commands': len(self.records), 'capacity': self.capacity,
                                 'superseded_before_setter_receipt': self.superseded, 'evicted': self.evicted,
                                 'diagnostic_errors': self.error_count, 'first_error': self.first_error},
                    'scope': 'command submission only; no motion, braking, rest or task acceptance'}
        return self._call(record)
