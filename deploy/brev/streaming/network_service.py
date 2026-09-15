#!/usr/bin/env python3
"""Own one narrow nftables table and publish its current host admission."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import threading
import time

import network_guard as guard


def expected_expressions(protocol):
    return [
        {'match': {'op': '!=', 'left': {'meta': {'key': 'iifname'}}, 'right': 'lo'}},
        {'match': {'op': '==', 'left': {'payload': {'protocol': protocol, 'field': 'dport'}},
                   'right': {'set': list(guard.PORTS)}}},
        {'drop': None},
    ]


def validate_rules(value):
    """Reject extra rules and wrong hooks, including an earlier accept in our chain."""
    expected = [
        {'table': {'family': 'inet', 'name': guard.TABLE}},
        {'chain': {'family': 'inet', 'table': guard.TABLE, 'name': guard.CHAIN,
                   'type': 'filter', 'hook': 'input', 'prio': -200, 'policy': 'accept'}},
        *[{'rule': {'family': 'inet', 'table': guard.TABLE, 'chain': guard.CHAIN,
                    'expr': expected_expressions(protocol)}} for protocol in ('tcp', 'udp')],
    ]
    actual = []
    for row in value.get('nftables', []):
        if set(row) == {'metainfo'}:
            continue
        actual.append({key: {k: v for k, v in item.items() if k not in ('handle', 'use')}
                       for key, item in row.items()})
    if actual != expected:
        raise ValueError('The owned camera firewall table differs from the reviewed rules')
    return True


def nft(*arguments, input_text=None):
    result = subprocess.run(['/usr/sbin/nft', *arguments], input=input_text,
                            capture_output=True, text=True, timeout=5)
    if result.returncode:
        raise RuntimeError('Camera nftables operation failed; no admission issued')
    return json.loads(result.stdout) if result.stdout.strip() else {}


def ensure_rules():
    tables = nft('-j', 'list', 'tables')
    exists = any(row.get('table', {}).get('family') == 'inet'
                 and row['table'].get('name') == guard.TABLE for row in tables.get('nftables', []))
    if not exists:
        nft('-f', '-', input_text=guard.RULESET)
    # Existing tables are never flushed or overwritten to conceal a mismatch.
    validate_rules(nft('-j', 'list', 'table', 'inet', guard.TABLE))


def publish():
    value = {'schema': 1, 'status': 'PASS', 'backend': 'nftables',
             'ports': list(guard.PORTS), 'rules_sha256': guard.RULESET_SHA256,
             'checked_monotonic': time.monotonic(), **guard.host_identity()}
    path = guard.DIRECTORY / f'.admission-{os.getpid()}.json'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        path.replace(guard.ADMISSION)
    finally:
        path.unlink(missing_ok=True)


def notify_ready():
    address = os.environ.get('NOTIFY_SOCKET')
    if address:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
            client.settimeout(2)
            client.connect('\0' + address[1:] if address.startswith('@') else address)
            client.sendall(b'READY=1')


def main():
    if os.geteuid() != 0 or socket.gethostname().split('.')[0].lower().startswith('matrix'):
        raise ValueError('Camera network admission must run as a supervised Brev host service')
    info = guard.DIRECTORY.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != 0
            or stat.S_IMODE(info.st_mode) & 0o022):
        raise ValueError('Unsafe camera network runtime directory')
    stopped = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopped.set())
    guard.ADMISSION.unlink(missing_ok=True)
    try:
        ensure_rules()
        publish()
        notify_ready()
        while not stopped.wait(5):
            validate_rules(nft('-j', 'list', 'table', 'inet', guard.TABLE))
            publish()
    finally:
        guard.ADMISSION.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
