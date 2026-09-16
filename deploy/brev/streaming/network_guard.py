"""Admit camera listeners only while a protected host firewall receipt is fresh."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

DIRECTORY = Path('/run/paai-camera-network')
ADMISSION = DIRECTORY / 'admission.json'
PORTS = (8554, 8555, 8556)
TABLE = 'paai_camera'
CHAIN = 'input'
RULESET = '''table inet paai_camera {
    chain input {
        type filter hook input priority -200; policy accept;
        iifname != "lo" tcp dport { 8554, 8555, 8556 } drop
        iifname != "lo" udp dport { 8554, 8555, 8556 } drop
    }
}
'''
RULESET_SHA256 = hashlib.sha256(RULESET.encode()).hexdigest()
MAX_AGE_S = 15


def host_identity():
    return {'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'netns_inode': os.stat('/proc/self/ns/net').st_ino}


def validate_receipt(value, ports, *, now, identity):
    if tuple(ports) != PORTS or not isinstance(value, dict):
        raise ValueError('Camera firewall requires the three admitted RTSP ports')
    expected = {'schema': 1, 'status': 'PASS', 'backend': 'nftables',
                'ports': list(PORTS), 'rules_sha256': RULESET_SHA256, **identity}
    if any(value.get(key) != wanted for key, wanted in expected.items()):
        raise ValueError('Camera firewall receipt does not match this host and profile')
    checked = value.get('checked_monotonic')
    if (type(checked) not in (int, float) or not math.isfinite(checked)
            or not 0 <= now - checked <= MAX_AGE_S):
        raise ValueError('Camera firewall receipt is stale or invalid')
    return value


def read_protected_receipt():
    for path in (*reversed(DIRECTORY.parents), DIRECTORY):
        info = path.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != 0
                or stat.S_IMODE(info.st_mode) & 0o022):
            raise ValueError('Camera firewall directory must be protected and root-owned')
    if not os.statvfs(DIRECTORY).f_flag & os.ST_RDONLY:
        raise ValueError('Camera firewall admission directory must be mounted read-only')
    fd = os.open(ADMISSION, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'r') as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0
                or stat.S_IMODE(info.st_mode) & 0o022 or not 0 < info.st_size <= 8192):
            raise ValueError('Camera firewall receipt must be a small root-owned regular file')
        return json.loads(stream.read(8193))


def verify_isolation(ports):
    """Called from the unprivileged, host-networked Isaac container."""
    return validate_receipt(read_protected_receipt(), ports,
                            now=time.monotonic(), identity=host_identity())
