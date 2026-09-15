#!/usr/bin/env python3
"""Install the optional host firewall monitor without granting the demo privileges."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pwd
import socket
import stat
import subprocess
import tempfile

UNIT_NAME = 'paai-camera-network.service'
LIBRARY = Path('/usr/local/lib/paai-camera-network')
UNIT_PATH = Path('/etc/systemd/system') / UNIT_NAME
DROPIN = Path('/etc/systemd/system/paai-demo-docker.service.d/camera-network.conf')
UNIT = '''[Unit]
Description=Physical Agentic AI private camera firewall admission
Before=paai-demo-docker.service
After=network-pre.target
StartLimitIntervalSec=300
StartLimitBurst=3

[Service]
Type=notify
ExecStart=/usr/bin/python3 /usr/local/lib/paai-camera-network/network_service.py
RuntimeDirectory=paai-camera-network
RuntimeDirectoryMode=0755
RuntimeDirectoryPreserve=yes
Restart=on-failure
RestartSec=5
TimeoutStartSec=20
TimeoutStopSec=10
User=root
Group=root
UMask=0022
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/run/paai-camera-network
PrivateTmp=yes
CapabilityBoundingSet=CAP_NET_ADMIN
RestrictAddressFamilies=AF_UNIX AF_NETLINK

[Install]
WantedBy=multi-user.target
'''
DEPENDENCY = '''[Unit]
Requires=paai-camera-network.service
After=paai-camera-network.service
'''


def run(arguments, timeout=30):
    result = subprocess.run(list(map(str, arguments)), stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError('Camera network installation failed; private output withheld')
    return result.stdout.strip()


def install_file(source, destination):
    if destination.exists() or destination.is_symlink():
        info = destination.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0
                or stat.S_IMODE(info.st_mode) & 0o022):
            raise ValueError('Existing camera network file is not protected')
        if destination.read_bytes() == source.read_bytes():
            return False
        raise ValueError('Existing camera network service differs; review before replacing it')
    run(['sudo', '-n', 'install', '-o', 'root', '-g', 'root', '-m', '0644', source, destination])
    return True


def protected_directory(path):
    for member in (*reversed(path.parents), path):
        try:
            info = member.lstat()
        except FileNotFoundError:
            if member == path:
                return False
            raise ValueError('Camera service ancestor directory is missing') from None
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != 0
                or stat.S_IMODE(info.st_mode) & 0o022):
            raise ValueError('Camera service directory and ancestors must be protected')
    return True


def install():
    if (socket.gethostname().split('.')[0].lower().startswith('matrix')
            or os.getuid() == 0 or pwd.getpwuid(os.getuid()).pw_name != 'ubuntu'):
        raise ValueError('Camera network setup requires the unprivileged Brev ubuntu account')
    if not Path('/usr/sbin/nft').is_file():
        raise ValueError('Install the Ubuntu nftables package before enabling camera video')
    run(['sudo', '-n', 'true'])
    for directory in (LIBRARY, DROPIN.parent, UNIT_PATH.parent):
        if not protected_directory(directory):
            run(['sudo', '-n', 'install', '-d', '-o', 'root', '-g', 'root', '-m', '0755', directory])
            if not protected_directory(directory):
                raise ValueError('Camera service directory creation failed')
    changed = False
    for name in ('network_guard.py', 'network_service.py'):
        changed |= install_file(Path(__file__).resolve().parent / name, LIBRARY / name)
    with tempfile.TemporaryDirectory(prefix='paai-camera-unit-') as temporary:
        files = Path(temporary)
        (files / 'unit').write_text(UNIT)
        (files / 'dependency').write_text(DEPENDENCY)
        changed |= install_file(files / 'unit', UNIT_PATH)
        changed |= install_file(files / 'dependency', DROPIN)
    if changed:
        run(['sudo', '-n', 'systemctl', 'daemon-reload'])
    run(['sudo', '-n', 'systemctl', 'enable', '--now', UNIT_NAME], timeout=30)
    active = run(['systemctl', 'is-active', UNIT_NAME])
    if active != 'active':
        raise ValueError('Camera network service did not become active')
    return {'status': 'PASS_HOST_SETUP', 'unit': UNIT_NAME, 'active': True,
            'live_media_verified': False, 'raw_rtsp_negative_probe_pending': True}


if __name__ == '__main__':
    print(json.dumps(install(), indent=2))
