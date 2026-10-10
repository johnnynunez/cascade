#!/usr/bin/env python3
"""Private CA + server certificate for an opt-in TLS `cascade.robot-runtime/1` endpoint (B71).

The split conversation deployment (docs/CONVERSATION.md) serves the robot runtime
over loopback in clear by default. Off loopback, `cascade-robot-service` refuses
to start without TLS; this creates the material it needs with the openssl recipe
of `scripts/nemoclaw_mcp.py certs` (B35):

    scripts/robot_endpoint_certs.py --out ~/.cascade/robot-tls --san IP:10.0.0.5
    cascade-robot-service --robot <profile> --host 10.0.0.5 --port 8781 \\
        --tls-cert ~/.cascade/robot-tls/server.pem --tls-key ~/.cascade/robot-tls/server.key \\
        --token-env CASCADE_ROBOT_ENDPOINT_TOKEN --run-root runs/robot
    # conversation host: copy ca.pem only (never ca.key or server.key), then set
    #   "robot_endpoint": "https://10.0.0.5:8781", "robot_tls_ca": "ca.pem"
    # or pin the printed certificate_sha256 as "robot_tls_fingerprint".

The CA is created once per --out directory and reused (--force rotates it); each
run issues a new server key and certificate named by --name, valid for exactly
the --san entries given (IP:<address> or DNS:<name>; the client checks the URL
host against them). The directory is 0700 and every key 0600. Prints one JSON
line with the paths and the server certificate's SHA-256 fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
import shutil
import ssl
import subprocess
import sys
from pathlib import Path

_DNS = re.compile(r"^(?=.{1,253}$)([A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*"
                  r"[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def san_entry(value: str) -> str:
    """Validate one subjectAltName: ``IP:<unicast address>`` or ``DNS:<name>`` (no wildcard)."""
    kind, _, name = value.partition(":")
    if kind == "IP":
        try:
            address = ipaddress.ip_address(name)
        except ValueError:
            raise ValueError(f"invalid IP subjectAltName {value!r}") from None
        if address.is_unspecified or address.is_multicast:
            raise ValueError(f"{value!r} is not one unicast address a client can dial")
        return f"IP:{address}"
    if kind == "DNS" and _DNS.match(name):
        return f"DNS:{name.lower()}"
    raise ValueError(f"subjectAltName must be IP:<address> or DNS:<name>, got {value!r}")


def certificate_sha256(path: Path) -> str:
    """SHA-256 of the first certificate in a PEM file (what the client pins)."""
    text = Path(path).read_text()
    begin = text.index("-----BEGIN CERTIFICATE-----")
    end = text.index("-----END CERTIFICATE-----", begin) + len("-----END CERTIFICATE-----")
    return hashlib.sha256(ssl.PEM_cert_to_DER_cert(text[begin:end])).hexdigest()


def _openssl(*args: str) -> None:
    if shutil.which("openssl") is None:
        raise SystemExit("openssl CLI not found")
    subprocess.run(["openssl", *args], check=True, capture_output=True)


def issue(out: Path, sans: list[str], *, name: str = "server", days: int = 397, force: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)
    ca_key, ca = out / "ca.key", out / "ca.pem"
    key, cert, csr, ext = (out / f"{name}{suffix}" for suffix in (".key", ".pem", ".csr", ".ext"))
    if not ca.exists() or force:
        _openssl("req", "-x509", "-newkey", "rsa:3072", "-nodes", "-days", "825", "-sha256",
                 "-subj", "/CN=CASCADE robot runtime private CA",
                 "-addext", "basicConstraints=critical,CA:TRUE,pathlen:0",
                 "-addext", "keyUsage=critical,keyCertSign,cRLSign",
                 "-keyout", str(ca_key), "-out", str(ca))
        ca_key.chmod(0o600)
        ca.chmod(0o644)
    ext.write_text("basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\n"
                   f"extendedKeyUsage=serverAuth\nsubjectAltName={','.join(sans)}\n")
    common_name = sans[0].partition(":")[2]
    try:
        _openssl("req", "-newkey", "rsa:2048", "-nodes", "-subj", f"/CN={common_name}",
                 "-keyout", str(key), "-out", str(csr))
        key.chmod(0o600)
        _openssl("x509", "-req", "-in", str(csr), "-CA", str(ca), "-CAkey", str(ca_key), "-CAcreateserial",
                 "-days", str(days), "-sha256", "-extfile", str(ext), "-out", str(cert))
    finally:
        csr.unlink(missing_ok=True)
        ext.unlink(missing_ok=True)
    return {"dir": str(out), "ca": str(ca), "cert": str(cert), "key": str(key), "san": sans,
            "certificate_sha256": certificate_sha256(cert)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="state directory (0700): ca.pem/ca.key + <name>.pem/.key")
    ap.add_argument("--san", action="append", required=True,
                    help="IP:<address> or DNS:<name> the client will dial (repeatable)")
    ap.add_argument("--name", default="server", help="server certificate file stem (default: server)")
    ap.add_argument("--days", type=int, default=397, help="server certificate validity in days (1..825)")
    ap.add_argument("--force", action="store_true", help="also rotate the CA")
    args = ap.parse_args(argv)
    try:
        sans = [san_entry(value) for value in args.san]
    except ValueError as exc:
        ap.error(str(exc))
    if not _NAME.match(args.name) or args.name == "ca":
        ap.error("--name must be 1-64 letters, digits, '-' or '_' and not 'ca'")
    if not 1 <= args.days <= 825:
        ap.error("--days must be in 1..825")
    print(json.dumps(issue(args.out.expanduser(), sans, name=args.name, days=args.days, force=args.force)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
