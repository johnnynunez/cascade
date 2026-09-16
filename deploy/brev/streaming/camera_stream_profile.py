"""Generate a private MediaMTX relay profile without importing Isaac Sim."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import ipaddress
import json
from pathlib import Path
from urllib.parse import urlsplit


CAMERAS = (("kitchen", "proof"), ("worktop", "cam0"), ("side", "side"))
TAILNET = ipaddress.IPv4Network("100.64.0.0/10")


@dataclass(frozen=True)
class StreamProfile:
    tailnet_ipv4: str
    visitor_origin: str
    rtsp_ports: tuple[int, int, int] = (8554, 8555, 8556)
    http_port: int = 8889
    media_port: int = 8189
    reserved_ports: tuple[int, ...] = (
        8041, 8091, 8092, 8210, 8611, 8888, 18789, 18790, 18791,
        47998, 47999, 48000, 49100, 49101, 49102,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.tailnet_ipv4, str):
            raise ValueError("Streaming requires a literal Tailscale IPv4 address")
        try:
            address = ipaddress.IPv4Address(self.tailnet_ipv4)
        except (ipaddress.AddressValueError, TypeError) as exc:
            raise ValueError("Streaming requires a literal Tailscale IPv4 address") from exc
        if address not in TAILNET:
            raise ValueError("Streaming address must belong to the Tailscale IPv4 range")
        if (not isinstance(self.visitor_origin, str)
                or any(char.isspace() or ord(char) < 32 or 127 <= ord(char) <= 159
                       for char in self.visitor_origin)):
            raise ValueError("Visitor origin must not contain whitespace or control characters")
        try:
            origin = urlsplit(self.visitor_origin)
            hostname, port = origin.hostname, origin.port
        except ValueError as exc:
            raise ValueError("Visitor origin has an invalid hostname or port") from exc
        tailnet_http = origin.scheme == "http" and hostname == self.tailnet_ipv4
        if (not hostname or (origin.scheme != "https" and not tailnet_http)
                or origin.username is not None or origin.password is not None
                or origin.path or "?" in self.visitor_origin or "#" in self.visitor_origin):
            raise ValueError("Visitor origin requires HTTPS or HTTP on the exact Tailscale IPv4, without credentials or a path")
        if (port is not None and not 1 <= port <= 65535) or origin.netloc.endswith(":"):
            raise ValueError("Visitor origin has an invalid port")
        if len(self.rtsp_ports) != len(CAMERAS):
            raise ValueError("Provide exactly one RTSP port for each camera")
        selected = (*self.rtsp_ports, self.http_port, self.media_port)
        all_ports = (*selected, *self.reserved_ports)
        if any(type(port) is not int or not 1 <= port <= 65535 for port in all_ports):
            raise ValueError("Ports must be integers between 1 and 65535")
        if len(set(selected)) != len(selected) or set(selected).intersection(self.reserved_ports):
            raise ValueError("Camera, relay, and existing service ports must not overlap")

    def mediamtx_config(self) -> dict:
        return {
            "logLevel": "info",
            "readTimeout": "10s",
            "writeTimeout": "10s",
            "writeQueueSize": 64,
            "api": False,
            "metrics": False,
            "pprof": False,
            "playback": False,
            "rtsp": False,
            "rtmp": False,
            "hls": False,
            "srt": False,
            "moq": False,
            "webrtc": True,
            "webrtcAddress": f"127.0.0.1:{self.http_port}",
            "webrtcAllowOrigins": [self.visitor_origin],
            "webrtcLocalUDPAddress": f"{self.tailnet_ipv4}:{self.media_port}",
            "webrtcLocalTCPAddress": "",
            "webrtcIPsFromInterfaces": False,
            "webrtcAdditionalHosts": [self.tailnet_ipv4],
            "webrtcICEServers2": [],
            "webrtcHandshakeTimeout": "10s",
            "webrtcTrackGatherTimeout": "2s",
            "authMethod": "internal",
            "authInternalUsers": [{
                "user": "any",
                "ips": ["127.0.0.1", "::1"],
                "permissions": [{"action": "read", "path": name} for name, _ in CAMERAS],
            }],
            "pathDefaults": {
                "sourceOnDemand": False,
                "rtspTransport": "tcp",
                "alwaysAvailable": False,
            },
            "paths": {
                name: {"source": f"rtsp://127.0.0.1:{port}/{name}"}
                for (name, _), port in zip(CAMERAS, self.rtsp_ports)
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tailnet-ipv4", required=True)
    parser.add_argument("--visitor-origin", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = StreamProfile(args.tailnet_ipv4, args.visitor_origin)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile.mediamtx_config(), indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
