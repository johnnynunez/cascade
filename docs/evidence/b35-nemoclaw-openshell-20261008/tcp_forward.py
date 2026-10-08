#!/usr/bin/env python3
"""Loopback -> OpenShell-bridge TCP forwarder for the authenticated llama-server.

NemoClaw's llama-cpp provider needs the server on 127.0.0.1:8081 AND on
host.openshell.internal:8081 (the openshell-docker bridge gateway). llama-server
binds one address; this relays the bridge address to loopback without binding
any LAN interface. Authentication stays native to llama-server (--api-key-file).
usage: tcp_forward.py LISTEN_HOST LISTEN_PORT TARGET_HOST TARGET_PORT
"""
import socket
import sys
import threading


def pipe(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()


def main():
    lhost, lport, thost, tport = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    srv = socket.create_server((lhost, lport), reuse_port=False)
    print(f"forwarding {lhost}:{lport} -> {thost}:{tport}", flush=True)
    while True:
        client, _ = srv.accept()
        try:
            upstream = socket.create_connection((thost, tport), timeout=10)
            upstream.settimeout(None)
        except OSError:
            client.close()
            continue
        threading.Thread(target=pipe, args=(client, upstream), daemon=True).start()
        threading.Thread(target=pipe, args=(upstream, client), daemon=True).start()


if __name__ == "__main__":
    main()
