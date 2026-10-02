"""Private inherited-FD worker; launch through OvrtxProcess only."""
import argparse
import os
from multiprocessing.connection import Connection

from .ovrtx_renderer import OvrtxRenderer


def serve(connection, renderer_factory=OvrtxRenderer):
    renderer = None
    os.set_inheritable(connection.fileno(), False)
    try:
        while True:
            try:
                op, data = connection.recv()
            except EOFError:
                return
            try:
                if op == "open" and renderer is None:
                    renderer = renderer_factory(**data)
                    renderer.open()
                    result = None
                elif op == "render" and renderer is not None:
                    result = renderer.render(data)
                elif op == "close":
                    return
                else:
                    raise ValueError("Invalid OVRTX worker lifecycle request")
                connection.send((True, result))
            except Exception as exc:
                connection.send((False, f"{type(exc).__name__}: {exc}"))
                return  # Partial native transactions permanently end this owner.
    finally:
        if renderer is not None:
            renderer.close()
        connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    serve(Connection(parser.parse_args().fd))
