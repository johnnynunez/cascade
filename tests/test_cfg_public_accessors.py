"""Readers of a `cascade.config.Cfg` section go through its public accessor
(`Cfg.as_dict()`, via `config.plain`), never its private `_data` (Task 5 fix
review Minor 3, #374 review Minor 5).

The reader gets a copy: nothing it does to the mapping (a workspace's corner
lists included) can change the running configuration the harness and the
skills read.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from cascade.config import Cfg

REPO = Path(__file__).resolve().parents[1]
#: Every site that read `getattr(<section>, "_data", <section>)` (the review's 5).
SITES = (
    "src/cascade/skills/runtime.py",
    "src/cascade/apps/map_tile.py",
    "src/cascade/apps/stream_server.py",
    "src/cascade/perception/probe.py",
    "src/cascade/perception/visual_interface.py",
)


def _private_data_reads(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return [node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("getattr", "hasattr") and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant) and node.args[1].value == "_data"]


def test_no_reader_reaches_into_a_cfg_sections_private_data():
    found = {rel: _private_data_reads(REPO / rel) for rel in SITES}
    assert all(not lines for lines in found.values()), found


def _workspace_cfg():
    return Cfg({"safety": {"workspace": {"min": [0.05, -0.3, 0.0], "max": [0.5, 0.3, 0.4]}}})


def test_the_probe_reads_a_copy_of_the_workspace():
    from cascade.perception.probe import PointProbe

    cfg = _workspace_cfg()
    ws = PointProbe(SimpleNamespace(cfg=cfg))._workspace()
    assert ws == {"min": [0.05, -0.3, 0.0], "max": [0.5, 0.3, 0.4]}
    ws["min"][0] = 9.0
    assert cfg.safety.workspace.min == [0.05, -0.3, 0.0]


def test_the_map_tile_reads_a_copy_of_its_settings():
    from cascade.apps.map_tile import _plain

    cfg = Cfg({"nvblox_map": {"enabled": True, "size": [320, 240]}})
    plain = _plain(cfg.nvblox_map)
    assert plain == {"enabled": True, "size": [320, 240]} and type(plain) is dict
    plain["size"][0] = 1
    assert cfg.nvblox_map.size == [320, 240]
    assert _plain("auto") == "auto" and _plain({"a": 1}) == {"a": 1}


def test_plain_is_a_copy_of_a_section_and_passes_anything_else():
    from cascade.config import plain

    cfg = _workspace_cfg()
    data = plain(cfg.safety)
    assert data == {"workspace": {"min": [0.05, -0.3, 0.0], "max": [0.5, 0.3, 0.4]}}
    data["workspace"]["max"][2] = 0.0
    assert cfg.safety.workspace.max == [0.5, 0.3, 0.4]
    raw = {"min": [0.0]}
    assert plain(raw) is raw and plain(None) is None
