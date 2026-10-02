"""OVRTX semantic output decoding; no physics or object detection inference."""
from __future__ import annotations

import re
import struct

import numpy as np


_PATH = re.compile(r"(?:/[A-Za-z_][A-Za-z_0-9]*)+")


def body_label(path):
    # OVRTX lowercases semantic class values. Encoding the exact USD path in
    # lowercase hex preserves case-sensitive prim identity without collisions.
    return "cascade_body_" + path.encode("utf-8").hex()


def semantic_layer(paths, *, world_paths=(), instance_paths=()):
    """Private stronger USD opinions: renderable prim -> exact body identity.

    The producer must inventory every renderable descendant of each body.
    This layer never edits the physics stage or infers label inheritance.
    """
    if not isinstance(paths, dict) or not paths:
        raise ValueError("OVRTX masks require a nonempty renderable-path inventory")
    root = {}
    def node_for(path):
        if not isinstance(path, str) or not _PATH.fullmatch(path):
            raise ValueError("Scene overrides require absolute prim paths")
        node = root
        for component in path.strip("/").split("/"):
            node = node.setdefault(component, {})
        return node.setdefault(None, {})

    for path, label in paths.items():
        if not all(isinstance(v, str) and _PATH.fullmatch(v) for v in (path, label)):
            raise ValueError("Semantic geometry and body identities must be absolute prim paths")
        node_for(path)["label"] = label
    for path in world_paths:
        node_for(path)["world"] = True
    for path in instance_paths:
        node_for(path)["instance"] = True

    def emit(children, depth=0):
        result = []
        indent = "    " * depth
        for name, node in children.items():
            if name is None:
                continue
            attrs = node.get(None, {})
            metadata = []
            if "label" in attrs:
                metadata.append('prepend apiSchemas = ["SemanticsAPI:class"]')
            if attrs.get("instance"):
                metadata.append('instanceable = false')
            schema = ' (\n' + '\n'.join(indent + '    ' + m for m in metadata) + '\n' + indent + ')' if metadata else ''
            result.append(f'{indent}over "{name}"{schema} {{')
            if "label" in attrs:
                result.extend((f'{indent}    string semantic:class:params:semanticType = "class"',
                               f'{indent}    string semantic:class:params:semanticData = "{body_label(attrs["label"])}"'))
            if attrs.get("world"):
                result.extend((f'{indent}    matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))',
                               f'{indent}    uniform token[] xformOpOrder = ["!resetXformStack!", "xformOp:transform"]'))
            result.extend(emit(node, depth + 1))
            result.append(indent + "}")
        return result

    return "\n".join(emit(root)) + "\n"


def decode_id_map(buffer):
    """Decode the documented IdentifierMap layout with explicit bounds checks.

    Each 24-byte entry is four uint32 ID words plus byte length and offset;
    the final uint32 is the entry count. SemanticSegmentation uses ID word0.
    Output strings are retained exactly apart from the SDK's trailing NULs.
    """
    value = np.asarray(buffer)
    if value.dtype != np.uint8 or value.size < 4 or value.size > 16 * 1024 * 1024:
        raise ValueError("Invalid OVRTX SemanticIdMap buffer")
    raw = np.ascontiguousarray(value).reshape(-1).tobytes()
    count = struct.unpack_from("<I", raw, len(raw) - 4)[0]
    table_end = count * 24
    if count > 65536 or table_end > len(raw) - 4:
        raise ValueError("SemanticIdMap entry table exceeds its buffer")
    labels = {}
    for index in range(count):
        identity, _, _, _, size, offset = struct.unpack_from("<6I", raw, index * 24)
        if offset < table_end or offset + size > len(raw) - 4:
            raise ValueError("SemanticIdMap label overlaps metadata or exceeds its buffer")
        if identity in labels:
            raise ValueError("SemanticIdMap has duplicate numeric identities")
        labels[identity] = raw[offset:offset + size].decode("utf-8").rstrip("\x00").strip()
    return labels


def body_masks(ids, label_buffer, shape, body_paths):
    """Visible masks for declared bodies from the same native sensor frame.

    Non-body semantic classes are background for this inventory. A body may
    be fully occluded. Conflicting body/class labels are rejected rather than
    interpreted as a usable zero mask. Physical attachment is not inferred.
    """
    ids = np.asarray(ids)
    if ids.dtype != np.uint32 or ids.shape != (*shape, 1):
        raise ValueError("SemanticSegmentation must be calibrated HxWx1 uint32")
    paths = tuple(body_paths)
    if len(set(paths)) != len(paths) or not all(isinstance(p, str) and _PATH.fullmatch(p) for p in paths):
        raise ValueError("Body identities must be distinct absolute prim paths")
    lookup = decode_id_map(label_buffer)
    selected = {path: [] for path in paths}
    for identity, label in lookup.items():
        matches = [path for path in paths if re.search(r"(?:^|;)\s*class:\s*" + body_label(path) + r"\s*;", label)]
        if len(matches) > 1:
            raise ValueError("Semantic ID refers to multiple physical bodies")
        if matches:
            # Do not accept a second class value silently layered on a body.
            if label.count("class:") != 1:
                raise ValueError("Physical body has ambiguous semantic class metadata")
            selected[matches[0]].append(identity)
    # The renderer must identify every ID it actually used, including classes
    # outside the robot/prop inventory. Zero is the documented background.
    if set(map(int, np.unique(ids))) - {0} - set(lookup):
        raise ValueError("SemanticSegmentation contains an unmapped visible ID")
    return {path: np.isin(ids[..., 0], values) for path, values in selected.items()}
