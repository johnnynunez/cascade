"""One admitted capture through the ordinary MCP spatial annotation handler.

Image-only fitting and retained-receipt comparison are diagnostics, not sensing
admission. Every intermediate result is retained, including partial failures.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import time

import numpy as np

from .planar_reference import compare_annotations, reference_from_rgb


def annotate_capture(server, read_result, *, save_record, operation_count):
    observation = read_result['observation']
    binding = dict(epoch=observation['epoch'], sequence=observation['sequence'],
                   capture_sha256=read_result['capture_sha256'])
    report = {'passed': False, 'physical_admission': False,
              'capture_binding': binding, 'annotations': [],
              'started_monotonic_s': time.monotonic()}
    before = operation_count()
    try:
        # This is the exact typed object already admitted by read_sensor, not a
        # reconstruction from mutable wire metadata or a second sensor read.
        hub = server._runtime.domains['sensing'].runtime.hub
        capture = hub.retained('overview', **binding)
        payload = capture.payload
        rgb = np.frombuffer(payload.rgb8, dtype=np.uint8).reshape(payload.height, payload.width, 3)
        reference = reference_from_rgb(rgb)
        report['reference'] = asdict(reference)
        for index, pixel in enumerate(reference.pixels()):
            args = {**binding, 'pixel': list(pixel),
                    'observation_id': f'capture_{capture.sequence}_held_{index}',
                    'label': 'declared planar reference sample, not semantic detection'}
            started = time.monotonic()
            response = server.call_tool('spatial.annotate_pixel', args)
            result = json.loads(response['content'][-1]['text'])
            row = {'args': args, 'result': result, 'started_monotonic_s': started,
                   'finished_monotonic_s': time.monotonic(),
                   'added_reader_operations': operation_count() - before}
            report['annotations'].append(row)
            save_record(index, row)
            if operation_count() != before:
                raise ValueError('spatial fan-out issued another reader RPC')
        report['comparison'] = compare_annotations(reference, capture,
            [r['result'] for r in report['annotations']], admitted_max_age_s=2.)
        report['passed'] = report['comparison']['passed']
    except (ValueError, KeyError, RuntimeError, TypeError) as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['added_reader_operations'] = operation_count() - before
        report['finished_monotonic_s'] = time.monotonic()
    return report
