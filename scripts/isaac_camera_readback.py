"""Own CPU annotator buffers for the JPEG/depth TCP transport.

CameraSensor's public ``out`` argument selects CPU readback in Replicator.
This avoids converting a borrowed CUDA render buffer after get_data returns;
that conversion segfaulted in depth_data.numpy() on controller/Isaac 6.1.
Rendering and simulation still execute on their selected GPU device.
"""
from __future__ import annotations


class CpuCameraReadback:
    def __init__(self, sensor, *, render_times=None, render_product_id=None):
        self._sensor = sensor
        self._buffers = {}
        self._render_times = dict(render_times or {})
        self.render_product_id = render_product_id

    def get_render_times(self):
        """Public SDK annotator outputs for this exact render product."""
        return {name: annotator.get_data() for name, annotator in self._render_times.items()}

    def detach_render_times(self):
        for annotator in self._render_times.values():
            annotator.detach()

    def get_data_bound(self, annotator, *, checkpoint=lambda: None):
        """Bracket one AOV read with this product's capture reference.

        CameraSensor RGB/depth arrays have no per-AOV timestamp in their info.
        This records the public product reference around each read, without
        updating the app or manufacturing an independent pixel clock.
        """
        import copy
        checkpoint()
        before = self.get_render_times()
        checkpoint()
        before = copy.deepcopy(before)
        data, info = self.get_data(annotator)
        checkpoint()
        after = self.get_render_times()
        checkpoint()
        after = copy.deepcopy(after)
        if before != after:
            raise RuntimeError('render product changed during AOV readback')
        return data, info, after

    def __getattr__(self, name):
        return getattr(self._sensor, name)

    def get_data(self, annotator):
        import warp as wp

        formats = {"rgb": (3, wp.uint8), "rgba": (4, wp.uint8),
                   "distance_to_image_plane": (1, wp.float32),
                   "instance_id_segmentation": (1, wp.uint32)}
        channels, dtype = formats[annotator]
        shape = (*self._sensor.resolution, channels)
        out = self._buffers.get(annotator)
        if out is None or out.shape != shape:
            out = wp.empty(shape, dtype=dtype, device="cpu")
            self._buffers[annotator] = out
        return self._sensor.get_data(annotator, out=out)
