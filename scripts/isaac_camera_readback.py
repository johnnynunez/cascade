"""Own CPU annotator buffers for the JPEG/depth TCP transport.

CameraSensor's public ``out`` argument selects CPU readback in Replicator.
This avoids converting a borrowed CUDA render buffer after get_data returns;
that conversion segfaulted in depth_data.numpy() on controller/Isaac 6.1.
Rendering and simulation still execute on their selected GPU device.
"""
from __future__ import annotations


class CpuCameraReadback:
    def __init__(self, sensor):
        self._sensor = sensor
        self._buffers = {}

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
