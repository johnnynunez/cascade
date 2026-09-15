"""Optional Isaac 6.1 NVENC camera outputs, activated by the simulator owner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from typing import Any

from camera_stream_profile import CAMERAS, StreamProfile


@dataclass
class _Output:
    product: Any
    writer: Any
    path: str
    camera_path: str
    state: str = "created"


class RtspCameraStreams:
    """Own separate compressed render products while retaining perception products.

    Call start, restart_camera, and stop only on the existing Isaac main thread.
    This module never creates a SimulationApp or advances the physics timeline.
    """

    def __init__(self, profile: StreamProfile) -> None:
        self.profile = profile
        self._outputs: dict[str, _Output] = {}
        self._stage = None
        self._state = "stopped"

    def _complete(self) -> bool:
        return (set(self._outputs) == {name for name, _ in CAMERAS}
                and all(output.state == "attached" and output.writer is not None
                        and output.product is not None for output in self._outputs.values()))

    def start(
        self,
        sensors: Mapping[str, Any],
        *,
        verify_rtsp_isolation: Callable[[tuple[int, int, int]], None],
    ) -> dict:
        """Attach after a deployment guard verifies that RTSP ports are protected.

        The guard must raise if any configured RTSP port can be reached from
        outside the host. Isaac's RTSP writer has no documented bind-address
        option; the deployment must enforce this with its firewall.
        """
        verify_rtsp_isolation(self.profile.rtsp_ports)
        if self._state == "attached" and self._complete():
            return self.status()
        if self._outputs or self._state != "stopped":
            raise RuntimeError("Call stop() to release incomplete camera outputs before starting")
        missing = [source for _, source in CAMERAS if source not in sensors]
        if missing:
            raise ValueError(f"Missing native cameras: {', '.join(missing)}")

        import omni.kit.app
        import omni.replicator.core as rep
        import omni.usd
        from pxr import UsdRender

        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("isaacsim.streaming.rtsp", True)
        from isaacsim.streaming.rtsp import RTSPStreamWriter
        from isaacsim.streaming.rtsp.impl.render_var_utils import ensure_render_var_on_product

        self._stage = omni.usd.get_context().get_stage()
        sources = {}
        native_paths = {str(sensor.render_product.GetPrim().GetPath()) for sensor in sensors.values()}
        for name, source in CAMERAS:
            product = sensors[source].render_product
            targets = product.GetCameraRel().GetTargets()
            if len(targets) != 1:
                raise RuntimeError(f"{name}: native render product must reference one camera")
            camera_path = str(targets[0])
            camera = self._stage.GetPrimAtPath(camera_path)
            tick_rate = camera.GetAttribute("omni:sensor:tickRate").Get()
            try:
                tick_rate = float(tick_rate)
            except (TypeError, ValueError, OverflowError):
                tick_rate = math.nan
            if (not math.isfinite(tick_rate) or tick_rate <= 0
                    or not math.isclose(tick_rate, 30.0, rel_tol=0, abs_tol=1e-6)):
                raise RuntimeError(f"{name}: configure the native camera at 30 Hz before enabling video")
            sources[name] = camera_path

        self._state = "starting"
        try:
            for (name, _), port in zip(CAMERAS, self.profile.rtsp_ports):
                camera_path = sources[name]
                # Compression belongs to the RenderVar, so never reuse the RGB-D product.
                product = rep.create.render_product(
                    camera=camera_path, resolution=(1280, 720), force_new=True,
                    name=f"paai_video_{name}", render_vars=[],
                )
                path = product.path
                if path in native_paths or any(output.path == path for output in self._outputs.values()):
                    raise RuntimeError(f"{name}: renderer reused an existing product")
                output = _Output(product, None, path, camera_path)
                self._outputs[name] = output
                rendered = UsdRender.Product(self._stage.GetPrimAtPath(path))
                if [str(target) for target in rendered.GetCameraRel().GetTargets()] != [camera_path]:
                    raise RuntimeError(f"{name}: streaming product references the wrong camera")
                success, _ = ensure_render_var_on_product(self._stage, path, "LdrColor", "h264")
                if not success:
                    raise RuntimeError(f"{name}: could not author the H.264 output")
                output.writer = RTSPStreamWriter(
                    port=port, mountPath=f"/{name}", encoding="h264", width=1280, height=720,
                )
                output.writer.attach([product])
                output.state = "attached"
        except BaseException as exc:
            self._state = "failed"
            try:
                self.stop()
            except Exception as cleanup:
                exc.add_note(f"Camera stream cleanup also failed: {cleanup}")
            raise
        self._state = "attached"
        return self.status()

    def restart_camera(self, name: str) -> None:
        """Recover a writer after media checks detect its documented failed state."""
        if self._state != "attached" or not self._complete():
            raise RuntimeError("Camera outputs must be completely attached before restarting a writer")
        output = self._outputs[name]
        self._state = output.state = "restarting"
        try:
            output.writer.detach()
            output.writer.attach([output.product])
        except BaseException:
            self._state = output.state = "failed"
            raise
        self._state = output.state = "attached"

    def status(self) -> dict:
        return {
            "state": "failed" if self._state == "attached" and not self._complete() else self._state,
            "live_verified": False,
            "requested_profile": {"width": 1280, "height": 720, "fps": 30, "codec": "h264"},
            "encoder_limits": {"bitrate_bps": None, "keyframe_interval_s": None},
            "cameras": {
                name: {"camera_path": output.camera_path, "render_product": output.path,
                       "state": output.state}
                for name, output in self._outputs.items()
            },
        }

    def stop(self) -> None:
        """Release owned outputs only, including after partial initialization."""
        errors = []
        self._state = "stopping"
        try:
            for name, output in list(self._outputs.items())[::-1]:
                output.state = "failed"
                try:
                    if output.writer is not None:
                        output.writer.detach()
                        output.writer = None
                    if output.product is not None:
                        output.product.destroy()
                        output.product = None
                    from pxr import Usd
                    for layer in (self._stage.GetSessionLayer(), self._stage.GetRootLayer()):
                        if not layer.GetPrimAtPath(output.path):
                            continue
                        with Usd.EditContext(self._stage, layer):
                            if not self._stage.RemovePrim(output.path):
                                raise RuntimeError(f"Could not remove video product from layer {layer.identifier}")
                        if layer.GetPrimAtPath(output.path):
                            raise RuntimeError(f"Video product remains in layer {layer.identifier}")
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
                else:
                    del self._outputs[name]
        finally:
            self._state = "failed" if self._outputs else "stopped"
            if not self._outputs:
                self._stage = None
        if errors:
            raise RuntimeError("Camera output cleanup failed: " + "; ".join(errors))
