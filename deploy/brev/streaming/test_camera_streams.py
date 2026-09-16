"""Exercise output ownership against fake native APIs; never load Isaac or a GPU."""

from collections import Counter
from contextlib import contextmanager
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from camera_stream_profile import CAMERAS, StreamProfile
from camera_streams import RtspCameraStreams


class _Layer:
    def __init__(self, identifier):
        self.identifier = identifier
        self.paths = set()

    def GetPrimAtPath(self, path):
        return path if path in self.paths else None


class _Prim:
    def __init__(self, path, camera_path=None, tick_rate=30.0):
        self.path, self.camera_path, self.tick_rate = path, camera_path, tick_rate

    def GetPrim(self):
        return self

    def GetPath(self):
        return self.path

    def GetCameraRel(self):
        return SimpleNamespace(GetTargets=lambda: [self.camera_path])

    def GetAttribute(self, name):
        if name != "omni:sensor:tickRate":
            raise AssertionError(name)
        return SimpleNamespace(Get=lambda: self.tick_rate)


class _Product:
    def __init__(self, sdk, name, path, native=False):
        self.sdk, self.name, self.path, self.native = sdk, name, path, native
        self.destroyed = False

    def destroy(self):
        if self.native or self.destroyed:
            raise AssertionError("Destroyed a borrowed or already released product")
        if any(writer.attached and writer.product is self for writer in self.sdk.writers):
            raise AssertionError("Destroyed a product still held by a writer")
        self.sdk.action("destroy", self.name)
        self.destroyed = True


class _Writer:
    def __init__(self, sdk, name):
        self.sdk, self.name = sdk, name
        self.product = None
        self.attached = False

    def attach(self, products):
        if len(products) != 1 or products[0].destroyed or self.attached:
            raise AssertionError("Invalid writer attachment")
        self.product, self.attached = products[0], True
        # Attachment can acquire a resource before the SDK reports failure.
        self.sdk.action("attach", self.name)

    def detach(self):
        self.sdk.action("detach", self.name)
        self.attached, self.product = False, None


class _Native:
    def __init__(self):
        self.events, self.effects = [], {}
        self.root, self.session = _Layer("root"), _Layer("session")
        self.layer = self.root
        self.prims, self.sensors, self.native_products = {}, {}, {}
        self.products, self.writers = [], []
        self.reuse, self.camera_override = {}, {}
        self.generations = Counter()
        for source in [source for _, source in CAMERAS] + ["wrist"]:
            camera_path, product_path = f"/Cameras/{source}", f"/Perception/{source}"
            self.prims[camera_path] = _Prim(camera_path)
            self.prims[product_path] = _Prim(product_path, camera_path)
            self.root.paths.add(product_path)
            self.sensors[source] = SimpleNamespace(render_product=self.prims[product_path])
            self.native_products[source] = _Product(self, source, product_path, native=True)

    def action(self, *event):
        self.events.append(event)
        queue = self.effects.get(event, [])
        effect = queue.pop(0) if queue else True
        if isinstance(effect, BaseException):
            raise effect
        return effect

    def fail(self, event, effect):
        self.effects.setdefault(event, []).append(effect)

    def guard(self, ports):
        if ports != (8554, 8555, 8556):
            raise AssertionError(ports)
        self.action("guard")

    def GetPrimAtPath(self, path):
        return self.prims[path]

    def GetSessionLayer(self):
        return self.session

    def GetRootLayer(self):
        return self.root

    def RemovePrim(self, path):
        effect = self.action("remove", self.layer.identifier, path)
        if effect is False:
            return False
        if effect != "unchanged":
            self.layer.paths.discard(path)
        return True

    @contextmanager
    def edit(self, stage, layer):
        if stage is not self:
            raise AssertionError("Wrong stage")
        previous, self.layer = self.layer, layer
        try:
            yield
        finally:
            self.layer = previous

    def create(self, *, camera, resolution, force_new, name, render_vars):
        name = name.removeprefix("paai_video_")
        if resolution != (1280, 720) or force_new is not True or render_vars != []:
            raise AssertionError("Native perception configuration was reused for video")
        self.action("create", name)
        if name in self.reuse:
            return self.reuse[name]()
        self.generations[name] += 1
        path = f"/Render/paai_video_{name}_{self.generations[name]}"
        product = _Product(self, name, path)
        self.products.append(product)
        self.prims[path] = _Prim(path, self.camera_override.get(name, camera))
        self.session.paths.add(path)
        self.root.paths.add(path)
        return product

    def ensure(self, stage, path, render_var, encoding):
        if stage is not self or render_var != "LdrColor" or encoding != "h264":
            raise AssertionError("Wrong render variable contract")
        product = next(product for product in self.products if product.path == path)
        return self.action("ensure", product.name), path + "/LdrColor"

    def writer(self, *, port, mountPath, encoding, width, height):
        name = mountPath.lstrip("/")
        if port not in (8554, 8555, 8556) or (encoding, width, height) != ("h264", 1280, 720):
            raise AssertionError("Wrong writer configuration")
        self.action("writer", name)
        writer = _Writer(self, name)
        self.writers.append(writer)
        return writer

    def modules(self):
        names = (
            "omni", "omni.kit", "omni.kit.app", "omni.replicator", "omni.replicator.core", "omni.usd",
            "pxr", "isaacsim", "isaacsim.streaming", "isaacsim.streaming.rtsp",
            "isaacsim.streaming.rtsp.impl", "isaacsim.streaming.rtsp.impl.render_var_utils",
        )
        modules = {name: ModuleType(name) for name in names}
        for name, module in modules.items():
            module.__path__ = []
            if "." in name:
                parent, attr = name.rsplit(".", 1)
                setattr(modules[parent], attr, module)
        manager = SimpleNamespace(set_extension_enabled_immediate=lambda *args: self.action("enable", *args))
        modules["omni.kit.app"].get_app = lambda: SimpleNamespace(get_extension_manager=lambda: manager)
        modules["omni.replicator.core"].create = SimpleNamespace(render_product=self.create)
        modules["omni.usd"].get_context = lambda: SimpleNamespace(get_stage=lambda: self)
        modules["pxr"].UsdRender = SimpleNamespace(Product=lambda prim: prim)
        modules["pxr"].Usd = SimpleNamespace(EditContext=self.edit)
        modules["isaacsim.streaming.rtsp"].RTSPStreamWriter = self.writer
        modules["isaacsim.streaming.rtsp.impl.render_var_utils"].ensure_render_var_on_product = self.ensure
        return modules


class CameraStreamOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.sdk = _Native()
        self.owner = RtspCameraStreams(StreamProfile("100.100.100.100", "http://100.100.100.100:8092"))
        self.modules = patch.dict(sys.modules, self.sdk.modules())
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def start(self):
        return self.owner.start(self.sdk.sensors, verify_rtsp_isolation=self.sdk.guard)

    def count(self, *event):
        return self.sdk.events.count(event)

    def assert_released(self):
        self.assertEqual(self.owner.status()["state"], "stopped")
        self.assertEqual(self.owner.status()["cameras"], {})
        self.assertTrue(all(product.destroyed for product in self.sdk.products))
        self.assertTrue(all(not writer.attached for writer in self.sdk.writers))
        self.assertFalse(any(path.startswith("/Render/") for path in self.sdk.root.paths | self.sdk.session.paths))
        self.assertEqual(self.sdk.root.paths, {product.path for product in self.sdk.native_products.values()})

    def test_complete_start_is_idempotent_and_stop_preserves_native_products(self):
        status = self.start()
        self.assertEqual(self.start(), status)
        self.assertEqual(status["state"], "attached")
        self.assertFalse(status["live_verified"])
        self.assertEqual({row["state"] for row in status["cameras"].values()}, {"attached"})
        self.assertEqual(self.count("guard"), 2)
        self.assertEqual(len(self.sdk.products), 3)
        self.owner.stop()
        self.owner.stop()
        self.assert_released()

    def test_guard_failure_prevents_native_activation(self):
        self.sdk.fail(("guard",), RuntimeError("Firewall not verified"))
        with self.assertRaisesRegex(RuntimeError, "Firewall"):
            self.start()
        self.assertEqual(self.sdk.events, [("guard",)])
        self.assert_released()

    def test_missing_camera_prevents_native_activation(self):
        del self.sdk.sensors["proof"]
        with self.assertRaisesRegex(ValueError, "Missing native cameras"):
            self.start()
        self.assertEqual(self.sdk.events, [("guard",)])
        self.assert_released()

    def test_invalid_tick_rates_allocate_no_video_products(self):
        for value in (None, "invalid", float("nan"), float("inf"), -float("inf"), 0, -30, 60):
            with self.subTest(value=value):
                self.sdk.prims["/Cameras/proof"].tick_rate = value
                with self.assertRaisesRegex(RuntimeError, "30 Hz"):
                    self.start()
                self.assert_released()
        self.assertFalse(any(event[0] == "create" for event in self.sdk.events))

    def test_attach_failure_and_interruption_release_partially_attached_writer(self):
        for error in (RuntimeError("Attach failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                self.sdk.fail(("attach", "worktop"), error)
                with self.assertRaises(type(error)):
                    self.start()
                self.assert_released()

    def test_failed_rollback_requires_release_before_new_start(self):
        self.sdk.fail(("attach", "worktop"), RuntimeError("Original attach failure"))
        self.sdk.fail(("detach", "worktop"), RuntimeError("Writer busy"))
        with self.assertRaisesRegex(RuntimeError, "Original attach failure") as raised:
            self.start()
        self.assertIn("Camera stream cleanup also failed", raised.exception.__notes__[0])
        status = self.owner.status()
        self.assertEqual(status["state"], "failed")
        self.assertEqual(list(status["cameras"]), ["worktop"])
        self.assertEqual(status["cameras"]["worktop"]["state"], "failed")
        with self.assertRaisesRegex(RuntimeError, r"Call stop\(\)"):
            self.start()
        with self.assertRaisesRegex(RuntimeError, "completely attached"):
            self.owner.restart_camera("worktop")
        self.assertEqual(len(self.sdk.products), 2)
        self.owner.stop()
        self.assert_released()
        self.assertEqual(self.start()["state"], "attached")
        self.owner.stop()
        self.assert_released()

    def test_setup_failures_release_products_without_constructed_writers(self):
        for event, effect in (("ensure", False), ("writer", RuntimeError("Construction failed"))):
            with self.subTest(event=event):
                before = self.count("detach", "worktop")
                self.sdk.fail((event, "worktop"), effect)
                with self.assertRaises(RuntimeError):
                    self.start()
                self.assertEqual(self.count("detach", "worktop"), before)
                self.assert_released()

    def test_failed_destroy_retries_product_without_repeating_detach(self):
        self.start()
        self.sdk.fail(("destroy", "worktop"), RuntimeError("Renderer busy"))
        with self.assertRaisesRegex(RuntimeError, "Renderer busy"):
            self.owner.stop()
        self.assertEqual(self.owner.status()["state"], "failed")
        self.assertEqual(list(self.owner.status()["cameras"]), ["worktop"])
        self.owner.stop()
        self.assertEqual(self.count("detach", "worktop"), 1)
        self.assertEqual(self.count("destroy", "worktop"), 2)
        self.assert_released()

    def test_layer_removal_failure_retains_specs_without_repeating_released_resources(self):
        for effect in (False, "unchanged", RuntimeError("Layer busy")):
            with self.subTest(effect=effect):
                self.start()
                product = self.sdk.products[-2]
                before = Counter(self.sdk.events)
                self.sdk.fail(("remove", "root", product.path), effect)
                with self.assertRaises(RuntimeError):
                    self.owner.stop()
                self.assertEqual(self.owner.status()["state"], "failed")
                self.assertIn(product.path, self.sdk.root.paths)
                self.owner.stop()
                for action in ("detach", "destroy"):
                    self.assertEqual(self.count(action, "worktop") - before[(action, "worktop")], 1)
                self.assertEqual(self.count("remove", "session", product.path), 1)
                self.assertEqual(self.count("remove", "root", product.path), 2)
                self.assert_released()

    def test_restart_failure_reports_failed_until_owned_outputs_are_stopped(self):
        for action in ("detach", "attach"):
            with self.subTest(action=action):
                self.start()
                self.sdk.fail((action, "worktop"), RuntimeError("Recovery failed"))
                with self.assertRaisesRegex(RuntimeError, "Recovery failed"):
                    self.owner.restart_camera("worktop")
                status = self.owner.status()
                self.assertEqual(status["state"], "failed")
                self.assertEqual(status["cameras"]["worktop"]["state"], "failed")
                self.assertEqual(status["cameras"]["side"]["state"], "attached")
                self.assertTrue(all(not product.destroyed for product in self.sdk.products[-3:]))
                with self.assertRaisesRegex(RuntimeError, r"Call stop\(\)"):
                    self.start()
                self.owner.stop()
                self.assert_released()

    def test_successful_restart_reuses_only_its_owned_product(self):
        self.start()
        self.owner.restart_camera("worktop")
        self.assertEqual(self.owner.status()["state"], "attached")
        self.assertEqual(len(self.sdk.products), 3)
        self.assertEqual(self.count("attach", "worktop"), 2)
        self.assertEqual(self.count("detach", "worktop"), 1)
        self.assertEqual(self.count("detach", "kitchen"), 0)
        self.owner.stop()
        self.assert_released()

    def test_any_reused_native_product_is_never_destroyed(self):
        for native in self.sdk.native_products.values():
            with self.subTest(path=native.path):
                self.sdk.reuse["kitchen"] = lambda: native
                with self.assertRaisesRegex(RuntimeError, "reused an existing product"):
                    self.start()
                self.assert_released()
                self.assertFalse(native.destroyed)

    def test_duplicate_owned_product_is_released_once(self):
        self.sdk.reuse["worktop"] = lambda: self.sdk.products[0]
        with self.assertRaisesRegex(RuntimeError, "reused an existing product"):
            self.start()
        self.assertEqual(self.count("destroy", "kitchen"), 1)
        self.assertEqual(len(self.sdk.products), 1)
        self.assert_released()

    def test_wrong_camera_relationship_releases_unattached_product(self):
        self.sdk.camera_override["worktop"] = "/Cameras/side"
        with self.assertRaisesRegex(RuntimeError, "wrong camera"):
            self.start()
        self.assertEqual(self.count("writer", "worktop"), 0)
        self.assert_released()

    def test_stop_interruption_retains_failed_resources_for_finite_retry(self):
        self.start()
        self.sdk.fail(("detach", "worktop"), KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            self.owner.stop()
        self.assertEqual(self.owner.status()["state"], "failed")
        self.assertEqual(set(self.owner.status()["cameras"]), {"kitchen", "worktop"})
        self.assertEqual(self.count("destroy", "worktop"), 0)
        self.owner.stop()
        self.assert_released()


if __name__ == "__main__":
    unittest.main()
