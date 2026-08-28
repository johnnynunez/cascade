"""Device resolution and the profile machinery that makes the framework portable.

Both of these exist so a config can travel between an SO-101 on a laptop and a
reBot on a DGX without being edited, so the failure they guard against is not a
crash -- it is a config that silently means something different on the new host.
"""

from __future__ import annotations

import pytest

from cascade import device as dev
from cascade.config import CONFIG_DIR, load_demo_config, load_profile


def _clear_caches() -> None:
    # `_torch` may currently BE a monkeypatched plain function (monkeypatch's
    # own finalizer runs after this fixture's teardown), so clearing is
    # best-effort on that one.
    for fn in (dev._torch, dev.best_device):
        clear = getattr(fn, "cache_clear", None)
        if clear is not None:
            clear()


@pytest.fixture(autouse=True)
def _clear_probe_caches(monkeypatch):
    """`_torch` and `best_device` are lru_cached for startup cost; tests that
    fake a host must not inherit a previous test's cached answer."""
    monkeypatch.delenv(dev.DEVICE_ENV, raising=False)
    _clear_caches()
    yield
    _clear_caches()


class FakeTorch:
    """Stands in for torch on a host we are pretending to be."""

    class _Backends:
        class _MPS:
            def __init__(self, ok):
                self._ok = ok

            def is_available(self):
                return self._ok

    class _Cuda:
        def __init__(self, n):
            self._n = n

        def is_available(self):
            return self._n > 0

        def device_count(self):
            return self._n

        def get_device_name(self, i):
            return f"FakeGPU{i}"

    def __init__(self, cuda_devices=0, mps=False):
        self.cuda = self._Cuda(cuda_devices)
        self.backends = self._Backends()
        self.backends.mps = self._Backends._MPS(mps)
        self.__version__ = "9.9.9"

        class _V:
            cuda = "13.0"
            hip = None

        self.version = _V()


def fake_host(monkeypatch, **kw):
    monkeypatch.setattr(dev, "_torch", lambda: FakeTorch(**kw))
    dev.best_device.cache_clear()


# ── device resolution ────────────────────────────────────────────────────


def test_cpu_only_host_resolves_auto_to_cpu(monkeypatch):
    monkeypatch.setattr(dev, "_torch", lambda: None)
    dev.best_device.cache_clear()
    assert dev.resolve_device("auto") == "cpu"


def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch):
    fake_host(monkeypatch, cuda_devices=2, mps=True)
    assert dev.resolve_device("auto") == "cuda:0"
    fake_host(monkeypatch, cuda_devices=0, mps=True)
    assert dev.resolve_device("auto") == "mps"
    fake_host(monkeypatch, cuda_devices=0, mps=False)
    assert dev.resolve_device("auto") == "cpu"


def test_an_absent_explicit_device_degrades_instead_of_raising(monkeypatch, caplog):
    """The booth rule: `device: cuda:0` opened on a laptop costs a warning and a
    slow run, never a dead session."""
    fake_host(monkeypatch, cuda_devices=0, mps=False)
    with caplog.at_level("WARNING"):
        assert dev.resolve_device("cuda:0", what="detector") == "cpu"
    assert "detector" in caplog.text and "not available" in caplog.text


def test_a_gpu_index_this_host_does_not_have_degrades(monkeypatch):
    """`cuda:3` is a config written for a bigger machine."""
    fake_host(monkeypatch, cuda_devices=1)
    assert dev.resolve_device("cuda:3") == "cuda:0"
    assert dev.resolve_device("cuda:0") == "cuda:0"


def test_present_explicit_device_is_passed_through_untouched(monkeypatch):
    fake_host(monkeypatch, cuda_devices=4)
    assert dev.resolve_device("cuda:2") == "cuda:2"


def test_env_var_overrides_every_profile(monkeypatch):
    """CASCADE_DEVICE=cpu is the documented way to rule out a GPU problem
    without editing YAML, so it must beat an explicit config value."""
    fake_host(monkeypatch, cuda_devices=2)
    monkeypatch.setenv(dev.DEVICE_ENV, "cpu")
    assert dev.resolve_device("cuda:0") == "cpu"
    assert dev.resolve_device("auto") == "cpu"


def test_unknown_backends_are_taken_at_face_value(monkeypatch):
    """Refusing a backend we cannot probe (xpu, hpu) would be worse than trying
    it -- the user asked for it explicitly."""
    fake_host(monkeypatch, cuda_devices=0)
    assert dev.resolve_device("xpu:0") == "xpu:0"


def test_rocm_reports_itself_as_cuda(monkeypatch):
    """ROCm rides the torch.cuda API, and "cuda:0" is what PyTorch wants to be
    told there. A `device: rocm` would be wrong."""
    torch = FakeTorch(cuda_devices=1)
    torch.version.hip = "6.2"
    monkeypatch.setattr(dev, "_torch", lambda: torch)
    dev.best_device.cache_clear()
    assert dev.best_device() == "cuda:0"
    assert "ROCm 6.2" in dev.host_summary()


def test_host_summary_never_raises_without_torch(monkeypatch):
    monkeypatch.setattr(dev, "_torch", lambda: None)
    assert "torch absent" in dev.host_summary()


def test_shipped_configs_do_not_pin_an_accelerator():
    """A hardcoded cuda:0 anywhere in configs/ reintroduces the exact problem
    the device layer exists to remove."""
    offenders = []
    for path in CONFIG_DIR.rglob("*.yaml"):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "device:" in stripped and "cuda" in stripped:
                offenders.append(f"{path.relative_to(CONFIG_DIR)}:{n}: {stripped}")
    assert not offenders, "use `device: auto`:\n" + "\n".join(offenders)


# ── profile inheritance and overrides ────────────────────────────────────


def test_extends_pulls_the_parent_in_underneath():
    """The SO-101 ships as mock/MuJoCo/serial variants of ONE physical arm; the
    kinematic constants must come from a single file or the copies drift."""
    base = load_profile("arms", "so101")
    mock = load_profile("arms", "so101_mock")
    assert mock.get("type") == "mock"          # the child's own value wins
    assert base.get("type") == "so101"         # the parent is untouched
    for key in ("model", "ee_frame", "n_joints", "home_q", "tool_axis_order"):
        assert mock.get(key) == base.get(key), key


def test_extends_merges_nested_blocks_rather_than_replacing_them():
    """so101_mock restates only part of `gripper:`; a replace would drop
    max_width_m and silently let it grasp things the jaw cannot hold."""
    mock = load_profile("arms", "so101_mock")
    assert mock.get("gripper").get("max_width_m") is not None
    assert mock.get("gripper").get("open_pos") == 0.60


def test_camera_profiles_can_extend_too():
    small = load_profile("cameras", "mock_small")
    full = load_profile("cameras", "mock")
    assert small.get("type") == "mock"
    assert small.get("table_depth_m") == full.get("table_depth_m")
    assert small.get("box_px") != full.get("box_px")
    # inherited from mock: the synthetic renderer cannot witness motion
    assert small.get("static_scene") is True


def test_an_extends_cycle_is_reported_not_hung(tmp_path):
    (tmp_path / "arms").mkdir()
    (tmp_path / "arms" / "a.yaml").write_text("extends: b\n")
    (tmp_path / "arms" / "b.yaml").write_text("extends: a\n")
    with pytest.raises(ValueError, match="cycle"):
        load_profile("arms", "a", tmp_path)


def test_a_missing_parent_names_what_is_available(tmp_path):
    (tmp_path / "arms").mkdir()
    (tmp_path / "arms" / "a.yaml").write_text("extends: nope\n")
    with pytest.raises(FileNotFoundError, match="available"):
        load_profile("arms", "a", tmp_path)


def test_arm_overrides_restate_rig_geometry_over_demo_defaults():
    """demo.yaml describes a reBot; the SO-101's top-down envelope is smaller
    than that arm's, so inheriting the defaults would put every grasp target
    outside what its wrist can orient."""
    rebot = load_demo_config(camera="mock", arm="mock", llm="mock")
    so101 = load_demo_config(camera="mock", arm="so101_mock", llm="mock")

    assert so101.grasp.get("topdown_z_max") < rebot.grasp.get("topdown_z_max")
    assert so101.safety.get("max_joint_vel") < rebot.safety.get("max_joint_vel")
    assert so101.safety.get("workspace").get("max")[0] < \
        rebot.safety.get("workspace").get("max")[0]
    # An override block must not leak into the arm profile itself.
    assert so101.arm.get("overrides") is None or "safety" not in so101.arm.as_dict()


def test_overrides_do_not_persist_across_loads():
    """Profiles are loaded fresh per call; a mutated module-level dict would
    make the SECOND arm in a session inherit the first one's geometry."""
    load_demo_config(camera="mock", arm="so101_mock", llm="mock")
    rebot = load_demo_config(camera="mock", arm="mock", llm="mock")
    assert rebot.grasp.get("topdown_z_max") > 0.05
    assert rebot.arm.get("n_joints") in (None, 6)
