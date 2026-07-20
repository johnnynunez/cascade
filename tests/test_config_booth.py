"""WRC_BOOTH tuning overlay: configs/booth.yaml deep-merged over demo.yaml
(docs/BOOTH_RUNBOOK.md §1). Dev defaults must be untouched without the env."""

from wrc_demo.config import load_demo_config


def test_dev_defaults_without_booth_env(monkeypatch):
    monkeypatch.delenv("WRC_BOOTH", raising=False)
    cfg = load_demo_config()
    assert cfg.grasp.persist_seconds == 120
    assert cfg.grasp.max_pick_attempts == 8
    assert cfg.grasp.graspgenx.timeout_ms == 8000
    assert cfg.memory.horizon_s == 15.0
    assert cfg.perception_loop.belief_fallback_age_s == 3.0
    assert "booth_mode" not in cfg


def test_booth_overlay_applies_and_merges(monkeypatch):
    monkeypatch.setenv("WRC_BOOTH", "1")
    cfg = load_demo_config()
    assert cfg.booth_mode is True
    # the five runbook §1 overrides
    assert cfg.grasp.persist_seconds == 45
    assert cfg.grasp.max_pick_attempts == 3
    assert cfg.grasp.graspgenx.timeout_ms == 2000
    assert cfg.memory.horizon_s == 60.0
    assert cfg.perception_loop.belief_fallback_age_s == 20.0
    # deep-merge, not replace: sibling keys must survive
    assert cfg.grasp.backend == "graspgenx"
    assert cfg.grasp.graspgenx.port == 5556
    assert cfg.perception_loop.rate_hz == 3.0
    assert cfg.grasp.drop_zone == [0.18, -0.17]


def test_booth_env_falsey_values_disable(monkeypatch):
    for v in ("0", "false", "False", "", "0 ", " false", "no", "off", "OFF"):
        monkeypatch.setenv("WRC_BOOTH", v)
        cfg = load_demo_config()
        assert "booth_mode" not in cfg, f"WRC_BOOTH={v!r} must not enable booth"


def test_booth_then_dev_no_cross_contamination(monkeypatch):
    """A booth load must not leak overrides into a later dev load in the
    same process (deep-merge mutates the freshly loaded dict only)."""
    monkeypatch.setenv("WRC_BOOTH", "1")
    booth = load_demo_config()
    assert booth.grasp.persist_seconds == 45

    monkeypatch.delenv("WRC_BOOTH")
    dev = load_demo_config()
    assert dev.grasp.persist_seconds == 120
    assert dev.grasp.max_pick_attempts == 8
    assert dev.grasp.graspgenx.timeout_ms == 8000
    assert "booth_mode" not in dev
    # and the earlier booth cfg object keeps its values
    assert booth.grasp.persist_seconds == 45


def test_booth_env_with_missing_overlay_fails_loud(monkeypatch, tmp_path):
    """An explicitly requested overlay must never no-op silently: WRC_BOOTH
    with booth.yaml missing (broken checkout) raises."""
    import shutil

    import pytest

    from wrc_demo.config import CONFIG_DIR

    cdir = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, cdir)
    (cdir / "booth.yaml").unlink()
    monkeypatch.setenv("WRC_BOOTH", "1")
    with pytest.raises(FileNotFoundError, match="booth.yaml"):
        load_demo_config(config_dir=cdir)
