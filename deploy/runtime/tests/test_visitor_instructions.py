"""Install current visitor guidance without altering operator-owned text."""
import importlib.util
from pathlib import Path
import stat

import pytest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    'visitor_policy_sync', ROOT / 'demo/kitchen/visitor_instructions.py')
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)


def current_block():
    return (policy.BEGIN_MARKER + '\n' + policy.SOURCE_PATH.read_text().strip()
            + '\n' + policy.END_MARKER).encode()


def test_camera_status_guidance_does_not_promote_cached_statistics_to_live_evidence():
    text = policy.SOURCE_PATH.read_text()
    assert 'cached client delivery statistics' in text
    assert 'One result cannot establish a fresh capture' in text
    assert 'camera_snapshot' in text and 'get_observation' in text
    assert 'authority for camera availability' not in text


def test_current_template_is_added_once_without_changing_operator_text(tmp_path):
    target = tmp_path / 'AGENTS.md'
    existing = 'Operator notes\r\nKeep the project attribution.\r\n'.encode()
    target.write_bytes(existing)

    assert policy.sync_visitor_instructions(tmp_path)
    assert target.read_bytes() == current_block() + b'\n\n' + existing
    before = target.stat()
    assert not policy.sync_visitor_instructions(tmp_path)
    assert target.stat().st_mtime_ns == before.st_mtime_ns
    assert target.stat().st_ino == before.st_ino


def test_current_template_replaces_only_its_managed_block_and_keeps_mode(tmp_path):
    target = tmp_path / 'AGENTS.md'
    prefix = b'# Local operator rules\r\nKeep this section.\r\n\r\n'
    suffix = b'\r\n\r\n# Other integration\r\nDo not edit this section.\r\n'
    old = (policy.BEGIN_MARKER + '\nEarlier visitor policy.\n'
           + policy.END_MARKER).encode()
    target.write_bytes(prefix + old + suffix)
    target.chmod(0o640)

    assert policy.sync_visitor_instructions(tmp_path)
    assert target.read_bytes() == prefix + current_block() + suffix
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert not policy.sync_visitor_instructions(tmp_path)


@pytest.mark.parametrize('contents', [
    policy.BEGIN_MARKER + '\nUnfinished policy.\n',
    policy.END_MARKER + '\nOperator note.\n',
    policy.END_MARKER + '\n' + policy.BEGIN_MARKER,
    policy.BEGIN_MARKER + '\n' + policy.BEGIN_MARKER + '\n' + policy.END_MARKER,
])
def test_malformed_managed_blocks_are_left_untouched(tmp_path, contents):
    target = tmp_path / 'AGENTS.md'
    original = contents.encode()
    target.write_bytes(original)

    with pytest.raises(ValueError, match='markers'):
        policy.sync_visitor_instructions(tmp_path)
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


def test_sync_does_not_follow_a_workspace_instruction_symlink(tmp_path):
    outside = tmp_path / 'operator.md'
    outside.write_text('Separate operator instructions.\n')
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    target = workspace / 'AGENTS.md'
    target.symlink_to(outside)

    with pytest.raises(ValueError, match='symlink'):
        policy.sync_visitor_instructions(workspace)
    assert target.is_symlink()
    assert outside.read_text() == 'Separate operator instructions.\n'
