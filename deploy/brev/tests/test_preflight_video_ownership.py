"""Running admission requires the exact Compose files selected by the profile."""

import importlib.util
import json
from pathlib import Path
import shlex

import pytest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("video_ownership_preflight", HERE / "preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


@pytest.fixture
def owned_container(tmp_path, monkeypatch):
    root = tmp_path / "deployment-root"
    container_id = "a" * 64
    metadata = {
        "id": container_id, "running": True, "pid": 700,
        "project": "paai-demo", "service": "demo",
        "working_dir": str(root / "deployment"),
        "config_files": str(root / "deployment/compose.yaml"),
        "mounts": [
            {"Type": "bind", "Source": str(root / source), "Destination": destination}
            for source, destination in (("source", "/workspace"), ("data", "/data"),
                                        ("models", "/data/models"))
        ],
    }
    proc = tmp_path / "proc"
    for pid in (700, 701):
        path = proc / str(pid) / "cgroup"
        path.parent.mkdir(parents=True)
        path.write_text(f"0::/system.slice/docker-{container_id}.scope\n")
    monkeypatch.setattr(preflight, "PROC_ROOT", proc)
    calls = []

    def output(command, timeout=15):
        calls.append(command)
        assert command[:3] == ["docker", "--host", preflight.DOCKER_SOCKET]
        if command[3] == "ps":
            return container_id
        assert command[3] == "inspect" and command[-1] == container_id
        return json.dumps(metadata)

    monkeypatch.setattr(preflight, "output", output)
    return root, metadata, calls


@pytest.mark.parametrize("selection,files", [
    ({}, ("compose.yaml",)),
    ({"streaming": {"enabled": False}}, ("compose.yaml",)),
    ({"streaming": {"enabled": True}}, ("compose.yaml", "compose.streaming.yaml")),
], ids=["omitted-video", "disabled-video", "enabled-video"])
def test_profile_admits_exact_compose_files(owned_container, selection, files):
    root, metadata, calls = owned_container
    metadata["config_files"] = ",".join(str(root / "deployment" / name) for name in files)
    profile = {"storage": {"root": str(root)}, **selection}

    result = preflight.validate_gpu_ownership(profile, "701,python3,1200", str(root / "docker"))

    assert result["verified"] is True
    assert result["gpu_compute_pids"] == [701]
    assert result["container_id"] == metadata["id"]
    assert [command[3] for command in calls] == ["ps", "inspect", "inspect"]


@pytest.mark.parametrize("enabled,files", [
    (True, ("compose.yaml", "compose.streaming.yaml", "extra.yaml")),
    (True, ("different.yaml", "compose.streaming.yaml")),
    (True, ("compose.yaml", "different.yaml")),
    (True, ("compose.streaming.yaml", "compose.yaml")),
    (True, ("compose.yaml",)),
    (False, ("compose.yaml", "compose.streaming.yaml")),
], ids=["extra-file", "different-base", "different-overlay", "reordered-files",
        "missing-overlay", "unselected-overlay"])
def test_profile_rejects_other_compose_files(owned_container, enabled, files):
    root, metadata, calls = owned_container
    metadata["config_files"] = ",".join(str(root / "deployment" / name) for name in files)
    profile = {"storage": {"root": str(root)}, "streaming": {"enabled": enabled}}

    with pytest.raises(ValueError, match="Container metadata does not identify"):
        preflight.validate_gpu_ownership(profile, "701,python3,1200", str(root / "docker"))

    assert [command[3] for command in calls] == ["ps", "inspect"]


@pytest.mark.parametrize("enabled", ["true", 1, None])
def test_video_selection_requires_explicit_boolean(owned_container, enabled):
    root, _, calls = owned_container
    profile = {"storage": {"root": str(root)}, "streaming": {"enabled": enabled}}

    with pytest.raises(ValueError, match="explicit boolean"):
        preflight.validate_gpu_ownership(profile, "701,python3,1200", str(root / "docker"))

    assert calls == []


@pytest.fixture
def public_encoders(owned_container, monkeypatch):
    root, metadata, _ = owned_container
    profile = {"storage": {"root": str(root)}, "public_visitor": {
        "enabled": True, "video": True, "local_auth": True}}
    parent = ["/usr/bin/python3", str(root / "tools/visitor/visitor.py"),
              "--auth-file=" + str(root / "data/private/public-visitor/auth.json"), "--video"]
    unit = {"ActiveState": "active", "MainPID": "800", "User": "ubuntu",
            "ExecStart": "{ path=/usr/bin/python3 ; argv[]=" + shlex.join(parent) + " ; }",
            "ControlGroup": "/system.slice/paai-visitor.service",
            "FragmentPath": "/etc/systemd/system/paai-visitor.service", "DropInPaths": ""}
    spec = importlib.util.spec_from_file_location("preflight_test_encoder", HERE / "visitor_video.py")
    video = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(video)

    def write_process(pid, ppid, argv, executable):
        path = preflight.PROC_ROOT / str(pid)
        path.mkdir()
        (path / "stat").write_text(str(pid) + " (fixture) " + " ".join(
            ["S", str(ppid), *(["0"] * 17), "1234"]))
        (path / "cmdline").write_bytes(b"\0".join(value.encode() for value in argv) + b"\0")
        (path / "exe").symlink_to(executable)
        (path / "cgroup").write_text("0::/system.slice/paai-visitor.service\n")

    write_process(800, 1, parent, Path("/usr/bin/python3").resolve())
    commands = {}
    for pid, camera in zip((801, 802, 803), video.CAMERAS):
        commands[pid] = video.encoder_command("/usr/bin/ffmpeg", "http://127.0.0.1:8091", camera)
        write_process(pid, 800, commands[pid], "/usr/bin/ffmpeg")
    original = preflight.output

    def output(command, timeout=15):
        if command[:2] == ["systemctl", "show"]:
            assert command[2] == "paai-visitor.service"
            return "\n".join(key + "=" + value for key, value in unit.items())
        return original(command, timeout)

    monkeypatch.setattr(preflight, "output", output)
    processes = "701,python3,1200\n" + "\n".join(f"{pid},/usr/bin/ffmpeg,570" for pid in commands)
    return {"profile": profile, "root": root, "unit": unit, "commands": commands,
            "processes": processes, "write_process": write_process}


def admit_public(fixture):
    return preflight.validate_gpu_ownership(fixture["profile"], fixture["processes"],
                                            str(fixture["root"] / "docker"))


def test_exact_public_camera_children_join_owned_container_admission(public_encoders):
    result = admit_public(public_encoders)
    assert result["verified"] and result["gpu_compute_pids"] == [701, 801, 802, 803]
    assert result["public_video"] == {"service": "paai-visitor.service", "main_pid": 800,
                                       "encoder_pids": [801, 802, 803],
                                       "cameras": ["kitchen", "side", "worktop"]}


@pytest.mark.parametrize("settings", [{}, {"enabled": False, "video": True},
                                      {"enabled": True, "video": False},
                                      {"enabled": "true", "video": True}])
def test_public_encoding_is_not_admitted_without_both_explicit_flags(public_encoders, settings):
    public_encoders["profile"]["public_visitor"] = settings
    with pytest.raises(ValueError, match="outside the exact owned container"):
        admit_public(public_encoders)


@pytest.mark.parametrize("key,value", [("ActiveState", "inactive"), ("MainPID", "0"),
                                      ("User", "root"), ("ControlGroup", "/other.service"),
                                      ("DropInPaths", "/other/override.conf"),
                                      ("FragmentPath", "/other/paai-visitor.service"),
                                      ("ExecStart", "{ argv[]=/usr/bin/python3 /other/visitor.py --video ; }")])
def test_public_encoding_refuses_an_unowned_or_inactive_service(public_encoders, key, value):
    public_encoders["unit"][key] = value
    with pytest.raises(ValueError, match="expected active visitor service"):
        admit_public(public_encoders)


@pytest.mark.parametrize("flag,value", [("-i", "http://unrelated.invalid/stream/kitchen"),
                                       ("-c:v", "libx264"), ("-b:v", "9M"), ("-r", "30")])
def test_public_encoding_refuses_modified_source_codec_bitrate_or_fps(public_encoders, flag, value):
    command = public_encoders["commands"][801].copy()
    command[command.index(flag) + 1] = value
    (preflight.PROC_ROOT / "801/cmdline").write_bytes(b"\0".join(s.encode() for s in command) + b"\0")
    with pytest.raises(ValueError, match="not an owned public camera encoder"):
        admit_public(public_encoders)


@pytest.mark.parametrize("group", ["/system.slice/paai-visitor.service-extra", "/other.service",
                                   "/system.slice/paai-visitor.service/descendant"])
def test_public_encoding_requires_the_exact_service_cgroup(public_encoders, group):
    (preflight.PROC_ROOT / "801/cgroup").write_text("0::" + group + "\n")
    with pytest.raises(ValueError, match="not an owned public camera encoder"):
        admit_public(public_encoders)


def test_public_encoding_requires_direct_parentage(public_encoders):
    path = preflight.PROC_ROOT / "801/stat"
    path.write_text(path.read_text().replace("S 800 ", "S 999 "))
    with pytest.raises(ValueError, match="not an owned public camera encoder"):
        admit_public(public_encoders)


def test_public_encoding_requires_the_system_ffmpeg_binary(public_encoders, monkeypatch):
    original = preflight.os.readlink
    monkeypatch.setattr(preflight.os, "readlink", lambda path: "/tmp/ffmpeg"
                        if Path(path) == preflight.PROC_ROOT / "801/exe" else original(path))
    with pytest.raises(ValueError, match="not an owned public camera encoder"):
        admit_public(public_encoders)


def test_public_encoding_rejects_duplicate_cameras(public_encoders):
    (preflight.PROC_ROOT / "802/cmdline").write_bytes((preflight.PROC_ROOT / "801/cmdline").read_bytes())
    with pytest.raises(ValueError, match="not an owned public camera encoder"):
        admit_public(public_encoders)


def test_public_encoding_refuses_a_fourth_gpu_process(public_encoders):
    public_encoders["write_process"](804, 800, public_encoders["commands"][801], "/usr/bin/ffmpeg")
    public_encoders["processes"] += "\n804,/usr/bin/ffmpeg,570"
    with pytest.raises(ValueError, match="at most three"):
        admit_public(public_encoders)


def test_public_encoding_refuses_main_pid_changes_during_admission(public_encoders, monkeypatch):
    original = preflight.output
    calls = []

    def changed(command, timeout=15):
        result = original(command, timeout)
        if command[:2] == ["systemctl", "show"]:
            calls.append(command)
            if len(calls) == 2:
                return result.replace("MainPID=800", "MainPID=999")
        return result

    monkeypatch.setattr(preflight, "output", changed)
    with pytest.raises(ValueError, match="changed during GPU ownership"):
        admit_public(public_encoders)
