"""Service ownership and real child HTTP recovery; no inference or robot physics."""
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from cascade.apps.conversation import parser, serve
from cascade.conversation.service import configuration


def configured(tmp_path, **values):
    path = tmp_path / "service.json"
    path.write_text(json.dumps({"version": 1, "provider_url": "ws://127.0.0.1:8765/v1/realtime",
                                "run_root": "runs", **values}))
    return path


def test_portable_configuration_has_no_cwd_or_secret_substitution(tmp_path, monkeypatch):
    path = configured(tmp_path, config_dir="profiles", token_env="TEST_PROVIDER_SECRET",
                      start_stopped=True, allow_tools=["sensing.read_sensor"])
    monkeypatch.setenv("TEST_PROVIDER_SECRET", "not-a-config-value")
    monkeypatch.chdir(tmp_path.parent)
    result = configuration(parser().parse_args(["--config", str(path), "--port", "0"]))
    assert result["run_root"] == tmp_path / "runs"
    assert result["config_dir"] == tmp_path / "profiles"
    assert result["port"] == 0 and result["start_stopped"] is True
    assert result["token_env"] == "TEST_PROVIDER_SECRET"
    assert result["service_config_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "not-a-config-value" not in repr(result)


@pytest.mark.parametrize("values", [
    {"allow_motion": "false"}, {"start_stopped": 1}, {"port": True},
    {"token_env": 123}, {"allow_tools": ["sensing.read_sensor"] * 2},
    {"allow_motion": True, "barge_in": "speech_only"}, {"run_dir": "also"},
    {"provider_url": "ws://remote.invalid/realtime"}, {"unexpected": True},
])
def test_service_rejects_ambiguous_authority_or_configuration(tmp_path, values):
    with pytest.raises(ValueError):
        configuration(parser().parse_args(["--config", str(configured(tmp_path, **values))]))


def test_duplicate_configuration_field_is_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"version":1,"allow_motion":false,"allow_motion":true}')
    with pytest.raises(ValueError, match="duplicate"):
        configuration(parser().parse_args(["--config", str(path)]))


@pytest.mark.parametrize("gateway_result", [None, {"ok": False, "complete": False}])
@pytest.mark.parametrize("startup_error", [False, True])
def test_service_closes_runtime_after_failed_gateway_and_exits_nonzero(tmp_path, monkeypatch, gateway_result, startup_error):
    import signal
    from types import SimpleNamespace
    from cascade.apps import robot_runtime
    from cascade import config
    from cascade.conversation import domain, gateway
    closed, handlers = [], {}

    class Runtime:
        stopped = True

        def close(self):
            closed.append("runtime")
            return {"ok": True}

    class Gateway:
        token = "a" * 32

        def __init__(self, *args):
            pass

        async def start(self, **kwargs):
            if startup_error:
                raise OSError("secret-provider-query")
            callback, args = handlers[signal.SIGTERM]
            callback(*args)
            return "http://127.0.0.1:8780"

        async def close(self):
            closed.append("gateway")
            if gateway_result is None:
                raise OSError("secret-provider-query")
            return gateway_result

    monkeypatch.setattr(config, "load_robot_config", lambda *a, **kw: SimpleNamespace(robot_id="fixture", as_dict=lambda: {}))
    monkeypatch.setattr(robot_runtime, "build_robot_runtime", lambda *a: (Runtime(), None))
    monkeypatch.setattr(domain, "ConversationDomain", lambda *a, **kw: SimpleNamespace(tools=()))
    monkeypatch.setattr(gateway, "ConversationGateway", Gateway)
    args = configuration(parser().parse_args([
        "--provider-url", "ws://127.0.0.1:8765/v1/realtime", "--run-dir", str(tmp_path / "run")]))
    async def run():
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "add_signal_handler", lambda sig, cb, *a: handlers.update({sig: (cb, a)}))
        monkeypatch.setattr(loop, "remove_signal_handler", lambda sig: handlers.pop(sig))
        return await serve(SimpleNamespace(**args))
    assert asyncio.run(run()) == 1
    receipt = json.loads((tmp_path / "run" / "closure.json").read_text())
    assert closed == ["gateway", "runtime"]
    assert receipt["ok"] is False and receipt["complete"] is False
    assert receipt["runtime"]["ok"] is True
    assert (tmp_path / "run" / "ready.json").exists() is not startup_error
    assert (receipt["service_error"] is not None) is startup_error
    assert "secret-provider-query" not in json.dumps(receipt)


def test_failed_provider_close_still_closes_media_and_blocks_replacement(monkeypatch):
    aiohttp = pytest.importorskip("aiohttp")
    from test_conversation_protocol import rig

    async def scenario():
        async with rig(gateway=True) as (_, runtime, _, _, _, _, gate, origin):
            async with aiohttp.ClientSession(headers={"Authorization": "Bearer " + gate.token}) as client:
                assert (await client.post(origin + "/api/session", json={"robot_id": "fixture"})).status == 200
                session = gate.session
                provider_close, media_close = session.provider.close, session.media.close
                media_closed = []

                async def fail_close():
                    await provider_close()
                    raise OSError("secret-provider-url")

                async def observe_media_close():
                    media_closed.append(True)
                    await media_close()

                monkeypatch.setattr(session.provider, "close", fail_close)
                monkeypatch.setattr(session.media, "close", observe_media_close)
                receipt = await (await client.delete(origin + "/api/session")).json()
                assert receipt["ok"] is False and receipt["complete"] is False
                assert session.done.is_set() and runtime.stopped and media_closed == [True]
                assert "secret-provider-url" not in json.dumps(receipt)
                assert (await client.post(origin + "/api/session", json={"robot_id": "fixture"})).status == 409
                closure = await gate.close()
                assert closure["ok"] is False
                assert closure["stop_worker_closed"] is True
                assert await gate.close() == closure
    asyncio.run(scenario())


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError, TimeoutError])
def test_provider_socket_failure_does_not_skip_owned_http_client(error):
    from types import SimpleNamespace
    from cascade.conversation.provider import RealtimeConfig, RealtimeWebSocket

    async def scenario():
        calls = []

        async def close_socket():
            calls.append("socket")
            raise error("secret-transport-detail")

        async def close_client():
            calls.append("client")

        provider = RealtimeWebSocket(RealtimeConfig("ws://127.0.0.1:8765/v1/realtime"))
        provider._ws = SimpleNamespace(close=close_socket)
        provider._client = SimpleNamespace(close=close_client)
        receipt = await provider.close()
        assert calls == ["socket", "client"]
        assert not receipt["ok"] and not receipt["complete"]
        assert provider._ws is provider._client is None
        assert "secret-transport-detail" not in json.dumps(receipt)
        assert (await provider.close())["ok"] is False
    asyncio.run(scenario())


def test_shutdown_refuses_session_whose_request_body_arrives_late(monkeypatch):
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web
    from test_conversation_protocol import rig

    async def scenario():
        async with rig(gateway=True) as (_, _, _, wire, _, _, gate, origin):
            entered, release = asyncio.Event(), asyncio.Event()
            original = web.Request.json

            async def delayed_body(request, *args, **kwargs):
                entered.set()
                await release.wait()
                return await original(request, *args, **kwargs)

            monkeypatch.setattr(web.Request, "json", delayed_body)
            async with aiohttp.ClientSession(headers={"Authorization": "Bearer " + gate.token}) as client:
                request = asyncio.create_task(client.post(origin + "/api/session", json={"robot_id": "fixture"}))
                await asyncio.wait_for(entered.wait(), 3)
                closing = asyncio.create_task(gate.close())
                await asyncio.sleep(0)  # run the close admission fence, then release this body
                release.set()
                response = await asyncio.wait_for(request, 3)
                assert response.status == 503
                assert (await asyncio.wait_for(closing, 3))["ok"]
                assert gate.session is None and wire.sessions == []
    asyncio.run(scenario())


def test_shutdown_closes_live_media_and_provider_before_success():
    aiohttp = pytest.importorskip("aiohttp")
    from test_conversation_protocol import rig

    async def scenario():
        async with rig(gateway=True) as (_, runtime, _, _, _, _, gate, origin):
            async with aiohttp.ClientSession(headers={"Authorization": "Bearer " + gate.token}) as client:
                binding = await (await client.post(origin + "/api/session", json={"robot_id": "fixture"})).json()
                async with client.ws_connect(origin + "/api/media?ticket=" + binding["ticket"]) as ws:
                    read = asyncio.create_task(ws.receive())
                    receipt = await asyncio.wait_for(gate.close(), 3)
                    assert (await asyncio.wait_for(read, 3)).type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED}
                    assert receipt["ok"] and receipt["complete"]
                    assert runtime.stopped and gate.session.done.is_set()
                    assert gate.session.provider._ws is None and gate._runner.addresses == []
    asyncio.run(scenario())


def test_child_service_restart_uses_new_private_run_and_requires_explicit_reset(tmp_path):
    aiohttp = pytest.importorskip("aiohttp")
    from test_conversation_protocol import call, created, rig, terminal
    repo = Path(__file__).resolve().parents[1]
    profiles = tmp_path / "profiles"
    for directory, name in (("robots", "conversation_mock"), ("llm", "mock")):
        target = profiles / directory / (name + ".yaml")
        target.parent.mkdir(parents=True)
        target.write_bytes((repo / "configs" / directory / target.name).read_bytes())

    async def scenario():
        async with rig(gateway=True) as (_, _, _, wire, _, _, fixture_gate, _):
            path = configured(tmp_path, config_dir="profiles", port=0, start_stopped=True,
                              allow_tools=["sensing.read_sensor"],
                              provider_url=fixture_gate.provider_factory().config.url)
            tokens, runs = [], []
            for iteration in range(2):
                # The fixture's observation queue outlives its WebSocket; do
                # not mistake the closed connection's continuation for input.
                while not wire.messages.empty():
                    wire.messages.get_nowait()
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "cascade.apps.conversation", "--config", str(path), cwd=tmp_path,
                    env={**os.environ, "PYTHONPATH": str(repo / "src")},
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                try:
                    line = await asyncio.wait_for(process.stdout.readline(), 15)
                    assert line.startswith(b"Open "), line
                    url, token = line.decode().strip()[5:].split("/#")
                    tokens.append(token)
                    current = [p for p in (tmp_path / "runs").iterdir() if p not in runs]
                    assert len(current) == 1
                    run = current[0]
                    runs.append(run)
                    ready = json.loads((run / "ready.json").read_text())
                    assert ready["origin"] == url and ready["runtime_stopped"] is True
                    assert ready["provider_connected"] is False and ready["physical_admission"] is False
                    assert run.stat().st_mode & 0o077 == 0
                    assert token not in (run / "ready.json").read_text()
                    async with aiohttp.ClientSession(headers={"Authorization": "Bearer " + token}) as client:
                        if iteration:
                            rejected = await client.get(url + "/api/status", headers={"Authorization": "Bearer " + tokens[0]})
                            assert rejected.status == 403
                        status = await (await client.get(url + "/api/status")).json()
                        assert status["stopped"] is True and status["ready"] is False
                        assert (await (await client.post(url + "/api/reset", json={"generation": status["generation"]})).json())["ok"]
                        binding = await (await client.post(url + "/api/session", json={"robot_id": "conversation_mock"})).json()
                        async with client.ws_connect(url + "/api/media?ticket=" + binding["ticket"]) as ws:
                            await ws.send_json({"type": "text", "session_id": binding["session_id"], "text": "read the IMU"})
                            request = await wire.next("response.create")
                            await created(wire, request=request)
                            await call(wire, args={"sensor_id": "imu"})
                            await terminal(wire)
                            output = await wire.next("conversation.item.create")
                            result = json.loads(output["item"]["output"])
                            assert result["ok"] and result["observation"]["measurement_kind"] == "synthetic"
                            # Stop remains a separate HTTP path; the old connection cannot regain authority.
                            stop = await (await client.post(url + "/api/stop")).json()
                            assert stop["ok"]
                            assert (await (await client.get(url + "/api/status")).json())["stopped"]
                finally:
                    if process.returncode is None:
                        process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), 15)
                    except TimeoutError:
                        process.kill()
                        await process.wait()
                        raise
                assert process.returncode == 0, (await process.stderr.read()).decode()
                receipt = json.loads((run / "closure.json").read_text())
                assert receipt["ok"] and receipt["complete"]
                assert receipt["signals"] == ["SIGTERM"] and receipt["physical_rest_verified"] is False
            assert tokens[0] != tokens[1] and runs[0] != runs[1]
            assert all((run / "closure.json").exists() for run in runs)
    asyncio.run(scenario())
