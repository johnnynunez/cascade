"""Real runtime/gateway activation with synthetic owners; no model or microphone."""
import asyncio
import threading
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from cascade.apps.robot_runtime import DomainAdapter
from cascade.conversation.activation import BoundedHandEpisode, build_bounded_hand_service
from cascade.conversation.domain import ConversationDomain
from cascade.conversation.gateway import ConversationGateway
from cascade.conversation.provider import RealtimeConfig, RealtimeWebSocket
from cascade.robotics.contracts import ResourceDescriptor
from cascade.robotics.runtime import RobotRuntime


class Controller:
    owner_wall_s = 30.

    def __init__(self, clock):
        self.clock = clock
        self.starts = self.stops = self.closes = 0
        self.fault = False

    def start(self):
        self.starts += 1
        self._started = self.clock()

    def stop(self):
        self.stops += 1
        return {"ok": True, "physical_stop_verified": False}

    def read(self):
        if self.fault:
            raise ValueError("retained owner fault")
        return (object(),)

    def close(self):
        self.closes += 1
        return {"ok": True, "owner_thread_closed": True, "physical_stop_verified": False}


def fixture(*, clock=time.monotonic, before=None, after=None, reset=True):
    resource = ResourceDescriptor("hand/fingers", "articulated_hand", "fixture", synthetic=True,
        controller_id="fixture/controller", writer_id="hand/fingers", admission="software_only")
    spec = {"name": "set_hand_posture", "description": "Synthetic posture fixture",
            "parameters": {"type": "object", "properties": {"posture": {"type": "string"}},
                           "required": ["posture"], "additionalProperties": False}}
    descriptor = DomainAdapter("hand", {"kind": "hand", "model_identity_sha256": "a"*64},
                               (resource,), (spec,), {"set_hand_posture"})
    controller = Controller(clock)
    calls = []
    def build(start):
        calls.append("build")
        if before:
            before()
        start(controller)
        if after:
            after()
        return SimpleNamespace(controller=controller, close=controller.close,
            reset_stop=lambda: {"ok": reset}, stop=controller.stop,
            execute=lambda name, args: {"ok": True, "synthetic": True,
                                        "postcondition": {"status": "unverified"}})
    episode = BoundedHandEpisode(descriptor, build, clock=clock)
    descriptor.runtime = episode
    runtime = RobotRuntime({"hand": descriptor})
    episode.runtime = runtime
    return episode, runtime, controller, calls


def test_passive_catalog_stop_and_reset_never_construct_an_owner():
    episode, runtime, controller, calls = fixture()
    try:
        assert runtime.execute("list_resources")["ok"]
        assert not runtime.execute("hand.set_hand_posture", {"posture": "index_flex"})["ok"]
        stopped = runtime.stop()
        assert stopped["ok"] and stopped["domains"]["hand"]["owner_started"] is False
        assert not runtime.reset_stop(expected_generation=runtime.cancellation_token)["ok"]
        assert calls == [] and controller.starts == 0
    finally:
        assert runtime.close()["ok"]


def test_activation_uses_real_reset_and_reconnect_cannot_renew_owner_clock():
    base = time.monotonic()
    now = [base]
    episode, runtime, controller, calls = fixture(clock=lambda: now[0])
    try:
        runtime.stop()
        observed = runtime.cancellation_token
        receipt = episode.activate(observed, deadline=base+10.)
        assert receipt["ok"] and receipt["reset"]["ok"]
        assert runtime.cancellation_token == observed + 1
        assert receipt["owner_started_monotonic_s"] == base
        assert receipt["owner_deadline_monotonic_s"] == base+30.
        assert runtime.execute("hand.set_hand_posture", {"posture": "index_flex"})["ok"]
        runtime.stop()
        assert runtime.reset_stop(expected_generation=runtime.cancellation_token)["ok"]
        with pytest.raises(ValueError):
            episode.activate(runtime.cancellation_token, deadline=base+10.)
        assert calls == ["build"] and controller.starts == 1
        now[0] = base+30.
        with pytest.raises(ValueError):
            episode.check_live()
        assert episode.status()["owner_deadline_monotonic_s"] == base+30.
    finally:
        runtime.close()


@pytest.mark.parametrize("phase", ["before", "after"])
def test_stop_during_construction_fences_start_or_closes_late_owner(phase):
    entered, release = threading.Event(), threading.Event()
    def block():
        entered.set()
        assert release.wait(2)
    episode, runtime, controller, calls = fixture(**{phase: block})
    errors = []
    def activate():
        try:
            episode.activate(runtime.cancellation_token, deadline=time.monotonic()+10.)
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=activate)
    thread.start()
    try:
        assert entered.wait(2)
        assert runtime.stop()["ok"]
        release.set(); thread.join(2)
        assert not thread.is_alive() and len(errors) == 1
        assert controller.starts == (phase == "after")
        if phase == "after":
            assert controller.closes == 1
        assert episode.status()["fault"] and not episode.status()["active"]
        with pytest.raises(ValueError):
            episode.activate(runtime.cancellation_token, deadline=time.monotonic()+10.)
    finally:
        release.set(); thread.join(2); runtime.close()


@pytest.mark.parametrize("mutation", ["profile", "deadline", "reset", "partial_start"])
def test_failed_activation_is_sticky_and_retains_closed_owner(mutation):
    base = time.monotonic()
    now = [base]
    episode, runtime, controller, _ = fixture(clock=lambda: now[0], reset=mutation != "reset")
    if mutation == "partial_start":
        start = controller.start
        def partial_start():
            start()
            raise KeyboardInterrupt("after actual start")
        controller.start = partial_start
    elif mutation in {"profile", "deadline"}:
        build = episode.builder
        def changed(start):
            result = build(start)
            if mutation == "profile":
                episode.descriptor.profile["model_identity_sha256"] = "b"*64
            else:
                now[0] = base+10.
            return result
        episode.builder = changed
    try:
        with pytest.raises((ValueError, KeyboardInterrupt)):
            episode.activate(runtime.cancellation_token, deadline=base+10.)
        assert not episode.status()["active"] and episode.status()["fault"]
        assert controller.starts == 1 and controller.closes == 1
    finally:
        runtime.close()


def test_close_while_constructor_is_pending_cannot_publish_or_recreate():
    entered, release = threading.Event(), threading.Event()
    def blocked():
        entered.set()
        assert release.wait(2)
    episode, runtime, controller, _ = fixture(before=blocked)
    outcomes = []
    def start():
        try:
            episode.activate(0, deadline=time.monotonic()+10.)
        except BaseException as exc:
            outcomes.append(type(exc).__name__)
    producer = threading.Thread(target=start); producer.start()
    assert entered.wait(2)
    closed = []
    closing = threading.Event()
    stop = episode.stop
    def observe_closing():
        closing.set()
        return stop()
    episode.stop = observe_closing
    closer = threading.Thread(target=lambda: closed.append(episode.close())); closer.start()
    # The condition protects the transition; no scheduler-dependent sleep.
    assert closing.wait(2)
    with episode._condition:
        assert episode._closed
    release.set(); producer.join(2); closer.join(2)
    assert outcomes == ["ValueError"] and closed and not producer.is_alive() and not closer.is_alive()
    assert controller.starts == 0 and episode.status()["active"] is False
    runtime.close()


def test_wrong_builder_owner_closes_both_and_fault_remains_in_final_receipt():
    episode, runtime, controller, _ = fixture()
    other = Controller(time.monotonic)
    build = episode.builder
    def wrong(start):
        result = build(start)
        result.controller = other
        result.close = other.close
        return result
    episode.builder = wrong
    with pytest.raises(ValueError, match="another owner"):
        episode.activate(0, deadline=time.monotonic()+10.)
    assert controller.closes == other.closes == 1
    closure = episode.close()
    assert closure["ok"] is False and closure["episode"]["fault"]
    assert closure["episode"]["owner_started_monotonic_s"] is not None
    assert episode.close() == closure
    runtime.close()


def test_completed_close_is_idempotent_without_reinvoking_owned_domain():
    episode, runtime, controller, _ = fixture()
    episode.activate(0, deadline=time.monotonic()+10.)
    first = episode.close()
    stopped = controller.stops
    assert first["ok"] and first["episode"]["closed"]
    assert first["episode"]["remaining_s"] == 0.
    assert episode.close() == first
    assert controller.closes == 1 and controller.stops == stopped
    runtime.close()


def test_real_passive_hand_composition_has_no_native_constructor(tmp_path, monkeypatch):
    from cascade.config import Cfg
    from cascade.apps import hand_runtime
    monkeypatch.setattr(hand_runtime, "prepare_hand_model", lambda *_: pytest.fail("native constructor"))
    cfg = Cfg({"robot_id": "leap_hand_right", "domains": {"hand": {
        "kind": "hand", "recipe": "leap_right_bounded_free_motion_v2",
        "asset_root": str(tmp_path), "model_identity_sha256": "a"*64}}})
    runtime, episode = build_bounded_hand_service(cfg, tmp_path)
    try:
        assert runtime.execute("list_resources")["ok"]
        assert not episode.status()["attempted"]
    finally:
        assert runtime.close()["ok"]


@asynccontextmanager
async def gateway_fixture(**options):
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web
    from test_conversation_protocol import Wire
    episode, runtime, controller, calls = fixture(**options)
    domain = ConversationDomain(runtime, robot_id="fixture", allow_tools=("hand.set_hand_posture",), allow_motion=True)
    wire = Wire()
    app = web.Application(); app.router.add_get("/v1/realtime", wire.handle)
    runner = web.AppRunner(app); await runner.setup(); await web.TCPSite(runner, "127.0.0.1", 0).start()
    provider_url = f"ws://127.0.0.1:{runner.addresses[0][1]}/v1/realtime"
    gate = ConversationGateway(domain, lambda: RealtimeWebSocket(RealtimeConfig(provider_url)), activation=episode)
    gate._test_wire = wire
    origin = await gate.start(port=0)
    try:
        async with aiohttp.ClientSession(headers={"Authorization": "Bearer "+gate.token}) as client:
            response = await client.post(origin+"/api/session", json={"robot_id": "fixture"})
            binding = await response.json()
            ws = await client.ws_connect(origin+"/api/media?ticket="+binding["ticket"])
            yield episode, runtime, controller, calls, gate, client, origin, binding, ws
    finally:
        await gate.close(); runtime.close(); await runner.cleanup()


def test_real_provider_preparation_and_mic_message_do_not_activate_until_explicit_start():
    async def run():
        async with gateway_fixture() as (episode, runtime, controller, calls, gate, client, origin, binding, ws):
            body = {"session_id": binding["session_id"], "generation": 0, "capture_id": "a"*32}
            assert (await client.post(origin+"/api/robot/start", json=body)).status == 409
            await ws.send_json({"type": "microphone_prepared", "session_id": binding["session_id"], "capture_id": "a"*32})
            assert (await ws.receive_json())["type"] == "microphone_prepared"
            assert calls == [] and controller.starts == 0
            old = gate.session
            response = await client.post(origin+"/api/robot/start", json=body)
            result = await response.json()
            assert response.status == 200 and result["ok"] and result["reconnect_required"]
            assert old.closed and old.authority_revoked and controller.starts == 1
            first_deadline = episode.status()["owner_deadline_monotonic_s"]
            fresh = await client.post(origin+"/api/session", json={"robot_id": "fixture"})
            assert fresh.status == 200 and (await fresh.json())["session_id"] != old.session_id
            assert gate.session.runtime_generation == runtime.cancellation_token
            assert episode.status()["owner_deadline_monotonic_s"] == first_deadline
            assert calls == ["build"]
            assert (await client.post(origin+"/api/robot/start", json=body)).status == 409
    asyncio.run(run())


@pytest.mark.parametrize("field,value", [("generation", True), ("generation", 10),
                                        ("session_id", "old"), ("capture_id", "b"*32)])
def test_activation_http_rejects_stale_or_ambiguous_authority(field, value):
    async def run():
        async with gateway_fixture() as (_, _, controller, calls, _, client, origin, binding, ws):
            await ws.send_json({"type": "microphone_prepared", "session_id": binding["session_id"], "capture_id": "a"*32})
            await ws.receive_json()
            body = {"session_id": binding["session_id"], "generation": 0, "capture_id": "a"*32, field: value}
            assert (await client.post(origin+"/api/robot/start", json=body)).status == 409
            assert calls == [] and controller.starts == 0
    asyncio.run(run())


@pytest.mark.parametrize("phase", ["before", "after"])
def test_real_http_disconnect_cancels_pending_constructor_even_when_old_session_is_closed(phase):
    entered, release = threading.Event(), threading.Event()
    def blocked():
        entered.set()
        assert release.wait(3)
    async def run():
        async with gateway_fixture(**{phase: blocked}) as (episode, runtime, controller, _, gate, client, origin, binding, ws):
            await ws.send_json({"type": "microphone_prepared", "session_id": binding["session_id"], "capture_id": "a"*32})
            await ws.receive_json()
            request = asyncio.create_task(client.post(origin+"/api/robot/start", json={
                "session_id": binding["session_id"], "generation": 0, "capture_id": "a"*32}))
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                assert gate.session.closed
                observed = runtime.cancellation_token
                reset = await client.post(origin+"/api/reset", json={"generation": observed})
                assert reset.status == 409 and runtime.cancellation_token == observed
                disconnected = await (await client.delete(origin+"/api/session")).json()
                assert disconnected["activation_cancelled"] is True
                assert runtime.cancellation_token > observed
                release.set()
                assert (await request).status == 409
                assert episode.status()["fault"] and not episode.status()["active"]
                assert controller.starts == (phase == "after")
                assert controller.closes == (phase == "after")
                assert (await client.post(origin+"/api/session", json={"robot_id": "fixture"})).status == 409
            finally:
                release.set(); await request
    asyncio.run(run())


def test_chromium_start_uses_real_worklet_and_provider_connection_with_synthetic_audio(tmp_path):
    import wave
    from pathlib import Path
    playwright = pytest.importorskip("playwright.async_api")
    wav = tmp_path / "synthetic-microphone.wav"
    with wave.open(str(wav), "wb") as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(24000)
        stream.writeframes(b"\x00\x00" * 24000)

    async def run():
        async with gateway_fixture() as (episode, _, controller, calls, gate, _, origin, _, preliminary_ws):
            # The fixture's first TCP session is retired before the real page
            # owns a new one. No model, real input device or robot is opened.
            await preliminary_ws.close()
            await gate.session.close("fixture_browser_handoff")
            async with playwright.async_playwright() as api:
                if not Path(api.chromium.executable_path).is_file():
                    pytest.skip("Chromium is an optional local browser fixture")
                browser = await api.chromium.launch(headless=True, args=[
                    "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                    "--use-file-for-fake-audio-capture="+str(wav)])
                try:
                    context = await browser.new_context()
                    page = await context.new_page(); page.set_default_timeout(5000)
                    errors = []; page.on("pageerror", lambda error: errors.append(str(error)))
                    await page.goto(origin+"/#"+gate.token)
                    await page.locator("#connect").click()
                    await playwright.expect(page.locator("#mic")).to_be_enabled()
                    old = gate.session
                    await page.locator("#mic").click()
                    await playwright.expect(page.locator("#start-robot")).to_be_enabled()
                    assert controller.starts == 0 and calls == []
                    assert gate._test_wire.messages.empty(), "prepared audio must not be transmitted"
                    await page.locator("#start-robot").click()
                    await playwright.expect(page.locator("#mic")).to_have_text("Mute microphone")
                    event = await gate._test_wire.next("input_audio_buffer.append")
                    assert event["audio"] and controller.starts == 1 and calls == ["build"]
                    assert gate.session is not old and old.closed and old.authority_revoked
                    assert episode.status()["active"] and episode.status()["owner_lifetime_s"] == 30.
                    assert not errors
                    await page.locator("#disconnect").click()
                    await playwright.expect(page.locator("#connect")).to_be_enabled()
                    assert controller.starts == 1
                    await context.close()
                finally:
                    await browser.close()
    asyncio.run(run())
