"""Run the real bridge loop with deterministic Kit/time boundaries, not Isaac."""
import ast
from types import SimpleNamespace
import threading

import pytest

from conftest import REPO, load_isaac_bridge_definitions


def run_loop(update_times, *, job_times=None, capture_time=0., step=0,
             initial_anchor=None, initial_frame_time=0., nested_job=None,
             capture_results=None):
    clock = [0.]
    events, captures, observed_ages = [], [], []
    frame_time = [initial_frame_time]
    update_started = [0.]
    index = [0]
    job_times = job_times or [0.] * len(update_times)
    results = iter(capture_results) if capture_results is not None else None
    env = dict(time=SimpleNamespace(monotonic=lambda: clock[0]),
               args=SimpleNamespace(cam_every=6), _last_camera_capture_started=initial_anchor,
               _annotators={'cam0': None},
               _tl=SimpleNamespace(is_playing=lambda: True), _was_playing=True,
               _REQUIRE_CUDA=False, _state_lock=threading.Lock(),
               _targets=dict(q=None, grip_frac=None, stopped=True), step=step)
    load_isaac_bridge_definitions({"_camera_capture_due"}, env)

    def wrist():
        events.append(("wrist", index[0]))

    def update():
        events.append(("update", index[0]))
        update_started[0] = clock[0]
        clock[0] += update_times[index[0]]
        index[0] += 1

    def capture():
        assert events[-2:] == [("wrist", index[0] - 1), ("update", index[0] - 1)]
        events.append(("capture", index[0] - 1))
        if results is None or next(results):
            frame_time[0] = update_started[0]  # Conservative snapshot anchor.
            env['_pending_camera_publications'].discard('cam0')
        # This scheduling model has no renderer latency; it cannot certify
        # actual camera freshness. Real history/readback is tested separately.
        captures.append((env["step"], clock[0]))
        clock[0] += capture_time

    def jobs():
        events.append(("jobs", index[0] - 1))
        observed_ages.append(clock[0] - frame_time[0])
        clock[0] += job_times[index[0] - 1]
        if nested_job:
            nested_job(env, clock, index[0])

    env.update(app=SimpleNamespace(update=update, is_running=lambda: index[0] < len(update_times)),
               _update_wrist_cam=wrist, _refresh_frames=capture, _run_exec_jobs=jobs,
               _step_with_frame_history=update)
    tree = ast.parse((REPO / "scripts/isaac_bridge.py").read_text())
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.While)
             and any(isinstance(t, ast.Call) and ast.unparse(t) == "app.is_running()"
                     for t in ast.walk(n.test))]
    assert len(loops) == 1
    exec(compile(ast.Module(body=loops, type_ignores=[]), "real_isaac_loop", "exec"), env)
    return SimpleNamespace(events=events, captures=captures, ages=observed_ages,
                           clock=clock[0], steps=index[0], env=env)


def test_fast_loop_keeps_every_nominal_camera_step_and_initial_capture():
    run = run_loop([.01] * 18)
    assert [step for step, _ in run.captures] == [1, 6, 12, 18]
    assert run.steps == 18
    assert sum(kind == "jobs" for kind, _ in run.events) == 18


def test_slow_main_thread_job_does_not_wait_for_the_sixth_physics_iteration():
    # Same mechanism as correction08: an RPC occupies a main-loop iteration,
    # so six physics iterations no longer imply fresh wall-clock camera data.
    run = run_loop([.23] * 6, job_times=[0., 1., 0., 0., 0., 0.],
                   step=16410, initial_anchor=0.)
    assert run.captures[0][0] == 16413  # Before legacy step 16416.
    assert run.captures[-1][0] == 16416  # Nominal capture is still preserved.
    assert max(run.ages) < .7  # Only this zero-render-lag scheduling model.
    assert run.steps == 6


@pytest.mark.parametrize("capture_time", [.8, 2.5])
def test_slow_capture_has_no_added_cooldown_or_fabricated_freshness(capture_time):
    run = run_loop([.01] * 4, capture_time=capture_time)
    assert len(run.captures) == run.steps == 4  # At most once per existing step.
    assert run.ages == pytest.approx([capture_time + .01] * 4)
    assert [t for _, t in run.captures] == pytest.approx(
        [.01 + i * (.01 + capture_time) for i in range(4)])
    # No guarantee of <2s is claimed if synchronous Kit/readback itself stalls.


def test_nested_capture_updates_the_same_anchor_as_the_outer_loop():
    def nested(env, clock, index):
        if index == 1:
            clock[0] += .6
            assert env["_camera_capture_due"]()
            clock[0] += .1  # Nested render/readback after starting its attempt.
            env['_pending_camera_publications'].discard('cam0')

    run = run_loop([.05] * 3, nested_job=nested)
    assert [step for step, _ in run.captures] == [1]
    assert run.env["_last_camera_capture_started"] == pytest.approx(.65)


def test_duplicate_render_retries_on_next_existing_update_without_new_cooldown():
    # A long RPC is followed by an old render token, then a new token on the
    # next existing update. The old scheduler waited until step 29088.
    run = run_loop([.1] * 4, job_times=[1.2, 0., 0., 0.], step=29083,
                   initial_anchor=0., capture_results=[False, True])
    assert [step for step, _ in run.captures] == [29085, 29086]
    assert run.ages[:2] == pytest.approx([.1, 1.4])  # Duplicate was not retimestamped.
    assert run.ages[2:] == pytest.approx([.1, .2])
    assert run.steps == 4  # No catch-up updates or extra physics steps.
    assert run.env['_last_camera_capture_started'] == pytest.approx(1.3)


def test_repeated_tokens_do_not_turn_attempts_into_publications():
    run = run_loop([.1] * 4, capture_results=[False] * 4)
    assert len(run.captures) == run.steps == 4
    assert run.ages == pytest.approx([.1, .2, .3, .4])
    assert run.env['_pending_camera_publications'] == {'cam0'}


@pytest.mark.parametrize('now', [float('nan'), float('inf'), -1.])
def test_invalid_schedule_clock_cannot_schedule_an_update(now):
    env = dict(time=SimpleNamespace(monotonic=lambda: now), args=SimpleNamespace(cam_every=6))
    load_isaac_bridge_definitions({'_camera_capture_due'}, env)
    with pytest.raises(ValueError, match='finite monotonic'):
        env['_camera_capture_due'](6)


def test_regressive_schedule_clock_cannot_relabel_an_attempt():
    env = dict(time=SimpleNamespace(monotonic=lambda: 4.), args=SimpleNamespace(cam_every=6),
               _last_camera_capture_started=5.)
    load_isaac_bridge_definitions({'_camera_capture_due'}, env)
    with pytest.raises(ValueError, match='regressed'):
        env['_camera_capture_due'](6)
