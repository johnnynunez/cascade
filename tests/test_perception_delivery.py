"""Live perception must not inherit a previous target or previous task image."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
from cascade.agent.orchestrator import AgentOrchestrator
from cascade.memory import EpisodicMemory


def planner_runtime():
    rt = SimpleNamespace(memory=EpisodicMemory(), last_frame=object(), trace=Mock(), calls=[],
                         begin_task_budget=Mock(), end_task_budget=Mock(), camera_reads=0)
    rt.jpeg = b'old-task-frame'
    rt.frame_jpeg = lambda: rt.jpeg
    rt.beliefs = SimpleNamespace(summary=lambda: [
        {'label':'ball', 'state':'visible', 'age_s':0.2},
        {'label':'old ghost', 'state':'remembered', 'age_s':400},
    ])

    def observe(*, require_fresh=False):
        assert require_fresh
        rt.camera_reads += 1
        rt.jpeg = f'fresh-frame-{rt.camera_reads}'.encode()
        return rt.last_frame

    def execute(name, args):
        rt.calls.append(name)
        rt.memory.add('observation', 'historical image', thumb_jpeg=b'history')
        return {'ok': True, 'task_complete':name == 'task_done'}

    rt.observe, rt.execute = observe, execute
    return rt


def test_planner_refreshes_image_after_readonly_tools_and_respects_frame_limit():
    rt = planner_runtime()
    llm = MockLLM([
        LLMResponse('', [ToolCall('list_objects', {})]),
        LLMResponse('', [ToolCall('task_done', {'success':True, 'summary':'looked'})]),
    ])
    llm.max_frames = 1
    agent = AgentOrchestrator(llm, rt, decompose=False, verify_milestones=False)
    report = agent.run_task('inspect only')
    assert report.success
    for i, request in enumerate(llm.requests, 1):
        attached = [m for m in request['messages'] if m.get('images')]
        assert len(attached) == 1
        assert attached[0]['images'] == [f'fresh-frame-{i}'.encode()]
        assert '"label": "ball"' in attached[0]['content']
        assert 'old ghost' not in attached[0]['content']
    assert rt.camera_reads == 2


def test_unavailable_fresh_camera_blocks_planner_and_tool_dispatch():
    rt = planner_runtime()

    def observe(**kwargs):
        raise RuntimeError('camera frame is stale')

    rt.observe = observe
    llm = MockLLM([LLMResponse('', [ToolCall('grasp_object', {'label':'ball'})])])
    report = AgentOrchestrator(llm, rt, decompose=False, verify_milestones=False).run_task('pick ball')
    assert not report.success and 'camera frame is stale' in report.summary
    assert rt.calls == [] and llm.requests == []


def test_grounder_disables_thinking_and_implicit_transport_retries(monkeypatch):
    from cascade.perception.vlm_ground import VLMGrounder
    from cascade.types import Frame
    options, sent = {}, {}

    def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"found":true,"bbox_2d":[500,400,600,600]}'))])

    def sdk(**kwargs):
        options.update(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr('openai.OpenAI', sdk)
    grounder = VLMGrounder('http://localhost:8082/v1', 'cosmos3-edge', enable_thinking=False)
    det = grounder.ground(Frame(np.zeros((480,640,3),np.uint8),None,np.eye(3)), 'ball')
    assert options['max_retries'] == 0
    assert sent['extra_body']['chat_template_kwargs']['enable_thinking'] is False
    np.testing.assert_allclose(det.bbox, [320,192,384,288])
    assert sent['messages'][0]['content'][0]['image_url']['url'].startswith('data:image/jpeg;base64,')


def test_configured_cosmos_grounder_initializes_and_lifts_depth(monkeypatch):
    from cascade.config import load_demo_config
    from cascade.skills.runtime import SkillRuntime
    from cascade.types import Detection, Frame
    options = {}

    class Grounder:
        def __init__(self, **kwargs):
            options.update(kwargs)

        def ground(self, frame, query):
            return Detection(label=query, conf=.5, bbox=np.array([10,10,30,30]))

    monkeypatch.setattr('cascade.perception.vlm_ground.VLMGrounder', Grounder)
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.cfg = load_demo_config(arm='fr3', camera='d435i', llm='local_cosmos')
    rt._grounder = None
    rt.watcher = None
    rt.memory = EpisodicMemory()
    frame = Frame(np.zeros((40,40,3),np.uint8),np.full((40,40),.5),np.eye(3),T_base_cam=np.eye(4))
    result = rt._vlm_ground_fix(frame, 'ball')
    assert result is not None
    observed, fix = result
    assert observed is frame
    assert len(fix.points) >= 30
    assert options['base_url'] == 'http://127.0.0.1:8082/v1'
    assert options['enable_thinking'] is False
