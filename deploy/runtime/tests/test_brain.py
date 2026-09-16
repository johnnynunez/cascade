"""Qwen allocation and private-log boundaries for the supervised worker."""
import importlib.util
from pathlib import Path
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "brain.py"


@pytest.fixture
def brain():
    spec = importlib.util.spec_from_file_location("cascade_gpu_brain", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def test_command_places_weights_cache_and_supported_operations_on_cuda(brain):
    plan = brain.make_plan(require_files=False)
    argv = plan["argv"]
    for flag, value in (("--device", "CUDA0"), ("--n-gpu-layers", "all"),
                        ("--fit", "off"), ("--override-tensor", ".*=CUDA0"),
                        ("--cache-ram", "0"), ("--host", "127.0.0.1")):
        assert argv[argv.index(flag) + 1] == value
    assert {"--kv-offload", "--op-offload", "--mmproj-offload"} <= set(argv)
    assert not any(x.startswith("--no-") and "offload" in x for x in argv)
    assert plan["cpu_fallback_allowed"] is False


FULL = """load_tensors: offloaded 41/41 layers to GPU
load_tensors: CUDA0 model buffer size = 35512.12 MiB
llama_kv_cache: CUDA0 KV buffer size = 1024.00 MiB
llama_context: CUDA0 compute buffer size = 600.00 MiB
llama_context: CUDA_Host compute buffer size = 1.01 MiB
## SPLIT #0: CUDA0 # 4 inputs
"""



def test_allocation_evidence_requires_weights_kv_and_cuda_schedule(brain):
    result = brain.audit_log(FULL)
    assert result["all_layers_cuda"] and result["kv_cuda"] and result["cuda_schedule"]
    assert result["cpu_fallback"] is False



@pytest.mark.parametrize("bad", [
    FULL.replace("41/41", "40/41"),
    FULL.replace("CUDA0 KV", "CPU KV"),
    FULL + "load_tensors: CPU_Mapped model buffer size = 10.00 MiB\n",
    FULL + "load_tensors: CUDA_Host model buffer size = 10.00 MiB\n",
    FULL + "llama_kv_cache: CUDA_Host KV buffer size = 10.00 MiB\n",
    FULL + "## SPLIT #1: CPU # 1 inputs\n",
    FULL.replace("## SPLIT #0: CUDA0 # 4 inputs\n", ""),
    "ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected\n",
])
def test_rejects_partial_offload_or_missing_runtime_proof(brain, bad):
    with pytest.raises(ValueError):
        brain.audit_log(bad)



@pytest.mark.parametrize("line", [
    '8.56.770.674 D srv  log_server_r: request: {"messages":[{"content":"GPU instructions"}]}',
    '8.56.770.674 D srv  log_server_r: response: {"content":"CUDA response"}',
    'request body: CPU CUDA GPU private conversation',
    'prompt: CUDA model buffer size = 999 MiB',
])
def test_runtime_log_never_keeps_model_input_or_output_for_gpu_keywords(brain, line):
    assert not brain.keep_runtime_line(line)



def child_plan(brain, monkeypatch, tmp_path, body):
    """An actual bounded Python child; never starts an inference/GPU runtime."""
    import sys
    monkeypatch.setattr(brain, 'HERE', tmp_path)
    monkeypatch.setattr(brain.signal, 'signal', lambda *args: None)
    script=tmp_path/'synthetic_runtime.py'
    script.write_text('import os, signal, time\ntime.sleep(.05)\n'+body)
    return {'argv':[sys.executable,str(script)],'profile':'qwen','unit':'synthetic-test-only',
            'plan_sha256':'synthetic','state_file':str(tmp_path/'state.json')}



def test_invalid_native_log_bytes_do_not_kill_model_or_leak_private_line(brain, monkeypatch, tmp_path):
    import json
    body='os.write(1, '+repr(FULL.encode())+')\n'
    body+="os.write(1, b'PRIVATE_INVALID_LOG '+bytes([255,195])+b'( CUDA private payload\\n')\n"
    body+="os.write(1, b'## SPLIT #1: CUDA0 # 4 inputs\\n')\ntime.sleep(.05)\n"
    plan=child_plan(brain,monkeypatch,tmp_path,body)
    assert brain.serve(plan)==0
    state=json.loads(Path(plan['state_file']).read_text())
    log=Path(state['log_file']).read_text()
    assert 'PRIVATE_INVALID_LOG' not in log and 'private payload' not in log
    events=[json.loads(line)['log_decode_recovery'] for line in log.splitlines() if 'log_decode_recovery' in line]
    assert len(events)==1 and events[0]['invalid_byte_count']==2
    assert events[0]['ranges'][0]['byte_escapes']=='\\xff\\xc3'
    assert events[0]['private_line_content_recorded'] is False
    assert state['log_decode_recoveries']==1 and state['supervisor_error'] is None
    assert brain.audit_log(log)['cuda_schedule'] is True
    assert brain.process_identity(state['identity']['pid']) is None



def test_malformed_private_bytes_do_not_disable_cuda_guard_or_leave_stubborn_child(brain, monkeypatch, tmp_path):
    import json
    body="signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    body+="os.write(1, b'PRIVATE_INVALID_LOG '+bytes([255])+b'\\n')\n"
    body+="os.write(1, b'## SPLIT #0: CPU # 1 inputs\\n')\ntime.sleep(60)\n"
    plan=child_plan(brain,monkeypatch,tmp_path,body)
    assert brain.serve(plan)==2
    state=json.loads(Path(plan['state_file']).read_text())
    assert state['gpu_violation'] and 'not executing on CUDA' in state['gpu_violation']
    assert state['return_code']==-9  # bounded SIGKILL after this child ignored SIGTERM
    assert brain.process_identity(state['identity']['pid']) is None



def test_unexpected_log_consumer_failure_reaps_only_its_child(brain, monkeypatch, tmp_path):
    import json
    plan=child_plan(brain,monkeypatch,tmp_path,"os.write(1, b'test line\\n')\ntime.sleep(60)\n")
    def fail(_line):raise RuntimeError('synthetic filter failure')
    monkeypatch.setattr(brain,'keep_runtime_line',fail)
    with pytest.raises(RuntimeError,match='synthetic filter failure'):
        brain.serve(plan)
    state=json.loads(Path(plan['state_file']).read_text())
    assert state['supervisor_error']=='RuntimeError'
    assert state['return_code']!=0 and state['gpu_violation'] is None
    assert brain.process_identity(state['identity']['pid']) is None
