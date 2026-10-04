"""Execute the browser capture worklet without microphone hardware."""
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
STATIC = Path(__file__).resolve().parents[1] / "src/cascade/conversation/static"
TIMINGS = Path(__file__).parent / "fixtures/conversation_playback_timing_20261002.json"


@pytest.mark.skipif(NODE is None, reason="Node is optional for browser activation validation")
@pytest.mark.parametrize("outcome", ["success", "disconnect", "track_end", "worklet_failure"])
def test_prepared_microphone_requires_explicit_start_and_new_session_without_replaying_pcm(outcome):
    script = r"""
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const outcome=process.argv[2], elements=new Map(), sockets=[], requests=[], nodes=[], sent=[];
let sessionCount=0, started=false, startResolve, ended, trackStops=0;
const track={readyState:'live',stop(){trackStops++;this.readyState='ended';},
  addEventListener(kind,cb){assert.equal(kind,'ended');ended=cb;}};
const acquired={getTracks:()=>[track],getAudioTracks:()=>[track]};
const response=value=>({ok:true,json:async()=>value});
const sandbox={Uint8Array,DataView,Math,Set,Map,JSON,atob,btoa,console,
  crypto:{randomUUID:()=> 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'},
  location:{hash:'#private',pathname:'/',origin:'http://127.0.0.1:8780'},history:{replaceState(){}},
  document:{getElementById:id=>{if(!elements.has(id))elements.set(id,{textContent:'',value:''});return elements.get(id);}},
  fetch:async(url,options={})=>{
    requests.push({url,method:options.method,body:options.body&&JSON.parse(options.body)});
    if(url.endsWith('/status'))return response({robot_id:'fixture',tools:['hand.posture'],generation:0,
      robot_episode:{lifecycle:'bounded_hand',attempted:started,active:started}});
    if(url.endsWith('/robot/start')){
      sockets.at(-1).readyState=3;sockets.at(-1).onclose();
      return await new Promise(resolve=>{startResolve=()=>{started=true;resolve(response({ok:true,active:true,remaining_s:27}));};});
    }
    if(url.endsWith('/session')&&options.method==='POST')return response({session_id:'session'+(++sessionCount),ticket:'ticket'});
    return response({ok:true});
  },
  AudioContext:class {constructor(){this.sampleRate=24000;this.destination={};this.audioWorklet={addModule:async()=>{}};}
    async resume(){} createMediaStreamSource(){return {connect(){},disconnect(){}};}},
  AudioWorkletNode:class {constructor(){if(outcome==='worklet_failure')throw Error('worklet refused');this.port={};nodes.push(this);}
    connect(){} disconnect(){}},
  navigator:{mediaDevices:{getUserMedia:async()=>acquired}},
  PcmPlaybackQueue:class {flush(){}enqueue(){}},
};
sandbox.WebSocket=class {static OPEN=1;constructor(){this.readyState=0;sockets.push(this);}
  send(raw){sent.push(JSON.parse(raw));}close(){this.readyState=3;this.onclose?.();}};
vm.createContext(sandbox);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),sandbox);
const settle=async()=>{for(let i=0;i<12;i++)await Promise.resolve();};
async function run(){
  await settle();await elements.get('connect').onclick();
  const old=sockets.at(-1);old.readyState=1;old.onopen();
  await elements.get('mic').onclick();
  assert.equal(nodes.length,0,'preparation must not connect a capture node');
  assert.equal(requests.filter(r=>r.url.endsWith('/robot/start')).length,0);
  assert.equal(sent.length,1);assert.equal(sent[0].type,'microphone_prepared');
  await old.onmessage({data:JSON.stringify({type:'microphone_prepared',capture_id:'a'.repeat(32),session_id:'session1'})});
  assert.equal(elements.get('start-robot').disabled,false);
  const pending=elements.get('start-robot').onclick();await settle();assert(startResolve);
  assert.equal(trackStops,0,'expected old provider close must keep explicitly prepared mic');
  if(outcome==='disconnect')await elements.get('disconnect').onclick();
  if(outcome==='track_end'){track.readyState='ended';ended();await settle();}
  startResolve();await pending;await settle();
  if(outcome==='disconnect'||outcome==='track_end'){
    assert.equal(sessionCount,1,'cancelled Start must not reconnect');assert(trackStops>0);assert.equal(nodes.length,0);
    assert(requests.some(r=>r.url.endsWith('/session')&&r.method==='DELETE'));
    return;
  }
  assert.equal(sessionCount,2);assert.equal(sockets.length,2);
  const current=sockets.at(-1);current.readyState=1;current.onopen();await settle();
  if(outcome==='worklet_failure'){
    assert(requests.some(r=>r.url.endsWith('/stop')));assert(trackStops>0);return;
  }
  assert.equal(nodes.length,1);assert.equal(trackStops,0);
  nodes[0].port.onmessage({data:new Uint8Array([0,0]).buffer});
  assert.equal(sent.length,2);assert.equal(sent[1].type,'audio');
  assert.equal(sent[1].session_id,'session2');assert.equal(sent[1].sequence,0);
  assert.equal(requests.filter(r=>r.url.endsWith('/robot/start')).length,1);
  await elements.get('disconnect').onclick();assert.equal(trackStops,1);
}
run().catch(error=>{console.error(error);process.exitCode=1;});
"""
    subprocess.run([NODE, "-e", script, str(STATIC / "app.js"), outcome], check=True, timeout=10)


@pytest.mark.skipif(NODE is None, reason="Node is optional for browser worklet validation")
def test_worklet_pcm_endianness_saturation_and_fixed_size_without_mic():
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const chunks = []; let Worklet;
const sandbox = {AudioWorkletProcessor: class {constructor(){this.port={postMessage: b => chunks.push(b)}}},
  registerProcessor: (name, implementation) => {assert.equal(name, 'pcm-capture'); Worklet=implementation;},
  Int16Array, DataView, ArrayBuffer, Math};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), sandbox);
const processor = new Worklet();
const input = new Float32Array(960); input[0]=-2;input[1]=2;input[2]=.5;input[3]=-.5;
assert.equal(processor.process([[input]]),true);
assert.equal(chunks.length,1);assert.equal(chunks[0].byteLength,1920);
const bytes = new DataView(chunks[0]);
assert.equal(bytes.getInt16(0,true),-32768);assert.equal(bytes.getInt16(2,true),32767);
assert.equal(bytes.getInt16(4,true),16384);assert.equal(bytes.getInt16(6,true),-16384);
processor.process([[new Float32Array(959)]]);assert.equal(chunks.length,1);
processor.process([[new Float32Array(1)]]);assert.equal(chunks.length,2);
"""
    subprocess.run([NODE, "-e", script, str(STATIC / "capture.js")], check=True, timeout=10)
    subprocess.run([NODE, "--check", str(STATIC / "app.js")], check=True, timeout=10)


@pytest.mark.skipif(NODE is None, reason="Node is optional for browser playback validation")
def test_native_provider_timing_replay_bounded_queue_and_flush_without_speaker():
    script = r"""
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const fixture = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
function harness() {
  let next = 0; const timers = new Map(), nodes = [], errors = [];
  const ctx = {currentTime: 0, destination: {},
    createBuffer: (channels, count, rate) => {
      assert.equal(channels, 1); assert.equal(rate, 24000);
      const data = new Float32Array(count);
      return {length: count, duration: count/rate, getChannelData: () => data, data};
    },
    createBufferSource: () => {
      const node = {connect() {}, start(at) {
        this.at = at; this.end = at + this.buffer.duration;
        assert(this.end <= ctx.currentTime + 2 + 1e-9, 'scheduling horizon exceeded');
        nodes.push(this);
      }, stop() {this.stopped = true;}};
      return node;
    }};
  const sandbox = {Uint8Array, DataView, Math,
    setTimeout: (fn, ms) => {const id=++next; timers.set(id,{fn,at:ctx.currentTime+ms/1000});return id;},
    clearTimeout: id => timers.delete(id)};
  vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), sandbox);
  const q = new sandbox.PcmPlaybackQueue(ctx, error => errors.push(error));
  function advance(to) {
    for (;;) {
      const entry = [...timers].filter(([,t]) => t.at <= to).sort((a,b) => a[1].at-b[1].at)[0];
      if (!entry) break;
      ctx.currentTime = entry[1].at; timers.delete(entry[0]);
      for (const n of nodes) if (!n.ended && n.end <= ctx.currentTime) {n.ended=true;n.onended();}
      entry[1].fn();
    }
    ctx.currentTime=to;
    for (const n of nodes) if (!n.ended && n.end <= to) {n.ended=true;n.onended();}
  }
  return {ctx,q,nodes,timers,errors,advance};
}
for (const trial of fixture.cases) {
  const h=harness(); let oldEnd=0, oldPeak=0, total=0;
  for (const e of trial.events) {
    h.advance(e.arrival_s);
    oldEnd=Math.max(e.arrival_s+.02,oldEnd)+e.pcm_bytes/48000;
    oldPeak=Math.max(oldPeak,oldEnd-e.arrival_s); total+=e.pcm_bytes/48000;
    assert(h.q.enqueue(new Uint8Array(e.pcm_bytes),0));
    assert(h.q.pendingSeconds <= 15); assert(h.q.sources.size+h.q.queue.length <=512);
  }
  assert(oldPeak>2, 'negative control must reject the observed native bursts');
  assert(Math.abs(oldPeak-trial.old_two_second_scheduler_max_backlog_s)<1e-8);
  h.advance(h.ctx.currentTime+20);
  assert(Math.abs(h.nodes.reduce((sum,n)=>sum+n.buffer.duration,0)-total)<1e-8);
  assert.equal(h.q.pendingSeconds,0); assert.equal(h.q.sources.size,0);
  assert.equal(h.timers.size,0); assert.equal(h.errors.length,0);
  console.log(trial.case+': native timing replay passed; old bound rejected');
}
// A provider cannot allocate an unbounded amount of PCM or tiny audio objects.
const over=harness();
for(let i=0;i<7;i++) over.q.enqueue(new Uint8Array(96000),0);
assert.throws(()=>over.q.enqueue(new Uint8Array(96000),0), /audio budget/);
assert(over.q.queue.length>0); assert.equal(over.timers.size,1);
over.q.flush(1); const count=over.nodes.length;
assert.equal(over.q.pendingSeconds,0); assert.equal(over.q.sources.size,0);
assert(over.nodes.every(n=>n.stopped)); assert.equal(over.timers.size,0);
assert.equal(over.q.enqueue(new Uint8Array(1920),0),false);
over.advance(30); assert.equal(over.nodes.length,count);
const pcm=new Uint8Array([0,128,255,127]); over.q.enqueue(pcm,1);
assert.deepEqual([...over.nodes.at(-1).buffer.data],[-1,32767/32768]);
const tiny=harness(); for(let i=0;i<512;i++) tiny.q.enqueue(new Uint8Array(2),0);
assert.throws(()=>tiny.q.enqueue(new Uint8Array(2),0), /audio budget/);
tiny.q.flush(); over.q.flush();
"""
    subprocess.run(
        [NODE, "-e", script, str(STATIC / "playback.js"), str(TIMINGS)],
        check=True, timeout=10,
    )


@pytest.mark.skipif(NODE is None, reason="Node is optional for browser playback validation")
def test_browser_flush_stop_and_reconnect_fence_pending_audio_context_resume():
    script = r"""
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const elements=new Map(), resumes=[], admitted=[], requests=[];
const sandbox={Uint8Array,DataView,Math,Set,Map,JSON,atob,
  location:{hash:'#private',pathname:'/',origin:'http://127.0.0.1:8780'},history:{replaceState(){}},
  document:{getElementById: id => {if(!elements.has(id))elements.set(id,{textContent:'',value:''});return elements.get(id);}},
  fetch:async url=>{requests.push(url);return {ok:true,json:async()=>({robot_id:'test',tools:[]})};},
  AudioContext:class {constructor(){this.sampleRate=24000;} resume(){return new Promise(r=>resumes.push(r));}},
  PcmPlaybackQueue:class {constructor(){this.generation=0;} flush(){} enqueue(b,g){admitted.push({b,g});}},
};
vm.createContext(sandbox); vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),sandbox);
async function check() {
  vm.runInContext('playbackAllowed=true',sandbox);
  let pending=vm.runInContext("play({generation:0,audio:'AAE='})",sandbox);
  vm.runInContext('generation=1;flush()',sandbox);resumes.shift()();await pending;
  assert.equal(admitted.length,0,'barge-in must discard audio awaiting context resume');
  pending=vm.runInContext("play({generation:1,audio:'AAE='})",sandbox);
  await elements.get('stop').onclick();
  assert(requests.includes('/api/stop'), 'stop must not wait for AudioContext resume');
  resumes.shift()();await pending;
  assert.equal(admitted.length,0,'local stop must suppress pending playback');
  vm.runInContext('playbackAllowed=true;generation=0;flush()',sandbox);
  pending=vm.runInContext("play({generation:0,audio:'AAE='})",sandbox);
  // A reconnect may reset provider generation to the same number; local revision still changes.
  vm.runInContext('flush();generation=0',sandbox);resumes.shift()();await pending;
  assert.equal(admitted.length,0);
  pending=vm.runInContext("play({generation:0,audio:'AAE='})",sandbox);
  resumes.shift()();await pending;assert.equal(admitted.length,1);
}
check().catch(error=>{console.error(error);process.exitCode=1;});
"""
    subprocess.run([NODE, "-e", script, str(STATIC / "app.js")], check=True, timeout=10)


@pytest.mark.skipif(NODE is None, reason="Node is optional for browser microphone lifecycle validation")
@pytest.mark.parametrize("phase", ["resume", "module", "permission"])
@pytest.mark.parametrize("action", ["stop", "disconnect"])
def test_microphone_async_setup_cannot_reactivate_after_stop_or_disconnect(phase, action):
    script = r"""
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const phase=process.argv[2], action=process.argv[3], elements=new Map(), requests=[];
let release, getMediaCalls=0, trackStops=0, nodes=0, sends=0;
const acquired={getTracks:()=>[{stop(){trackStops++;}}]};
const gate=name=>name===phase ? new Promise(r=>release=r) : Promise.resolve();
const sandbox={Uint8Array,DataView,Math,Set,Map,JSON,atob,WebSocket:{OPEN:1},
  location:{hash:'#private',pathname:'/',origin:'http://127.0.0.1:8780'},history:{replaceState(){}},
  document:{getElementById:id=>{if(!elements.has(id))elements.set(id,{textContent:'',value:''});return elements.get(id);}},
  fetch:async url=>{requests.push(url);return {ok:true,json:async()=>({robot_id:'test',tools:[]})};},
  AudioContext:class {constructor(){this.sampleRate=24000;this.audioWorklet={addModule:()=>gate('module')};}
    resume(){return gate('resume');} createMediaStreamSource(){throw new Error('superseded source connected');}},
  AudioWorkletNode:class{constructor(){nodes++;}},
  navigator:{mediaDevices:{getUserMedia:async()=>{getMediaCalls++;await gate('permission');return acquired;}}},
  sendSpy:()=>sends++,
};
vm.createContext(sandbox);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),sandbox);
async function check(){
  vm.runInContext("session={session_id:'one'};socket={readyState:1,bufferedAmount:0,send:sendSpy,close(){}};playbackAllowed=true",sandbox);
  const pending=elements.get('mic').onclick();
  for(let i=0;i<12 && !release;i++)await Promise.resolve();
  assert(release,'selected microphone setup boundary was not reached');
  await elements.get(action).onclick();
  assert(requests.includes(action==='stop'?'/api/stop':'/api/session'));
  release();await pending;
  assert.equal(getMediaCalls,phase==='permission'?1:0);
  assert.equal(trackStops,phase==='permission'?1:0,'late permission tracks must immediately stop');
  assert.equal(nodes,0);assert.equal(sends,0);
  assert.equal(vm.runInContext('capturePending',sandbox),false);
  assert.equal(vm.runInContext('stream',sandbox),undefined);
}
check().catch(error=>{console.error(error);process.exitCode=1;});
"""
    subprocess.run(
        [NODE, "-e", script, str(STATIC / "app.js"), phase, action], check=True, timeout=10,
    )


@pytest.mark.skipif(NODE is None, reason="Node is optional for browser connection lifecycle validation")
@pytest.mark.parametrize("case", ["old_socket", "pending_connect_stop", "pending_reset_stop",
                                 "pending_reset_status_stop", "pending_reset_reply_stop", "authority_revoked"])
def test_browser_connection_ownership_survives_late_callbacks_and_operator_stop(case):
    script = r"""
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const kind=process.argv[2], elements=new Map(), sockets=[], requests=[];
let release, next=0, deleteCount=0, statusCount=0;
class Socket {
  static OPEN=1;
  constructor(){this.readyState=1;this.bufferedAmount=0;sockets.push(this);}
  close(){this.readyState=3;this.closed=true;}
  send(){}
}
const sandbox={Uint8Array,DataView,Math,Set,Map,JSON,atob,WebSocket:Socket,
  location:{hash:'#private',pathname:'/',origin:'http://127.0.0.1:8780'},history:{replaceState(){}},
  document:{getElementById:id=>{if(!elements.has(id))elements.set(id,{textContent:'',value:''});return elements.get(id);}},
  fetch:async(url,options)=>{
    requests.push({url,method:options.method,body:options.body});
    let value={robot_id:'test',tools:[],ok:true,generation:17};
    if(url==='/api/status' && ++statusCount>1 && kind==='pending_reset_status_stop')
      await new Promise(r=>release=r);
    if(url==='/api/reset' && kind==='pending_reset_reply_stop')await new Promise(r=>release=r);
    if(url==='/api/session' && options.method==='POST') {
      value={session_id:'session-'+(++next),ticket:'ticket'};
      if(kind==='pending_connect_stop')await new Promise(r=>release=r);
    }
    if(url==='/api/session' && options.method==='DELETE' && ++deleteCount===1 && kind==='pending_reset_stop')
      await new Promise(r=>release=r);
    return {ok:true,json:async()=>value};
  },
};
vm.createContext(sandbox);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),sandbox);
const click=id=>elements.get(id).onclick();
async function waitGate(){for(let i=0;i<20 && !release;i++)await Promise.resolve();assert(release);}
async function check(){
  if(kind==='authority_revoked') {
    await click('connect'); const current=sockets[0]; current.onopen();
    const count=requests.length;
    await current.onmessage({data:JSON.stringify({type:'authority_revoked',session_id:'session-1',reconnect_required:true})});
    assert.equal(requests.length,count,'speech-only revocation must not issue HTTP stop/reset');
    assert.equal(elements.get('mic').disabled,true); assert.equal(elements.get('send').disabled,true);
    assert.equal(elements.get('disconnect').disabled,false);
    assert.equal(vm.runInContext('playbackAllowed',sandbox),false);
    assert(elements.get('status').textContent.includes('Disconnect and reconnect'));
  } else if(kind==='old_socket') {
    await click('connect');const old=sockets[0];old.onopen();
    await click('disconnect');await click('reset');await click('connect');
    const current=sockets[1];current.onopen();const count=requests.length;
    const revision=vm.runInContext('playbackRevision',sandbox);
    await old.onmessage({data:'invalid JSON must never reach the new session'});
    await old.onerror();old.onclose();old.onopen();
    assert.equal(requests.length,count,'old socket must not stop or delete a newer session');
    assert.equal(vm.runInContext('socket',sandbox),current);
    assert.equal(vm.runInContext('playbackRevision',sandbox),revision);
    assert.equal(elements.get('mic').disabled,false);
    assert.equal(current.closed,undefined);
    const reset=requests.find(r=>r.url==='/api/reset');
    assert.deepEqual(JSON.parse(reset.body),{generation:17});
  } else if(kind==='pending_connect_stop') {
    const pending=click('connect');await waitGate();
    await click('stop');assert(requests.some(r=>r.url==='/api/stop'));
    assert.equal(elements.get('connect').disabled,true);
    await click('connect');assert.equal(next,1,'unfinished session owner must block another POST');
    const priorDeletes=deleteCount;release();await pending;
    assert(deleteCount>priorDeletes,'late-created provider session must be explicitly closed');
    assert.equal(sockets.length,0,'stopped pending connect must not construct a socket');
    assert.equal(vm.runInContext('socket',sandbox),undefined);
    assert.equal(elements.get('connect').disabled,false);
  } else if(kind==='pending_reset_reply_stop') {
    const pending=click('reset');await waitGate();await click('stop');release();await pending;
    assert.equal(vm.runInContext('playbackAllowed',sandbox),false);
    assert.equal(vm.runInContext('socket',sandbox),undefined);
    assert.equal(sockets.length,0);
    assert(elements.get('status').textContent.startsWith('Stopped.'));
  } else {
    const pending=click('reset');await waitGate();await click('stop');release();await pending;
    assert(!requests.some(r=>r.url==='/api/reset'),'superseded reset must not undo a newer stop');
    assert(requests.some(r=>r.url==='/api/stop'));assert.equal(elements.get('connect').disabled,false);
  }
  assert.equal(vm.runInContext('pendingOperations',sandbox),0);
}
check().catch(error=>{console.error(error);process.exitCode=1;});
"""
    subprocess.run(
        [NODE, "-e", script, str(STATIC / "app.js"), case], check=True, timeout=10,
    )
