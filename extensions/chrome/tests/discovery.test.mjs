import test from 'node:test';
import assert from 'node:assert/strict';
import {cleanURL,statusHint,contract,bootstrap,discover,LABELS} from '../discovery.mjs';

test('automatically discovered demo cameras use the approved English labels',()=>{
 assert.deepEqual(LABELS,{kitchen:'Kitchen',worktop:'Worktop',side:'Side'});
 const cameras=Object.keys(LABELS).map(name=>({name,stream_url:'/stream/'+name}));
 const config=contract({camera_discovery:{version:1,base_url:'https://demo.example',state_url:'/state',default_camera:'worktop',cameras}},'https://demo.example/api/status');
 assert.deepEqual(config.labels,LABELS);
});

test('deployment discovery supports metadata, active remote hosts and the localhost tunnel',()=>{
 assert.equal(statusHint({pageURL:'http://127.0.0.1:18790/chat'}),'http://127.0.0.1:18790/api/status');
 assert.equal(statusHint({pageURL:'https://demo.example/chat',advertised:'/robot/demo-status'}),'https://demo.example/robot/demo-status');
 assert.equal(statusHint({pageURL:'http://100.90.1.2:18791/chat'}),'http://100.90.1.2:18791/api/status');
 assert.equal(statusHint({pageURL:'https://demo.example:9443/chat'}),'https://demo.example:9443/api/status');
 assert.throws(()=>statusHint());
 for(const url of ['file:///state','http://user:password@demo/state','https://demo/state?token=x','https://demo/#token=x'])assert.throws(()=>cleanURL(url));
});
const source='https://demo.example/robot/demo-status';
test('saved camera servers retain their own advertised port',async()=>{
 const original=globalThis.chrome;
 globalThis.chrome={storage:{local:{get:async()=>({cameraURL:'http://demo.example:8090'})}}};
 try{assert.equal((await bootstrap()).statusURL,'http://demo.example:8090/state');}
 finally{globalThis.chrome=original;}
});
const advertisement={camera_discovery:{version:1,base_url:'/robot/cameras',state_url:'/robot/cameras/state',default_camera:'overhead',cameras:[{name:'overhead',stream_url:'/robot/cameras/stream/overhead'},{name:'wrist',stream_url:'/robot/cameras/stream/wrist'}]}};
test('contract learns configured ports, path prefixes, names and preferred view',()=>{
 const c=contract(advertisement,source);
 assert.equal(c.base,'https://demo.example/robot/cameras');assert.equal(c.defaultCamera,'overhead');
 assert.equal(c.streams.wrist,'https://demo.example/robot/cameras/stream/wrist');assert.equal(c.permission,'https://demo.example/*');
 const alternate=structuredClone(advertisement);alternate.camera_discovery.base_url='https://demo.example:9443/cam';alternate.camera_discovery.state_url='https://demo.example:9443/cam/state';
 assert.equal(contract(alternate,source).base,'https://demo.example:9443/cam');
});
test('discovery rejects credentials, redirected hosts and path injection before camera fetches',()=>{
 for(const patch of [{base_url:'https://unexpected.example/cam'},{state_url:'https://demo.example/state?token=x'},{cameras:[{name:'../x',stream_url:'/x'}]},{cameras:[{name:'wrist',stream_url:'https://unexpected.example/wrist'}]}]){
  const bad=structuredClone(advertisement);Object.assign(bad.camera_discovery,patch);assert.throws(()=>contract(bad,source));
 }
});
test('fresh profile discovers only live-announced cameras and persists public metadata without typed input',async()=>{
 const original={chrome:globalThis.chrome,fetch:globalThis.fetch};const saved={},requests=[];
 globalThis.chrome={storage:{local:{get:async()=>saved,set:async values=>Object.assign(saved,values)}}};
 globalThis.fetch=async(url,options)=>{requests.push({url,options});return new Response(JSON.stringify(url===source?advertisement:{cameras:{overhead:{frame_id:11}}}),{headers:{'content-type':'application/json'}});};
 try{
  const hint=await bootstrap({pageURL:'https://demo.example',advertised:source});const {config}=await discover(hint);
  assert.deepEqual(config.names,['overhead']);assert.equal(config.defaultCamera,'overhead');assert.equal(saved.discoveryHint,source);
  assert.equal(saved.demoDiscovery.streams.wrist,'https://demo.example/robot/cameras/stream/wrist');
  assert.deepEqual(await bootstrap(),hint);assert.ok(requests.every(r=>r.options.credentials==='omit'&&r.options.redirect==='error'));
  assert.ok(!JSON.stringify(saved).includes('token'));
 }finally{Object.assign(globalThis,original);}
});
test('the Spark gateway page falls back to the local camera server on the same host grant',async()=>{
 const original={chrome:globalThis.chrome,fetch:globalThis.fetch};const saved={},requests=[];
 const cameras=['kitchen','worktop','side'];
 globalThis.chrome={storage:{local:{get:async()=>saved,set:async values=>Object.assign(saved,values)}}};
 globalThis.fetch=async url=>{requests.push(url);
  if(url==='http://127.0.0.1:8091/api/status')return new Response(JSON.stringify({camera_discovery:{version:1,base_url:'/',state_url:'/state',default_camera:'worktop',cameras:cameras.map(name=>({name,stream_url:'/stream/'+name}))}}),{headers:{'content-type':'application/json'}});
  if(url==='http://127.0.0.1:8091/state')return new Response(JSON.stringify({cameras:Object.fromEntries(cameras.map(n=>[n,{frame_id:3,online:true}]))}),{headers:{'content-type':'application/json'}});
  return new Response('Not Found',{status:404,headers:{'content-type':'text/plain'}});};
 try{
  const hint=await bootstrap({pageURL:'http://127.0.0.1:18790/chat/main'});
  assert.equal(hint.statusURL,'http://127.0.0.1:18790/api/status');assert.equal(hint.permission,'http://127.0.0.1/*');
  const {config}=await discover(hint);
  assert.deepEqual(requests,['http://127.0.0.1:18790/api/status','http://127.0.0.1:8091/api/status','http://127.0.0.1:8091/state']);
  assert.equal(config.base,'http://127.0.0.1:8091');assert.equal(config.streams.side,'http://127.0.0.1:8091/stream/side');
  assert.equal(config.permission,hint.permission,'the fallback must not need another host grant');
  assert.deepEqual(config.names,cameras);
  assert.equal(saved.discoveryHint,hint.statusURL,'keep the page hint so reopening the panel does not restart discovery');
 }finally{Object.assign(globalThis,original);}
});
test('a loopback OpenClaw page on another port (NemoClaw 18789) finds the real rig through cascade\'s live view',async()=>{
 // Real arm, no Spark camera server: the cameras are cascade's own live view
 // (stream_server.py, :8090), whose /state lists them and /stream/<name> is MJPEG.
 const original={chrome:globalThis.chrome,fetch:globalThis.fetch};const saved={},requests=[];
 const cameras=['d455f_scene','d455f_wrist'];
 globalThis.chrome={storage:{local:{get:async()=>saved,set:async values=>Object.assign(saved,values)}}};
 globalThis.fetch=async url=>{requests.push(url);
  if(url==='http://127.0.0.1:8090/state')return new Response(JSON.stringify({cameras:Object.fromEntries(cameras.map(n=>[n,{frame_id:7,online:true}])),t:1}),{headers:{'content-type':'application/json'}});
  // the OpenClaw Control UI answers its SPA shell, not JSON
  if(url==='http://127.0.0.1:18789/api/status')return new Response('<html></html>',{headers:{'content-type':'text/html'}});
  return new Response('Not Found',{status:404,headers:{'content-type':'text/plain'}});};
 try{
  const hint=await bootstrap({pageURL:'http://127.0.0.1:18789/chat?session=main'});
  const {config}=await discover(hint);
  assert.deepEqual(requests,['http://127.0.0.1:18789/api/status','http://127.0.0.1:8091/api/status','http://127.0.0.1:8090/state','http://127.0.0.1:8090/state']);
  assert.equal(config.base,'http://127.0.0.1:8090');
  assert.equal(config.streams.d455f_wrist,'http://127.0.0.1:8090/stream/d455f_wrist');
  assert.equal(config.permission,hint.permission,'the fallback must not need another host grant');
  assert.deepEqual(config.names,cameras);assert.equal(config.defaultCamera,'d455f_scene');
 }finally{Object.assign(globalThis,original);}
});
test('the Spark camera server still wins over the live view when both answer',async()=>{
 const original={chrome:globalThis.chrome,fetch:globalThis.fetch};const requests=[];
 globalThis.chrome={storage:{local:{get:async()=>({}),set:async()=>{}}}};
 globalThis.fetch=async url=>{requests.push(url);
  if(url==='http://127.0.0.1:8091/api/status')return new Response(JSON.stringify({camera_discovery:{version:1,base_url:'/',state_url:'/state',default_camera:'worktop',cameras:[{name:'worktop',stream_url:'/stream/worktop'}]}}),{headers:{'content-type':'application/json'}});
  if(url==='http://127.0.0.1:8091/state'||url==='http://127.0.0.1:8090/state')return new Response(JSON.stringify({cameras:{worktop:{frame_id:1}}}),{headers:{'content-type':'application/json'}});
  return new Response('Not Found',{status:404,headers:{'content-type':'text/plain'}});};
 try{
  const {config}=await discover(await bootstrap({pageURL:'http://127.0.0.1:18789/'}));
  assert.equal(config.base,'http://127.0.0.1:8091');
  assert.ok(!requests.includes('http://127.0.0.1:8090/state'));
 }finally{Object.assign(globalThis,original);}
});
test('the local camera fallback never applies to remote pages or to the camera server itself',async()=>{
 const original={chrome:globalThis.chrome,fetch:globalThis.fetch};const requests=[];
 globalThis.chrome={storage:{local:{get:async()=>({}),set:async()=>{}}}};
 globalThis.fetch=async url=>{requests.push(url);return new Response('Not Found',{status:404,headers:{'content-type':'text/plain'}});};
 try{
  for(const pageURL of ['https://demo.example/openclaw/','http://100.90.1.2:18790/chat','http://localhost:18790/chat','http://127.0.0.1:8091/','http://127.0.0.1:8090/','https://127.0.0.1:18789/']){
   requests.length=0;
   await assert.rejects(discover(await bootstrap({pageURL})));
   assert.equal(requests.length,1,pageURL+' must not try a second server');
  }
 }finally{Object.assign(globalThis,original);}
});
