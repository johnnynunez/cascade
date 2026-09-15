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
