import test from 'node:test';
import assert from 'node:assert/strict';

async function boot(sidePanels, key) {
  const elements=new Map(), scripts=[], contextQueries=[];
  const original={document:globalThis.document,chrome:globalThis.chrome};
  globalThis.document={getElementById(id){
    if(!elements.has(id)) elements.set(id,{value:'',textContent:'',dataset:{},events:{},addEventListener(event,fn){this.events[event]=fn;}});
    return elements.get(id);
  }};
  globalThis.chrome={
    storage:{local:{get:async()=>({}),set:async()=>{}},session:{set:async()=>{}}},
    permissions:{contains:async()=>false},
    tabs:{sendMessage:async()=>{},query:async()=>[{id:17,windowId:9,url:'http://127.0.0.1:18790/chat'}]},
    runtime:{getContexts:async query=>{contextQueries.push(query);return sidePanels;}},
    scripting:{executeScript:async args=>{
      if(args.func?.toString().includes('link[rel='))return [{result:null}];
      scripts.push(args);
      return [{result:args.files ? {ok:true} : true}];
    }},
  };
  try {
    await import(`../popup.mjs?test=${key}`);
    return {elements,scripts,contextQueries,restore(){Object.assign(globalThis,original);}};
  } catch(error) {Object.assign(globalThis,original);throw error;}
}

test('opening the toolbar beside a native panel stays unobtrusive; explicit attach still works',async()=>{
  const env=await boot([{contextType:'SIDE_PANEL',windowId:-1}],'native-side-panel');
  try {
    assert.deepEqual(env.contextQueries,[{contextTypes:['SIDE_PANEL']}]);
    assert.equal(env.scripts.length,0,'settings must not inject a duplicate floating panel');
    assert.equal(env.elements.get('integration').textContent,'The camera is already open in the side panel.');
    await env.elements.get('attach').events.click();
    assert.equal(env.scripts.length,2,'the explicit attach control must still detect and inject');
    assert.deepEqual(env.scripts[1].files,['inject.js']);
    assert.equal(env.scripts[1].target.tabId,17);
  } finally {env.restore();}
});

test('a side panel with an explicit different window does not block this window',async()=>{
  const env=await boot([{contextType:'SIDE_PANEL',windowId:72}],'other-window');
  try {assert.equal(env.scripts.length,2);} finally {env.restore();}
});

test('opening the toolbar without a native panel still adds the integrated camera',async()=>{
  const env=await boot([],'no-side-panel');
  try {
    assert.equal(env.scripts.length,2);
    assert.deepEqual(env.scripts[1].files,['inject.js']);
    assert.equal(env.elements.get('integration').textContent,'Panel added to OpenClaw. You can keep chatting.');
    assert.equal(env.elements.get('status').textContent,'Cameras ready to connect');
    assert.equal(env.elements.get('connection').textContent,'Allow this demo to show its cameras.');
  } finally {env.restore();}
});
