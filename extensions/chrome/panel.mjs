import {cameraNames, MJPEGParser, freshness} from './core.mjs';
import {bootstrap,discover,LABELS} from './discovery.mjs';
const $=id=>document.getElementById(id);
const embedded=new URLSearchParams(location.search).get('surface')==='embedded';
if (!embedded) { document.body.classList.add('side'); for(const id of ['move','side','collapse','close']) $(id).hidden=true; }
let config,hint, generation=0, controllers=[], timers=[], imageURL=null, selected='', paused=false, collapsed=false, left=false;
let lastFrame=0,lastState=0,lastAdvance=0,hasAdvanced=false,lastId=null,cameraError=false,networkError='',retryAt=0;
let authorized=false;
function later(fn,ms){ const t=setTimeout(()=>{timers=timers.filter(id=>id!==t);fn();},ms);timers.push(t);return t; }
function cancel(){generation++;controllers.forEach(c=>c.abort());controllers=[];timers.forEach(clearTimeout);timers=[];}
function forgetFrame(){if(imageURL)URL.revokeObjectURL(imageURL);imageURL=null;$('camera-image').removeAttribute('src');$('camera-image').hidden=true;$('empty').hidden=false;lastFrame=0;}
function state(kind,title,detail){$('retry').hidden=!['offline','stale'].includes(kind);$('status').dataset.state=kind;$('status').textContent=title;$('detail').textContent=detail;$('viewer').dataset.stale=String(kind!=='live');$('frame-note').hidden=!lastFrame||kind==='live';$('frame-note').textContent=kind==='paused'?'Video paused · last frame':'Last frame · freshness unconfirmed';}
function refresh(){
 if(paused||collapsed||document.hidden){state('paused','Paused','The connection pauses while the camera is out of view.');return;}
 if(networkError){const sec=Math.max(1,Math.ceil((retryAt-Date.now())/1000));state(lastFrame?'stale':'offline',lastFrame?'Not updating':'Offline',`${networkError} Retrying in ${sec} s.`);return;}
 const f=freshness({now:performance.now(),lastFrame,lastState,lastAdvance,hasAdvanced,error:cameraError});
 if(f==='live')state('live','Live',`The image is updating live.`);
 else if(f==='stale')state('stale','Not updating','A recent image could not be confirmed. The last frame may be frozen.');
 else if(f==='unverified')state('unverified','Checking','Image received. Checking that the video is advancing.');
 else state('connecting','Connecting','Waiting for the first frame from the server…');
}
async function request(url,gen,stream=false){
 const control=new AbortController();controllers.push(control);
 const timer=setTimeout(()=>control.abort(),7000);
 try{
  const response=await fetch(url,{signal:control.signal,cache:'no-store',credentials:'omit',redirect:'error',referrerPolicy:'no-referrer'});
  if(gen!==generation)throw new DOMException('Stopped','AbortError');
  if(!response.ok)throw new Error(`Server: HTTP ${response.status}.`);
  if(stream){clearTimeout(timer);return {response,control};}
  if(!response.headers.get('content-type')?.includes('application/json'))throw new Error('/state did not return JSON.');
  const reader=response.body.getReader();let total=0,chunks=[];
  while(true){const {done,value}=await reader.read();if(done)break;total+=value.length;if(total>1024*1024){control.abort();throw new Error('/state exceeds 1 MB.');}chunks.push(value);}
  const all=new Uint8Array(total);let offset=0;for(const c of chunks){all.set(c,offset);offset+=c.length;}
  return JSON.parse(new TextDecoder().decode(all));
 }finally{clearTimeout(timer);if(!stream)controllers=controllers.filter(c=>c!==control);}
}
function updateCameras(data){
 const names=cameraNames(data).filter(name=>config.streams[name]);
 if(!names.length)throw new Error('The server has not announced any cameras yet.');
 if(!names.includes(selected))selected=config.defaultCamera||names[0];
 if([...$('camera').options].map(o=>o.value).join('\n')!==names.join('\n')){
  $('camera').replaceChildren(...names.map(n=>{const o=document.createElement('option');o.value=n;o.textContent=LABELS[n]||n;return o;}));
 }
 $('camera').value=selected;$('camera').disabled=false;
 const camera=data.cameras[selected];const id=camera?.frame_id;
 cameraError=!!camera?.error;
 lastState=performance.now();
 if(!lastAdvance)lastAdvance=lastState;
 if(Number.isFinite(id)&&id>0){if(lastId!==null&&id!==lastId){lastAdvance=performance.now();hasAdvanced=true;}lastId=id;}
 else {lastId=null;hasAdvanced=false;}
}
async function renderFrame(bytes,gen){
 const url=URL.createObjectURL(new Blob([bytes],{type:'image/jpeg'}));
 const probe=new Image();probe.src=url;
 try {await probe.decode();if(gen!==generation){URL.revokeObjectURL(url);return;}const old=imageURL;imageURL=url;$('camera-image').src=url;$('camera-image').hidden=false;$('empty').hidden=true;lastFrame=performance.now();if(old)URL.revokeObjectURL(old);refresh();}
 catch(e){URL.revokeObjectURL(url);throw new Error('The JPEG frame could not be decoded.');}
}
async function run(gen,attempt=0){
 if(gen!==generation)return;
 try{
  networkError='';refresh();
  const data=await request(config.stateURL,gen);if(gen!==generation)return;updateCameras(data);
  const {response,control}=await request(config.streams[selected],gen,true);
  if(!response.headers.get('content-type')?.includes('multipart/x-mixed-replace'))throw new Error('The endpoint did not return MJPEG.');
  const poll=async()=>{try{const data=await request(config.stateURL,gen);if(gen!==generation)return;
    if(!cameraNames(data).includes(selected)){cameraError=true;}else updateCameras(data);
   }catch{if(gen===generation)cameraError=true;}if(gen===generation)later(poll,1000);};
  later(poll,1000);
  const reader=response.body.getReader(),parser=new MJPEGParser();
  while(gen===generation){
   const deadline=setTimeout(()=>control.abort(),8000);
   let item;try{item=await reader.read();}finally{clearTimeout(deadline);}
   if(item.done)throw new Error('The server closed the video stream.');
   const frames=parser.push(item.value);
   // Drop superseded frames if network buffering delivers multiple at once.
   if(frames.length){await renderFrame(frames[frames.length-1],gen);attempt=0;}
  }
 }catch(e){
  if(gen!==generation)return;
  fail(attempt);
 }
}
function scheduleRefresh(gen){later(()=>{if(gen!==generation)return;refresh();scheduleRefresh(gen);},1000);}
function fail(attempt=0){
 cancel();const next=generation,delay=Math.min(30000,2000*2**Math.min(attempt,4));
 networkError='The demo is not responding.';retryAt=Date.now()+delay;
 $('empty-title').textContent='Camera unavailable';$('empty-detail').textContent='Reconnecting automatically. You can keep chatting.';
 refresh();later(()=>start(false,attempt+1),delay);scheduleRefresh(next);
}
async function start(reset=false,attempt=0){
 cancel();const gen=generation;networkError='';lastId=null;hasAdvanced=false;lastAdvance=0;lastState=0;cameraError=false;
 if(reset)forgetFrame();
 if(!authorized){state('idle','Permission needed','Open the cameras from the OpenClaw Demo extension.');return;}
 if(paused||collapsed||document.hidden){refresh();return;}
 try{
  hint=await bootstrap();if(gen!==generation)return;
  if(!await chrome.permissions.contains({origins:[hint.permission]})){
   if(gen!==generation)return;state('idle','Permission needed','Connect the cameras to view the demo.');
   $('connect').hidden=false;$('retry').hidden=true;$('permission-note').hidden=false;
   $('empty-title').textContent='Cameras beside your chat.';$('empty-detail').textContent='Kitchen, Worktop, and Side are set up automatically.';return;
  }
  if(gen!==generation)return;
  $('connect').hidden=true;$('retry').hidden=false;$('permission-note').hidden=true;
  state('connecting','Connecting','Finding the demo views…');
  const control=new AbortController();controllers.push(control);
  const discovered=await discover(hint,control.signal);if(gen!==generation)return;config=discovered.config;
  if(!selected)selected=(await chrome.storage.local.get('selectedCamera')).selectedCamera||config.defaultCamera;
  if(gen!==generation)return;updateCameras(discovered.state);scheduleRefresh(gen);run(gen,attempt);
 }catch{if(gen===generation)fail(attempt);}
}
$('retry').addEventListener('click',()=>{paused=false;$('pause').textContent='Ⅱ';$('pause').title='Pause video';$('pause').setAttribute('aria-label','Pause video');start();});
$('pause').addEventListener('click',()=>{paused=!paused;$('pause').textContent=paused?'▷':'Ⅱ';$('pause').title=paused?'Resume video':'Pause video';$('pause').setAttribute('aria-label',paused?'Resume video':'Pause video');start();});
$('camera').addEventListener('change',async()=>{selected=$('camera').value;await chrome.storage.local.set({selectedCamera:selected});start(true);});
$('connect').addEventListener('click',async()=>{
 try{const granted=await chrome.permissions.request({origins:[hint.permission]});if(granted)start(true);else state('idle','Permission needed','Connect the cameras whenever you are ready.');}
 catch{state('offline','Offline','Open the extension and try connecting again.');}
});
$('collapse').addEventListener('click',()=>chrome.runtime.sendMessage({type:'collapse'}));
$('close').addEventListener('click',()=>{cancel();chrome.runtime.sendMessage({type:'close'});});
$('move').addEventListener('click',()=>{left=!left;chrome.runtime.sendMessage({type:'move',left});});
$('side').addEventListener('click',()=>chrome.runtime.sendMessage({type:'open-sidepanel'}).then(r=>{if(!r?.ok)$('detail').textContent=r?.error||'Open the side panel from the extension.';}));
window.addEventListener('keydown',e=>{if(e.key==='Escape'&&embedded){chrome.runtime.sendMessage({type:'collapse'});e.stopPropagation();}});
window.addEventListener('message',e=>{if(!embedded||e.source!==parent)return;if(e.data?.cascade==='pause'){collapsed=true;start();}if(e.data?.cascade==='resume'){collapsed=false;start();$('collapse').focus();}});
document.addEventListener('visibilitychange',()=>start());
window.addEventListener('pagehide',()=>{cancel();forgetFrame();});
chrome.storage.onChanged.addListener((changes,area)=>{if(area==='local'&&changes.discoveryHint&&changes.discoveryHint.newValue!==changes.discoveryHint.oldValue)start(true);if(area==='local'&&changes.selectedCamera&&changes.selectedCamera.newValue!==selected){selected=changes.selectedCamera.newValue;start(true);}});
chrome.permissions.onRemoved.addListener(()=>start(true));
chrome.permissions.onAdded.addListener(()=>start(true));
authorized=!!(await chrome.runtime.sendMessage({type:'authorize-panel'}))?.ok;
await start();
