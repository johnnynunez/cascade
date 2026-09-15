/* Beginner guide: read-only status, live views, and text copying. The launch client owns actions. */
(function(root){
  'use strict';
  const LOCAL_CHAT='/openclaw';
  function launcherURL(current){
    return /^https?:$/.test(current.protocol)?current.origin+'/':'./index.html';
  }
  function normalizeStatus(raw,now=Date.now()){
    const timestamp=Date.parse(raw?.generated_at),empty={valid:false,ready:false,discovery:null};
    if(raw?.version!==1||!Number.isFinite(timestamp)||timestamp>now+30000||now-timestamp>120000)return empty;
    const value={valid:true,ready:raw.interactive_ready===true,stopped:raw.phase==='stopped',needsAttention:['failed','stop_failed','needs_attention'].includes(raw.phase),discovery:null};
    const candidate=raw.camera_discovery;
    try{
      if(candidate?.version!==1||!Array.isArray(candidate.cameras)||!candidate.cameras.length||candidate.cameras.length>32)return value;
      const base=new URL(candidate.base_url);
      if(!['http:','https:'].includes(base.protocol)||base.username||base.password||base.search||base.hash||base.href.replace(/\/$/,'')!==candidate.base_url||candidate.state_url!==candidate.base_url+'/state')return value;
      const names=new Set(),cameras=[];
      for(const row of candidate.cameras){
        if(!row||typeof row.name!=='string'||!/^\w[\w.-]{0,63}$/.test(row.name)||names.has(row.name)||row.stream_url!==candidate.base_url+'/stream/'+row.name)return value;
        names.add(row.name);cameras.push({name:row.name,label:typeof row.label==='string'?row.label.slice(0,50):row.name,stream_url:row.stream_url});
      }
      value.discovery={base_url:candidate.base_url,cameras,default_camera:names.has(candidate.default_camera)?candidate.default_camera:cameras[0].name,live:candidate.live===true};
    }catch{}
    return value;
  }
  if(typeof module!=='undefined'&&module.exports){module.exports={normalizeStatus,LOCAL_CHAT,launcherURL};return;}
  const $=id=>document.getElementById(id);
  let img=$('scene-image');
  let snapshot=normalizeStatus(null),controller,pollTimer,toastTimer,sequence=0,selected='',sceneMode='',expectedStream='',viewSignature='';
  let frameTimer,frameDeadline,frameRequest,frameVersion=0;
  try{selected=localStorage.getItem('authentic-guide-camera')||'';}catch{}
  const served=/^https?:$/.test(location.protocol);
  const launcher=launcherURL(location);
  for(const id of ['brand-launcher','launcher-link','prepare-link'])$(id).href=launcher;
  $('chat-link').href=served?new URL(LOCAL_CHAT,location.origin).href:LOCAL_CHAT;
  function announce(message){clearTimeout(toastTimer);$('notification').textContent=message;$('notification').hidden=false;toastTimer=setTimeout(()=>{$('notification').hidden=true;},4500);}
  function stopFrames(){
    frameVersion++;expectedStream='';clearTimeout(frameTimer);clearTimeout(frameDeadline);
    if(frameRequest){frameRequest.onload=null;frameRequest.onerror=null;frameRequest.removeAttribute('src');frameRequest=null;}
  }
  function showOffline(){
    stopFrames();
    sceneMode='offline';expectedStream='';img.hidden=true;img.removeAttribute('src');
    $('scene-name').textContent='The demo kitchen';$('scene-state').textContent='Cameras offline';
    $('scene-state').dataset.live='false';$('scene-caption').textContent='Wait for a live camera before judging a movement.';
    $('view-buttons').hidden=true;$('camera-placeholder').hidden=false;
    $('camera-state-title').textContent='Waiting for the kitchen';
    $('camera-state-detail').textContent='The guide stays available while the cameras reconnect.';
    $('scene-media').setAttribute('aria-busy','false');
  }
  function requestFrame(row,version){
    if(version!==frameVersion||document.hidden||!snapshot.discovery?.live)return;
    const frame=new Image();frameRequest=frame;
    const current=()=>version===frameVersion&&frameRequest===frame;
    frame.onload=()=>{
      if(!current())return;
      if(!frame.naturalWidth){showOffline();return;}
      clearTimeout(frameDeadline);frameRequest=null;frame.onload=null;frame.onerror=null;
      frame.id=img.id;frame.alt='Simulator camera: '+row.label;img.replaceWith(frame);img=frame;
      sceneMode='live';$('camera-placeholder').hidden=true;$('scene-media').setAttribute('aria-busy','false');$('scene-state').textContent='Live';$('scene-state').dataset.live='true';$('scene-caption').textContent='Live simulator view · Watch the movement, then ask for the physics result.';
      frameTimer=setTimeout(()=>requestFrame(row,version),350);
    };
    frame.onerror=()=>{if(current())showOffline();};
    frameDeadline=setTimeout(()=>{if(current())showOffline();},6000);
    frame.src=snapshot.discovery.base_url+'/snapshot/'+row.name+'.jpg?t='+Date.now();
  }
  function choose(name){
    const discovery=snapshot.discovery,row=discovery?.cameras.find(camera=>camera.name===name);
    if(!row||!discovery.live||document.hidden){showOffline();return;}
    selected=name;try{localStorage.setItem('authentic-guide-camera',selected);}catch{}
    for(const button of $('view-buttons').children)button.setAttribute('aria-pressed',String(button.dataset.camera===selected));
    $('view-buttons').hidden=false;
    if(expectedStream===row.stream_url)return;
    stopFrames();img.hidden=true;img.removeAttribute('src');sceneMode='connecting';expectedStream=row.stream_url;$('scene-name').textContent=row.label;$('scene-state').textContent='Connecting';$('scene-state').dataset.live='false';$('scene-caption').textContent='Connecting to the simulator camera…';$('camera-placeholder').hidden=false;$('camera-state-title').textContent='Connecting the camera';$('camera-state-detail').textContent='The image will appear shortly.';$('scene-media').setAttribute('aria-busy','true');
    requestFrame(row,frameVersion);
  }
  function render(){
    $('status-title').textContent=snapshot.ready?'Ready to interact':snapshot.stopped?'The demo is stopped':snapshot.needsAttention?'The demo needs attention':'The guide is available';
    $('status-detail').textContent=snapshot.ready?'Start with the observation order. The live camera lets you follow the result.':snapshot.stopped?'Refresh this guide and wait for the booth operator to restore the demo.':snapshot.needsAttention?'Refresh this guide and check the status before sending a movement.':'Click Open OpenClaw to follow the connection checks. The lessons stay available here.';
    const discovery=snapshot.discovery;
    if(discovery){
      $('camera-link').href=discovery.base_url+'/';
      const signature=JSON.stringify(discovery.cameras.map(row=>[row.name,row.label]));
      if(signature!==viewSignature){
        viewSignature=signature;$('view-buttons').replaceChildren(...discovery.cameras.map(row=>{const button=document.createElement('button');button.type='button';button.dataset.camera=row.name;button.textContent=row.label;button.setAttribute('aria-pressed','false');button.addEventListener('click',()=>choose(row.name));return button;}));
      }
      if(!discovery.cameras.some(row=>row.name===selected))selected=discovery.default_camera;
    }
    if(discovery?.live&&!document.hidden)choose(selected);else showOffline();
  }
  async function refresh(){
    const request=++sequence;controller?.abort();controller=new AbortController();const timer=setTimeout(()=>controller.abort(),6000);
    try{
      const response=await fetch('/api/status',{cache:'no-store',credentials:'omit',redirect:'error',signal:controller.signal});const text=await response.text();
      if(!response.ok||text.length>65536)throw Error('invalid status');const raw=JSON.parse(text);if(request!==sequence)return;snapshot=normalizeStatus(raw);

    }catch{if(request===sequence)snapshot=normalizeStatus(null);}
    finally{clearTimeout(timer);if(request===sequence)render();}
  }
  for(const button of document.querySelectorAll('[data-copy]'))button.addEventListener('click',async()=>{
    const text=$(button.dataset.copy).textContent;let copied=false;
    try{await navigator.clipboard.writeText(text);copied=true;}catch{const field=document.createElement('textarea');field.value=text;field.setAttribute('aria-label','Text to copy');document.body.append(field);field.select();try{copied=document.execCommand('copy');}catch{}field.remove();button.focus();}
    if(copied){button.textContent='Copied';button.dataset.copied='true';setTimeout(()=>{button.textContent='Copy order';delete button.dataset.copied;},3000);}
    announce(copied?'Copied. Paste into the OpenClaw message box.':'Select the order text and copy it manually.');
  });
  function poll(){clearInterval(pollTimer);if(!document.hidden&&served){refresh();pollTimer=setInterval(refresh,4000);}else{sequence++;controller?.abort();stopFrames();img.hidden=true;img.removeAttribute('src');sceneMode='';if(!document.hidden){showOffline();render();}}}
  document.addEventListener('oneclick-ready',refresh);document.addEventListener('paai-learner-ready',refresh);
  document.addEventListener('visibilitychange',poll);window.addEventListener('pagehide',()=>{clearInterval(pollTimer);clearTimeout(toastTimer);sequence++;controller?.abort();stopFrames();img.hidden=true;img.removeAttribute('src');});window.addEventListener('pageshow',event=>{if(event.persisted)poll();});
  showOffline();poll();
})(globalThis);
