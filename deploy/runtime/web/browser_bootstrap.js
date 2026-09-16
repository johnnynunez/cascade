/* Native one-time browser handoff: secret is fetched into memory, never a URL. */
(() => {
  'use strict';
  const params = new URLSearchParams(location.search);
  if(params.get('connect') !== '1') return;
  history.replaceState(null,'',location.pathname);
  const controller=new AbortController(), started=Date.now(), limit=90000;
  let panel, status, stopped=false;
  function show(){
    if(panel||!document.body)return;
    panel=document.createElement('section');panel.setAttribute('role','region');panel.setAttribute('aria-label','Opening OpenClaw');
    panel.style.cssText='position:fixed;inset:0;z-index:2147483647;overflow:auto;color-scheme:light;background:#f7f6f2;color:#242722;padding:12vh 10vw;font:18px/1.5 system-ui';
    const heading=document.createElement('h1');heading.textContent='Opening OpenClaw';
    status=document.createElement('p');status.setAttribute('role','status');status.textContent='Preparing secure browser access…';
    panel.append(heading,status);document.body.append(panel);
  }
  if(document.body)show();else document.addEventListener('DOMContentLoaded',show,{once:true});
  function finish(authenticated){
    if(stopped)return;stopped=true;controller.abort();clearTimeout(watchdog);
    window.opener?.postMessage({type:authenticated?'paai-openclaw-ready':'paai-openclaw-error'},location.origin);
    if(authenticated){panel?.remove();document.documentElement.dataset.paaiAuthenticated='true';return;}
    show();if(status)status.textContent='Secure browser access could not be confirmed. Return to the guide and click Retry.';
    const retry=document.createElement('a');retry.href='/guide/';retry.textContent='Return to the guide';
    retry.style.cssText='display:inline-block;padding:12px 24px;border-radius:10px;background:#b4e35a;color:#242722;text-decoration:none';panel?.append(retry);
  }
  const watchdog=setTimeout(()=>finish(false),limit);
  const sleep=()=>new Promise(resolve=>setTimeout(resolve,50));
  async function run(){
    const response=await fetch('/api/openclaw-bootstrap',{method:'POST',credentials:'same-origin',cache:'no-store',redirect:'error',
      headers:{'Content-Type':'application/json','X-PAAI-Action':'openclaw'},body:'{}',signal:controller.signal,referrerPolicy:'no-referrer'});
    if(!response.ok)throw Error('access');
    let grant=await response.json();
    if(typeof grant.bootstrapToken!=='string'||grant.bootstrapProfile!=='owner'||grant.expiresAtMs<Date.now())throw Error('access');
    let gateway;
    while(!stopped&&Date.now()-started<limit){
      const candidate=document.querySelector('openclaw-app')?.context?.gateway;
      if(candidate?.snapshot?.client&&typeof candidate.connect==='function'){gateway=candidate;break;}
      await sleep();
    }
    if(!gateway||stopped)throw Error('interface');
    if(status)status.textContent='Confirming the OpenClaw connection…';
    gateway.connect({gatewayUrl:(location.protocol==='https:'?'wss://':'ws://')+location.host+'/openclaw/',
      token:'',password:'',bootstrapToken:grant.bootstrapToken,bootstrapProfile:'owner'});
    grant=null;
    const client=gateway.snapshot.client,revision=gateway.connectionRevision;
    while(!stopped&&Date.now()-started<limit){
      const snapshot=gateway.snapshot;
      if(snapshot.client===client&&gateway.connectionRevision===revision&&snapshot.phase==='connected'&&
        snapshot.hello?.type==='hello-ok'&&snapshot.hello?.auth?.scopes?.includes('operator.admin')){finish(true);return;}
      await sleep();
    }
    throw Error('connection');
  }
  run().catch(()=>finish(false));
})();
