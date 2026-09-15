(async () => {
  let app = document.querySelector('openclaw-app');
  let tree = app?.shadowRoot || app;
  // The gateway document can finish loading before its lazy UI has mounted.
  // Same-origin reloads retain activeTab, so wait for that actual app briefly.
  const deadline = Date.now() + 10000;
  while (/^https?:$/.test(location.protocol) && /\bOpenClaw\b/i.test(document.title) && !tree?.querySelector('.shell, .login-gate') && Date.now() < deadline) {
    await new Promise(resolve => setTimeout(resolve, 100));
    app = document.querySelector('openclaw-app'); tree = app?.shadowRoot || app;
  }
  // OpenClaw 2026.9.3 renders .login-gate before the authenticated .shell.
  if (!/^https?:$/.test(location.protocol) || !/\bOpenClaw\b/i.test(document.title) || !tree?.querySelector('.shell, .login-gate')) return {ok:false, reason:'This page is not the OpenClaw interface.'};
  const existing = document.getElementById('cascade-camera-companion');
  if (existing) return {ok:true};
  const previousFocus = document.activeElement;
  const host = document.createElement('div'); host.id = 'cascade-camera-companion';
  host.style.cssText = 'all:initial!important;position:fixed!important;inset:84px 16px auto auto!important;width:min(440px,calc(100vw - 24px))!important;height:min(560px,calc(100dvh - 250px))!important;z-index:2147483000!important;display:block!important;color-scheme:light!important;';
  const root = host.attachShadow({mode:'open'});
  const style = document.createElement('style');
  style.textContent = ':host{contain:layout style}iframe{display:block;width:100%;height:100%;border:0;border-radius:12px;box-shadow:0 18px 64px #24272226;color-scheme:light}button{font:600 13px system-ui;color:#242722;background:#efeee7;border:1px solid #b3b9a8;border-radius:8px;padding:12px 18px;cursor:pointer;box-shadow:0 8px 24px #2427221a}button:focus-visible{outline:3px solid #b4e35a;outline-offset:3px}[hidden]{display:none}';
  const frame = document.createElement('iframe'); frame.title = 'Physical Agentic AI · OpenClaw Demo';
  frame.src = chrome.runtime.getURL('panel.html?surface=embedded');
  const launcher = document.createElement('button'); launcher.textContent = 'OpenClaw Demo · Camera'; launcher.hidden = true;
  launcher.setAttribute('aria-expanded', 'false');
  root.append(style, frame, launcher); document.body.append(host);
  const expand = () => { frame.hidden = false; launcher.hidden = true; host.style.setProperty('height','min(560px,calc(100dvh - 250px))','important'); host.style.setProperty('width','min(440px,calc(100vw - 24px))','important'); frame.contentWindow.postMessage({cascade:'resume'}, chrome.runtime.getURL('').slice(0,-1)); };
  const collapse = () => { frame.contentWindow.postMessage({cascade:'pause'}, chrome.runtime.getURL('').slice(0,-1)); frame.hidden=true; launcher.hidden=false;host.style.setProperty('height','auto','important');host.style.setProperty('width','auto','important');launcher.focus(); };
  launcher.addEventListener('click', expand); host.addEventListener('cascade-expand', expand);
  const listener = (m, sender) => {
    if (sender.id !== chrome.runtime.id || !host.isConnected) return;
    if (m.type === 'expand') expand();
    if (m.type === 'collapse') collapse();
    if (m.type === 'close') { cleanup(); previousFocus?.isConnected && previousFocus.focus({preventScroll:true}); }
    if (m.type === 'move') host.style.setProperty('inset', m.left ? '84px auto auto 12px' : '84px 16px auto auto','important');
  };
  const cleanup = () => { chrome.runtime.onMessage.removeListener(listener); observer.disconnect(); host.remove(); };
  chrome.runtime.onMessage.addListener(listener);
  // Remove our overlay if a SPA replaces the app with an unrelated surface.
  const observer = new MutationObserver(() => { if (!app.isConnected || !host.isConnected) cleanup(); });
  observer.observe(document.body, {childList:true});
  return {ok:true};
})();
