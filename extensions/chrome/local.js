// The local Spark launcher grants loopback access when loading this extension.
// Never inject into another loopback service, even though match patterns have no ports.
(() => {
  if (location.origin !== 'http://127.0.0.1:8092' || location.pathname !== '/') return;
  if (!document.querySelector('body[data-paai-chat]')) return;
  if (document.getElementById('paai-camera-extension')) return;
  const frame = document.createElement('iframe');
  frame.id = 'paai-camera-extension';
  frame.title = 'PAAI camera extension: Kitchen, Worktop and Side';
  frame.src = chrome.runtime.getURL('local.html');
  frame.style.cssText = 'display:block;width:100%;height:390px;border:0;margin:0 0 20px;border-radius:12px';
  const main = document.querySelector('main');
  if (!main) return;
  main.prepend(frame);
  // The extension supplies all three views; avoid a second decoder on the page.
  document.querySelectorAll('main > figure, main > nav[aria-label="Camera selection"], #visitor-cameras, #status').forEach(node => {
    node.hidden = true;
    node.style.setProperty('display','none','important');
  });
  document.documentElement.dataset.paaiCameraExtension = chrome.runtime.id;
})();
