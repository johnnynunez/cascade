import {isAccessLink} from './history.mjs';

chrome.history.onVisited.addListener(async visit => {
  try {
    const saved = await chrome.storage.local.get('openclawOrigins');
    if (isAccessLink(visit.url, saved.openclawOrigins || [])) {
      await chrome.history.deleteUrl({url: visit.url});
    }
  } catch {
    // Do not log an exception carrying a credential-bearing visit URL.
  }
});

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.setAccessLevel({accessLevel:'TRUSTED_CONTEXTS'});
});
chrome.runtime.onMessage.addListener((message, sender, reply) => {
  if (sender.id !== chrome.runtime.id || !sender.url?.startsWith(chrome.runtime.getURL(''))) return;
  if (message.type === 'authorize-panel') {
    if (!sender.tab) { reply({ok:true}); return; }
    chrome.storage.session.get(`allowed:${sender.tab.id}`).then(s => {
      const allowed = s[`allowed:${sender.tab.id}`];
      reply({ok: !!allowed && sender.tab.url?.startsWith(allowed + '/')});
    }); return true;
  }
  if (message.type === 'open-sidepanel') {
    const options = sender.tab ? {tabId:sender.tab.id} : {windowId:message.windowId};
    chrome.sidePanel.open(options).then(() => {
      if (sender.tab) chrome.tabs.sendMessage(sender.tab.id,{type:'close'}).catch(()=>{});
      reply({ok:true});
    }, () => reply({ok:false, error:'Open “Side panel” from the OpenClaw Demo extension in Chrome.'}));
    return true;
  }
  if (['collapse','close','move'].includes(message.type) && sender.tab) {
    chrome.tabs.sendMessage(sender.tab.id, message).catch(()=>{});
  }
});
chrome.tabs.onRemoved.addListener(id => chrome.storage.session.remove(`allowed:${id}`));

// activeTab survives same-origin reloads. Restore only the explicitly allowed
// OpenClaw tab; inject.js independently verifies the actual app again.
chrome.tabs.onUpdated.addListener((id,change,tab)=>{
 if(change.status!=='complete'||!tab.url)return;
 chrome.storage.session.get('allowed:'+id).then(saved=>{
  if(saved['allowed:'+id]===new URL(tab.url).origin)chrome.scripting.executeScript({target:{tabId:id},files:['inject.js']}).catch(()=>{});
 });
});
