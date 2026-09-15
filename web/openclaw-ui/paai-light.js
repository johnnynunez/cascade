/* Physical Agentic AI: use OpenClaw's native light mode on every visit. */
(() => {
  'use strict';
  const root = document.documentElement;
  const prefix = 'openclaw.control.settings.v1';
  const started = Date.now();
  let nativeTheme = null;
  let unsubscribe = null;
  let timer = null;
  let applying = false;

  // Keep existing settings intact. Only presentation fields are changed; no
  // authentication value is changed, exposed, or moved to another origin.
  function storeLightPreference() {
    try {
      const base = root.getAttribute('data-openclaw-control-ui-base-path') || '';
      const gateway = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + base;
      const keys = new Set(Object.keys(localStorage).filter(key => key === prefix || key.startsWith(prefix + ':')));
      keys.add(prefix + ':' + gateway.replace(/\/$/, ''));
      for (const key of keys) {
        let settings;
        try { settings = JSON.parse(localStorage.getItem(key) || '{}'); } catch { continue; }
        if (!settings || typeof settings !== 'object' || Array.isArray(settings)) continue;
        if (settings.theme === 'claw' && settings.themeMode === 'light' && settings.accent === '#b4e35a') continue;
        localStorage.setItem(key, JSON.stringify({...settings, theme: 'claw', themeMode: 'light', accent: '#b4e35a'}));
      }
    } catch { /* Storage may be disabled; the native controller and CSS still pin light. */ }
  }

  function paintLight() {
    for (const [name, value] of Object.entries({
      'data-paai-light': 'true', 'data-theme': 'light',
      'data-theme-mode': 'light', 'data-theme-resolved': 'light'
    })) if (root.getAttribute(name) !== value) root.setAttribute(name, value);
    if (root.style.colorScheme !== 'light') root.style.colorScheme = 'light';
    if (!root.classList.contains('wa-light')) root.classList.add('wa-light');
    if (root.classList.contains('wa-dark')) root.classList.remove('wa-dark');
    for (const meta of document.querySelectorAll('meta[name="color-scheme"]')) meta.content = 'light';
    for (const meta of document.querySelectorAll('meta[name="theme-color"]')) {
      meta.content = '#f7f6f2';
      meta.removeAttribute('media');
    }
  }

  function useNativeLight() {
    if (applying) return;
    const theme = document.querySelector('openclaw-app')?.context?.theme;
    if (!theme || typeof theme.setMode !== 'function') return;
    if (theme !== nativeTheme) {
      unsubscribe?.();
      nativeTheme = theme;
      unsubscribe = typeof theme.subscribe === 'function' ? theme.subscribe(useNativeLight) : null;
    }
    if (theme.mode !== 'light' || theme.resolvedMode !== 'light') {
      applying = true;
      try { theme.setMode('light'); } finally { applying = false; }
    }
    paintLight();
  }

  storeLightPreference();
  paintLight();
  const observer = new MutationObserver(() => { paintLight(); useNativeLight(); });
  observer.observe(root, {attributes: true, attributeFilter: ['data-theme', 'data-theme-mode', 'data-theme-resolved', 'class', 'style']});
  function attach() {
    useNativeLight();
    if (!nativeTheme && Date.now() - started < 15000) timer = setTimeout(attach, 100);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', attach, {once: true});
  else attach();
  addEventListener('storage', event => { if (!event.key || event.key.startsWith(prefix)) { storeLightPreference(); useNativeLight(); } });
  addEventListener('pagehide', () => { clearTimeout(timer); observer.disconnect(); unsubscribe?.(); nativeTheme = null; unsubscribe = null; });
  addEventListener('pageshow', event => {
    if (!event.persisted) return;
    paintLight();
    observer.observe(root, {attributes: true, attributeFilter: ['data-theme', 'data-theme-mode', 'data-theme-resolved', 'class', 'style']});
    useNativeLight();
  });
})();
