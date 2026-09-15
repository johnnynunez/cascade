// Native OpenClaw removes the fragment with replaceState, but Chromium can
// retain the original visit in its disk History database. Delete only that
// exact credential-bearing visit. Never search, export, log or store history.
const DEFAULT_ORIGINS = Object.freeze([
  'http://127.0.0.1:18790', 'http://localhost:18790'
]);

export function isAccessLink(raw, recognizedOrigins = []) {
  try {
    const url = new URL(raw);
    if (url.username || url.password || !['http:', 'https:'].includes(url.protocol)) return false;
    const allowed = [...DEFAULT_ORIGINS, ...recognizedOrigins.slice(0, 8)];
    return allowed.includes(url.origin) && !!new URLSearchParams(url.hash.slice(1)).get('token');
  } catch { return false; }
}
