export function parseCameraURL(raw) {
  const u = new URL(raw.trim());
  if (!['http:', 'https:'].includes(u.protocol) || u.username || u.password || u.search || u.hash) {
    throw new Error('Use an HTTP or HTTPS URL without credentials, parameters, or fragments.');
  }
  let path = u.pathname.replace(/\/+$/, '');
  let camera = '';
  const match = path.match(/^(.*)\/stream\/([A-Za-z0-9_.-]+)$/);
  if (match) { path = match[1]; camera = match[2]; }
  if (/(?:^|\/)(?:snapshot|depth|annotated)(?:\/|$)/.test(path) || /\/stream(?:\/|$)/.test(path)) {
    throw new Error('Use the server’s base URL or /stream/name.');
  }
  return {url: u.origin + (match ? `${path}/stream/${camera}` : path), base: u.origin + path,
    camera, permission: `${u.protocol}//${u.hostname}/*`};
}
export function cameraNames(state) {
  if (!state || typeof state.cameras !== 'object' || Array.isArray(state.cameras)) return [];
  return Object.keys(state.cameras || {}).filter(n => /^[A-Za-z0-9_.-]+$/.test(n) && n !== '..').slice(0, 32);
}
// The real server sends Content-Length for every wrcframe part. Bound both headers and frames.
// The Spark camera viewer (scripts/spark_cameras.py) also writes X-Frame-Age on
// each part: seconds since that viewer first served the image, which it keeps
// serving for up to 5 s while the simulator skips frames. `age` is the
// X-Frame-Age of the last frame `push` returned, or null when its part had none.
export class MJPEGParser {
  buffer = new Uint8Array(0);
  length = null;
  age = null;
  partAge = null;
  push(chunk) {
    if (this.buffer.length + chunk.length > 10 * 1024 * 1024) throw new Error('Frame too large.');
    const joined = new Uint8Array(this.buffer.length + chunk.length);
    joined.set(this.buffer); joined.set(chunk, this.buffer.length); this.buffer = joined;
    const frames = [];
    while (true) {
      if (this.length === null) {
        let end = -1;
        for (let i = 0; i + 3 < this.buffer.length; i++) {
          if (this.buffer[i] === 13 && this.buffer[i+1] === 10 && this.buffer[i+2] === 13 && this.buffer[i+3] === 10) { end = i; break; }
        }
        if (end < 0) { if (this.buffer.length > 8192) throw new Error('Invalid MJPEG header.'); break; }
        const header = new TextDecoder().decode(this.buffer.subarray(0, end));
        const size = header.match(/content-length:\s*(\d+)/i);
        if (!size || !/content-type:\s*image\/jpeg/i.test(header)) throw new Error('Expected MJPEG with Content-Length.');
        this.length = Number(size[1]);
        const age = header.match(/(?:^|\n)x-frame-age:[ \t]*(\d{1,6}(?:\.\d{1,6})?)[ \t]*(?:\r|$)/i);
        this.partAge = age ? Number(age[1]) : null;
        if (this.length < 4 || this.length > 8 * 1024 * 1024) throw new Error('Invalid JPEG size.');
        this.buffer = this.buffer.slice(end + 4);
      }
      if (this.buffer.length < this.length) break;
      const frame = this.buffer.slice(0, this.length);
      if (frame[0] !== 255 || frame[1] !== 216) throw new Error('The server did not send a JPEG.');
      frames.push(frame); this.buffer = this.buffer.slice(this.length); this.length = null; this.age = this.partAge;
    }
    return frames;
  }
}
export function freshness({now, lastFrame, lastState, lastAdvance, hasAdvanced, error}) {
  if (!lastFrame) return 'connecting';
  if (now - lastFrame > 5000 || now - lastState > 4000 || error || (lastAdvance > 0 && now - lastAdvance > 5000)) return 'stale';
  return hasAdvanced ? 'live' : 'unverified';
}
// Over 1 s old, the frame on screen is a held one: say how old instead of Live.
export function frameAgeNote(ageS) {
  return typeof ageS === 'number' && Number.isFinite(ageS) && ageS > 1 ? `Last frame · ${Math.floor(ageS)} s old` : '';
}
// The age of the frame on screen now: its part's X-Frame-Age plus the time it
// has been shown (`shownMs`), so a frame keeps ageing when no further part
// arrives. null when the part had no age (any other server).
export function frameAgeNow(ageS, shownMs) {
  if (typeof ageS !== 'number' || !Number.isFinite(ageS)) return null;
  return ageS + Math.max(0, shownMs) / 1000;
}
