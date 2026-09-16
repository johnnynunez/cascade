// Verify the edge assertion before granting a one-time OpenClaw browser handoff.
// Input (including the assertion) is stdin only; output is one boolean.
import {createPublicKey, verify} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

async function edgeKeys(origin) {
  const response = await fetch(origin + '/cdn-cgi/access/certs', {
    signal: AbortSignal.timeout(5000), redirect: 'error', headers: {'Accept': 'application/json'},
  });
  if (!response.ok) throw Error();
  const reader = response.body.getReader();
  const chunks = []; let total = 0;
  for (;;) {
    const {done, value} = await reader.read();
    if (done) break;
    total += value.length;
    if (total > 262144) { await reader.cancel(); throw Error(); }
    chunks.push(value);
  }
  return JSON.parse(Buffer.concat(chunks).toString()).keys;
}

export async function verifyAssertion({assertion, issuer, audience}, loadKeys = edgeKeys) {
 try {
  const origin = new URL(issuer);
  if (origin.protocol !== 'https:' || !/^[a-z0-9-]+\.cloudflareaccess\.com$/.test(origin.hostname) ||
      origin.port || !['', '/'].includes(origin.pathname) || origin.search || origin.hash ||
      origin.username || origin.password || typeof audience !== 'string' || !/^[a-f0-9]{64}$/.test(audience)) throw Error();
  if (typeof assertion !== 'string' || assertion.length > 16000) throw Error();
  const pieces = assertion.split('.');
  if (pieces.length !== 3 || pieces.some(piece => !/^[A-Za-z0-9_-]+$/.test(piece))) throw Error();
  const header = JSON.parse(Buffer.from(pieces[0], 'base64url'));
  const claims = JSON.parse(Buffer.from(pieces[1], 'base64url'));
  const now = Math.floor(Date.now() / 1000);
  if (header.alg !== 'RS256' || typeof header.kid !== 'string' || header.kid.length > 200 ||
      claims.iss !== origin.origin || !Number.isFinite(claims.exp) || claims.exp <= now ||
      !Number.isFinite(claims.iat) || claims.iat > now + 30 ||
      (claims.nbf !== undefined && (!Number.isFinite(claims.nbf) || claims.nbf > now + 30)) ||
      !(Array.isArray(claims.aud) ? claims.aud : [claims.aud]).includes(audience) ||
      typeof claims.sub !== 'string' || !claims.sub) throw Error();
  const keys = await loadKeys(origin.origin);
  const jwk = Array.isArray(keys) && keys.find(key => key.kid === header.kid && key.kty === 'RSA' && (!key.alg || key.alg === 'RS256'));
  if (!jwk) throw Error();
  return verify('RSA-SHA256', Buffer.from(pieces[0] + '.' + pieces[1]),
    createPublicKey({key: jwk, format: 'jwk'}), Buffer.from(pieces[2], 'base64url'));
 } catch { return false; }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  let raw = '';
  for await (const chunk of process.stdin) {
    raw += chunk;
    if (raw.length > 32768) process.exit(1);
  }
  let authenticated = false;
  try { authenticated = await verifyAssertion(JSON.parse(raw)); } catch {}
  raw = '';
  process.stdout.write(JSON.stringify({authenticated}) + '\n');
  process.exit(authenticated ? 0 : 1);
}
