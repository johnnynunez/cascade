// Read-only native SDK authentication proof. Credential material stays in memory.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const deployment = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const output = process.stdout.write.bind(process.stdout);
const discard = (_chunk, encoding, callback) => { if (typeof encoding === 'function') encoding(); else callback?.(); return true; };
process.stdout.write = discard;
process.stderr.write = discard;
let result;
try {
  process.env.OPENCLAW_PROFILE = 'cascade';
  process.env.OPENCLAW_STATE_DIR = path.dirname(deployment.token_file);
  delete process.env.OPENCLAW_GATEWAY_TOKEN;
  delete process.env.OPENCLAW_GATEWAY_PASSWORD;
  const token = fs.readFileSync(deployment.token_file, 'utf8').trim();
  const sdk = path.join(deployment.root, 'runtime/openclaw/node_modules/openclaw/dist/plugin-sdk/gateway-runtime.js');
  const {callGatewayFromCli} = await import(pathToFileURL(sdk).href);
  await callGatewayFromCli('health', {
    url: 'ws://127.0.0.1:18791/openclaw/', token, timeout: '12000', json: true,
    config: {gateway: {mode: 'local', port: 18790, auth: {mode: 'token'}}},
  }, {}, {sharedStateMode: 'read-only', progress: false, scopes: ['operator.read'], timeoutMs: 12000});
  result = {authenticated_websocket: true, endpoint: 'instance OpenClaw proxy', method: 'health'};
} catch (error) {
  result = {authenticated_websocket: false, error: error?.name || 'GatewayCheckFailed'};
}
output(JSON.stringify(result) + '\n');
process.exit(result.authenticated_websocket ? 0 : 1);
