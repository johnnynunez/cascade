#!/usr/bin/env node
// Renders cascade's OpenClaw config into $HOME/.openclaw/openclaw.json at the
// sandbox's first start (docs/OPENSHELL.md).
//
// The template carries placeholders, never secrets: `${CASCADE_BRAIN_API_KEY}`
// and `${CASCADE_MCP_TOKEN}` are expanded by OpenClaw from its environment,
// where OpenShell put opaque provider placeholders; the supervisor swaps them
// for the real credentials only on the wire, at the one endpoint each provider
// profile authorizes. The only secret written here is the chat's gateway
// token, minted inside the sandbox (never on a host argv) into a 0600 file.
//
// An existing config is kept: a restart must not rotate the chat token out
// from under an open browser tab. Set CASCADE_OPENCLAW_FORCE=1 to re-render.
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const env = process.env;
const home = env.HOME || "/sandbox";
const dir = path.join(home, ".openclaw");
const out = path.join(dir, "openclaw.json");
const template = env.CASCADE_OPENCLAW_TEMPLATE || "/usr/local/share/cascade/openclaw.json";

function die(msg) {
  console.error(`render-config: ${msg}`);
  process.exit(2);
}

function port(name, fallback) {
  const raw = env[name] ?? String(fallback);
  const n = Number(raw);
  if (!/^\d{1,5}$/.test(raw) || n < 1 || n > 65535) die(`${name}=${JSON.stringify(raw)} is not a TCP port`);
  return n;
}

if (fs.existsSync(out) && env.CASCADE_OPENCLAW_FORCE !== "1") {
  console.log(`render-config: keeping ${out}`);
  process.exit(0);
}

const brain = port("CASCADE_BRAIN_PORT", 8081);
const mcp = port("CASCADE_MCP_PORT", 18741);
const chat = port("CASCADE_CHAT_PORT", 18789);
const model = env.CASCADE_MODEL || "qwen3.8-27b";
// OpenClaw splits `provider/model` on the first '/', so an id that looks like
// a path (Qwen/Qwen3.8-27B) would name a provider that does not exist.
if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(model)) die(`CASCADE_MODEL=${JSON.stringify(model)} must be a bare id`);

const cfg = JSON.parse(fs.readFileSync(template, "utf8"));
const provider = cfg.models.providers.cascade;
provider.baseUrl = `http://host.openshell.internal:${brain}/v1`;
provider.models[0].id = model;
cfg.agents.defaults.model.primary = `cascade/${model}`;
cfg.mcp.servers.cascade.url = `http://host.openshell.internal:${mcp}/mcp`;
cfg.gateway.auth.token = crypto.randomBytes(24).toString("hex");
cfg.gateway.controlUi.allowedOrigins = [`http://127.0.0.1:${chat}`, `http://localhost:${chat}`];

fs.mkdirSync(path.join(dir, "workspace"), { recursive: true, mode: 0o700 });
const tmp = `${out}.tmp-${process.pid}`;
fs.writeFileSync(tmp, JSON.stringify(cfg, null, 2) + "\n", { mode: 0o600 });
fs.renameSync(tmp, out);
console.log(`render-config: wrote ${out} (brain :${brain}, mcp :${mcp}, chat :${chat}, model ${model})`);
