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
//
// CASCADE_AGENT_PROFILE selects the agent's robot profile: `generic` (unset,
// the default; the config above) or `franka`, the Franka FR3 real arm
// (scripts/launch.sh --sim none --arm fr3): the operator command plugin
// (franka-plugin/, next to the template) compiles explicit routine commands
// into native tool calls, its instructions become the workspace AGENTS.md,
// the chat sees only the FR3 operator tools, and heartbeats, scheduled jobs
// and the generic memory plugin are off (actions start only in the chat).
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
const model = env.CASCADE_MODEL || "nemotron-3.5-bolt";
// What the chat shows and sends for the brain actually served (CASCADE_MODEL
// is the id that server reports). Nemotron's ultra_v3 reasoning parser ignores
// the server's default thinking-off for non-streaming requests, so OpenClaw
// sends {"enable_thinking": false} with every request to it. A brain missing
// here is shown by its own id, never under another model's name.
const BRAINS = {
  "nemotron-3.5-bolt": {
    name: "Nemotron 3.5 Bolt 74B-A9B (cascade brain)",
    params: { chat_template_kwargs: { enable_thinking: false } },
  },
  "qwen3.8-27b": { name: "Qwen3.8-27B (cascade brain)" },
};
const known = Object.hasOwn(BRAINS, model) ? BRAINS[model] : null;
// The brain's context window: OpenClaw sizes every prompt from it, so a 32k
// brain (Cosmos, the Spark's Qwen) declared as the template's 64k overflows.
const rawWindow = env.CASCADE_CONTEXT_WINDOW;
const contextWindow = rawWindow === undefined ? null : Number(rawWindow);
if (rawWindow !== undefined && (!/^\d{4,7}$/.test(rawWindow) || contextWindow < 4096 || contextWindow > 2097152)) {
  die(`CASCADE_CONTEXT_WINDOW=${JSON.stringify(rawWindow)} is not a context window (4096..2097152 tokens)`);
}
// OpenClaw splits `provider/model` on the first '/', so an id that looks like
// a path (Qwen/Qwen3.8-27B) would name a provider that does not exist.
if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(model)) die(`CASCADE_MODEL=${JSON.stringify(model)} must be a bare id`);
const profile = env.CASCADE_AGENT_PROFILE || "generic";
if (!["generic", "franka"].includes(profile)) die(`CASCADE_AGENT_PROFILE=${JSON.stringify(profile)} must be generic or franka`);

const cfg = JSON.parse(fs.readFileSync(template, "utf8"));
const provider = cfg.models.providers.cascade;
provider.baseUrl = `http://host.openshell.internal:${brain}/v1`;
provider.models[0].id = model;
provider.models[0].name = known ? known.name : `${model} (cascade brain)`;
if (contextWindow !== null) provider.models[0].contextWindow = contextWindow;
cfg.agents.defaults.model.primary = `cascade/${model}`;
if (known && known.params) cfg.agents.defaults.models = { [`cascade/${model}`]: { params: known.params } };
cfg.mcp.servers.cascade.url = `http://host.openshell.internal:${mcp}/mcp`;
cfg.gateway.auth.token = crypto.randomBytes(24).toString("hex");
cfg.gateway.controlUi.allowedOrigins = [`http://127.0.0.1:${chat}`, `http://localhost:${chat}`];

// The FR3 operator's tool catalog: observation, localization, grasp preview,
// pick, place, gripper, home, verification and STOP. reset_stop is absent:
// OpenShell refuses it for the agent, Resume & Home is the operator's page.
const FRANKA_TOOLS = [
  "world_state", "camera_snapshot", "get_observation", "localize_object", "locate_pixel",
  "preview_grasp", "grasp_at_pixel", "grasp_object", "pick_and_place", "place_at",
  "place_on_object", "move_home", "open_gripper", "close_gripper", "emergency_stop",
  "verify_last_action",
];
let instructions = null;
if (profile === "franka") {
  const pluginDir = env.CASCADE_FRANKA_PLUGIN_DIR || path.join(path.dirname(template), "franka-plugin");
  instructions = fs.readFileSync(path.join(pluginDir, "instructions.md"), "utf8");
  if (!instructions.trim()) die(`${pluginDir}/instructions.md is empty`);
  const selected = `cascade/${model}`;
  cfg.tools.allow = FRANKA_TOOLS.map((name) => `cascade__${name}`);
  cfg.agents.defaults.heartbeat = { every: "0m" };
  cfg.cron = { enabled: false };
  // Short, deterministic replies; Nemotron's thinking-off kwarg is kept.
  const models = cfg.agents.defaults.models || {};
  const params = (models[selected] && models[selected].params) || {};
  models[selected] = { params: { ...params, temperature: 0, maxTokens: 512 } };
  cfg.agents.defaults.models = models;
  // One robot, one tool at a time.
  cfg.mcp.servers.cascade.supportsParallelToolCalls = false;
  cfg.plugins = {
    load: { paths: [pluginDir] },
    entries: {
      "memory-core": { enabled: false },
      "cascade-franka-prompt": {
        enabled: true,
        hooks: { allowConversationAccess: true, allowPromptInjection: true },
        config: { workspaceDir: cfg.agents.defaults.workspace, modelProviderId: "cascade", modelId: model },
      },
    },
  };
}

fs.mkdirSync(path.join(dir, "workspace"), { recursive: true, mode: 0o700 });
if (instructions !== null) {
  // The plugin injects this file into every prompt. An operator's edit inside
  // the running sandbox is kept; a new image renders into a new sandbox.
  const agents = path.join(dir, "workspace", "AGENTS.md");
  if (!fs.existsSync(agents)) fs.writeFileSync(agents, instructions, { mode: 0o600 });
}
const tmp = `${out}.tmp-${process.pid}`;
fs.writeFileSync(tmp, JSON.stringify(cfg, null, 2) + "\n", { mode: 0o600 });
fs.renameSync(tmp, out);
console.log(`render-config: wrote ${out} (brain :${brain}, mcp :${mcp}, chat :${chat}, model ${model}${profile === "generic" ? "" : `, profile ${profile}`})`);
