// Live check of the brain's OpenShell profile: the sandbox reaches the brain only on GET/POST /v1/**.
// Run it INSIDE the sandbox, once per brain, after `scripts/openshell_rig.py up` (about 1 min):
//   scripts/openshell_rig.py exec -- node -e "$(cat scripts/brain_v1_live_check.js)" <served id>
// (served id: nemotron-3.5-bolt, qwen3.8-27b, ...). Prints PASS/FAIL per row; exit 0 = every row passes, 1 = a row
// failed, 2 = usage. docs/OPENSHELL.md, "Live check", says how to read a failure.
//
// Raw http.request on purpose: fetch() and new URL() normalize `/v1/../slots` and `/v1/%2e%2e/slots` to `/slots`
// before sending, so a fetch of them only re-tests /slots (#405 review, Minor 3). Read-only requests only (never
// POST /slots); each chat asks for at most 8 tokens.
//
// Environment (set inside the sandbox): CASCADE_BRAIN_API_KEY (OpenShell's placeholder, swapped for the real key
// on the brain's endpoint only), CASCADE_BRAIN_PORT (8081 when unset), CASCADE_BRAIN_HOST (host.openshell.internal
// when unset; the tests point it at a fake).
"use strict";
const http = require("http");

const TIMEOUT_MS = 90000; // a chat can wait behind the agent's own turn on a busy brain

function chat(model, stream) {
  return JSON.stringify({
    model,
    messages: [{ role: "user", content: "Reply with OK" }],
    max_tokens: 8,
    stream,
    chat_template_kwargs: { enable_thinking: false },
  });
}

// expect: "ok" = 200; "stream" = 200 text/event-stream; "denied" = 403 policy_denied from the proxy;
// "denied-or-400" = the same, or 400 (the proxy may refuse a dot path as malformed).
function rows(model) {
  return [
    { method: "GET", path: "/v1/models", expect: "ok" },
    { method: "POST", path: "/v1/chat/completions", body: chat(model, false), expect: "ok" },
    { method: "POST", path: "/v1/chat/completions", body: chat(model, true), expect: "stream", label: "(stream)" },
    { method: "GET", path: "/slots", expect: "denied" },
    { method: "GET", path: "/props", expect: "denied" },
    { method: "GET", path: "/metrics", expect: "denied" },
    { method: "POST", path: "/tokenize", body: JSON.stringify({ content: "x" }), expect: "denied" },
    { method: "GET", path: "/v1/../slots", expect: "denied-or-400" },
    { method: "GET", path: "/v1/%2e%2e/slots", expect: "denied-or-400" },
  ];
}

function send(target, key, row) {
  return new Promise((resolve) => {
    const headers = { authorization: "Bearer " + key, "content-type": "application/json" };
    const req = http.request(
      { host: target.host, port: target.port, method: row.method, path: row.path, headers, agent: false },
      (res) => {
        let body = "";
        res.setEncoding("utf8");
        res.on("data", (chunk) => (body += chunk));
        res.on("end", () =>
          resolve({ status: res.statusCode, type: String(res.headers["content-type"] || "").split(";")[0], body }));
      },
    );
    req.setTimeout(TIMEOUT_MS, () => req.destroy(new Error("no answer in " + TIMEOUT_MS / 1000 + " s")));
    req.on("error", (e) => resolve({ error: e.message }));
    req.end(row.body);
  });
}

function judge(row, a) {
  if (a.error) return [false, "no answer (" + a.error + ")"];
  const v1 = row.expect === "ok" || row.expect === "stream";
  if (v1) {
    if (a.status === 401) return [false, "401: the key placeholder was not swapped (brain provider or profile not in force)"];
    if (a.status !== 200) return [false, "expected 200"];
    if (row.expect === "stream" && a.type !== "text/event-stream") return [false, "expected 200 text/event-stream"];
    return [true, ""];
  }
  if (a.status === 403 && a.body.includes("policy_denied")) return [true, ""];
  if (row.expect === "denied-or-400" && a.status === 400) return [true, ""];
  if (a.status === 403) return [false, "403 without policy_denied: not the OpenShell proxy's denial"];
  return [false, "expected 403 policy_denied: the request reached the brain, the /v1-only profile is not in force"];
}

async function main(argv, env) {
  const target = { host: env.CASCADE_BRAIN_HOST || "host.openshell.internal", port: Number(env.CASCADE_BRAIN_PORT || 8081) };
  const where = "brain http://" + target.host + ":" + target.port;
  const model = argv[0];
  const key = env.CASCADE_BRAIN_API_KEY;
  if (!model) {
    console.error("usage: node brain_v1_live_check.js <served id>   (" + where + ")");
    return 2;
  }
  if (!key) {
    console.error("CASCADE_BRAIN_API_KEY is not set: run this inside the sandbox (openshell_rig.py exec)   (" + where + ")");
    return 2;
  }
  console.log(where + ", served id " + model);
  let failed = 0;
  const all = rows(model);
  for (const row of all) {
    const a = await send(target, key, row);
    const [pass, why] = judge(row, a);
    failed += pass ? 0 : 1;
    const got = a.error ? "ERR" : a.status + " " + a.type + " " + a.body.slice(0, 60).replace(/\s+/g, " ");
    console.log((pass ? "PASS" : "FAIL") + "  " + [row.method, row.path, row.label || ""].join(" ").trim() + " -> " +
      got.trim() + (why ? "  <- " + why : ""));
  }
  console.log(failed ? failed + " of " + all.length + " rows FAIL" : all.length + "/" + all.length + " rows pass");
  return failed ? 1 : 0;
}

// `node -e "$(cat ...)" <id>` puts the id at argv[1] (no script path); `node brain_v1_live_check.js <id>` at argv[2].
if (!require.main || require.main === module) {
  main(process.argv.slice(require.main ? 2 : 1), process.env).then((code) => (process.exitCode = code));
}
