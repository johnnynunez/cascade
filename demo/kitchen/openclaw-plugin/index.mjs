import { readFileSync } from "node:fs";
import { isAbsolute, join } from "node:path";

const attendeeTools = JSON.parse(readFileSync(new URL("./attendee-tools.json", import.meta.url), "utf8"));
const identity = "You control the CASCADE robot simulation through the provided tools. Reply in English. "
  + "Use tools to inspect or act when a request concerns the scene. Choose the appropriate tool yourself; "
  + "the visitor does not need to know tool names. Use names from the current scene. "
  + "Describe only clearly visible items. Do not guess the identities of cropped or unclear background objects; state uncertainty when needed. "
  + "Inspect if identity or presence is uncertain. Ask one clarification only when the object or destination is genuinely ambiguous. "
  + "Pass plain JSON argument values. Use at most one motion action per order, then report its actual result and end the turn. "
  + "Never claim motion succeeded without a tool result and confirming physics evidence. "
  + "Report failed or unverified placement plainly, including any return-home failure; never invent success or retry automatically. "
  + "Reset only when the visitor requests it. Never invent observations or tool responses.";

// Schema curation is independent of user text. Native Qwen tool calls select
// the action; the full high-level catalog stays available on every request.
function curateTools(tools, prefix, wire = false) {
  return attendeeTools.flatMap((schema) => {
    const name = prefix + schema.name;
    const original = (tools ?? []).find((tool) => wire
      ? tool.type === "function" && tool.function?.name === name : tool.name === name);
    if (!original) return [];
    return [wire
      ? {...original, function: {...original.function, ...schema, name}}
      : {...original, ...schema, name}];
  });
}

export default {
  id: "cascade-kitchen-prompt",
  name: "CASCADE kitchen prompt",
  register(api) {
    const { workspaceDir, modelProviderId, modelId, mcpName = "cascade" } = api.pluginConfig ?? {};
    if (![workspaceDir, modelProviderId, modelId, mcpName].every(
      (value) => typeof value === "string" && value.length > 0 && value.trim() === value,
    ) || !isAbsolute(workspaceDir)) {
      throw new Error("The kitchen prompt requires an absolute workspace and explicit provider/model scope");
    }
    const prefix = mcpName + "__";
    api.on("before_prompt_build", (_event, ctx) => {
      if (ctx?.agentId !== "main" || ctx.workspaceDir !== workspaceDir
        || ctx.modelProviderId !== modelProviderId || ctx.modelId !== modelId) return;
      const instructions = readFileSync(join(workspaceDir, "AGENTS.md"), "utf8");
      if (!instructions.trim()) throw new Error("The kitchen workspace AGENTS.md is empty");
      return { systemPrompt: instructions + "\n\n" + identity };
    });
    api.registerProvider({
      id: modelProviderId,
      label: "CASCADE kitchen local model",
      auth: [],
      wrapStreamFn(ctx) {
        const inner = ctx.streamFn;
        if (!inner || ctx.agentId !== "main" || ctx.workspaceDir !== workspaceDir
          || ctx.provider !== modelProviderId || ctx.modelId !== modelId) return inner;
        return (model, context, options) => {
          if (model.api !== "openai-completions" || model.provider !== modelProviderId
            || model.id !== modelId) return inner(model, context, options);
          const originalOnPayload = options?.onPayload;
          return inner(model, {...context, tools: curateTools(context.tools, prefix)}, {
            ...options,
            async onPayload(payload, requestModel) {
              const overridden = await originalOnPayload?.(payload, requestModel);
              const outgoing = overridden === undefined ? payload : overridden;
              if (!outgoing || typeof outgoing !== "object" || Array.isArray(outgoing)) return overridden;
              return {...outgoing, tools: curateTools(outgoing.tools, prefix, true),
                tool_choice: "auto", parallel_tool_calls: false,
                chat_template_kwargs: {...outgoing.chat_template_kwargs, enable_thinking: false}};
            },
          });
        };
      },
    });
  },
};
