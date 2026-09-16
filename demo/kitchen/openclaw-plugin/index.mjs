import { readFileSync } from "node:fs";
import { isAbsolute, join } from "node:path";

const identity = "You control the CASCADE robot simulation through the provided tools. "
  + "For requests to read or act on the scene, call the requested tool and wait for its real result. "
  + "Pass plain JSON values in tool arguments. A string argument is just its value, without label or description annotations. "
  + "Never invent observations or tool responses. One motion tool per order. "
  + "After a tool returns, briefly report its actual result.";

function requestedTool(context) {
  const messages = (context.messages ?? []).filter((message) => !message.runtimeContextCarrier);
  const last = messages.at(-1);
  if (last?.role !== "user") return;
  const text = (typeof last.content === "string" ? last.content : (last.content ?? [])
    .filter((part) => part.type === "text").map((part) => part.text).join("\n"))
    .replace(/^\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s[^\]\n]+\]\s*/i, "");
  if (/\b(?:do\s+not|don't|never|must\s+not|should\s+not)\s+call\s+the\s+(?:native\s+MCP\s+)?tool\b/i.test(text)) return;
  // Only a positive sentence-level directive, never a quoted/code example or
  // a tool name copied out of model prose or a previous tool result.
  const directives = [...text.replace(/```[\s\S]*?```|~~~[\s\S]*?~~~/g, "")
    .matchAll(/(?:^|[.!?]\s+)Call the (?:native MCP )?tool ([A-Za-z0-9_.-]+)\b/g)];
  if (directives.length !== 1) return;
  const name = directives[0][1];
  if ((context.tools ?? []).some((tool) => tool.name === name)) {
    return {name, once: /\bexactly\s+once\b/i.test(text)};
  }
}

// Native OpenClaw extension points; tools and their results remain untouched.
export default {
  id: "cascade-kitchen-prompt",
  name: "CASCADE kitchen prompt",
  register(api) {
    const { workspaceDir, modelProviderId, modelId } = api.pluginConfig ?? {};
    if (![workspaceDir, modelProviderId, modelId].every(
      (value) => typeof value === "string" && value.length > 0 && value.trim() === value,
    ) || !isAbsolute(workspaceDir)) {
      throw new Error("The kitchen prompt requires an absolute workspace and explicit provider/model scope");
    }
    api.on("before_prompt_build", (_event, ctx) => {
      if (ctx?.agentId !== "main" || ctx.workspaceDir !== workspaceDir
        || ctx.modelProviderId !== modelProviderId || ctx.modelId !== modelId) return;
      // The launcher synchronizes the visitor block while preserving operator
      // additions. Read every turn so those additions are never stale or lost.
      const instructions = readFileSync(join(workspaceDir, "AGENTS.md"), "utf8");
      if (!instructions.trim()) throw new Error("The kitchen workspace AGENTS.md is empty");
      return { systemPrompt: identity + "\n\n" + instructions };
    });
    api.registerProvider({
      id: modelProviderId,
      label: "CASCADE kitchen local model",
      auth: [],
      wrapStreamFn(ctx) {
        const inner = ctx.streamFn;
        if (!inner || ctx.agentId !== "main" || ctx.workspaceDir !== workspaceDir
          || ctx.provider !== modelProviderId || ctx.modelId !== modelId) return inner;
        let firstRequest = true;
        let onceCall;
        return (model, context, options) => {
          const initial = firstRequest;
          firstRequest = false;
          if (model.api !== "openai-completions" || model.provider !== modelProviderId
            || model.id !== modelId) return inner(model, context, options);
          const directive = initial
            ? requestedTool(context) : undefined;
          const name = directive?.name;
          if (directive?.once) onceCall = {name, messageCount: (context.messages ?? []).length};
          const completed = onceCall && (context.messages ?? []).slice(onceCall.messageCount)
            .some((message) => message.role === "toolResult" && message.toolName === onceCall.name);
          if (!name && !completed) return inner(model, context, options);
          const originalOnPayload = options?.onPayload;
          return inner(model, context, {
            ...options,
            async onPayload(payload, requestModel) {
              const overridden = await originalOnPayload?.(payload, requestModel);
              const outgoing = overridden === undefined ? payload : overridden;
              if (!outgoing || typeof outgoing !== "object" || Array.isArray(outgoing)
                || (outgoing.tool_choice !== undefined && outgoing.tool_choice !== "auto")) return overridden;
              // An exactly-once order ends after its actual tool result. The
              // model still sees that result and writes the final response.
              if (completed) return {...outgoing, tool_choice: "none", parallel_tool_calls: false};
              if (!outgoing.tools?.some((tool) => tool.type === "function" && tool.function?.name === name)) return overridden;
              // This server ends named choices with finish_reason=stop, which
              // the host correctly treats as a final answer. Requiring the one
              // requested native schema retains tool_calls completion semantics.
              return {...outgoing,
                // Exactly once also excludes multiple calls in ONE response.
                ...(directive?.once ? {parallel_tool_calls: false} : {}),
                tools: outgoing.tools.filter((tool) => tool.type === "function" && tool.function?.name === name),
                tool_choice: "required"};
            },
          });
        };
      },
    });
  },
};
