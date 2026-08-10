/**
 * Hard treatment-budget enforcement for the gym.
 *
 * The original gym tool returns an error after its execution budget is used,
 * but that alone lets the model repeatedly request rejected calls.  This
 * companion extension blocks the first over-budget call at Pi's tool-call
 * boundary and asks the agent loop to terminate. Registered treatments load
 * this extension whenever their immutable bundle enables budget enforcement.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const DEFAULT_TOOL_LIMIT = 8;
const BUDGETED_TOOLS = new Set(["bash", "unbrowser"]);

function commandLineLimit(): number | undefined {
  const flag = "--gym-tool-limit";
  const prefix = `${flag}=`;
  for (let index = 0; index < process.argv.length; index++) {
    const value = process.argv[index];
    const raw = value === flag
      ? process.argv[index + 1]
      : value.startsWith(prefix)
        ? value.slice(prefix.length)
        : undefined;
    if (raw === undefined) continue;
    const parsed = Number.parseInt(raw, 10);
    return Number.isFinite(parsed) && parsed >= 0 ? parsed : undefined;
  }
  return undefined;
}

function configuredLimit(pi: ExtensionAPI): number {
  // Extension flags are scoped by Pi; this extension cannot reliably read a
  // flag registered by gym-tools.ts. The raw CLI is the authoritative shared
  // configuration passed by the orchestrator.
  const cliLimit = commandLineLimit();
  if (cliLimit !== undefined) return cliLimit;

  const raw = pi.getFlag("gym-tool-limit");
  // Pi may return a numeric value even when an extension flag was registered
  // as a string. Accept both representations rather than silently falling
  // back to the default limit.
  const parsed = typeof raw === "number"
    ? raw
    : typeof raw === "string"
      ? Number.parseInt(raw, 10)
      : NaN;
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : DEFAULT_TOOL_LIMIT;
}

export default function (pi: ExtensionAPI): void {
  let admittedCalls = 0;

  pi.on("session_start", (event) => {
    // Model/provider extensions can trigger a resource reload after the first
    // turn. Resetting on that reload grants an accidental extra tool call.
    // This extension instance is session-scoped, so only startup initializes
    // the counter.
    if (event.reason === "startup") admittedCalls = 0;
  });

  pi.on("tool_call", (event, ctx) => {
    if (!BUDGETED_TOOLS.has(event.toolName)) return;

    const limit = configuredLimit(pi);
    if (admittedCalls >= limit) {
      // `terminate` is advisory and some provider/tool-loop combinations may
      // still request another turn. Abort the active agent operation as the
      // authoritative hard stop; verification will inspect the workspace that
      // exists at this treatment boundary.
      ctx.abort();
      return {
        block: true,
        reason: `Tool call limit reached (${limit}); submit the current workspace without another model turn.`,
        terminate: true,
      };
    }

    admittedCalls += 1;
  });

  pi.on("tool_result", (event) => {
    if (!BUDGETED_TOOLS.has(event.toolName) || admittedCalls < configuredLimit(pi)) return;
    return {
      content: [
        ...event.content,
        {
          type: "text" as const,
          text: "Tool budget is now exhausted. Do not request another tool; finish with the current workspace.",
        },
      ],
    };
  });
}
