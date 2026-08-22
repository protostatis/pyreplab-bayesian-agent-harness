/**
 * Independent provider-turn and tool-attempt enforcement for gym runs.
 *
 * Pi validates tool arguments before the `tool_call` extension hook. A schema-
 * invalid request therefore consumes a provider turn and tool attempt without
 * consuming the execution budget. This extension caps those resources
 * independently and emits a machine-readable receipt for reconciliation.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const DEFAULT_TOOL_ADMISSION_LIMIT = 8;
const DEFAULT_PROVIDER_TURN_LIMIT = 9;
const RECEIPT_PREFIX = "PYREPLAB_GYM_BUDGET_V3 ";
const RECEIPT_SCHEMA_VERSION = "pyreplab-gym-budget-v3-receipt-v1";
const BUDGETED_TOOLS = new Set(["bash", "unbrowser", "semantic_table", "semantic_form"]);

type BudgetState = {
  providerRequestAdmissions: number;
  providerRequestBlocks: number;
  toolAttemptCount: number;
  toolAttemptIds: string[];
  admittedToolCallIds: string[];
  executedToolCallIds: string[];
  suppressedToolRequestIds: string[];
  invariantViolations: string[];
  receiptEmitted: boolean;
};

type BudgetGlobal = typeof globalThis & {
  __pyreplabGymBudgetV3State?: BudgetState;
};

function newState(): BudgetState {
  return {
    providerRequestAdmissions: 0,
    providerRequestBlocks: 0,
    toolAttemptCount: 0,
    toolAttemptIds: [],
    admittedToolCallIds: [],
    executedToolCallIds: [],
    suppressedToolRequestIds: [],
    invariantViolations: [],
    receiptEmitted: false,
  };
}

function budgetState(): BudgetState {
  const root = globalThis as BudgetGlobal;
  if (!root.__pyreplabGymBudgetV3State) {
    root.__pyreplabGymBudgetV3State = newState();
  }
  return root.__pyreplabGymBudgetV3State;
}

function resetBudgetState(): void {
  (globalThis as BudgetGlobal).__pyreplabGymBudgetV3State = newState();
}

function commandLineInteger(flag: string): number | undefined {
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

function configuredInteger(
  pi: ExtensionAPI,
  flag: string,
  fallback: number,
): number {
  const cliValue = commandLineInteger(`--${flag}`);
  if (cliValue !== undefined) return cliValue;
  const raw = pi.getFlag(flag);
  const parsed = typeof raw === "number"
    ? raw
    : typeof raw === "string"
      ? Number.parseInt(raw, 10)
      : NaN;
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

function uniquePush(values: string[], value: string, state: BudgetState, label: string): void {
  if (values.includes(value)) {
    state.invariantViolations.push(`duplicate_${label}:${value}`);
    return;
  }
  values.push(value);
}

function emitReceipt(pi: ExtensionAPI): void {
  const state = budgetState();
  if (state.receiptEmitted) return;
  state.receiptEmitted = true;

  const providerTurnLimit = configuredInteger(
    pi,
    "gym-provider-turn-limit",
    DEFAULT_PROVIDER_TURN_LIMIT,
  );
  const toolAdmissionLimit = configuredInteger(
    pi,
    "gym-tool-limit",
    DEFAULT_TOOL_ADMISSION_LIMIT,
  );
  const toolAttemptLimit = toolAdmissionLimit + 1;
  const admitted = new Set(state.admittedToolCallIds);
  const preAdmissionRejected = state.toolAttemptIds.filter((id) => !admitted.has(id));

  process.stderr.write(
    RECEIPT_PREFIX + JSON.stringify({
      schema_version: RECEIPT_SCHEMA_VERSION,
      provider_turn_limit: providerTurnLimit,
      provider_request_admissions: state.providerRequestAdmissions,
      provider_request_blocks: state.providerRequestBlocks,
      provider_gate_checks:
        state.providerRequestAdmissions + state.providerRequestBlocks,
      tool_attempt_limit: toolAttemptLimit,
      tool_attempt_count: state.toolAttemptCount,
      tool_attempt_ids: state.toolAttemptIds,
      tool_admission_limit: toolAdmissionLimit,
      admitted_tool_call_count: state.admittedToolCallIds.length,
      admitted_tool_call_ids: state.admittedToolCallIds,
      executed_tool_call_count: state.executedToolCallIds.length,
      executed_tool_call_ids: state.executedToolCallIds,
      pre_admission_rejected_tool_call_count: preAdmissionRejected.length,
      pre_admission_rejected_tool_call_ids: preAdmissionRejected,
      suppressed_tool_request_count: state.suppressedToolRequestIds.length,
      suppressed_tool_request_ids: state.suppressedToolRequestIds,
      invariant_violations: state.invariantViolations,
    }) + "\n",
  );
}

export default function (pi: ExtensionAPI): void {
  pi.registerFlag("gym-provider-turn-limit", {
    description: "Maximum provider requests admitted per session",
    type: "string",
    default: String(DEFAULT_PROVIDER_TURN_LIMIT),
  });

  pi.on("session_start", (event) => {
    if (event.reason === "startup") resetBudgetState();
  });

  pi.on("before_provider_request", (_event, ctx) => {
    const state = budgetState();
    const limit = configuredInteger(
      pi,
      "gym-provider-turn-limit",
      DEFAULT_PROVIDER_TURN_LIMIT,
    );
    const toolAttemptLimit = configuredInteger(
      pi,
      "gym-tool-limit",
      DEFAULT_TOOL_ADMISSION_LIMIT,
    ) + 1;
    if (
      state.providerRequestAdmissions >= limit
      || state.toolAttemptCount >= toolAttemptLimit
      || state.invariantViolations.length > 0
    ) {
      state.providerRequestBlocks += 1;
      if (state.providerRequestBlocks > 1) {
        state.invariantViolations.push("multiple_provider_request_blocks");
      }
      ctx.abort();
      return;
    }
    state.providerRequestAdmissions += 1;
  });

  pi.on("message_end", (event, ctx) => {
    if (event.message.role !== "assistant" || !Array.isArray(event.message.content)) {
      return;
    }
    const state = budgetState();
    const toolAdmissionLimit = configuredInteger(
      pi,
      "gym-tool-limit",
      DEFAULT_TOOL_ADMISSION_LIMIT,
    );
    const remaining = Math.max(
      0,
      toolAdmissionLimit + 1 - state.toolAttemptCount,
    );
    const toolCalls = event.message.content.filter(
      (item): item is typeof item & { id: string; type: "toolCall" } =>
        item.type === "toolCall" && typeof item.id === "string",
    );
    if (toolCalls.length <= remaining) return;

    // Tool batches are atomic. Pi emits one terminal tool-attempt event after
    // observing the abort; suppress every sibling so none can be admitted.
    const terminalToolCallId = toolCalls[0].id;
    for (const item of toolCalls.slice(1)) {
      uniquePush(state.suppressedToolRequestIds, item.id, state, "suppressed_tool_request_id");
    }
    ctx.abort();
    return {
      ...event.message,
      content: event.message.content.filter(
        (item) => item.type !== "toolCall" || item.id === terminalToolCallId,
      ),
    };
  });

  pi.on("tool_execution_start", (event, ctx) => {
    const state = budgetState();
    const toolAdmissionLimit = configuredInteger(
      pi,
      "gym-tool-limit",
      DEFAULT_TOOL_ADMISSION_LIMIT,
    );
    state.toolAttemptCount += 1;
    if (state.toolAttemptCount > toolAdmissionLimit + 1) {
      state.invariantViolations.push("tool_attempt_limit_bypassed");
      ctx.abort();
      return;
    }
    if (state.toolAttemptIds.includes(event.toolCallId)) {
      state.invariantViolations.push(`duplicate_tool_attempt_id:${event.toolCallId}`);
      ctx.abort();
      return;
    }
    uniquePush(state.toolAttemptIds, event.toolCallId, state, "tool_attempt_id");
  });

  pi.on("tool_call", (event, ctx) => {
    if (!BUDGETED_TOOLS.has(event.toolName)) return;

    const state = budgetState();
    const limit = configuredInteger(
      pi,
      "gym-tool-limit",
      DEFAULT_TOOL_ADMISSION_LIMIT,
    );
    if (ctx.signal?.aborted || state.admittedToolCallIds.length >= limit) {
      ctx.abort();
      return {
        block: true,
        reason: `Tool call limit reached (${limit}); submit the current workspace without another model turn.`,
        terminate: true,
      };
    }

    if (!state.toolAttemptIds.includes(event.toolCallId)) {
      state.invariantViolations.push(`admission_without_attempt:${event.toolCallId}`);
    }
    uniquePush(
      state.admittedToolCallIds,
      event.toolCallId,
      state,
      "admitted_tool_call_id",
    );
  });

  pi.on("tool_result", (event) => {
    if (!BUDGETED_TOOLS.has(event.toolName)) return;

    const state = budgetState();
    if (!state.admittedToolCallIds.includes(event.toolCallId)) {
      state.invariantViolations.push(`execution_without_admission:${event.toolCallId}`);
    }
    uniquePush(
      state.executedToolCallIds,
      event.toolCallId,
      state,
      "executed_tool_call_id",
    );

    const limit = configuredInteger(
      pi,
      "gym-tool-limit",
      DEFAULT_TOOL_ADMISSION_LIMIT,
    );
    if (state.admittedToolCallIds.length < limit) return;
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

  pi.on("agent_settled", () => emitReceipt(pi));
}
