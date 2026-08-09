/**
 * Pyreplab Bayesian Agent Harness — Gym Tools Extension
 *
 * Exposes exactly one model-callable tool named `bash` that routes commands
 * over a persistent SSH child process to a Python JSON-lines worker on a
 * configured Linux host. The worker executes commands inside an isolated task
 * sandbox (Bubblewrap).
 *
 * Architecture:
 *   Pi 0.84.1 on macOS
 *     -> SSH child process (persistent, spawned on session_start)
 *     -> Python worker on the configured SSH host (JSONL protocol)
 *     -> Bubblewrap-isolated task workspace
 *
 * Usage:
 *   pi --no-builtin-tools --no-extensions \
 *      -e <model-switch-extension> \
 *      -e pi_extensions/gym-tools.ts \
 *      --gym-root /path/to/root \
 *      --gym-workspace /path/to/workspace \
 *      --gym-project /path/to/project
 */

import { spawn, type ChildProcess } from "node:child_process";
import { createInterface, type Interface } from "node:readline";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

// ---------------------------------------------------------------------------
// Shell quoting — safe single-quote escape
// ---------------------------------------------------------------------------

/** Wrap a string in single quotes, escaping any embedded single quotes. */
function shellQuote(s: string): string {
  return `'${s.replace(/'/g, "'\\''")}'`;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

const NEWLINE_OR_NUL = /[\n\0]/;

/** Reject values that contain newlines or NUL bytes. */
function validateNoNewlines(label: string, value: string): void {
  if (NEWLINE_OR_NUL.test(value)) {
    throw new Error(`${label} must not contain newlines or NUL bytes`);
  }
}

/** Validate a required absolute remote path. */
function validateRemotePath(label: string, value: string): void {
  if (!value.startsWith("/")) {
    throw new Error(`${label} must be an absolute path starting with "/"`);
  }
  validateNoNewlines(label, value);
}

// ---------------------------------------------------------------------------
// JSON-RPC client state
// ---------------------------------------------------------------------------

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
  signal?: AbortSignal;
  onAbort?: () => void;
}

let child: ChildProcess | null = null;
let rl: Interface | null = null;
const pending = new Map<number, PendingRequest>();
let nextId = 0;
let stderrBuf = "";
const MAX_STDERR_BUF = 8192;

/** Append to bounded stderr buffer. */
function appendStderr(data: string): void {
  stderrBuf += data;
  if (stderrBuf.length > MAX_STDERR_BUF) {
    stderrBuf = stderrBuf.slice(-MAX_STDERR_BUF);
  }
}

// ---------------------------------------------------------------------------
// JSON-RPC send and receive
// ---------------------------------------------------------------------------

function sendRpc(msg: { id: number; method: string; params?: unknown }): void {
  if (!child?.stdin || child.stdin.destroyed) {
    throw new Error("Worker stdin is not available");
  }
  child.stdin.write(JSON.stringify(msg) + "\n");
}

function handleRpcLine(line: string): void {
  let msg: { id: number; ok: boolean; result?: unknown; error?: { type: string; message: string } };
  try {
    msg = JSON.parse(line);
  } catch {
    // Malformed line — ignore (stderr already captures worker diagnostics)
    return;
  }

  const p = pending.get(msg.id);
  if (!p) return;

  pending.delete(msg.id);
  if (p.signal && p.onAbort) {
    p.signal.removeEventListener("abort", p.onAbort);
  }

  if (msg.ok) {
    p.resolve(msg.result);
  } else {
    const errType = msg.error?.type ?? "RemoteError";
    const errMsg = msg.error?.message ?? "Unknown remote error";
    p.reject(new Error(`${errType}: ${errMsg}`));
  }
}

/**
 * Send a JSON-RPC request and wait for the response.
 * Requests are sequential in normal Pi use; this client supports
 * concurrent use but the worker handles one at a time.
 */
function rpcCall(
  method: string,
  params: unknown,
  signal?: AbortSignal,
): Promise<unknown> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new Error("Aborted"));
      return;
    }

    const id = ++nextId;

    const onAbort = () => {
      pending.delete(id);
      reject(new Error("Aborted"));
    };

    signal?.addEventListener("abort", onAbort, { once: true });

    pending.set(id, {
      resolve: (v: unknown) => {
        signal?.removeEventListener("abort", onAbort);
        resolve(v);
      },
      reject: (e: Error) => {
        signal?.removeEventListener("abort", onAbort);
        reject(e);
      },
      signal: signal ?? undefined,
      onAbort,
    });

    try {
      sendRpc({ id, method, params });
    } catch (e) {
      pending.delete(id);
      signal?.removeEventListener("abort", onAbort);
      reject(e);
    }
  });
}

/** Reject all outstanding requests with the given error. */
function rejectAllPending(err: Error): void {
  for (const [id, p] of pending) {
    if (p.signal && p.onAbort) {
      p.signal.removeEventListener("abort", p.onAbort);
    }
    p.reject(err);
    pending.delete(id);
  }
}

// ---------------------------------------------------------------------------
// Worker lifecycle
// ---------------------------------------------------------------------------

function buildRemoteCommand(
  python: string,
  project: string,
  root: string,
  workspace: string,
  commandTimeout: number,
  memoryMax: string,
  tasksMax: number,
  cpuQuota: string,
): string {
  return (
    `PYTHONPATH=${shellQuote(project + "/src")} ` +
    `${shellQuote(python)} -u -m pyreplab_harness serve-worker ` +
    `--root ${shellQuote(root)} ` +
    `--workspace ${shellQuote(workspace)} ` +
    `--max-timeout ${commandTimeout} ` +
    `--memory-max ${shellQuote(memoryMax)} ` +
    `--tasks-max ${tasksMax} ` +
    `--cpu-quota ${shellQuote(cpuQuota)}`
  );
}

function startWorker(
  host: string,
  python: string,
  project: string,
  root: string,
  workspace: string,
  commandTimeout: number,
  memoryMax: string,
  tasksMax: number,
  cpuQuota: string,
): void {
  const remoteCmd = buildRemoteCommand(
    python, project, root, workspace,
    commandTimeout, memoryMax, tasksMax, cpuQuota,
  );

  child = spawn("ssh", [
    "-T",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    host,
    remoteCmd,
  ], {
    stdio: ["pipe", "pipe", "pipe"],
  });

  // Line-buffered JSON-RPC reader on stdout
  rl = createInterface({ input: child.stdout! });
  rl.on("line", handleRpcLine);

  // Capture bounded stderr for diagnostics
  child.stderr!.on("data", (data: Buffer) => {
    appendStderr(data.toString("utf-8"));
  });

  // Child exit: reject all pending requests
  child.on("close", (code, sig) => {
    const reason = sig ? `killed by signal ${sig}` : `exited with code ${code}`;
    rejectAllPending(new Error(`Worker process ${reason}`));
    rl?.close();
    rl = null;
    child = null;
  });

  child.on("error", (err) => {
    rejectAllPending(new Error(`Failed to spawn worker: ${err.message}`));
    rl?.close();
    rl = null;
    child = null;
  });
}

/** Request graceful shutdown, then terminate process idempotently. */
async function shutdownWorker(): Promise<void> {
  if (!child) return;

  const c = child;
  const r = rl;

  // Try graceful shutdown request first
  try {
    await rpcCall("shutdown", {});
  } catch {
    // Ignore — worker may already be gone
  }

  // Terminate idempotently
  try {
    if (!c.killed && c.exitCode === null) {
      c.kill("SIGTERM");
      // Give it a brief moment, then SIGKILL
      setTimeout(() => {
        if (!c.killed && c.exitCode === null) {
          c.kill("SIGKILL");
        }
      }, 3000);
    }
  } catch {
    // Ignore
  }

  r?.close();
  child = null;
  rl = null;
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

export default function (pi: ExtensionAPI) {
  // ---- CLI flags ----

  pi.registerFlag("gym-host", {
    description: "SSH host running the Python gym worker",
    type: "string",
    default: "ubuntu-local",
  });

  pi.registerFlag("gym-python", {
    description: "Python interpreter on the remote host",
    type: "string",
    default: "python3",
  });

  pi.registerFlag("gym-project", {
    description: "Absolute path to pyreplab_harness project on remote (required)",
    type: "string",
  });

  pi.registerFlag("gym-root", {
    description: "Absolute path to gym root directory on remote (required)",
    type: "string",
  });

  pi.registerFlag("gym-workspace", {
    description: "Absolute path to task workspace directory on remote (required)",
    type: "string",
  });

  pi.registerFlag("gym-tool-limit", {
    description: "Maximum number of tool calls per session",
    type: "string",
    default: "8",
  });

  pi.registerFlag("gym-command-timeout", {
    description: "Maximum command execution timeout in seconds",
    type: "string",
    default: "30",
  });

  pi.registerFlag("gym-memory-max", {
    description: "Memory limit for the worker sandbox (e.g., 1G)",
    type: "string",
    default: "1G",
  });

  pi.registerFlag("gym-tasks-max", {
    description: "Maximum number of tasks for the worker sandbox",
    type: "string",
    default: "64",
  });

  pi.registerFlag("gym-cpu-quota", {
    description: "CPU quota for the worker sandbox (e.g., 200%)",
    type: "string",
    default: "200%",
  });

  pi.registerFlag("gym-max-output-tokens", {
    description: "Cap for model max_tokens in provider requests",
    type: "string",
    default: "2048",
  });

  // ---- Runtime state ----

  let toolCallCount = 0;

  function getConfig() {
    const host = (pi.getFlag("gym-host") as string) || "ubuntu-local";
    const python = (pi.getFlag("gym-python") as string) || "python3";
    const project = (pi.getFlag("gym-project") as string) || "";
    const root = (pi.getFlag("gym-root") as string) || "";
    const workspace = (pi.getFlag("gym-workspace") as string) || "";
    const toolLimit = parseInt((pi.getFlag("gym-tool-limit") as string) || "8", 10);
    const commandTimeout = parseInt((pi.getFlag("gym-command-timeout") as string) || "30", 10);
    const memoryMax = (pi.getFlag("gym-memory-max") as string) || "1G";
    const tasksMax = parseInt((pi.getFlag("gym-tasks-max") as string) || "64", 10);
    const cpuQuota = (pi.getFlag("gym-cpu-quota") as string) || "200%";
    const maxOutputTokens = parseInt((pi.getFlag("gym-max-output-tokens") as string) || "2048", 10);

    return { host, python, project, root, workspace, toolLimit, commandTimeout, memoryMax, tasksMax, cpuQuota, maxOutputTokens };
  }

  // ---- session_start: spawn persistent SSH worker ----

  pi.on("session_start", async (_event, ctx) => {
    toolCallCount = 0;

    const cfg = getConfig();

    // Validate required flags
    if (!cfg.host) {
      throw new Error("--gym-host is required");
    }
    validateNoNewlines("gym-host", cfg.host);

    if (!cfg.project) {
      throw new Error("--gym-project is required (absolute remote path)");
    }
    validateRemotePath("gym-project", cfg.project);

    if (!cfg.root) {
      throw new Error("--gym-root is required (absolute remote path)");
    }
    validateRemotePath("gym-root", cfg.root);

    if (!cfg.workspace) {
      throw new Error("--gym-workspace is required (absolute remote path)");
    }
    validateRemotePath("gym-workspace", cfg.workspace);

    validateNoNewlines("gym-python", cfg.python);

    // Spawn the persistent worker
    startWorker(
      cfg.host,
      cfg.python,
      cfg.project,
      cfg.root,
      cfg.workspace,
      cfg.commandTimeout,
      cfg.memoryMax,
      cfg.tasksMax,
      cfg.cpuQuota,
    );

    // Startup ping to confirm the worker is alive
    try {
      await rpcCall("ping", {});
    } catch (e) {
      await shutdownWorker();
      throw new Error(`Gym worker startup ping failed: ${(e as Error).message}`);
    }

    // Set active tools now that the runtime is fully initialized
    pi.setActiveTools(["bash"]);

    if (ctx.hasUI) {
      ctx.ui.notify(`Gym worker ready (${cfg.host})`, "info");
    }
  });

  // ---- Register the bash tool ----

  pi.registerTool({
    name: "bash",
    label: "Bash (sandbox)",
    description:
      "Execute a shell command in the isolated Ubuntu /workspace sandbox. " +
      "Use this to run commands, scripts, and tools. " +
      "All requested artifacts must be written under /workspace. " +
      "There are no local filesystem tools — this is the only execution tool available.",
    parameters: Type.Object({
      command: Type.String({ description: "Shell command to execute" }),
      timeout: Type.Optional(Type.Number({ description: "Timeout in seconds (clamped to configured max)" })),
    }),

    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      const cfg = getConfig();

      // Enforce tool-call limit
      if (toolCallCount >= cfg.toolLimit) {
        return {
          content: [{
            type: "text",
            text: `Tool call limit reached (${cfg.toolLimit}). No further commands can be executed in this session.`,
          }],
          details: { exit_code: -1 },
        };
      }

      toolCallCount++;

      // Clamp timeout to configured maximum
      const clampedTimeout = Math.min(
        params.timeout ?? cfg.commandTimeout,
        cfg.commandTimeout,
      );

      try {
        const result = await rpcCall("exec", {
          command: params.command,
          timeout: clampedTimeout,
        }, signal ?? undefined) as {
          stdout: string;
          stderr: string;
          exit_code: number;
          timed_out: boolean;
          truncated: boolean;
        };

        const parts: string[] = [];
        if (result.stdout) parts.push(result.stdout);
        if (result.stderr) parts.push(result.stderr);

        let prefix = "";
        if (result.timed_out) prefix += `[Command timed out after ${clampedTimeout}s]\n`;
        if (result.truncated) prefix += "[Output truncated]\n";

        return {
          content: [{
            type: "text",
            text: prefix + parts.join("\n") || "(no output)",
          }],
          details: {
            exit_code: result.exit_code,
          },
        };
      } catch (e) {
        return {
          content: [{
            type: "text",
            text: `Error executing command: ${(e as Error).message}`,
          }],
          details: {
            exit_code: -1,
            error: (e as Error).message,
          },
        };
      }
    },
  });

  // ---- before_agent_start: inject sandbox instructions ----

  pi.on("before_agent_start", async (event) => {
    const instruction =
      "\n\n" +
      "## Sandbox Environment\n" +
      "The `bash` tool executes commands in an isolated Ubuntu `/workspace` sandbox. " +
      "There are no local filesystem tools available. " +
      "All requested artifacts, files, and output must be written under `/workspace`. " +
      "The workspace is ephemeral and will be verified after the session ends.";

    return {
      systemPrompt: event.systemPrompt + instruction,
    };
  });

  // ---- before_provider_request: cap max_tokens ----

  pi.on("before_provider_request", (event) => {
    const cfg = getConfig();
    const cap = cfg.maxOutputTokens;

    if (!event.payload || typeof event.payload !== "object") return;

    const payload = event.payload as Record<string, unknown>;

    // Only cap if the payload has a max_tokens field and the cap is lower
    if ("max_tokens" in payload) {
      const current = payload.max_tokens;
      if (typeof current === "number" && current > cap) {
        return { ...payload, max_tokens: cap };
      }
    }
  });

  // ---- session_shutdown: clean up the worker ----

  pi.on("session_shutdown", async () => {
    await shutdownWorker();
  });
}
