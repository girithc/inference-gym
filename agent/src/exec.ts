import { spawn } from "node:child_process";

export type ExecResult = {
  code: number;
  stdout: string;
  stderr: string;
};

export type ExecOptions = {
  cwd: string;
  timeoutMs: number;
  env?: Record<string, string>;
};

const MAX_CAPTURE = 200_000;

/**
 * Run a program to completion. Captures output (tail only, capped at 200KB).
 * Rejects on spawn failure; non-zero exit codes are returned, not thrown.
 */
export function exec(cmd: string, args: string[], opts: ExecOptions): Promise<ExecResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, {
      cwd: opts.cwd,
      env: { ...process.env, ...opts.env },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    child.stdout?.on("data", (c: Buffer) => {
      stdout += c;
      if (stdout.length > MAX_CAPTURE) stdout = stdout.slice(-MAX_CAPTURE);
    });
    child.stderr?.on("data", (c: Buffer) => {
      stderr += c;
      if (stderr.length > MAX_CAPTURE) stderr = stderr.slice(-MAX_CAPTURE);
    });

    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      setTimeout(() => child.kill("SIGKILL"), 5_000);
    }, opts.timeoutMs);

    child.on("error", (e) => {
      clearTimeout(timer);
      reject(new Error(`failed to spawn ${cmd}: ${e.message}`));
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      resolve({ code: code ?? -1, stdout, stderr });
    });
  });
}

/** Tail of a string, for folding huge build logs into errors and observations. */
export function tail(s: string, max: number): string {
  return s.length <= max ? s : `...[earlier output trimmed]\n${s.slice(-max)}`;
}
