import fsp from "node:fs/promises";
import path from "node:path";
import { exec, tail } from "./exec.ts";
import type { Tool } from "./types.ts";

/** Resolve a workspace-relative path and refuse anything that escapes it. */
function safeJoin(root: string, rel: string): string {
  const p = path.resolve(root, rel.trim());
  if (p !== root && !p.startsWith(root + path.sep)) {
    throw new Error(`path escapes workspace: ${rel}`);
  }
  return p;
}

export function makeReadFileTool(root: string): Tool {
  return {
    name: "read_file",
    description: "Read a text file. Input: relative path.",
    async execute(input) {
      const p = safeJoin(root, input);
      const stat = await fsp.stat(p).catch(() => null);
      if (!stat) return { ok: false, output: `no such file: ${input.trim()}` };
      if (stat.size > 1_000_000) return { ok: false, output: "file too large (>1MB); read it in pieces with shell" };
      const content = await fsp.readFile(p, "utf8");
      return { ok: true, output: content };
    },
  };
}

export function makeWriteFileTool(root: string): Tool {
  return {
    name: "write_file",
    description: "Create or overwrite a text file. Input: first line is the relative path, remaining lines are the content.",
    async execute(input) {
      const nl = input.indexOf("\n");
      if (nl === -1) return { ok: false, output: "input must be: <relative path>\\n<file content>" };
      const rel = input.slice(0, nl).trim();
      const content = input.slice(nl + 1);
      const p = safeJoin(root, rel);
      await fsp.mkdir(path.dirname(p), { recursive: true });
      await fsp.writeFile(p, content);
      return { ok: true, output: `wrote ${rel} (${content.length} bytes)` };
    },
  };
}

const SKIP_DIRS = new Set([".git", "build", "node_modules", ".gym"]);

export function makeListFilesTool(root: string): Tool {
  return {
    name: "list_files",
    description: "List workspace files under a directory. Input: relative directory ('.' for the root).",
    async execute(input) {
      const base = safeJoin(root, input || ".");
      const out: string[] = [];
      await walk(base, root, out, 0);
      if (out.length === 0) return { ok: true, output: "(no files)" };
      return {
        ok: true,
        output: out.slice(0, 500).join("\n") + (out.length > 500 ? `\n... (${out.length} files total)` : ""),
      };
    },
  };
}

async function walk(dir: string, root: string, out: string[], depth: number): Promise<void> {
  if (depth > 4 || out.length > 500) return;
  const entries = await fsp.readdir(dir, { withFileTypes: true }).catch(() => []);
  for (const e of entries) {
    if (SKIP_DIRS.has(e.name)) continue;
    const full = path.join(dir, e.name);
    if (e.isDirectory()) await walk(full, root, out, depth + 1);
    else out.push(path.relative(root, full));
  }
}

export function makeShellTool(root: string, shellTimeoutMs = 10 * 60_000): Tool {
  return {
    name: "shell",
    description: "Run a bash command in the workspace root (long builds are fine). Input: the command.",
    async execute(input) {
      const cmd = input.trim();
      if (!cmd) return { ok: false, output: "empty command" };
      const r = await exec("/bin/bash", ["-c", cmd], { cwd: root, timeoutMs: shellTimeoutMs });
      const output = (r.stdout + (r.stderr ? `\n${r.stderr}` : "")).trim() || "(no output)";
      return { ok: r.code === 0, output: r.code === 0 ? output : `${output}\n[exit code: ${r.code}]` };
    },
  };
}

export function makeGitTool(root: string): Tool {
  return {
    name: "git",
    description: "Version control. Input: status | diff [path] | add | commit <message> | log [count]",
    async execute(input) {
      const parts = input.trim().split(/\s+/);
      const sub = parts[0] ?? "";
      const rest = parts.slice(1).join(" ");
      const run = async (args: string[]) => {
        const r = await exec("git", args, { cwd: root, timeoutMs: 60_000 });
        return { ok: r.code === 0, output: (r.stdout + (r.stderr ? `\n${r.stderr}` : "")).trim() || "(no output)" };
      };
      switch (sub) {
        case "status":
          return run(["status", "--short", "--branch"]);
        case "diff":
          return run(rest ? ["--no-pager", "diff", "--", rest] : ["--no-pager", "diff"]);
        case "add":
          return run(["add", "-A"]);
        case "commit": {
          const msg = rest.trim();
          if (!msg) return { ok: false, output: "usage: commit <message>" };
          return run(["commit", "-am", msg]);
        }
        case "log":
          return run(["--no-pager", "log", "--oneline", "-n", rest || "10"]);
        default:
          return { ok: false, output: "usage: status | diff [path] | add | commit <message> | log [count]" };
      }
    },
  };
}

/** Standard toolkit for an agent working inside a directory. */
export function makeToolkit(root: string, opts: { shellTimeoutMs?: number } = {}): Tool[] {
  return [
    makeShellTool(root, opts.shellTimeoutMs),
    makeReadFileTool(root),
    makeWriteFileTool(root),
    makeListFilesTool(root),
    makeGitTool(root),
  ];
}

export { tail };
