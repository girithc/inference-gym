import fsp from "node:fs/promises";
import path from "node:path";
import { exec } from "@gym/agent/exec";
import { paths } from "./config.ts";
import { evaluate, type EvalResult } from "./eval.ts";
import { buildLlama, hasLlamaRepo, llamaCommit } from "./llama.ts";
import type { Task } from "./task.ts";
import { getTask } from "./task.ts";

export type InstanceMeta = {
  name: string;
  task: string;
  createdAt: string;
  llamaCommit: string;
};

export type Baseline = {
  tg: number;
  pp: number | null;
  gpuInfo: string;
  correctnessText: string;
  commit: string;
};

export type Instance = {
  name: string;
  dir: string;
  task: Task;
  meta: InstanceMeta;
  baseline: Baseline;
};

export const gymDirOf = (dir: string) => path.join(dir, ".gym");

const NAME_RE = /^[a-zA-Z0-9][a-zA-Z0-9._-]*$/;

async function exists(p: string): Promise<boolean> {
  return fsp.stat(p).then(() => true).catch(() => false);
}

async function git(dir: string, args: string[]): Promise<string> {
  const r = await exec("git", args, { cwd: dir, timeoutMs: 60_000 });
  if (r.code !== 0) throw new Error(`git ${args.join(" ")} failed: ${r.stderr.slice(-500)}`);
  return r.stdout.trim();
}

/** Commit with a fallback identity when the host has no global git user configured. */
async function commitAll(dir: string, message: string): Promise<void> {
  await git(dir, ["add", "-A"]);
  const email = await exec("git", ["config", "user.email"], { cwd: dir, timeoutMs: 15_000 });
  const hasIdentity = email.code === 0 && email.stdout.trim() !== "";
  const idArgs = hasIdentity ? [] : ["-c", "user.name=gym-agent", "-c", "user.email=agent@gym.local"];
  const r = await exec("git", [...idArgs, "commit", "-m", message], { cwd: dir, timeoutMs: 60_000 });
  if (r.code !== 0 && !/nothing to commit/.test(r.stdout + r.stderr)) {
    throw new Error(`git commit failed: ${r.stderr.slice(-500)}`);
  }
}

async function writeJson(p: string, data: unknown): Promise<void> {
  await fsp.writeFile(p, JSON.stringify(data, null, 2) + "\n");
}

async function readJson<T>(p: string): Promise<T> {
  return JSON.parse(await fsp.readFile(p, "utf8")) as T;
}

/**
 * Create an instance: a full copy of the cached llama.cpp checkout (its own git repo),
 * seeded with the task, built from source, with a measured baseline committed to git.
 */
export async function createInstance(name: string, task: Task): Promise<Instance> {
  if (!NAME_RE.test(name)) throw new Error(`invalid instance name '${name}'`);
  const dir = path.join(paths.instances, name);
  if (await exists(dir)) throw new Error(`instance '${name}' already exists at ${dir}`);
  if (!(await hasLlamaRepo())) {
    throw new Error("llama.cpp is not cached yet; run `bun run gym setup` first");
  }

  console.log(`[gym] creating instance '${name}' at ${dir} ...`);
  await fsp.mkdir(paths.instances, { recursive: true });
  await fsp.cp(paths.llamaDir, dir, {
    recursive: true,
    filter: (src) => !/(^|[\\/])build([\\/]|$)/.test(src), // never copy build artifacts; instances build their own
  });

  await fsp.mkdir(gymDirOf(dir), { recursive: true });
  await writeJson(path.join(gymDirOf(dir), "instance.json"), {
    name,
    task: task.id,
    createdAt: new Date().toISOString(),
    llamaCommit: await llamaCommit(dir),
  } satisfies InstanceMeta);
  await fsp.writeFile(path.join(dir, "TASK.md"), taskMarkdown(task));
  await commitAll(dir, `gym: seed instance '${name}' (task ${task.id}, llama.cpp @ ${(await llamaCommit(dir)).slice(0, 8)})`);

  await buildLlama(dir);

  console.log("[gym] measuring baseline (bench + correctness) ...");
  const base = await evaluate(dir, task);
  const baseline: Baseline = {
    tg: base.bench.tg,
    pp: base.bench.pp,
    gpuInfo: base.bench.gpuInfo,
    correctnessText: base.correctness.text,
    commit: base.commit,
  };
  await writeJson(path.join(gymDirOf(dir), "baseline.json"), baseline);
  await saveResults(dir, base, "baseline");
  await commitAll(dir, `gym: baseline tg=${base.bench.tg.toFixed(2)} t/s (pass=${base.pass})`);

  return { name, dir, task, meta: await readJson(path.join(gymDirOf(dir), "instance.json")), baseline };
}

export function taskMarkdown(task: Task): string {
  return [
    `# Task: ${task.title}`,
    "",
    task.brief,
    "",
    "## Scoring",
    `- Benchmark: llama-bench, prompt ${task.bench.pp} tokens, ${task.bench.tg} generated, ${task.bench.reps} reps.`,
    "- Correctness: greedy completion of a fixed prompt must stay similar to the baseline output.",
    `- Results are written to .gym/results.json after every eval; history in .gym/history.jsonl.`,
    "",
  ].join("\n");
}

export async function loadInstance(name: string): Promise<Instance> {
  const dir = path.join(paths.instances, name);
  if (!(await exists(dir))) throw new Error(`no instance '${name}' at ${dir}`);
  const meta = await readJson<InstanceMeta>(path.join(gymDirOf(dir), "instance.json"));
  const task = getTask(meta.task);
  if (!(await exists(path.join(gymDirOf(dir), "baseline.json")))) {
    throw new Error(`instance '${name}' has no baseline; recreate it with \`bun run gym new ${name}\``);
  }
  const baseline = await readJson<Baseline>(path.join(gymDirOf(dir), "baseline.json"));
  return { name, dir, task, meta, baseline };
}

export async function saveResults(dir: string, result: EvalResult, label: string): Promise<void> {
  await writeJson(path.join(gymDirOf(dir), "results.json"), { label, ...result });
  await fsp.appendFile(path.join(gymDirOf(dir), "history.jsonl"), JSON.stringify({ label, ...result }) + "\n");
}

export type InstanceRow = {
  name: string;
  task: string;
  createdAt: string;
  baselineTg: number | null;
  bestTg: number | null;
  lastPass: boolean | null;
  lastAt: string | null;
};

export async function listInstances(): Promise<InstanceRow[]> {
  const rows: InstanceRow[] = [];
  let entries: string[] = [];
  try {
    entries = (await fsp.readdir(paths.instances)).sort();
  } catch {
    return rows;
  }
  for (const name of entries) {
    const dir = path.join(paths.instances, name);
    const metaPath = path.join(gymDirOf(dir), "instance.json");
    if (!(await exists(metaPath))) continue;
    const meta = await readJson<InstanceMeta>(metaPath);
    const baseline = await readJson<Baseline | null>(path.join(gymDirOf(dir), "baseline.json")).catch(() => null);
    let bestTg: number | null = null;
    let lastPass: boolean | null = null;
    let lastAt: string | null = null;
    const historyPath = path.join(gymDirOf(dir), "history.jsonl");
    if (await exists(historyPath)) {
      const lines = (await fsp.readFile(historyPath, "utf8")).split("\n").filter((l) => l.trim());
      for (const line of lines) {
        const r = JSON.parse(line) as EvalResult;
        lastPass = r.pass;
        lastAt = r.at;
        if (r.pass && (bestTg === null || r.bench.tg > bestTg)) bestTg = r.bench.tg;
      }
    }
    rows.push({
      name,
      task: meta.task,
      createdAt: meta.createdAt,
      baselineTg: baseline?.tg ?? null,
      bestTg,
      lastPass,
      lastAt,
    });
  }
  return rows;
}
