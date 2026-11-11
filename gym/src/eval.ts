import path from "node:path";
import { paths } from "./config.ts";
import { buildLlama, bin, runBench, startLlamaServer, llamaCommit } from "./llama.ts";
import type { Task } from "./task.ts";

export type Correctness = {
  passed: boolean;
  similarity: number;
  text: string;
};

export type EvalResult = {
  at: string;
  commit: string;
  bench: { pp: number | null; tg: number; gpuInfo: string; backends: string };
  correctness: Correctness;
  /** Overall gate: correctness must pass for a result to count. */
  pass: boolean;
};

/** Greedy, seeded completion through llama-server /completion. Deterministic on unchanged binaries. */
export async function generate(buildDir: string, model: string, prompt: string, nPredict: number): Promise<string> {
  const server = await startLlamaServer({ bin: bin(buildDir, "llama-server"), model });
  try {
    const res = await fetch(`http://127.0.0.1:${server.port}/completion`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt, n_predict: nPredict, temperature: 0, seed: 42, cache_prompt: false }),
    });
    if (!res.ok) throw new Error(`/completion HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
    const data = (await res.json()) as { content: string };
    return data.content.trim();
  } finally {
    await server.stop();
  }
}

/** Word-level Jaccard similarity on normalized text; robust to minor numeric drift between kernels. */
export function textSimilarity(a: string, b: string): number {
  const norm = (s: string) =>
    s
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, " ")
      .trim();
  const aw = new Set(norm(a).split(" ").filter(Boolean));
  const bw = new Set(norm(b).split(" ").filter(Boolean));
  if (aw.size === 0 || bw.size === 0) return 0;
  let inter = 0;
  for (const w of aw) if (bw.has(w)) inter++;
  return inter / (aw.size + bw.size - inter);
}

/**
 * Full evaluation of an instance: rebuild from source, benchmark (pp/tg), and check
 * output correctness against the baseline generation. Writes nothing; callers persist.
 */
export async function evaluate(
  dir: string,
  task: Task,
  opts: { baselineText?: string } = {},
): Promise<EvalResult> {
  const buildDir = await buildLlama(dir);
  const model = paths.model;

  const bench = await runBench({
    bin: bin(buildDir, "llama-bench"),
    model,
    pp: task.bench.pp,
    tg: task.bench.tg,
    reps: task.bench.reps,
  });

  const text = await generate(buildDir, model, task.correctness.prompt, task.correctness.nPredict);
  const similarity = opts.baselineText ? textSimilarity(text, opts.baselineText) : 1;
  const correctness: Correctness = {
    passed: similarity >= task.correctness.minSimilarity,
    similarity,
    text,
  };

  return {
    at: new Date().toISOString(),
    commit: await llamaCommit(dir),
    bench: { pp: bench.pp, tg: bench.tg, gpuInfo: bench.gpuInfo, backends: bench.backends },
    correctness,
    pass: correctness.passed,
  };
}
