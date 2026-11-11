/** A gym task: what the agent must optimize, what it may touch, and how it is scored. */
export type Task = {
  id: string;
  title: string;
  /** Mission text shown to the agent (without per-instance data like baselines). */
  brief: string;
  /** Files the agent is allowed to edit, relative to the instance root (a llama.cpp checkout). */
  allowedPaths: string[];
  bench: { pp: number; tg: number; reps: number };
  correctness: { prompt: string; nPredict: number; minSimilarity: number };
  /**
   * Minimum tg gain (percent vs baseline) that counts as an improvement.
   * Must sit above run-to-run Metal measurement noise (~2-3% at tg128 with 3 reps).
   */
  minTgGainPct: number;
};

export const metalKernelTask: Task = {
  id: "metal-kernels",
  title: "Optimize llama.cpp Metal kernels for token generation",
  allowedPaths: ["ggml/src/ggml-metal/ggml-metal.metal", "ggml/src/ggml-metal/ggml-metal-impl.h"],
  bench: { pp: 512, tg: 128, reps: 3 },
  minTgGainPct: 5,
  correctness: {
    prompt: "The theory of relativity states",
    nPredict: 96,
    minSimilarity: 0.8,
  },
  brief: `Increase token-generation (tg) throughput of llama.cpp on the Apple Metal GPU while keeping outputs correct.

The workspace is a llama.cpp checkout. You may ONLY edit the Metal kernel sources listed in the rules; everything else must stay byte-identical.

Workflow that works:
1. Read the Metal kernel file and find the kernels on the decode (tg) hot path. For small models, decode is memory-bandwidth bound and dominated by the matrix-vector kernels (mul_mv for q4_K/q8_0/f16) and the quantize/dequant helpers they call.
2. Make ONE small, focused change (threadgroup sizing, vector width, simdgroup reductions, fewer divergent branches, better memory access patterns).
3. Run the eval tool: it rebuilds, checks correctness against a fixed prompt, and benchmarks. Takes a few minutes - use it deliberately, not after every speculative edit.
4. If tg improved and correctness passed: commit with the git tool (add, then commit <what and why>). Otherwise revert your edit and try a different idea.

Stop and finish when you are out of credible ideas or have a committed improvement you cannot beat.`,
};

export const tasks: Record<string, Task> = {
  [metalKernelTask.id]: metalKernelTask,
};

export function getTask(id: string): Task {
  const task = tasks[id];
  if (!task) throw new Error(`unknown task '${id}'. available: ${Object.keys(tasks).join(", ")}`);
  return task;
}
