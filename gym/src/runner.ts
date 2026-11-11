import fsp from "node:fs/promises";
import path from "node:path";
import { Agent, OpenAICompatLLM, makeToolkit } from "@gym/agent";
import type { Step, Tool } from "@gym/agent";
import { exec } from "@gym/agent/exec";
import { llmConfig, paths, defaultMaxSteps } from "./config.ts";
import { evaluate } from "./eval.ts";
import { bin } from "./llama.ts";
import { loadInstance, saveResults, type Instance } from "./instance.ts";

function buildSystemPrompt(inst: Instance): string {
  const b = inst.baseline;
  return [
    "You are a GPU kernel optimization agent working autonomously inside a git repository.",
    "",
    "## Machine",
    `- ${b.gpuInfo || "Apple Silicon"}, macOS arm64, Metal GPU`,
    `- llama.cpp checkout (your working directory): ${inst.dir}`,
    `- Benchmark model: ${path.basename(paths.model)}`,
    "",
    "## Baseline (measured, do not re-measure)",
    `- tg${inst.task.bench.tg}: ${b.tg.toFixed(2)} t/s`,
    `- pp${inst.task.bench.pp}: ${b.pp !== null ? b.pp.toFixed(2) : "n/a"} t/s`,
    "",
    "## Rules",
    `- You may ONLY edit: ${inst.task.allowedPaths.join(", ")}`,
    "- Verify a change with the eval tool (rebuild + correctness + benchmark) before committing it.",
    "- Commit every verified improvement with the git tool (add, then commit <what and why>).",
    "- If a change did not help or broke correctness, revert it (shell: git checkout -- <file>).",
    "- Your action budget is limited: one focused change per eval, no re-running eval without edits.",
  ].join("\n");
}

function makeEvalTool(inst: Instance): Tool {
  return {
    name: "eval",
    description:
      "Rebuild llama.cpp from source, run the Metal benchmark (pp/tg tokens/s) and the correctness check. Takes a few minutes. Input: ignored. Output: JSON with tg, pp, correctness and delta vs baseline.",
    async execute() {
      const result = await evaluate(inst.dir, inst.task, { baselineText: inst.baseline.correctnessText });
      await saveResults(inst.dir, result, "agent");
      const tgDelta = ((result.bench.tg / inst.baseline.tg - 1) * 100).toFixed(2);
      return {
        ok: result.pass,
        output: JSON.stringify(
          {
            tg_ts: Number(result.bench.tg.toFixed(2)),
            tg_delta_pct: Number(tgDelta),
            pp_ts: result.bench.pp !== null ? Number(result.bench.pp.toFixed(2)) : null,
            correctness_passed: result.correctness.passed,
            correctness_similarity: Number(result.correctness.similarity.toFixed(3)),
            pass: result.pass,
            improved: result.pass && result.bench.tg > inst.baseline.tg * 1.02,
          },
          null,
          2,
        ),
      };
    },
  };
}

async function assertLlmReachable(): Promise<void> {
  const { baseUrl } = llmConfig();
  const origin = new URL(baseUrl).origin;
  try {
    await fetch(`${origin}/health`, { signal: AbortSignal.timeout(5_000) });
  } catch {
    throw new Error(
      `no LLM reachable at ${origin}. Start the local one with \`bun run gym serve\`, or point GYM_LLM_BASE_URL at any OpenAI-compatible endpoint.`,
    );
  }
}

export type RunSummary = {
  improved: boolean;
  tg: number;
  baselineTg: number;
  trajectoryPath: string;
};

/** Run the agent loop inside an instance, then verify and record the final state. */
export async function runInstance(name: string, opts: { maxSteps?: number } = {}): Promise<RunSummary> {
  const inst = await loadInstance(name);
  await assertLlmReachable();

  const llm = new OpenAICompatLLM({ ...llmConfig(), maxTokens: 1024, temperature: 0.3 });
  const tools: Tool[] = [...makeToolkit(inst.dir), makeEvalTool(inst)];
  const maxSteps = opts.maxSteps ?? defaultMaxSteps();

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const trajPath = path.join(inst.dir, ".gym", `trajectory-${stamp}.jsonl`);
  await fsp.appendFile(trajPath, "");

  const agent = new Agent({
    llm,
    tools,
    system: buildSystemPrompt(inst),
    goal: inst.task.brief,
    maxSteps,
    onStep: (s: Step) => {
      const head = s.thought ? s.thought.split("\n")[0]!.slice(0, 100) : "";
      console.log(`[agent ${s.index}] ${s.action}${s.input ? ` ${s.input.split("\n")[0]!.slice(0, 80)}` : ""} -> ${s.ok ? "ok" : "FAILED"}${head ? `\n         ${head}` : ""}`);
      fsp.appendFile(trajPath, JSON.stringify(s) + "\n").catch(() => {});
    },
  });

  console.log(`[gym] running agent on '${inst.name}' (max ${maxSteps} steps, model ${llm.model}) ...`);
  const traj = await agent.run();
  console.log(`[gym] agent finished: ${traj.summary}`);

  console.log("[gym] final verification eval ...");
  const final = await evaluate(inst.dir, inst.task, { baselineText: inst.baseline.correctnessText });
  await saveResults(inst.dir, final, "final");
  const improved = final.pass && final.bench.tg > inst.baseline.tg * 1.02;

  await exec("git", ["add", "-A"], { cwd: inst.dir, timeoutMs: 60_000 });
  const msg = `gym: ${improved ? "improvement" : "result"} tg=${final.bench.tg.toFixed(2)} t/s (baseline ${inst.baseline.tg.toFixed(2)}, pass=${final.pass})`;
  const c = await exec("git", ["commit", "-m", msg], { cwd: inst.dir, timeoutMs: 60_000 });
  if (c.code !== 0 && !/nothing to commit/.test(c.stdout + c.stderr)) {
    console.warn(`[gym] warning: could not record results commit: ${c.stderr.slice(-200)}`);
  }

  return { improved, tg: final.bench.tg, baselineTg: inst.baseline.tg, trajectoryPath: trajPath };
}
