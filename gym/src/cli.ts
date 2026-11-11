import path from "node:path";
import { exec } from "@gym/agent/exec";
import { llmConfig, agentLlmPort, MODEL_URL, paths } from "./config.ts";
import { downloadModel, bin, buildLlama, ensureLlamaRepo, runBench, startLlamaServer } from "./llama.ts";
import { createInstance, listInstances, loadInstance, saveResults } from "./instance.ts";
import { evaluate } from "./eval.ts";
import { runInstance } from "./runner.ts";
import { getTask, metalKernelTask } from "./task.ts";

const USAGE = `gym - a gym where AI agents optimize GPU kernels for llama.cpp inference on Metal

usage: bun run gym <command> [args]

  setup              clone + build llama.cpp (Metal), download the model, run a smoke bench
  serve              start a local llama-server as a fallback agent brain (Ctrl+C to stop)
  new <name>         create an instance: copy llama.cpp, git-seed, build, measure baseline
  run <name> [steps] run the agent loop in the instance (edit kernels, eval, commit)
  eval <name>        rebuild + benchmark + correctness check for the instance, as-is
  list               list instances with baseline / best tg scores
  task               show the current task brief
  help               this text

environment:
  MODEL_API_KEY      Meta Model API key (the default agent brain)
  GYM_LLM_BASE_URL   agent brain endpoint (default https://api.meta.ai/v1, OpenAI-compatible)
  GYM_LLM_API_KEY    overrides MODEL_API_KEY
  GYM_LLM_MODEL      model id (default muse-spark-1.3-contributor)
  GYM_LLM_REASONING_EFFORT  reasoning effort for Meta reasoning models (default high)
  GYM_LLM_MAX_TOKENS  max completion tokens per agent step (default 4096)
  GYM_LLM_TEMPERATURE  sampling temperature (unset by default; reasoning models may reject it)
  GYM_LLM_PORT       port for \`gym serve\` (default 8080)
  GYM_MAX_STEPS      agent action budget (default 24)
  GYM_MODEL          path to the .gguf model
  GYM_LLAMA_DIR      llama.cpp checkout to use (default cache/llama.cpp)`;

async function cmdSetup(): Promise<void> {
  if (process.platform !== "darwin") {
    throw new Error(`this gym targets macOS + Metal; you are on ${process.platform}`);
  }
  const bunV = await exec("bun", ["--version"], { cwd: paths.root, timeoutMs: 15_000 });
  console.log("[gym] doctor:");
  console.log(`  platform  ${process.platform} ${process.arch}`);
  console.log(`  bun       ${bunV.stdout.trim()}`);
  console.log(`  model     ${paths.model}`);
  console.log(`  llama.cpp ${paths.llamaDir}`);

  await ensureLlamaRepo();
  await buildLlama(paths.llamaDir);
  await downloadModel(MODEL_URL, paths.model);

  console.log("[gym] smoke benchmark (pp128/tg64, 1 rep) ...");
  const r = await runBench({
    bin: bin(path.join(paths.llamaDir, "build"), "llama-bench"),
    model: paths.model,
    pp: 128,
    tg: 64,
    reps: 1,
  });
  console.log(`[gym] backends: ${r.backends} on ${r.gpuInfo}`);
  console.log(`[gym] pp128=${r.pp?.toFixed(2)} t/s  tg64=${r.tg.toFixed(2)} t/s`);
  console.log("\nnext steps:");
  console.log("  1. bun run gym serve        # start the agent-LLM server (or point GYM_LLM_* elsewhere)");
  console.log("  2. bun run gym new demo     # create an instance with a measured baseline");
  console.log("  3. bun run gym run demo     # let the agent optimize");
  console.log("  4. bun run gym list         # compare instances");
}

async function cmdNew(name?: string): Promise<void> {
  if (!name) throw new Error("usage: gym new <name>");
  const inst = await createInstance(name, metalKernelTask);
  console.log(`[gym] instance '${inst.name}' ready.`);
  console.log(
    `      baseline tg=${inst.baseline.tg.toFixed(2)} t/s, pp=${inst.baseline.pp?.toFixed(2)} t/s on ${inst.baseline.gpuInfo}`,
  );
}

async function cmdEval(name?: string): Promise<void> {
  if (!name) throw new Error("usage: gym eval <name>");
  const inst = await loadInstance(name);
  console.log(`[gym] evaluating '${name}' ...`);
  const result = await evaluate(inst.dir, inst.task, { baselineText: inst.baseline.correctnessText });
  await saveResults(inst.dir, result, "manual");
  printEval(name, result, inst.baseline.tg);
}

async function cmdRun(name?: string, maxSteps?: number): Promise<void> {
  if (!name) throw new Error("usage: gym run <name>");
  const summary = await runInstance(name, maxSteps ? { maxSteps } : {});
  const pct = ((summary.tg / summary.baselineTg - 1) * 100).toFixed(2);
  console.log("\n=== run summary ===");
  console.log(`improved: ${summary.improved ? "YES" : "no"}`);
  console.log(`tg: ${summary.baselineTg.toFixed(2)} -> ${summary.tg.toFixed(2)} t/s (${pct}%)`);
  console.log(`trajectory: ${summary.trajectoryPath}`);
}

async function cmdList(): Promise<void> {
  const rows = await listInstances();
  if (rows.length === 0) {
    console.log("(no instances yet; create one with `bun run gym new <name>`)");
    return;
  }
  const pad = (s: string, n: number) => (s.length >= n ? s : s + " ".repeat(n - s.length));
  console.log(`${pad("name", 16)}${pad("task", 16)}${pad("baseline tg", 13)}${pad("best tg", 13)}last pass`);
  for (const r of rows) {
    console.log(
      `${pad(r.name, 16)}${pad(r.task, 16)}${pad(r.baselineTg !== null ? r.baselineTg.toFixed(2) : "-", 13)}${pad(
        r.bestTg !== null ? r.bestTg.toFixed(2) : "-",
        13,
      )}${r.lastPass === null ? "-" : r.lastPass ? "yes" : "NO"}`,
    );
  }
}

async function cmdServe(): Promise<void> {
  const buildDir = path.join(paths.llamaDir, "build");
  const cfg = llmConfig();
  const pathName = new URL(cfg.baseUrl).pathname;
  console.log(`[gym] starting agent-LLM server: ${path.basename(paths.model)}`);
  console.log(`[gym] OpenAI-compatible endpoint: http://127.0.0.1:${agentLlmPort()}${pathName}`);
  const server = await startLlamaServer({
    bin: bin(buildDir, "llama-server"),
    model: paths.model,
    port: agentLlmPort(),
  });
  console.log("[gym] ready. press Ctrl+C to stop.");
  const shutdown = async () => {
    console.log("\n[gym] stopping ...");
    await server.stop();
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
  await new Promise<void>(() => {});
}

function printEval(name: string, result: Awaited<ReturnType<typeof evaluate>>, baselineTg: number): void {
  const delta = ((result.bench.tg / baselineTg - 1) * 100).toFixed(2);
  console.log(
    `[gym] ${name}: tg=${result.bench.tg.toFixed(2)} t/s (${delta}% vs baseline) pp=${result.bench.pp?.toFixed(2)} t/s`,
  );
  console.log(
    `[gym] correctness: ${result.correctness.passed ? "PASS" : "FAIL"} (similarity ${result.correctness.similarity.toFixed(3)})`,
  );
}

async function main(): Promise<void> {
  const [cmd, arg, arg2] = process.argv.slice(2);
  switch (cmd) {
    case "setup":
      return cmdSetup();
    case "serve":
      return cmdServe();
    case "new":
      return cmdNew(arg);
    case "run":
      return cmdRun(arg, arg2 ? Number(arg2) : undefined);
    case "eval":
      return cmdEval(arg);
    case "list":
      return cmdList();
    case "task":
      console.log(getTask("metal-kernels").brief);
      return;
    case undefined:
    case "help":
    case "--help":
    case "-h":
      console.log(USAGE);
      return;
    default:
      throw new Error(`unknown command '${cmd}'\n\n${USAGE}`);
  }
}

main().catch((err) => {
  console.error(`[gym] error: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
});
