import fsp from "node:fs/promises";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { spawn, type ChildProcess } from "node:child_process";
import { exec, tail } from "@gym/agent/exec";
import { paths } from "./config.ts";

const LLAMA_REPO = "https://github.com/ggml-org/llama.cpp";
const CMAKE_TARBALL_URL =
  "https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-macos-universal.tar.gz";
const PORTABLE_CMAKE = () => path.join(paths.cache, "tools", "cmake", "CMake.app", "Contents", "bin", "cmake");

const BUILD_FLAGS = ["-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON", "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"];

async function exists(p: string): Promise<boolean> {
  return fsp
    .stat(p)
    .then(() => true)
    .catch(() => false);
}

async function which(cmd: string): Promise<string | null> {
  const r = await exec("/bin/bash", ["-lc", `command -v ${cmd}`], { cwd: paths.root, timeoutMs: 15_000 });
  const p = r.stdout.trim().split("\n")[0] ?? "";
  return r.code === 0 && p ? p : null;
}

/** Locate cmake on PATH, else use (or download) a portable copy under cache/tools. */
export async function ensureCmake(): Promise<string> {
  const onPath = await which("cmake");
  if (onPath) return onPath;
  const portable = PORTABLE_CMAKE();
  if (await exists(portable)) return portable;

  console.log("[gym] cmake not found; downloading portable cmake ...");
  await fsp.mkdir(path.join(paths.cache, "tools"), { recursive: true });
  const tgz = path.join(paths.cache, "tools", "cmake.tar.gz");
  await exec("curl", ["-fL", "--retry", "3", "-o", tgz, CMAKE_TARBALL_URL], { cwd: paths.cache, timeoutMs: 15 * 60_000 });
  await exec("tar", ["-xzf", "cmake.tar.gz", "-C", "tools"], { cwd: paths.cache, timeoutMs: 5 * 60_000 });
  await fsp.rename(path.join(paths.cache, "tools", "cmake-3.31.6-macos-universal"), path.join(paths.cache, "tools", "cmake"));
  if (!(await exists(portable))) throw new Error("portable cmake download failed");
  return portable;
}

export async function hasLlamaRepo(dir: string = paths.llamaDir): Promise<boolean> {
  return exists(path.join(dir, ".git"));
}

/** Shallow-clone llama.cpp into dir if it is not there yet. */
export async function ensureLlamaRepo(dir: string = paths.llamaDir): Promise<string> {
  if (await hasLlamaRepo(dir)) return dir;
  console.log(`[gym] cloning llama.cpp into ${dir} ...`);
  await fsp.mkdir(path.dirname(dir), { recursive: true });
  const r = await exec("git", ["clone", "--depth", "1", LLAMA_REPO, dir], { cwd: path.dirname(dir), timeoutMs: 15 * 60_000 });
  if (r.code !== 0) throw new Error(`git clone failed:\n${tail(r.stderr, 2_000)}`);
  return dir;
}

export async function llamaCommit(dir: string): Promise<string> {
  const r = await exec("git", ["rev-parse", "HEAD"], { cwd: dir, timeoutMs: 15_000 });
  return r.stdout.trim() || "unknown";
}

/** Configure once, then build the requested llama.cpp targets. Throws with the build log tail on failure. */
export async function buildLlama(dir: string, targets: string[] = ["llama-server", "llama-bench"]): Promise<string> {
  const cmake = await ensureCmake();
  const jobs = String(os.cpus().length);

  if (!(await exists(path.join(dir, "build", "CMakeCache.txt")))) {
    console.log(`[gym] configuring llama.cpp (${dir}) ...`);
    const r = await exec(cmake, ["-B", "build", ...BUILD_FLAGS], { cwd: dir, timeoutMs: 10 * 60_000 });
    if (r.code !== 0) throw new Error(`cmake configure failed:\n${tail(r.stdout + r.stderr, 4_000)}`);
  }
  console.log(`[gym] building ${targets.join(", ")} (${dir}) ...`);
  const r = await exec(cmake, ["--build", "build", "--config", "Release", "-j", jobs, "--target", ...targets], {
    cwd: dir,
    timeoutMs: 30 * 60_000,
  });
  if (r.code !== 0) throw new Error(`build failed:\n${tail(r.stdout + r.stderr, 8_000)}`);
  return path.join(dir, "build");
}

export function bin(buildDir: string, name: "llama-server" | "llama-bench" | "llama-cli"): string {
  return path.join(buildDir, "bin", name);
}

async function isGguf(p: string): Promise<boolean> {
  const fh = await fsp.open(p, "r").catch(() => null);
  if (!fh) return false;
  try {
    const buf = Buffer.alloc(4);
    await fh.read(buf, 0, 4, 0);
    return buf.toString("ascii") === "GGUF";
  } finally {
    await fh.close();
  }
}

/** Download a GGUF model (curl, resumable-by-retry) unless a valid one already exists. */
export async function downloadModel(url: string, dest: string): Promise<string> {
  if (await isGguf(dest)) return dest;
  await fsp.mkdir(path.dirname(dest), { recursive: true });
  const tmp = `${dest}.part`;
  console.log(`[gym] downloading model to ${dest} ...`);
  const r = await exec("curl", ["-fL", "--retry", "5", "-o", tmp, url], { cwd: paths.cache, timeoutMs: 60 * 60_000 });
  if (r.code !== 0) throw new Error(`model download failed:\n${tail(r.stderr, 1_000)}`);
  if (!(await isGguf(tmp))) throw new Error("downloaded file is not a GGUF model");
  await fsp.rename(tmp, dest);
  return dest;
}

export type LlamaServer = {
  port: number;
  stop(): Promise<void>;
};

export async function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(0, "127.0.0.1", () => {
      const addr = srv.address();
      const port = typeof addr === "object" && addr ? addr.port : 0;
      srv.close(() => (port ? resolve(port) : reject(new Error("could not allocate a free port"))));
    });
    srv.on("error", reject);
  });
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Start llama-server on 127.0.0.1 and wait until /health reports ok. */
export async function startLlamaServer(opts: {
  bin: string;
  model: string;
  port?: number;
  ctxSize?: number;
}): Promise<LlamaServer> {
  const port = opts.port ?? (await freePort());
  const proc: ChildProcess = spawn(
    opts.bin,
    ["-m", opts.model, "--host", "127.0.0.1", "--port", String(port), "--ctx-size", String(opts.ctxSize ?? 8192)],
    { stdio: ["ignore", "pipe", "pipe"] },
  );
  let logs = "";
  proc.stdout?.on("data", (c: Buffer) => {
    logs += c;
    if (logs.length > 20_000) logs = logs.slice(-20_000);
  });
  proc.stderr?.on("data", (c: Buffer) => {
    logs += c;
    if (logs.length > 20_000) logs = logs.slice(-20_000);
  });

  const stop = async (): Promise<void> => {
    if (proc.exitCode !== null) return;
    proc.kill("SIGTERM");
    await new Promise<void>((resolve) => {
      const t = setTimeout(() => {
        proc.kill("SIGKILL");
        resolve();
      }, 5_000);
      proc.once("close", () => {
        clearTimeout(t);
        resolve();
      });
    });
  };

  const deadline = Date.now() + 180_000;
  while (Date.now() < deadline) {
    if (proc.exitCode !== null) throw new Error(`llama-server exited with code ${proc.exitCode}:\n${tail(logs, 2_000)}`);
    try {
      const res = await fetch(`http://127.0.0.1:${port}/health`);
      if (res.ok) {
        const body = (await res.json()) as { status?: string };
        if (body.status === "ok") return { port, stop };
      }
    } catch {
      // not up yet
    }
    await sleep(500);
  }
  await stop();
  throw new Error(`llama-server did not become healthy within 180s:\n${tail(logs, 2_000)}`);
}

export type BenchResult = {
  pp: number | null;
  tg: number;
  gpuInfo: string;
  backends: string;
};

/**
 * Run llama-bench and parse its `-o json` output (fields: n_prompt, n_gen, avg_ts, backends, gpu_info, ...).
 * Refuses results where the Metal backend is not active.
 */
export async function runBench(opts: {
  bin: string;
  model: string;
  pp: number;
  tg: number;
  reps: number;
  timeoutMs?: number;
}): Promise<BenchResult> {
  const r = await exec(
    opts.bin,
    ["-m", opts.model, "-p", String(opts.pp), "-n", String(opts.tg), "-r", String(opts.reps), "-o", "json"],
    { cwd: paths.root, timeoutMs: opts.timeoutMs ?? 15 * 60_000 },
  );
  if (r.code !== 0) throw new Error(`llama-bench failed:\n${tail(r.stdout + r.stderr, 3_000)}`);

  let entries: Record<string, unknown>[];
  try {
    entries = JSON.parse(r.stdout) as Record<string, unknown>[];
  } catch {
    throw new Error(`could not parse llama-bench output:\n${tail(r.stdout, 1_000)}`);
  }

  let pp: number | null = null;
  let tg: number | null = null;
  let gpuInfo = "";
  let backends = "";
  for (const e of entries) {
    const nPrompt = Number(e.n_prompt);
    const nGen = Number(e.n_gen);
    const avgTs = Number(e.avg_ts);
    if (nPrompt > 0 && nGen === 0) pp = avgTs;
    if (nGen > 0 && nPrompt === 0) tg = avgTs;
    gpuInfo = String(e.gpu_info ?? gpuInfo);
    backends = String(e.backends ?? backends);
  }
  if (tg === null) throw new Error(`llama-bench output has no tg result:\n${tail(r.stdout, 1_000)}`);
  if (!/\bMTL\b|\bMETAL\b/i.test(backends)) {
    throw new Error(`Metal backend is not active (backends: "${backends}"); refusing to benchmark`);
  }
  return { pp, tg: tg!, gpuInfo, backends };
}
