import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url)); // gym/src
export const ROOT = path.resolve(HERE, "..", "..");

export const paths = {
  root: ROOT,
  cache: path.join(ROOT, "cache"),
  instances: path.join(ROOT, "instances"),
  llamaDir: process.env.GYM_LLAMA_DIR ?? path.join(ROOT, "cache", "llama.cpp"),
  modelsDir: path.join(ROOT, "cache", "models"),
  get model(): string {
    return process.env.GYM_MODEL ?? path.join(this.modelsDir, "qwen2.5-0.5b-instruct-q4_k_m.gguf");
  },
};

export const MODEL_URL =
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf";

export function llmConfig() {
  return {
    baseUrl: process.env.GYM_LLM_BASE_URL ?? "http://127.0.0.1:8080/v1",
    apiKey: process.env.GYM_LLM_API_KEY ?? "local",
    model: process.env.GYM_LLM_MODEL ?? "local",
  };
}

export const agentLlmPort = () => Number(process.env.GYM_LLM_PORT ?? 8080);
export const defaultMaxSteps = () => Number(process.env.GYM_MAX_STEPS ?? 24);
