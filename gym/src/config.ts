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

/** Meta Model API: OpenAI-compatible Chat Completions. https://dev.meta.ai/docs */
export const META_MODEL_API_BASE = "https://api.meta.ai/v1";
export const META_CONTRIBUTOR_MODEL = "muse-spark-1.3-contributor";

export function llmConfig() {
  const temperature = process.env.GYM_LLM_TEMPERATURE;
  return {
    baseUrl: process.env.GYM_LLM_BASE_URL ?? META_MODEL_API_BASE,
    // Meta Model API key: GYM_LLM_API_KEY overrides .env names (MODEL_API_KEY per docs, META_API_KEY accepted)
    apiKey: process.env.GYM_LLM_API_KEY ?? process.env.MODEL_API_KEY ?? process.env.META_API_KEY ?? "",
    model: process.env.GYM_LLM_MODEL ?? META_CONTRIBUTOR_MODEL,
    reasoningEffort: process.env.GYM_LLM_REASONING_EFFORT ?? "high",
    maxTokens: Number(process.env.GYM_LLM_MAX_TOKENS ?? 4096),
    temperature: temperature !== undefined && temperature !== "" ? Number(temperature) : undefined,
  };
}

export const agentLlmPort = () => Number(process.env.GYM_LLM_PORT ?? 8080);
export const defaultMaxSteps = () => Number(process.env.GYM_MAX_STEPS ?? 24);
