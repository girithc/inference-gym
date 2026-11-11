import type { LLMClient, LLMMessage, LLMOptions } from "./types.ts";

const DEFAULT_TIMEOUT_MS = 120_000;
const DEFAULT_MAX_RETRIES = 4;

/** Client for any OpenAI-compatible /chat/completions endpoint (llama-server, vLLM, OpenAI, ...). */
export class OpenAICompatLLM implements LLMClient {
  constructor(private readonly opts: LLMOptions) {}

  get model(): string {
    return this.opts.model;
  }

  async chat(messages: LLMMessage[]): Promise<string> {
    const url = joinUrl(this.opts.baseUrl, "/chat/completions");
    const maxRetries = this.opts.maxRetries ?? DEFAULT_MAX_RETRIES;
    const maxTokens = this.opts.maxTokens ?? 1024;
    const temperature = this.opts.temperature ?? 0.3;

    let lastErr: unknown;
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.opts.timeoutMs ?? DEFAULT_TIMEOUT_MS);
      try {
        const res = await fetch(url, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            authorization: `Bearer ${this.opts.apiKey}`,
          },
          body: JSON.stringify({ model: this.opts.model, messages, temperature, max_tokens: maxTokens }),
          signal: controller.signal,
        });
        if (!res.ok) {
          const body = await res.text().catch(() => "");
          throw new Error(`HTTP ${res.status} ${res.statusText} ${body.slice(0, 300)}`);
        }
        const data = (await res.json()) as { choices?: { message?: { content?: string } }[] };
        const content = data.choices?.[0]?.message?.content ?? "";
        if (!content.trim()) throw new Error("empty completion");
        return content;
      } catch (err) {
        lastErr = err;
        if (attempt < maxRetries) await sleep(1_500 * (attempt + 1));
      } finally {
        clearTimeout(timer);
      }
    }
    throw new Error(`LLM request failed after ${maxRetries + 1} attempts: ${String(lastErr)}`);
  }
}

function joinUrl(base: string, path: string): string {
  return base.replace(/\/+$/, "") + path;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
