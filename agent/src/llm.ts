import type { LLMClient, LLMMessage, LLMOptions } from "./types.ts";

const DEFAULT_TIMEOUT_MS = 300_000;
const DEFAULT_MAX_RETRIES = 4;

/** Client for any OpenAI-compatible /chat/completions endpoint (Meta Model API, llama-server, vLLM, OpenAI, ...). */
export class OpenAICompatLLM implements LLMClient {
  constructor(private readonly opts: LLMOptions) {}

  get model(): string {
    return this.opts.model;
  }

  async chat(messages: LLMMessage[]): Promise<string> {
    const url = joinUrl(this.opts.baseUrl, "/chat/completions");
    const maxRetries = this.opts.maxRetries ?? DEFAULT_MAX_RETRIES;
    const maxTokens = this.opts.maxTokens ?? 1024;

    let lastErr: unknown;
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.opts.timeoutMs ?? DEFAULT_TIMEOUT_MS);
      try {
        const body: Record<string, unknown> = { model: this.opts.model, messages, max_tokens: maxTokens };
        if (this.opts.temperature !== undefined) body.temperature = this.opts.temperature;
        if (this.opts.reasoningEffort !== undefined) body.reasoning_effort = this.opts.reasoningEffort;

        const res = await fetch(url, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            authorization: `Bearer ${this.opts.apiKey}`,
          },
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        if (!res.ok) {
          const text = await res.text().catch(() => "");
          lastErr = new Error(`HTTP ${res.status} ${res.statusText} ${text.slice(0, 300)}`);
          // 4xx (bad key, bad request) will not improve by retrying; 408/429/5xx might.
          if (!(res.status === 408 || res.status === 429 || res.status >= 500)) break;
        } else {
          const data = (await res.json()) as { choices?: { message?: { content?: string } }[] };
          const content = data.choices?.[0]?.message?.content ?? "";
          if (!content.trim()) throw new Error("empty completion");
          return content;
        }
      } catch (err) {
        lastErr = err;
      } finally {
        clearTimeout(timer);
      }
      if (attempt < maxRetries) await sleep(1_500 * (attempt + 1));
    }
    throw new Error(`LLM request failed after ${maxRetries + 1} attempts: ${String(lastErr)}`);
  }
}

function joinUrl(base: string, path: string): string {
  return base.replace(/\/+$/, "") + path;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
