import type { AgentOptions, LLMMessage, Step, Tool, Trajectory } from "./types.ts";
import { parseAction } from "./parser.ts";

export const FINISH_ACTION = "finish";

function protocolSpec(tools: Tool[]): string {
  const names = [...tools.map((t) => t.name), FINISH_ACTION].join(", ");
  return [
    "# Output protocol",
    "Every reply must contain exactly one action, in this format:",
    "",
    "THOUGHT: <brief reasoning>",
    "ACTION: <name>",
    "<tool input on the following lines; multi-line allowed; empty if the tool needs none>",
    "",
    `Valid actions: ${names}.`,
    `"${FINISH_ACTION}" ends the task; its input is a short final summary.`,
    "One action per reply, never wrapped in code fences. Wait for the observation before acting again.",
  ].join("\n");
}

function truncate(s: string, max: number): string {
  return s.length <= max ? s : `${s.slice(0, max)}\n...[truncated ${s.length - max} chars]`;
}

/**
 * The agent loop: LLM -> parse action -> run tool -> append observation -> repeat,
 * until the model calls "finish" or the step budget is exhausted.
 */
export class Agent {
  private readonly messages: LLMMessage[] = [];

  constructor(private readonly opts: AgentOptions) {}

  async run(): Promise<Trajectory> {
    const trajectory: Trajectory = { goal: this.opts.goal, steps: [], summary: "", finished: false };
    const maxSteps = this.opts.maxSteps ?? 24;
    const maxObs = this.opts.maxObservationChars ?? 6_000;

    this.messages.push({ role: "system", content: `${this.opts.system}\n\n${protocolSpec(this.opts.tools)}` });
    this.messages.push({ role: "user", content: this.opts.goal });

    let parseFailures = 0;
    for (let i = 0; i < maxSteps; i++) {
      let text: string;
      try {
        text = await this.opts.llm.chat(this.messages);
      } catch (err) {
        trajectory.summary = `agent stopped: LLM error: ${String(err)}`;
        return trajectory;
      }

      const parsed = parseAction(text);
      if (!parsed) {
        parseFailures++;
        this.messages.push({ role: "assistant", content: text });
        this.messages.push({
          role: "user",
          content: `ERROR: your reply contained no parseable action. Reply again using exactly:\nTHOUGHT: ...\nACTION: <one of: ${this.opts.tools.map((t) => t.name).join(", ")}, ${FINISH_ACTION}>`,
        });
        if (parseFailures >= 3) {
          trajectory.summary = "agent stopped: model failed to produce a parseable action 3 times in a row";
          return trajectory;
        }
        continue;
      }
      parseFailures = 0;

      if (parsed.action === FINISH_ACTION) {
        trajectory.finished = true;
        trajectory.summary = parsed.input || "finished";
        return trajectory;
      }

      const tool = this.opts.tools.find((t) => t.name === parsed.action);
      let result;
      if (!tool) {
        result = {
          ok: false,
          output: `unknown action '${parsed.action}'. available actions: ${this.opts.tools.map((t) => t.name).join(", ")}, ${FINISH_ACTION}`,
        };
      } else {
        try {
          result = await tool.execute(parsed.input);
        } catch (err) {
          result = { ok: false, output: `tool crashed: ${String(err)}` };
        }
      }
      result.output = truncate(result.output, maxObs);

      const step: Step = {
        index: i,
        thought: parsed.thought,
        action: parsed.action,
        input: parsed.input,
        observation: result.output,
        ok: result.ok,
      };
      trajectory.steps.push(step);
      this.opts.onStep?.(step);

      this.messages.push({ role: "assistant", content: text });
      this.messages.push({ role: "user", content: `OBSERVATION (${parsed.action}, ok=${result.ok}):\n${result.output}` });
    }

    trajectory.summary = "max steps reached without finish";
    return trajectory;
  }
}
