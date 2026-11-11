/** Result of a tool invocation, shown back to the agent as an observation. */
export type ToolResult = {
  ok: boolean;
  output: string;
};

/**
 * A capability the agent can invoke.
 * Tools take a single free-form string input; each tool defines its own format.
 */
export type Tool = {
  name: string;
  description: string;
  execute(input: string): Promise<ToolResult>;
};

/** One agent turn: a thought, an action, and the observation it produced. */
export type Step = {
  index: number;
  thought: string;
  action: string;
  input: string;
  observation: string;
  ok: boolean;
};

export type Trajectory = {
  goal: string;
  steps: Step[];
  summary: string;
  finished: boolean;
};

export type LLMMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type LLMOptions = {
  baseUrl: string;
  apiKey: string;
  model: string;
  timeoutMs?: number;
  maxRetries?: number;
  maxTokens?: number;
  temperature?: number;
};

export interface LLMClient {
  readonly model: string;
  chat(messages: LLMMessage[]): Promise<string>;
}

export type AgentOptions = {
  llm: LLMClient;
  tools: Tool[];
  system: string;
  goal: string;
  maxSteps?: number;
  maxObservationChars?: number;
  onStep?: (step: Step) => void;
};
