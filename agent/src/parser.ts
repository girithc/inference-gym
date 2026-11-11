/**
 * Text action protocol. Works with any instruction-following model, including
 * small local ones that cannot do provider-native tool calling:
 *
 *   THOUGHT: why
 *   ACTION: tool-name
 *   <raw tool input, possibly multi-line; empty for no-input tools>
 *
 * "ACTION: finish" ends the episode; its input is the final summary.
 */
export type ParsedAction = {
  thought: string;
  action: string;
  input: string;
};

const ACTION_RE = /^\s*\**\s*ACTION:\s*(.+?)\s*\**\s*$/i;

function stripFences(text: string): string {
  return text
    .replace(/^\s*```[a-zA-Z0-9_-]*\s*\n/, "")
    .replace(/\n?```\s*$/, "")
    .trim();
}

export function parseAction(text: string): ParsedAction | null {
  const lines = stripFences(text).split("\n");
  const idx = lines.findIndex((l) => ACTION_RE.test(l));
  if (idx === -1) return null;

  const match = lines[idx]!.match(ACTION_RE)!;
  const action = match[1]!.match(/[a-zA-Z0-9_-]+/)?.[0]?.toLowerCase() ?? "";

  const thoughtLines = lines.slice(0, idx);
  const first = thoughtLines.findIndex((l) => l.trim() !== "");
  if (first !== -1) {
    thoughtLines[first] = thoughtLines[first]!.replace(/^\s*\**\s*THOUGHT:\s*/i, "");
  }
  const thought = thoughtLines.join("\n").trim();
  const input = lines.slice(idx + 1).join("\n").trim();

  return { thought, action, input };
}
