import { describe, expect, test } from "bun:test";
import { parseAction } from "../src/parser.ts";

describe("parseAction", () => {
  test("parses thought, action and multiline input", () => {
    const parsed = parseAction(
      ["THOUGHT: read the kernel first", "ACTION: read_file", "ggml/src/ggml-metal/ggml-metal.metal"].join("\n"),
    );
    expect(parsed).not.toBeNull();
    expect(parsed!.action).toBe("read_file");
    expect(parsed!.thought).toBe("read the kernel first");
    expect(parsed!.input).toBe("ggml/src/ggml-metal/ggml-metal.metal");
  });

  test("parses action without thought", () => {
    const parsed = parseAction("ACTION: eval");
    expect(parsed!.action).toBe("eval");
    expect(parsed!.thought).toBe("");
    expect(parsed!.input).toBe("");
  });

  test("parses finish with summary", () => {
    const parsed = parseAction("THOUGHT: done\nACTION: finish\nImproved mv kernel q4_K; tg +3.1%");
    expect(parsed!.action).toBe("finish");
    expect(parsed!.input).toBe("Improved mv kernel q4_K; tg +3.1%");
  });

  test("handles markdown bold and code fences", () => {
    const text = "```text\n**ACTION:** shell\ngit status --short\n```";
    const parsed = parseAction(text);
    expect(parsed!.action).toBe("shell");
    expect(parsed!.input).toBe("git status --short");
  });

  test("multiline tool input (write_file) is preserved verbatim", () => {
    const content = ["first line", "second line", "}"].join("\n");
    const parsed = parseAction(`THOUGHT: rewrite\nACTION: write_file\nggml/src/ggml-metal/x.h\n${content}`);
    expect(parsed!.input).toBe(`ggml/src/ggml-metal/x.h\n${content}`);
  });

  test("returns null when no action line exists", () => {
    expect(parseAction("I would like to check the kernel.")).toBeNull();
  });

  test("is case-insensitive on the action keyword", () => {
    expect(parseAction("action: git")!.action).toBe("git");
  });

  test("takes the first action line when the model echoes the protocol", () => {
    const text = [
      "ACTION: list_files",
      ".",
      "",
      "Remember the format: ACTION: <name>",
    ].join("\n");
    const parsed = parseAction(text);
    expect(parsed!.action).toBe("list_files");
    expect(parsed!.input).toBe(".\n\nRemember the format: ACTION: <name>");
  });
});
