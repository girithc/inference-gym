import { describe, expect, test } from "bun:test";
import { textSimilarity } from "../src/eval.ts";

describe("textSimilarity", () => {
  test("identical text scores 1", () => {
    expect(textSimilarity("The theory of relativity states that", "the theory of relativity, states that")).toBe(1);
  });

  test("garbage vs coherent text is low", () => {
    const s = textSimilarity(
      "The theory of relativity states that space and time are interwoven",
      "\n\n\n!@# !@# !@#\n!?!! ???? 0x00",
    );
    expect(s).toBeLessThan(0.3);
  });

  test("minor numeric drift keeps a high score", () => {
    const a = "The theory of relativity states that energy equals mass times the speed of light squared";
    const b = "The theory of relativity states that energy equals mass times the speed off light squared";
    expect(textSimilarity(a, b)).toBeGreaterThan(0.8);
  });

  test("empty input scores 0", () => {
    expect(textSimilarity("", "something")).toBe(0);
    expect(textSimilarity("something", "")).toBe(0);
  });
});
