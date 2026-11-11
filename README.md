# gym

A gym where AI agents optimize GPU kernels for LLM inference on Apple Silicon (Metal), using llama.cpp.

Agents get a real environment — a llama.cpp checkout, a Metal GPU, a benchmark, a correctness gate, a git repo — and autonomy to find bottlenecks and improve them. Every change is measured with `llama-bench` and gated by a deterministic correctness check. Every instance is its own git repository, so progress is a commit history.

No mocks, no simulated scores: numbers come from real `llama-bench` runs on your Metal GPU, and correctness comes from real greedy generation compared against the baseline.

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│ repo root (this monorepo, bun workspaces)                          │
│                                                                    │
│  agent/                 reusable agent framework, zero deps        │
│    loop.ts               Agent: LLM → parse action → run tool →    │
│                          observe → repeat, until finish/budget     │
│    llm.ts                OpenAI-compatible client (any endpoint)   │
│    parser.ts             THOUGHT/ACTION text protocol (tiny-model  │
│                          friendly, no provider tool-calling)       │
│    tools.ts              shell, read/write/list files, git —       │
│                          sandboxed to the instance directory       │
│                                                                    │
│  gym/                   the gym: instances, tasks, eval, llama.cpp │
│    llama.ts              clone/build llama.cpp (Metal), llama-bench│
│                          JSON parser, llama-server lifecycle,      │
│                          model download (verified GGUF magic)      │
│    task.ts               Task definition: mission, allowed files,  │
│                          bench params, correctness spec            │
│    instance.ts           create/load/list instances (one git repo  │
│                          each), measure + commit baselines         │
│    eval.ts               rebuild → bench → correctness → score     │
│    runner.ts             wires framework + tools + task, records   │
│                          trajectories and result commits           │
│    cli.ts                setup / serve / new / run / eval / list   │
│                                                                    │
│  cache/                 shared: llama.cpp clone, portable cmake,   │
│                         models (gitignored)                        │
│  instances/<name>/      one instance = one agent's workspace       │
│    …llama.cpp source…    its own git repo (upstream + agent commits│
│    ggml/src/ggml-metal/  ← the kernels the agent optimizes         │
│    .gym/baseline.json    measured baseline (tg/pp t/s, output text)│
│    .gym/results.json     latest eval                               │
│    .gym/history.jsonl    every eval ever run in this instance      │
│    .gym/trajectory-*.jsonl  step-by-step agent trace               │
└────────────────────────────────────────────────────────────────────┘
```

Design rules: two packages (`agent` = model-agnostic framework, `gym` = environment), zero npm runtime dependencies (Node/Bun builtins only), every long operation is a real subprocess, every score is a real measurement.

## The optimization target (MVP task: `metal-kernels`)

The agent may edit only the Metal kernel sources of llama.cpp (`ggml-metal.metal`, `ggml-metal-impl.h`). Scoring:

- **Throughput**: `llama-bench -p 512 -n 128 -r 3` → tg128 tokens/s (and pp512 for reference). For small models, decode is memory-bandwidth bound and dominated by matrix-vector kernels.
- **Correctness gate**: greedy, seeded completion of a fixed prompt through the freshly built `llama-server` must stay similar (word-Jaccard ≥ 0.8) to the baseline generation. A kernel change that breaks outputs cannot win, no matter how fast it is.

A run only counts as an **improvement** when correctness passes **and** tg ≥ baseline × 1.02.

## Quickstart (macOS, Apple Silicon)

```bash
bun install

# 1. one-time: clone + build llama.cpp with Metal, download Qwen2.5-0.5B, smoke bench
bun run gym setup

# 2. start the agent's brain (local llama-server, OpenAI-compatible API)
bun run gym serve          # leave running; Ctrl+C to stop

# 3. in a second terminal: create an instance (builds + measures baseline, commits it)
bun run gym new demo

# 4. let the agent loose (edit kernels → eval → commit loop)
bun run gym run demo

# 5. compare instances / scores
bun run gym list
```

To inspect or continue an instance's work yourself:

```bash
cd instances/demo
git log --oneline          # baseline → agent commits → results
git diff <baseline-sha>    # exactly what the agent changed
bun run gym eval demo      # re-verify any time
```

## Using a different (smarter) brain

The small local model works end-to-end but is a weak optimizer — the point of the gym is to test LLMs, so point the framework anywhere OpenAI-compatible:

```bash
GYM_LLM_BASE_URL=https://api.openai.com/v1 \
GYM_LLM_API_KEY=sk-... \
GYM_LLM_MODEL=gpt-4.1-mini \
bun run gym run demo
```

Defaults: `http://127.0.0.1:8080/v1` / model `local` (what `gym serve` runs).

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `GYM_LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | agent brain endpoint |
| `GYM_LLM_API_KEY` | `local` | bearer token |
| `GYM_LLM_MODEL` | `local` | model id sent to the endpoint |
| `GYM_LLM_PORT` | `8080` | port for `gym serve` |
| `GYM_MAX_STEPS` | `24` | agent action budget per run |
| `GYM_MODEL` | `cache/models/qwen2.5-0.5b-instruct-q4_k_m.gguf` | benchmark model |
| `GYM_LLAMA_DIR` | `cache/llama.cpp` | llama.cpp checkout to seed instances from |

## Extending

- **New task**: add a `Task` to `gym/src/task.ts` (mission, allowed paths, bench/correctness params). `gym new` seeds instances with it.
- **New tools for agents**: `Tool` is `{ name, description, execute(input) }` — e.g. `gym/runner.ts` injects the `eval` tool the same way.
- **New framework features** (memory, planning, parallel instances): `agent/src/loop.ts` is the whole loop; keep it small.
