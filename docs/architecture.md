# Runtime architecture

## Implemented boundaries

| Component | Responsibility |
|---|---|
| `BaseAgent` | One bounded provider/tool conversation and its trace |
| `OpenAiProvider` | Real Chat Completions requests, optional Bearer authentication, raw HTTP errors |
| `ToolRegistry` | Deterministic named tools and extensibility through `BaseAgent::with_tools` |
| `Gym` | Canonical root, relative paths, rejection of parent/symlink path escapes |
| `Tool` | Asynchronous interface receiving arguments and the current Gym explicitly |
| `RunProcess` | Child process groups, timeouts, secret-scrubbed environment, complete output files |
| Hugging Face downloader | Repository URL parsing, commit-pinned downloads, file hashes, Keychain credentials |
| `TraceWriter` | Persistent flushed JSONL records for calls, observations, and raw failures |

Each tool receives the current Gym as a separate argument. The downloader does not
choose a global artifact destination; the agent maps the authenticated repository
to the invoking Gym's model directory. Tool inputs are resolved against one
canonical Gym root before file access.

The CLI opts into process execution. Its child environments retain only an
explicitly selected operating-system/toolchain environment, and caller-provided
variables whose names suggest credentials are rejected. Provider credentials are
marked sensitive on HTTP headers and never included in traces.

## State categories

- **Scoped source:** C++ inference and llama.cpp worktrees, once implemented,
  belong in a concrete Gym and are eligible for Git history.
- **Pinned artifacts:** native weights belong under that Gym's `model/weights/` and
  remain outside Git; their repository, resolved commit, paths, sizes, and hashes
  are recorded in its manifest.
- **Execution evidence:** commands, stdout/stderr, and run traces belong beneath
  that Gym's runtime directory and remain outside Git.
- **Learning:** findings acquired in one Gym stay private by default; promoting
  general findings to shared knowledge is a later controller responsibility.
- **Base agent improvement:** traces and evaluations are durable source material
  for later agent improvement, never a claim that downloading a model trained an
  agent or changed its weights.

## Deliberate limits

This version has no mock provider, fake filesystem, or simulated build. It does not
provide a model provider or inference server on its own; operators configure a real
OpenAI-compatible, tool-capable provider.

In particular, implementing a filesystem boundary is not equivalent to protecting
macOS from arbitrary generated code. Once commands run, they execute natively as
the invoking user. Per-account macOS provisioning, network/resource policy,
long-running inference supervision, protected evaluation fixtures, automatic
promotion/rollback, and inter-Gym GPU scheduling remain separate responsibilities
rather than implied capabilities of this crate.

The initial Hugging Face implementation is an authenticated model-repository
acquisition tool, not a format converter or proof of engine/model compatibility.
It does not promise that every repository can run on llama.cpp or Metal.
