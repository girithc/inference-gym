# inference

Local LLM inference tooling on Apple Silicon (M1 Max, Metal).

## Layout

- [agents/tools/huggingface/](agents/tools/huggingface/) — authenticated Hugging Face
  model downloader: URL in, commit-pinned snapshot with SHA-256 manifest out. The only
  crate currently in the Cargo workspace.
- [interface/cli/](interface/cli/) — command-line interface (empty, to be built).
- [gyms/](gyms/) — per-model Gym directories; [_template](gyms/_template/) is the
  reusable scaffold.

The previous agent runtime and control plane were removed; the project is being
rebuilt around the downloader and a new CLI.
