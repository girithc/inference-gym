# core-agent-huggingface

Minimal authenticated Hugging Face **model** downloader. Downloads a repository
snapshot at a resolved commit into `destination/<commit-sha>`, writes
`manifest.json` with per-file byte counts and SHA-256 hashes, and returns raw
errors. No CLI, no tests, no cache-reuse machinery.

## API

- `token_from_environment_or_keychain() -> Result<String, HuggingfaceError>`
- `store_token_in_keychain(token: &str) -> Result<(), HuggingfaceError>`
- `repository_id_from_url(repository_url: &str) -> Result<String, HuggingfaceError>`
- `Downloader::new(token: String) -> Result<Self, HuggingfaceError>`
- `Downloader::download(&self, repository_url: &str, revision: Option<&str>, destination: &Path) -> Result<DownloadReport, HuggingfaceError>`

`DownloadReport`: `root` (returned, omitted from JSON), `repository_url`,
`repository_id`, `requested_revision`, `resolved_revision`, `files`
(each with `relative_path`, `bytes`, `sha256`).

## Credentials

A nonempty `HF_TOKEN` wins; otherwise the macOS Keychain generic-password item
with service `core-agent-huggingface`, account `default`. Tokens are never
logged or included in errors.

## Behavior

Only `https://huggingface.co/<namespace>/<repository>` URLs are accepted.
Metadata (`/api/models/<id>/revision/<revision-or-main>`) must resolve to a
40-character commit SHA; every sibling is then streamed from
`<id>/resolve/<sha>/<path>` into place via a `.part` temporary file and atomic
rename. Remote paths with absolute, `..`, empty, backslash, or control
characters are rejected. HTTP failures keep their status and response body.
