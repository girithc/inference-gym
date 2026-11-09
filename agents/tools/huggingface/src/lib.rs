//! Minimal authenticated Hugging Face model downloader.
//!
//! Downloads a model repository snapshot at a resolved commit into
//! `destination/<commit-sha>`, records a manifest with per-file SHA-256
//! hashes, and returns raw errors instead of substituting defaults.
//! Credentials come from `HF_TOKEN` or the macOS Keychain and are never
//! logged.

use std::env::{self, VarError};
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use reqwest::{Client, StatusCode, Url};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::io::AsyncWriteExt;

const DEFAULT_REVISION: &str = "main";
const HUGGINGFACE_ORIGIN: &str = "https://huggingface.co/";
const MANIFEST_NAME: &str = "manifest.json";
const KEYCHAIN_SERVICE: &str = "core-agent-huggingface";
const KEYCHAIN_ACCOUNT: &str = "default";
const METADATA_TIMEOUT: Duration = Duration::from_secs(60);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const READ_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Error)]
pub enum HuggingfaceError {
    #[error("the Hugging Face token must not be empty")]
    EmptyToken,
    #[error("HF_TOKEN is not valid Unicode")]
    EnvironmentTokenNotUnicode,
    #[error("Keychain token is not valid UTF-8: {0}")]
    KeychainTokenNotUtf8(#[from] std::str::Utf8Error),
    #[cfg(target_os = "macos")]
    #[error(transparent)]
    Keychain(#[from] security_framework::base::Error),
    #[error("native macOS Keychain access is unsupported on this operating system")]
    UnsupportedKeychain,
    #[error("invalid Hugging Face model repository URL: {0}")]
    InvalidRepositoryUrl(String),
    #[error("unsafe remote file path {path:?}: {reason}")]
    InvalidRemotePath { path: String, reason: &'static str },
    #[error("the Hugging Face API returned an invalid commit SHA")]
    InvalidCommitSha,
    #[error("Hugging Face returned HTTP {status}: {body}")]
    HttpStatus { status: StatusCode, body: String },
    #[error(transparent)]
    Request(#[from] reqwest::Error),
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),
    #[error(transparent)]
    Filesystem(#[from] io::Error),
}

/// One downloaded file: repository-relative path, byte count, and SHA-256.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadedFile {
    pub relative_path: String,
    pub bytes: u64,
    pub sha256: String,
}

/// Everything known about one download. `root` is returned to the caller but
/// omitted from the serialized manifest so a moved cache keeps no stale path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadReport {
    #[serde(skip)]
    pub root: PathBuf,
    pub repository_url: String,
    pub repository_id: String,
    pub requested_revision: String,
    pub resolved_revision: String,
    pub files: Vec<DownloadedFile>,
}

/// Reusable HTTPS client with a bearer token. `Debug` redacts the token.
#[derive(Clone)]
pub struct Downloader {
    client: Client,
    token: String,
}

impl fmt::Debug for Downloader {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Downloader")
            .field("client", &self.client)
            .field("token", &"<redacted>")
            .finish()
    }
}

/// Reads a nonempty `HF_TOKEN`, falling back to the macOS Keychain.
pub fn token_from_environment_or_keychain() -> Result<String, HuggingfaceError> {
    match env::var("HF_TOKEN") {
        Ok(token) if !token.is_empty() => return Ok(token),
        Err(VarError::NotUnicode(_)) => return Err(HuggingfaceError::EnvironmentTokenNotUnicode),
        Ok(_) | Err(VarError::NotPresent) => {}
    }
    keychain_token()
}

#[cfg(target_os = "macos")]
fn keychain_token() -> Result<String, HuggingfaceError> {
    let bytes =
        security_framework::passwords::get_generic_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT)?;
    Ok(std::str::from_utf8(&bytes)?.to_owned())
}

#[cfg(not(target_os = "macos"))]
fn keychain_token() -> Result<String, HuggingfaceError> {
    Err(HuggingfaceError::UnsupportedKeychain)
}

/// Creates or updates the Keychain item for this tool's token.
#[cfg(target_os = "macos")]
pub fn store_token_in_keychain(token: &str) -> Result<(), HuggingfaceError> {
    if token.is_empty() {
        return Err(HuggingfaceError::EmptyToken);
    }
    security_framework::passwords::set_generic_password(
        KEYCHAIN_SERVICE,
        KEYCHAIN_ACCOUNT,
        token.as_bytes(),
    )?;
    Ok(())
}

#[cfg(not(target_os = "macos"))]
pub fn store_token_in_keychain(_token: &str) -> Result<(), HuggingfaceError> {
    Err(HuggingfaceError::UnsupportedKeychain)
}

/// Parses `https://huggingface.co/<namespace>/<repository>` with an optional
/// trailing slash and rejects everything else.
pub fn repository_id_from_url(repository_url: &str) -> Result<String, HuggingfaceError> {
    let invalid = |reason: &str| HuggingfaceError::InvalidRepositoryUrl(reason.to_owned());
    let parsed = Url::parse(repository_url).map_err(|error| invalid(&error.to_string()))?;
    if parsed.scheme() != "https" {
        return Err(invalid("the URL scheme must be https"));
    }
    if parsed.host_str() != Some("huggingface.co") {
        return Err(invalid("the URL host must be huggingface.co"));
    }
    if parsed.port().is_some() {
        return Err(invalid("a nonstandard port is not allowed"));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(invalid("user information is not allowed"));
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        return Err(invalid("query strings and fragments are not allowed"));
    }
    let mut segments: Vec<&str> = parsed
        .path_segments()
        .map(Iterator::collect)
        .unwrap_or_default();
    if segments.last() == Some(&"") {
        segments.pop();
    }
    if segments.len() != 2 {
        return Err(invalid("expected exactly <namespace>/<repository>"));
    }
    for segment in &segments {
        let valid = !segment.is_empty()
            && *segment != "."
            && *segment != ".."
            && segment
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'));
        if !valid {
            return Err(invalid(
                "repository path segments contain unsupported characters",
            ));
        }
    }
    Ok(format!("{}/{}", segments[0], segments[1]))
}

impl Downloader {
    pub fn new(token: String) -> Result<Self, HuggingfaceError> {
        if token.is_empty() {
            return Err(HuggingfaceError::EmptyToken);
        }
        let client = Client::builder()
            .user_agent(concat!(
                "core-agent-huggingface/",
                env!("CARGO_PKG_VERSION")
            ))
            .https_only(true)
            .redirect(reqwest::redirect::Policy::limited(5))
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(READ_TIMEOUT)
            .build()?;
        Ok(Self { client, token })
    }

    /// Download every repository file into `destination/<resolved-commit-sha>`.
    pub async fn download(
        &self,
        repository_url: &str,
        revision: Option<&str>,
        destination: &Path,
    ) -> Result<DownloadReport, HuggingfaceError> {
        let repository_id = repository_id_from_url(repository_url)?;
        let requested_revision = revision.unwrap_or(DEFAULT_REVISION);
        let metadata = self.metadata(&repository_id, requested_revision).await?;
        let root = tokio::fs::canonicalize(destination)
            .await?
            .join(&metadata.sha);
        tokio::fs::create_dir_all(&root).await?;

        let mut files = Vec::with_capacity(metadata.siblings.len());
        for sibling in metadata.siblings {
            let relative = safe_remote_path(&sibling.rfilename)?;
            let local = root.join(&relative);
            if let Some(parent) = local.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            let (bytes, sha256) = self
                .stream_file(&repository_id, &metadata.sha, &sibling.rfilename, &local)
                .await?;
            files.push(DownloadedFile {
                relative_path: sibling.rfilename,
                bytes,
                sha256,
            });
        }
        files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));

        let report = DownloadReport {
            root,
            repository_url: repository_url.to_owned(),
            repository_id,
            requested_revision: requested_revision.to_owned(),
            resolved_revision: metadata.sha,
            files,
        };
        write_manifest(&report).await?;
        Ok(report)
    }

    async fn metadata(
        &self,
        repository_id: &str,
        revision: &str,
    ) -> Result<ModelMetadata, HuggingfaceError> {
        let mut url = Url::parse(HUGGINGFACE_ORIGIN).expect("the static base URL is valid");
        let (namespace, repository) = split_repository_id(repository_id);
        url.path_segments_mut()
            .expect("the base URL supports path segments")
            .extend(["api", "models", namespace, repository, "revision", revision]);
        let response = self
            .client
            .get(url)
            .bearer_auth(&self.token)
            .timeout(METADATA_TIMEOUT)
            .send()
            .await?;
        let status = response.status();
        let body = response.text().await?;
        if !status.is_success() {
            return Err(HuggingfaceError::HttpStatus { status, body });
        }
        let metadata: ModelMetadata = serde_json::from_str(&body)?;
        if metadata.sha.len() == 40 && metadata.sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            Ok(metadata)
        } else {
            Err(HuggingfaceError::InvalidCommitSha)
        }
    }

    async fn stream_file(
        &self,
        repository_id: &str,
        sha: &str,
        remote_path: &str,
        local: &Path,
    ) -> Result<(u64, String), HuggingfaceError> {
        let mut url = Url::parse(HUGGINGFACE_ORIGIN).expect("the static base URL is valid");
        let (namespace, repository) = split_repository_id(repository_id);
        {
            let mut segments = url
                .path_segments_mut()
                .expect("the base URL supports path segments");
            segments.extend([namespace, repository, "resolve", sha]);
            segments.extend(remote_path.split('/'));
        }
        let mut response = self.client.get(url).bearer_auth(&self.token).send().await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await?;
            return Err(HuggingfaceError::HttpStatus { status, body });
        }
        let parent = local
            .parent()
            .expect("a validated relative path has a parent");
        let name = local
            .file_name()
            .expect("a validated relative path has a file name");
        let temporary = parent.join(format!("{}.part", name.to_string_lossy()));
        let streamed = stream_to_file(&mut response, &temporary).await;
        match streamed {
            Ok((bytes, sha256)) => {
                tokio::fs::rename(&temporary, local).await?;
                Ok((bytes, sha256))
            }
            Err(error) => {
                let _ = tokio::fs::remove_file(&temporary).await;
                Err(error)
            }
        }
    }
}

async fn stream_to_file(
    response: &mut reqwest::Response,
    path: &Path,
) -> Result<(u64, String), HuggingfaceError> {
    let mut file = tokio::fs::File::create(path).await?;
    let mut hasher = Sha256::new();
    let mut bytes = 0u64;
    while let Some(chunk) = response.chunk().await? {
        file.write_all(&chunk).await?;
        hasher.update(&chunk);
        bytes += chunk.len() as u64;
    }
    file.flush().await?;
    file.sync_all().await?;
    Ok((bytes, format!("{:x}", hasher.finalize())))
}

async fn write_manifest(report: &DownloadReport) -> Result<(), HuggingfaceError> {
    let mut bytes = serde_json::to_vec_pretty(report)?;
    bytes.push(b'\n');
    let path = report.root.join(MANIFEST_NAME);
    let temporary = path.with_file_name(format!("{MANIFEST_NAME}.tmp"));
    tokio::fs::write(&temporary, bytes).await?;
    tokio::fs::rename(&temporary, &path).await?;
    Ok(())
}

fn split_repository_id(repository_id: &str) -> (&str, &str) {
    repository_id
        .split_once('/')
        .expect("a validated repository ID contains one slash")
}

fn safe_remote_path(remote_path: &str) -> Result<PathBuf, HuggingfaceError> {
    let reject = |reason: &'static str| HuggingfaceError::InvalidRemotePath {
        path: remote_path.to_owned(),
        reason,
    };
    if remote_path.is_empty() {
        return Err(reject("the path is empty"));
    }
    if remote_path.starts_with('/') || remote_path.starts_with('\\') {
        return Err(reject("absolute paths are not allowed"));
    }
    if remote_path.chars().any(char::is_control) {
        return Err(reject("control characters are not allowed"));
    }
    let mut local = PathBuf::new();
    for segment in remote_path.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return Err(reject("empty, '.', and '..' segments are not allowed"));
        }
        if segment.contains('\\') {
            return Err(reject("backslashes are not allowed"));
        }
        local.push(segment);
    }
    Ok(local)
}

#[derive(Debug, Deserialize)]
struct ModelMetadata {
    sha: String,
    siblings: Vec<ModelSibling>,
}

#[derive(Debug, Deserialize)]
struct ModelSibling {
    rfilename: String,
}
