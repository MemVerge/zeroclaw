use crate::traits::{
    ChatMessage, ChatRequest as ProviderChatRequest, ChatResponse as ProviderChatResponse,
    ModelProvider, ProviderCapabilities, StreamChunk, StreamError, StreamEvent, StreamOptions,
    StreamResult, TokenUsage, ToolCall as ProviderToolCall,
};
use anyhow::Context;
use async_trait::async_trait;
use base64::Engine as _;
use futures_util::stream::{self, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use zeroclaw_api::tool::ToolSpec;

/// Anthropic's API documentation lists 1.0 as the default sampling temperature.
const TEMPERATURE_DEFAULT: f64 = 1.0;
/// Anthropic's public API endpoint. Overrideable via `model_providers.<name>.base_url`.
pub(crate) const BASE_URL: &str = "https://api.anthropic.com";
const SSE_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(90);
/// Legacy replay metadata accepted for histories written by earlier builds.
/// New histories use [`AnthropicReplayEnvelope`] so persistence does not rely
/// on tool-call extension fields or offsets into normalized display text.
const ANTHROPIC_BLOCK_INDEX_KEY: &str = "__anthropic_content_block_index";
const ANTHROPIC_EXTRA_CONTENT_KEY: &str = "anthropic";
const ANTHROPIC_TEXT_BLOCK_MARKER: &str = "__anthropic_text_block";
const ANTHROPIC_TEXT_START_KEY: &str = "start";
const ANTHROPIC_TEXT_END_KEY: &str = "end";
const ANTHROPIC_REPLAY_PROVIDER: &str = "anthropic";
const ANTHROPIC_REPLAY_KIND: &str = "content_blocks";
const ANTHROPIC_REPLAY_VERSION: u8 = 1;

use crate::stream_guard::{AbortOnDrop, SseFinish};
use zeroclaw_api::StopReason;

fn anthropic_tool_call_index(extra: Option<&serde_json::Value>) -> Option<u64> {
    extra?
        .get(ANTHROPIC_EXTRA_CONTENT_KEY)?
        .get(ANTHROPIC_BLOCK_INDEX_KEY)?
        .as_u64()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Provider-wire source of truth for Anthropic continuation blocks.
/// `ChatResponse.text` remains the independently normalized display value.
struct AnthropicReplayEnvelope {
    provider: String,
    kind: String,
    version: u8,
    blocks: Vec<AnthropicReplayBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicReplayBlock {
    Thinking {
        index: u64,
        thinking: String,
        signature: String,
    },
    Text {
        index: u64,
        text: String,
    },
    ToolUse {
        index: u64,
        id: String,
    },
    RedactedThinking {
        index: u64,
        data: String,
    },
}

impl AnthropicReplayEnvelope {
    fn encode(blocks: Vec<AnthropicReplayBlock>) -> Option<String> {
        if blocks.is_empty() {
            return None;
        }
        serde_json::to_string(&Self {
            provider: ANTHROPIC_REPLAY_PROVIDER.to_string(),
            kind: ANTHROPIC_REPLAY_KIND.to_string(),
            version: ANTHROPIC_REPLAY_VERSION,
            blocks,
        })
        .ok()
    }

    fn encode_reasoning_history(blocks: Vec<AnthropicReplayBlock>) -> Option<String> {
        if !blocks.iter().any(AnthropicReplayBlock::is_reasoning) {
            return None;
        }
        Self::encode(blocks)
    }

    fn decode(content: &str) -> Option<Vec<AnthropicReplayBlock>> {
        let mut blocks = content
            .lines()
            .filter_map(Self::decode_snapshot)
            .flatten()
            .collect::<Vec<_>>();
        blocks.sort_by_key(AnthropicReplayBlock::index);
        let has_duplicate_index = blocks
            .windows(2)
            .any(|pair| pair[0].index() == pair[1].index());
        (!blocks.is_empty() && !has_duplicate_index).then_some(blocks)
    }

    fn decode_snapshot(content: &str) -> Option<Vec<AnthropicReplayBlock>> {
        let envelope = serde_json::from_str::<Self>(content).ok()?;
        (envelope.provider == ANTHROPIC_REPLAY_PROVIDER
            && envelope.kind == ANTHROPIC_REPLAY_KIND
            && envelope.version == ANTHROPIC_REPLAY_VERSION)
            .then_some(envelope.blocks)
    }
}

impl AnthropicReplayBlock {
    fn index(&self) -> u64 {
        match self {
            Self::Thinking { index, .. }
            | Self::Text { index, .. }
            | Self::ToolUse { index, .. }
            | Self::RedactedThinking { index, .. } => *index,
        }
    }

    fn is_reasoning(&self) -> bool {
        matches!(self, Self::Thinking { .. } | Self::RedactedThinking { .. })
    }
}

pub struct AnthropicModelProvider {
    /// `[providers.models.anthropic.<alias>]` config-key alias.
    alias: String,
    credential: Option<String>,
    base_url: String,
    max_tokens: u32,
    timeout_secs: u64,
    /// Provider-level reasoning effort forwarded from `[runtime] reasoning_effort`.
    /// Serialized verbatim as Anthropic's top-level `output_config.effort`.
    /// Independent of per-request thinking.
    reasoning_effort: Option<String>,
    /// Caller-supplied headers stamped on every request (validated once at
    /// build time; see `extra_headers`). Applied per request rather than as
    /// client default headers so the pooled runtime proxy client stays shared.
    extra_headers: Vec<(reqwest::header::HeaderName, reqwest::header::HeaderValue)>,
}

#[cfg(test)]
#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
}

#[cfg(test)]
#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[cfg(test)]
#[derive(Debug, Deserialize)]
struct ChatResponse {
    content: Vec<ContentBlock>,
}

#[cfg(test)]
#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Serialize)]
struct NativeChatRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<SystemPrompt>,
    messages: Vec<NativeMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<NativeToolSpec>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<NativeThinkingConfig>,
    /// Anthropic places reasoning effort under a top-level `output_config`
    /// object, not inside `thinking`. `None` omits the field entirely.
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<OutputConfig>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum NativeThinkingConfig {
    Enabled { budget_tokens: u32 },
    Adaptive { display: ThinkingDisplay },
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
enum ThinkingDisplay {
    Summarized,
}

#[derive(Debug, Serialize)]
struct OutputConfig {
    effort: String,
}

#[derive(Debug, Default)]
struct StreamingThinkingBlock {
    index: u64,
    thinking: String,
    signature: String,
}

impl StreamingThinkingBlock {
    fn from_start(index: u64, block: &serde_json::Value) -> Self {
        Self {
            index,
            thinking: block
                .get("thinking")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
            signature: block
                .get("signature")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
        }
    }

    fn replay_block(self) -> Option<AnthropicReplayBlock> {
        if self.thinking.is_empty() && self.signature.is_empty() {
            return None;
        }
        Some(AnthropicReplayBlock::Thinking {
            index: self.index,
            thinking: self.thinking,
            signature: self.signature,
        })
    }
}

#[derive(Debug)]
enum StreamingReasoningBlock {
    Thinking(StreamingThinkingBlock),
    Redacted { index: u64, data: String },
}

impl StreamingReasoningBlock {
    fn from_start(index: u64, block: &serde_json::Value) -> Option<Self> {
        match block.get("type").and_then(serde_json::Value::as_str) {
            Some("thinking") => Some(Self::Thinking(StreamingThinkingBlock::from_start(
                index, block,
            ))),
            Some("redacted_thinking") => {
                block
                    .get("data")
                    .and_then(serde_json::Value::as_str)
                    .map(|data| Self::Redacted {
                        index,
                        data: data.to_string(),
                    })
            }
            _ => None,
        }
    }

    fn index(&self) -> u64 {
        match self {
            Self::Thinking(block) => block.index,
            Self::Redacted { index, .. } => *index,
        }
    }

    fn replay_block(self) -> Option<AnthropicReplayBlock> {
        match self {
            Self::Thinking(block) => block.replay_block(),
            Self::Redacted { index, data } => {
                Some(AnthropicReplayBlock::RedactedThinking { index, data })
            }
        }
    }
}

/// Returns true for models where ZeroClaw should send
/// `thinking: {type:"adaptive"}`. Unknown models default to `true` so future
/// Anthropic releases do not receive the legacy fixed-budget shape.
///
/// Claude 4.6 accepts both modes, but Anthropic deprecates manual thinking on
/// 4.6 and recommends adaptive. Claude 4.5 and earlier remain on manual mode.
fn anthropic_model_uses_adaptive_thinking(model: &str) -> bool {
    let id = model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .to_ascii_lowercase();
    !is_legacy_thinking_model(&id)
}

fn anthropic_model_is_claude_4_6(model: &str) -> bool {
    let id = model.rsplit('/').next().unwrap_or(model);
    let mut tokens = id.split('-');
    tokens.next() == Some("claude")
        && tokens.next().is_some()
        && tokens.next() == Some("4")
        && tokens.next() == Some("6")
}

/// True for models that must use the fixed-budget
/// `{type:"enabled", budget_tokens}` shape: Claude 3.x and Claude 4.5 or older.
///
/// New-style Claude IDs are `claude-{family}-{major}[-{minor}][-{date}]`. The
/// original bare Claude 4 has no minor (e.g. `claude-sonnet-4`,
/// `claude-opus-4-20250514`) and is minor-less, so it is legacy; so are the
/// `4-1` through `4-5` minors. Claude 4.6 and later use adaptive. Older
/// `claude-3-*` naming is matched by prefix and never places `4` in the major
/// slot, so it cannot collide with the new-style branch.
fn is_legacy_thinking_model(id: &str) -> bool {
    // Old `claude-3-*` naming covers every Claude 3 variant, dated or not.
    if id.starts_with("claude-3") {
        return true;
    }
    // New-style `claude-{family}-{major}[-{minor}][-{date}]`.
    let mut tokens = id.split('-');
    if tokens.next() != Some("claude") {
        return false;
    }
    let _ = tokens.next(); // family: sonnet / opus / haiku / ...
    if tokens.next() != Some("4") {
        return false; // major 5+, non-numeric, or absent -> adaptive default
    }
    match tokens.next() {
        // bare `claude-sonnet-4`
        None => true,
        // `claude-{family}-4-{YYYYMMDD}` with no minor
        Some(date) if date.len() == 8 && date.bytes().all(|b| b.is_ascii_digit()) => true,
        // `claude-{family}-4-{minor}[-{date}]`: manual-only iff minor < 6
        Some(minor) => minor.parse::<u32>().is_ok_and(|m| m < 6),
    }
}

#[derive(Debug, Serialize)]
struct NativeMessage {
    role: String,
    content: Vec<NativeContentOut>,
}

#[derive(Debug, Serialize)]
struct ImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum NativeContentOut {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "image")]
    Image { source: ImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Thinking block for round-tripping extended thinking in conversation
    /// history. Required when thinking is enabled and assistant messages
    /// contain tool_use blocks.
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
}

#[derive(Debug)]
struct IndexedNativeContent {
    /// Original position in Anthropic's assistant `content` array.
    index: Option<u64>,
    block: NativeContentOut,
}

#[derive(Debug, Clone, Copy)]
struct IndexedTextSpan {
    index: u64,
    start: usize,
    end: usize,
}

#[derive(Debug, Default)]
struct StreamingTextBlock {
    index: u64,
    text: String,
}

impl StreamingTextBlock {
    fn new(index: u64, initial_text: &str) -> Self {
        Self {
            index,
            text: initial_text.to_string(),
        }
    }

    fn replay_block(self) -> AnthropicReplayBlock {
        AnthropicReplayBlock::Text {
            index: self.index,
            text: self.text,
        }
    }
}

#[derive(Debug, Serialize)]
struct NativeToolSpec {
    name: String,
    description: String,
    /// `Arc`-shared with the tool registry's stored schema when no cleaning
    /// is required — serialized transparently, deep-cloned only for schemas
    /// the Anthropic cleaner actually rewrites
    input_schema: std::sync::Arc<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Debug, Clone, Serialize)]
struct CacheControl {
    #[serde(rename = "type")]
    cache_type: String,
}

impl CacheControl {
    fn ephemeral() -> Self {
        Self {
            cache_type: "ephemeral".to_string(),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum SystemPrompt {
    String(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Debug, Serialize)]
struct SystemBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Debug, Deserialize)]
struct NativeChatResponse {
    #[serde(default)]
    content: Vec<NativeContentIn>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    /// Tokens *after* the last cache breakpoint — NOT the total prompt.
    /// Per Anthropic prompt-caching docs:
    /// total_input = cache_read + cache_creation + input_tokens.
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    /// Tokens served from the prompt cache this request.
    #[serde(default)]
    cache_read_input_tokens: Option<u64>,
    /// Tokens written to the prompt cache this request (cache miss path).
    /// Disjoint from `cache_read_input_tokens` and `input_tokens`.
    #[serde(default)]
    cache_creation_input_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct NativeContentIn {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    /// Signature for integrity verification of thinking blocks.
    #[serde(default)]
    signature: Option<String>,
    /// Opaque encrypted payload for safety-redacted thinking blocks.
    #[serde(default)]
    data: Option<String>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    input: Option<serde_json::Value>,
}

/// Typed builder for [`AnthropicModelProvider`].
///
/// `alias` is the only positional argument. Everything else has a
/// sensible default: the base URL falls back to Anthropic's published
/// endpoint, no credential leaves the provider unauthenticated (fine
/// for local mocks), and token/timeout limits use the workspace baselines.
#[must_use]
pub struct AnthropicBuilder {
    alias: String,
    credential: Option<String>,
    base_url: Option<String>,
    max_tokens: Option<u32>,
    timeout_secs: Option<u64>,
    reasoning_effort: Option<String>,
    extra_headers: std::collections::HashMap<String, String>,
}

impl AnthropicBuilder {
    /// Explicit API credential. Whitespace-only inputs are normalized
    /// to `None` so a stray `Some("   ")` from config cannot produce a
    /// bogus `Bearer    ` header.
    pub fn credential(mut self, credential: Option<&str>) -> Self {
        self.credential = credential
            .map(str::trim)
            .filter(|k| !k.is_empty())
            .map(ToString::to_string);
        self
    }

    /// Override the API endpoint. Trailing slashes are stripped so
    /// callers need not care whether config supplied them.
    pub fn base_url(mut self, base_url: &str) -> Self {
        self.base_url = Some(base_url.trim_end_matches('/').to_string());
        self
    }

    /// Override the maximum output tokens for API requests. Defaults to
    /// [`zeroclaw_api::model_provider::BASELINE_MAX_TOKENS`] when unset.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Override the HTTP request timeout for LLM API calls. Defaults to
    /// [`zeroclaw_api::model_provider::BASELINE_TIMEOUT_SECS`] when unset.
    pub fn timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = Some(timeout_secs);
        self
    }

    /// Provider-level reasoning effort, forwarded from `[runtime] reasoning_effort`.
    /// Serialized verbatim as Anthropic's top-level `output_config.effort`.
    /// Mirrors the Azure/OpenAI `reasoning_effort` builder method for
    /// cross-provider parity.
    pub fn reasoning_effort(mut self, effort: Option<String>) -> Self {
        self.reasoning_effort = effort;
        self
    }

    /// Extra HTTP headers to send on every request, e.g. a host's per-turn
    /// correlation tags. Same contract as the compatible provider's
    /// `extra_headers`: reserved names (`Authorization`, `x-api-key`, framing)
    /// and invalid entries are dropped with a warning.
    pub fn extra_headers(mut self, headers: std::collections::HashMap<String, String>) -> Self {
        self.extra_headers = headers;
        self
    }

    pub fn build(self) -> AnthropicModelProvider {
        AnthropicModelProvider {
            alias: self.alias,
            credential: self.credential,
            base_url: self.base_url.unwrap_or_else(|| BASE_URL.to_string()),
            max_tokens: self
                .max_tokens
                .unwrap_or(zeroclaw_api::model_provider::BASELINE_MAX_TOKENS),
            timeout_secs: self
                .timeout_secs
                .unwrap_or(zeroclaw_api::model_provider::BASELINE_TIMEOUT_SECS),
            reasoning_effort: self.reasoning_effort,
            extra_headers: crate::extra_headers::typed_extra_headers(&self.extra_headers),
        }
    }
}

impl AnthropicModelProvider {
    /// Entry point. Only `alias` is required; every other field is set
    /// via a labelled chain method on the returned [`AnthropicBuilder`].
    pub fn builder(alias: &str) -> AnthropicBuilder {
        AnthropicBuilder {
            alias: alias.to_string(),
            credential: None,
            base_url: None,
            max_tokens: None,
            timeout_secs: None,
            reasoning_effort: None,
            extra_headers: std::collections::HashMap::new(),
        }
    }

    fn is_setup_token(token: &str) -> bool {
        token.starts_with("sk-ant-oat01-")
    }

    fn apply_auth(
        &self,
        request: reqwest::RequestBuilder,
        credential: &str,
    ) -> reqwest::RequestBuilder {
        let is_setup = Self::is_setup_token(credential);
        let len = credential.len();
        let head: String = credential.chars().take(8).collect();
        let tail: String = credential
            .chars()
            .rev()
            .take(4)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        ::zeroclaw_log::record!(DEBUG, ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note).with_attrs(::serde_json::json!({"header": if is_setup { "Authorization" } else { "x-api-key" }, "credential_len": len, "credential_head": head, "credential_tail": tail})), "Anthropic auth header applied");
        if is_setup {
            request
                .header("Authorization", format!("Bearer {credential}"))
                .header(
                    "anthropic-beta",
                    "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14",
                )
                .header("anthropic-dangerous-direct-browser-access", "true")
        } else {
            request.header("x-api-key", credential)
        }
    }

    /// For OAuth tokens, Anthropic requires the system prompt to start with the
    /// Claude Code identity prefix. This prepends it to any existing system prompt.
    ///
    /// Anthropic allows at most four explicit `cache_control` breakpoints. When a
    /// full system block follows the identity prefix, caching the identity alone is
    /// redundant (the later system breakpoint already covers that prefix). Keep the
    /// identity block uncached in that case so a slot remains for MemBox's stable
    /// request boundary plus an incremental tool-loop tail.
    fn apply_oauth_system_prompt(system: Option<SystemPrompt>) -> Option<SystemPrompt> {
        let identity_text = "You are Claude Code, Anthropic's official CLI for Claude.".to_string();
        match system {
            Some(SystemPrompt::Blocks(mut blocks)) => {
                blocks.insert(
                    0,
                    SystemBlock {
                        block_type: "text".to_string(),
                        text: identity_text,
                        cache_control: None,
                    },
                );
                if !blocks.iter().any(|block| block.cache_control.is_some())
                    && let Some(last) = blocks.last_mut()
                {
                    last.cache_control = Some(CacheControl::ephemeral());
                }
                Some(SystemPrompt::Blocks(blocks))
            }
            Some(SystemPrompt::String(s)) => Some(SystemPrompt::Blocks(vec![
                SystemBlock {
                    block_type: "text".to_string(),
                    text: identity_text,
                    cache_control: None,
                },
                SystemBlock {
                    block_type: "text".to_string(),
                    text: s,
                    cache_control: Some(CacheControl::ephemeral()),
                },
            ])),
            None => Some(SystemPrompt::Blocks(vec![SystemBlock {
                block_type: "text".to_string(),
                text: identity_text,
                // Identity is the only system block, so it owns the system breakpoint.
                cache_control: Some(CacheControl::ephemeral()),
            }])),
        }
    }

    /// Cache conversations with more than 1 non-system message (i.e. after first exchange)
    fn should_cache_conversation(messages: &[ChatMessage]) -> bool {
        messages.iter().filter(|m| m.role != "system").count() > 1
    }

    /// Apply cache control to the last message content block
    fn apply_cache_to_last_message(messages: &mut [NativeMessage]) {
        if let Some(last_msg) = messages.last_mut()
            && let Some(last_content) = last_msg.content.last_mut()
        {
            match last_content {
                NativeContentOut::Text { cache_control, .. }
                | NativeContentOut::ToolResult { cache_control, .. } => {
                    *cache_control = Some(CacheControl::ephemeral());
                }
                NativeContentOut::ToolUse { .. }
                | NativeContentOut::Image { .. }
                | NativeContentOut::Thinking { .. }
                | NativeContentOut::RedactedThinking { .. } => {}
            }
        }
    }

    fn native_messages_have_stable_cache_boundary(messages: &[NativeMessage]) -> bool {
        messages.iter().any(|message| {
            message.content.iter().any(|block| {
                matches!(
                    block,
                    NativeContentOut::Text {
                        cache_control: Some(_),
                        ..
                    }
                )
            })
        })
    }

    /// Anthropic accepts at most four explicit breakpoints per request.
    const MAX_EXPLICIT_CACHE_BREAKPOINTS: usize = 4;

    fn content_block_has_cache(block: &NativeContentOut) -> bool {
        matches!(
            block,
            NativeContentOut::Text {
                cache_control: Some(_),
                ..
            } | NativeContentOut::ToolResult {
                cache_control: Some(_),
                ..
            } | NativeContentOut::ToolUse {
                cache_control: Some(_),
                ..
            }
        )
    }

    fn count_explicit_cache_breakpoints(
        system: Option<&SystemPrompt>,
        tools: Option<&[NativeToolSpec]>,
        messages: &[NativeMessage],
    ) -> usize {
        let system_count = match system {
            Some(SystemPrompt::Blocks(blocks)) => blocks
                .iter()
                .filter(|block| block.cache_control.is_some())
                .count(),
            Some(SystemPrompt::String(_)) | None => 0,
        };
        let tool_count = tools
            .map(|specs| {
                specs
                    .iter()
                    .filter(|tool| tool.cache_control.is_some())
                    .count()
            })
            .unwrap_or(0);
        let message_count = messages
            .iter()
            .flat_map(|message| message.content.iter())
            .filter(|block| Self::content_block_has_cache(block))
            .count();
        system_count + tool_count + message_count
    }

    /// When MemBox already marked a stable request boundary, still cache the
    /// growing tool-loop tail — but never the volatile MemoryBox context that
    /// shares the same user message as that boundary.
    fn apply_incremental_tail_cache_after_stable_boundary(messages: &mut [NativeMessage]) {
        let Some(last_msg) = messages.last() else {
            return;
        };
        let same_message_has_boundary = last_msg.content.iter().any(|block| {
            matches!(
                block,
                NativeContentOut::Text {
                    cache_control: Some(_),
                    ..
                }
            )
        });
        if same_message_has_boundary {
            return;
        }
        Self::apply_cache_to_last_message(messages);
    }

    fn apply_conversation_cache_control(
        source_messages: &[ChatMessage],
        native_messages: &mut [NativeMessage],
    ) {
        if !Self::should_cache_conversation(source_messages) {
            return;
        }
        if Self::native_messages_have_stable_cache_boundary(native_messages) {
            Self::apply_incremental_tail_cache_after_stable_boundary(native_messages);
            return;
        }
        Self::apply_cache_to_last_message(native_messages);
    }

    /// Drop lowest-priority conversation tail breakpoints first when a request
    /// would otherwise exceed Anthropic's four-breakpoint cap. Stable MemBox
    /// boundaries, system, and tool breakpoints are preserved.
    fn enforce_max_explicit_cache_breakpoints(
        system: Option<&SystemPrompt>,
        tools: Option<&[NativeToolSpec]>,
        messages: &mut [NativeMessage],
    ) {
        while Self::count_explicit_cache_breakpoints(system, tools, messages)
            > Self::MAX_EXPLICIT_CACHE_BREAKPOINTS
        {
            let Some((message_index, content_index)) =
                Self::lowest_priority_tail_breakpoint(messages)
            else {
                break;
            };
            match &mut messages[message_index].content[content_index] {
                NativeContentOut::Text { cache_control, .. }
                | NativeContentOut::ToolResult { cache_control, .. } => {
                    *cache_control = None;
                }
                _ => break,
            }
        }
    }

    fn lowest_priority_tail_breakpoint(messages: &[NativeMessage]) -> Option<(usize, usize)> {
        // Prefer stripping incremental tool_result tails first.
        for (message_index, message) in messages.iter().enumerate().rev() {
            for (content_index, block) in message.content.iter().enumerate().rev() {
                if matches!(
                    block,
                    NativeContentOut::ToolResult {
                        cache_control: Some(_),
                        ..
                    }
                ) {
                    return Some((message_index, content_index));
                }
            }
        }
        // Then strip trailing text caches that are not a MemBox boundary placed
        // immediately before volatile uncached sibling content.
        for (message_index, message) in messages.iter().enumerate().rev() {
            for (content_index, block) in message.content.iter().enumerate().rev() {
                if !matches!(
                    block,
                    NativeContentOut::Text {
                        cache_control: Some(_),
                        ..
                    }
                ) {
                    continue;
                }
                let has_later_uncached_sibling =
                    message.content[content_index + 1..].iter().any(|sibling| {
                        matches!(
                            sibling,
                            NativeContentOut::Text {
                                cache_control: None,
                                ..
                            } | NativeContentOut::ToolResult {
                                cache_control: None,
                                ..
                            }
                        )
                    });
                if has_later_uncached_sibling {
                    continue;
                }
                return Some((message_index, content_index));
            }
        }
        None
    }

    fn parse_reasoning_content_block(block: &serde_json::Value) -> Option<IndexedNativeContent> {
        let index = block
            .get(ANTHROPIC_BLOCK_INDEX_KEY)
            .and_then(serde_json::Value::as_u64);
        let block = match block.get("type").and_then(serde_json::Value::as_str) {
            Some(ANTHROPIC_TEXT_BLOCK_MARKER) => return None,
            Some("redacted_thinking") => block
                .get("data")
                .and_then(serde_json::Value::as_str)
                .map(|data| NativeContentOut::RedactedThinking {
                    data: data.to_string(),
                })?,
            Some("thinking") => NativeContentOut::Thinking {
                thinking: block
                    .get("thinking")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                signature: block
                    .get("signature")
                    .and_then(serde_json::Value::as_str)
                    .filter(|signature| !signature.is_empty())
                    .map(str::to_string),
            },
            None if block.get("thinking").is_some() || block.get("signature").is_some() => {
                NativeContentOut::Thinking {
                    thinking: block
                        .get("thinking")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    signature: block
                        .get("signature")
                        .and_then(serde_json::Value::as_str)
                        .filter(|signature| !signature.is_empty())
                        .map(str::to_string),
                }
            }
            _ => return None,
        };
        Some(IndexedNativeContent { index, block })
    }

    fn native_tool_call(call: ProviderToolCall) -> NativeContentOut {
        let input = serde_json::from_str::<serde_json::Value>(&call.arguments)
            .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));
        NativeContentOut::ToolUse {
            id: call.id,
            name: call.name,
            input,
            cache_control: None,
        }
    }

    fn parse_text_span(block: &serde_json::Value) -> Option<IndexedTextSpan> {
        if block.get("type").and_then(serde_json::Value::as_str)
            != Some(ANTHROPIC_TEXT_BLOCK_MARKER)
        {
            return None;
        }
        Some(IndexedTextSpan {
            index: block.get(ANTHROPIC_BLOCK_INDEX_KEY)?.as_u64()?,
            start: usize::try_from(block.get(ANTHROPIC_TEXT_START_KEY)?.as_u64()?).ok()?,
            end: usize::try_from(block.get(ANTHROPIC_TEXT_END_KEY)?.as_u64()?).ok()?,
        })
    }

    fn indexed_tool_call(call: ProviderToolCall) -> IndexedNativeContent {
        let index = anthropic_tool_call_index(call.extra_content.as_ref());
        IndexedNativeContent {
            index,
            block: Self::native_tool_call(call),
        }
    }

    fn rebuild_replay_envelope(
        replay: Vec<AnthropicReplayBlock>,
        mut tool_calls: Vec<ProviderToolCall>,
    ) -> Option<Vec<NativeContentOut>> {
        let mut blocks = Vec::with_capacity(replay.len());
        for block in replay {
            blocks.push(match block {
                AnthropicReplayBlock::Thinking {
                    thinking,
                    signature,
                    ..
                } => NativeContentOut::Thinking {
                    thinking,
                    signature: (!signature.is_empty()).then_some(signature),
                },
                AnthropicReplayBlock::Text { text, .. } => NativeContentOut::Text {
                    text,
                    cache_control: None,
                },
                AnthropicReplayBlock::RedactedThinking { data, .. } => {
                    NativeContentOut::RedactedThinking { data }
                }
                AnthropicReplayBlock::ToolUse { id, .. } => {
                    let position = tool_calls.iter().position(|call| call.id == id)?;
                    Self::native_tool_call(tool_calls.remove(position))
                }
            });
        }
        tool_calls.is_empty().then_some(blocks)
    }

    fn ordered_indexed_blocks(mut blocks: Vec<IndexedNativeContent>) -> Vec<NativeContentOut> {
        blocks.sort_by_key(|block| block.index);
        blocks.into_iter().map(|block| block.block).collect()
    }

    fn append_text_block(blocks: &mut Vec<NativeContentOut>, text: Option<&str>) {
        if let Some(text) = text.map(str::trim).filter(|text| !text.is_empty()) {
            blocks.push(NativeContentOut::Text {
                text: text.to_string(),
                cache_control: None,
            });
        }
    }

    fn parse_reasoning_blocks(value: &serde_json::Value) -> Vec<IndexedNativeContent> {
        value
            .get("reasoning_content")
            .and_then(serde_json::Value::as_str)
            .into_iter()
            .flat_map(str::lines)
            .filter_map(|part| serde_json::from_str::<serde_json::Value>(part).ok())
            .filter_map(|block| Self::parse_reasoning_content_block(&block))
            .collect()
    }

    fn parse_text_spans(value: &serde_json::Value) -> Vec<IndexedTextSpan> {
        value
            .get("reasoning_content")
            .and_then(serde_json::Value::as_str)
            .into_iter()
            .flat_map(str::lines)
            .filter_map(|part| serde_json::from_str::<serde_json::Value>(part).ok())
            .filter_map(|block| Self::parse_text_span(&block))
            .collect()
    }

    fn indexed_text_blocks(
        text: Option<&str>,
        mut spans: Vec<IndexedTextSpan>,
    ) -> Option<Vec<IndexedNativeContent>> {
        let Some(text) = text else {
            return spans.is_empty().then_some(Vec::new());
        };
        if spans.is_empty() {
            return None;
        }
        spans.sort_by_key(|span| span.start);
        let mut cursor = 0;
        let mut blocks = Vec::with_capacity(spans.len());
        for span in spans {
            let gap = text.get(cursor..span.start)?;
            if !gap.is_empty() && gap != "\n" {
                return None;
            }
            let part = text.get(span.start..span.end)?;
            blocks.push(IndexedNativeContent {
                index: Some(span.index),
                block: NativeContentOut::Text {
                    text: part.to_string(),
                    cache_control: None,
                },
            });
            cursor = span.end;
        }
        (cursor == text.len()).then_some(blocks)
    }

    fn rebuild_assistant_blocks(
        reasoning: Vec<IndexedNativeContent>,
        tools: Vec<IndexedNativeContent>,
        text: Option<&str>,
        text_spans: Vec<IndexedTextSpan>,
    ) -> Vec<NativeContentOut> {
        let indexed_text = Self::indexed_text_blocks(text, text_spans);
        let can_restore_order = indexed_text.is_some()
            && reasoning
                .iter()
                .chain(&tools)
                .all(|block| block.index.is_some());
        if can_restore_order {
            return Self::ordered_indexed_blocks(
                reasoning
                    .into_iter()
                    .chain(indexed_text.unwrap_or_default())
                    .chain(tools)
                    .collect(),
            );
        }
        let mut blocks = reasoning.into_iter().map(|block| block.block).collect();
        Self::append_text_block(&mut blocks, text);
        blocks.extend(tools.into_iter().map(|block| block.block));
        blocks
    }

    fn convert_tools(tools: Option<&[ToolSpec]>) -> Option<Vec<NativeToolSpec>> {
        let items = tools?;
        if items.is_empty() {
            return None;
        }
        let mut native_tools: Vec<NativeToolSpec> = items
            .iter()
            .map(|tool| NativeToolSpec {
                name: tool.name.clone(),
                description: tool.description.clone(),
                input_schema: zeroclaw_api::schema::SchemaCleanr::clean_shared(
                    &tool.parameters,
                    zeroclaw_api::schema::CleaningStrategy::Anthropic,
                ),
                cache_control: None,
            })
            .collect();

        // Cache the last tool definition (caches all tools)
        if let Some(last_tool) = native_tools.last_mut() {
            last_tool.cache_control = Some(CacheControl::ephemeral());
        }

        Some(native_tools)
    }

    fn parse_assistant_tool_call_message(content: &str) -> Option<Vec<NativeContentOut>> {
        let value = serde_json::from_str::<serde_json::Value>(content).ok()?;
        let tool_calls = value
            .get("tool_calls")
            .and_then(|v| serde_json::from_value::<Vec<ProviderToolCall>>(v.clone()).ok())?;
        let reasoning_content = value
            .get("reasoning_content")
            .and_then(serde_json::Value::as_str);
        if let Some(replay) = reasoning_content.and_then(AnthropicReplayEnvelope::decode)
            && let Some(blocks) = Self::rebuild_replay_envelope(replay, tool_calls.clone())
        {
            return Some(blocks);
        }
        let text = value
            .get("content")
            .and_then(serde_json::Value::as_str)
            .filter(|text| !text.trim().is_empty());
        let reasoning_blocks = Self::parse_reasoning_blocks(&value);
        let text_spans = Self::parse_text_spans(&value);
        let tool_blocks = tool_calls
            .into_iter()
            .map(Self::indexed_tool_call)
            .collect::<Vec<_>>();
        Some(Self::rebuild_assistant_blocks(
            reasoning_blocks,
            tool_blocks,
            text,
            text_spans,
        ))
    }

    fn parse_tool_result_message(content: &str) -> Option<NativeMessage> {
        let value = serde_json::from_str::<serde_json::Value>(content).ok()?;
        let tool_use_id = value
            .get("tool_call_id")
            .and_then(serde_json::Value::as_str)?
            .to_string();
        let result = value
            .get("content")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_string();
        Some(NativeMessage {
            role: "user".to_string(),
            content: vec![NativeContentOut::ToolResult {
                tool_use_id,
                content: result,
                cache_control: None,
            }],
        })
    }

    fn convert_messages(messages: &[ChatMessage]) -> (Option<SystemPrompt>, Vec<NativeMessage>) {
        let mut system_text = None;
        let mut native_messages = Vec::new();
        let last_boundary_index = messages.iter().enumerate().rev().find_map(|(index, msg)| {
            (msg.role == "user" && zeroclaw_api::has_membox_prompt_cache_boundary(&msg.content))
                .then_some(index)
        });

        for (index, msg) in messages.iter().enumerate() {
            if ChatMessage::should_skip_internal_pruning_marker(messages, index) {
                continue;
            }
            match msg.role.as_str() {
                "system" => {
                    if system_text.is_none() {
                        system_text = Some(msg.content.clone());
                    }
                }
                "assistant" => {
                    if let Some(blocks) = Self::parse_assistant_tool_call_message(&msg.content) {
                        native_messages.push(NativeMessage {
                            role: "assistant".to_string(),
                            content: blocks,
                        });
                    } else if !msg.content.trim().is_empty() {
                        native_messages.push(NativeMessage {
                            role: "assistant".to_string(),
                            content: vec![NativeContentOut::Text {
                                text: msg.content.clone(),
                                cache_control: None,
                            }],
                        });
                    }
                }
                "tool" => {
                    let tool_msg = if let Some(tr) = Self::parse_tool_result_message(&msg.content) {
                        tr
                    } else if !msg.content.trim().is_empty() {
                        NativeMessage {
                            role: "user".to_string(),
                            content: vec![NativeContentOut::Text {
                                text: msg.content.clone(),
                                cache_control: None,
                            }],
                        }
                    } else {
                        continue;
                    };
                    // Tool results map to role "user"; merge consecutive ones
                    // into a single message so Anthropic doesn't reject the
                    // request for having adjacent same-role messages.
                    if native_messages
                        .last()
                        .is_some_and(|m| m.role == tool_msg.role)
                    {
                        native_messages
                            .last_mut()
                            .unwrap()
                            .content
                            .extend(tool_msg.content);
                    } else {
                        native_messages.push(tool_msg);
                    }
                }
                _ => {
                    // Parse image markers from user message content
                    let (content_body, had_boundary) =
                        zeroclaw_api::strip_membox_prompt_cache_boundary(&msg.content);
                    let (text, image_refs) = crate::multimodal::parse_image_markers(&content_body);
                    let cache_at_boundary = had_boundary && last_boundary_index == Some(index);
                    let mut content_blocks: Vec<NativeContentOut> = Vec::new();

                    // Add image content blocks for each image reference
                    for img_ref in &image_refs {
                        let (media_type, data) = if img_ref.starts_with("data:") {
                            // Data URI format: data:image/jpeg;base64,/9j/4AAQ...
                            if let Some(comma) = img_ref.find(',') {
                                let header = &img_ref[5..comma];
                                let mime =
                                    header.split(';').next().unwrap_or("image/jpeg").to_string();
                                let b64 = img_ref[comma + 1..].trim().to_string();
                                (mime, b64)
                            } else {
                                continue;
                            }
                        } else if std::path::Path::new(img_ref.trim()).exists() {
                            // Local file path
                            match std::fs::read(img_ref.trim()) {
                                Ok(bytes) => {
                                    let b64 =
                                        base64::engine::general_purpose::STANDARD.encode(&bytes);
                                    let ext = std::path::Path::new(img_ref.trim())
                                        .extension()
                                        .and_then(|e| e.to_str())
                                        .unwrap_or("jpg");
                                    let mime = match ext {
                                        "png" => "image/png",
                                        "gif" => "image/gif",
                                        "webp" => "image/webp",
                                        _ => "image/jpeg",
                                    }
                                    .to_string();
                                    (mime, b64)
                                }
                                Err(_) => continue,
                            }
                        } else {
                            continue;
                        };

                        content_blocks.push(NativeContentOut::Image {
                            source: ImageSource {
                                source_type: "base64".to_string(),
                                media_type,
                                data,
                            },
                        });
                    }

                    // Add text content block (skip empty text when images are present)
                    if text.is_empty() && !image_refs.is_empty() {
                        content_blocks.push(NativeContentOut::Text {
                            text: "[image]".to_string(),
                            cache_control: if cache_at_boundary {
                                Some(CacheControl::ephemeral())
                            } else {
                                None
                            },
                        });
                    } else if !text.trim().is_empty() {
                        content_blocks.push(NativeContentOut::Text {
                            text,
                            cache_control: if cache_at_boundary {
                                Some(CacheControl::ephemeral())
                            } else {
                                None
                            },
                        });
                    }

                    // Merge into previous user message if present (e.g.
                    // when a user message immediately follows tool results
                    // which are also role "user" in Anthropic's format).
                    if native_messages.last().is_some_and(|m| m.role == "user") {
                        native_messages
                            .last_mut()
                            .unwrap()
                            .content
                            .extend(content_blocks);
                    } else {
                        native_messages.push(NativeMessage {
                            role: "user".to_string(),
                            content: content_blocks,
                        });
                    }
                }
            }
        }

        Self::degrade_orphaned_tool_results(&mut native_messages);
        Self::backfill_orphaned_tool_uses(&mut native_messages);

        // Always use Blocks format with cache_control for system prompts
        let system_prompt = system_text.map(|text| {
            SystemPrompt::Blocks(vec![SystemBlock {
                block_type: "text".to_string(),
                text,
                cache_control: Some(CacheControl::ephemeral()),
            }])
        });

        (system_prompt, native_messages)
    }

    /// Preserve orphaned tool output as ordinary user text so the final
    /// Anthropic payload cannot contain a `tool_result` without a matching
    /// `tool_use` in the immediately preceding assistant message.
    fn degrade_orphaned_tool_results(messages: &mut [NativeMessage]) {
        for index in 0..messages.len() {
            let declared_ids = Self::preceding_tool_use_ids(messages, index);
            let orphaned_ids = Self::degrade_unmatched_results(&mut messages[index], &declared_ids);
            if !orphaned_ids.is_empty() {
                ::zeroclaw_log::record!(
                    WARN,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Validate)
                        .with_category(::zeroclaw_log::EventCategory::Provider)
                        .with_outcome(::zeroclaw_log::EventOutcome::Success)
                        .with_attrs(::serde_json::json!({
                            "message_index": index,
                            "orphan_tool_result_ids": orphaned_ids,
                        })),
                    "anthropic: degraded orphaned tool_result blocks to user text"
                );
            }
        }
    }

    fn preceding_tool_use_ids(
        messages: &[NativeMessage],
        index: usize,
    ) -> std::collections::HashSet<String> {
        let Some(previous_index) = index.checked_sub(1) else {
            return std::collections::HashSet::new();
        };
        messages
            .get(previous_index)
            .filter(|message| message.role == "assistant")
            .into_iter()
            .flat_map(|message| &message.content)
            .filter_map(|block| match block {
                NativeContentOut::ToolUse { id, .. } => Some(id.clone()),
                _ => None,
            })
            .collect()
    }

    fn degrade_unmatched_results(
        message: &mut NativeMessage,
        declared_ids: &std::collections::HashSet<String>,
    ) -> Vec<String> {
        let mut matched_results = Vec::new();
        let mut remaining_blocks = Vec::new();
        let mut orphaned_ids = Vec::new();
        for block in std::mem::take(&mut message.content) {
            match block {
                NativeContentOut::ToolResult {
                    tool_use_id,
                    content,
                    cache_control,
                } if declared_ids.contains(&tool_use_id) => {
                    matched_results.push(NativeContentOut::ToolResult {
                        tool_use_id,
                        content,
                        cache_control,
                    });
                }
                NativeContentOut::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } => {
                    remaining_blocks.push(Self::orphaned_tool_result_text(&tool_use_id, content));
                    orphaned_ids.push(tool_use_id);
                }
                other => remaining_blocks.push(other),
            }
        }
        matched_results.append(&mut remaining_blocks);
        message.content = matched_results;
        orphaned_ids
    }

    fn orphaned_tool_result_text(tool_use_id: &str, content: String) -> NativeContentOut {
        NativeContentOut::Text {
            text: format!("[Tool result for {tool_use_id}]\n{content}"),
            cache_control: None,
        }
    }

    /// Pair any orphaned `tool_use` with a stub `tool_result` so interrupted
    /// turns can't wedge the session with a hard 400 on replay. Defensive
    /// backstop for the canonical-history guard in the runtime.
    fn backfill_orphaned_tool_uses(messages: &mut Vec<NativeMessage>) {
        let mut idx = 0;
        while idx < messages.len() {
            let pending: Vec<String> = messages[idx]
                .content
                .iter()
                .filter_map(|block| match block {
                    NativeContentOut::ToolUse { id, .. } => Some(id.clone()),
                    _ => None,
                })
                .collect();

            if pending.is_empty() {
                idx += 1;
                continue;
            }

            let answered: std::collections::HashSet<String> = messages
                .get(idx + 1)
                .map(|next| {
                    next.content
                        .iter()
                        .filter_map(|block| match block {
                            NativeContentOut::ToolResult { tool_use_id, .. } => {
                                Some(tool_use_id.clone())
                            }
                            _ => None,
                        })
                        .collect()
                })
                .unwrap_or_default();

            let stubs: Vec<NativeContentOut> = pending
                .into_iter()
                .filter(|id| !answered.contains(id))
                .map(|tool_use_id| NativeContentOut::ToolResult {
                    tool_use_id,
                    content: "[tool result missing from history — the turn was \
                              interrupted before this tool finished]"
                        .to_string(),
                    cache_control: None,
                })
                .collect();

            if !stubs.is_empty() {
                if messages
                    .get(idx + 1)
                    .is_some_and(|next| next.role == "user")
                {
                    let next = &mut messages[idx + 1];
                    let mut merged = stubs;
                    merged.append(&mut next.content);
                    next.content = merged;
                } else {
                    messages.insert(
                        idx + 1,
                        NativeMessage {
                            role: "user".to_string(),
                            content: stubs,
                        },
                    );
                }
            }

            idx += 1;
        }
    }

    fn parse_native_response(response: NativeChatResponse) -> ProviderChatResponse {
        let mut text = String::new();
        let mut replay_blocks = Vec::new();
        let mut tool_calls = Vec::new();

        let usage = response.usage.map(|u| {
            let uncached = u.input_tokens.unwrap_or(0);
            let cache_read = u.cache_read_input_tokens.unwrap_or(0);
            let cache_create = u.cache_creation_input_tokens.unwrap_or(0);
            let total = uncached
                .saturating_add(cache_read)
                .saturating_add(cache_create);
            let any_reported = u.input_tokens.is_some()
                || u.cache_read_input_tokens.is_some()
                || u.cache_creation_input_tokens.is_some();
            TokenUsage {
                input_tokens: if any_reported { Some(total) } else { None },
                output_tokens: u.output_tokens,
                cached_input_tokens: u.cache_read_input_tokens,
            }
        });

        for (index, block) in response.content.into_iter().enumerate() {
            let index = u64::try_from(index).unwrap_or(u64::MAX);
            match block.kind.as_str() {
                "text" => {
                    if let Some(raw_text) = block.text {
                        replay_blocks.push(AnthropicReplayBlock::Text {
                            index,
                            text: raw_text.clone(),
                        });
                        let part = raw_text.trim().to_string();
                        if !part.is_empty() {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(&part);
                        }
                    }
                }
                "thinking" => {
                    // Preserve signed thinking blocks for multi-turn/tool-use
                    // replay. On adaptive models `display` defaults to
                    // "omitted": the response returns `thinking: ""` with an
                    // encrypted `signature` that must be replayed unchanged or
                    // the continuation 400s. Keep the block whenever it carries
                    // a signature, even when the thinking text is empty.
                    let thinking = block
                        .thinking
                        .as_deref()
                        .or(block.text.as_deref())
                        .unwrap_or("");
                    let signature = block.signature.as_deref().unwrap_or("");
                    if !thinking.is_empty() || !signature.is_empty() {
                        replay_blocks.push(AnthropicReplayBlock::Thinking {
                            index,
                            thinking: thinking.to_string(),
                            signature: signature.to_string(),
                        });
                    }
                }
                "redacted_thinking" => {
                    if let Some(data) = block.data {
                        replay_blocks.push(AnthropicReplayBlock::RedactedThinking { index, data });
                    }
                }
                "tool_use" => {
                    let name = block.name.unwrap_or_default();
                    if name.is_empty() {
                        continue;
                    }
                    let arguments = block
                        .input
                        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
                    let id = block.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                    replay_blocks.push(AnthropicReplayBlock::ToolUse {
                        index,
                        id: id.clone(),
                    });
                    tool_calls.push(ProviderToolCall {
                        id,
                        name,
                        arguments: arguments.to_string(),
                        extra_content: None,
                    });
                }
                _ => {}
            }
        }

        let reasoning_content = AnthropicReplayEnvelope::encode_reasoning_history(replay_blocks);

        ProviderChatResponse {
            text: (!text.is_empty()).then_some(text),
            tool_calls,
            usage,
            reasoning_content,
        }
    }

    /// Resolve thinking parameters for an API request. Returns the effective
    /// temperature (forced to 1.0 when thinking is active), the thinking
    /// config for the request body, and the effective max_tokens. Manual mode
    /// raises the limit above budget_tokens; Claude 4.6 adaptive mode preserves
    /// the former budget as an output-capacity floor without serializing it.
    fn resolve_thinking(
        &self,
        thinking: Option<zeroclaw_api::model_provider::NativeThinkingParams>,
        temperature: Option<f64>,
        model: &str,
    ) -> (Option<f64>, Option<NativeThinkingConfig>, u32) {
        match thinking {
            Some(params) if anthropic_model_uses_adaptive_thinking(model) => {
                ::zeroclaw_log::record!(
                    INFO,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                        .with_attrs(::serde_json::json!({"model": model})),
                    "Adaptive thinking enabled; forcing temperature=1.0"
                );
                // Adaptive thinking can omit display output by default. Request
                // the only generally available visible form explicitly; Anthropic
                // does not expose raw chain-of-thought through this API.
                // Keep the former manual-thinking budget as a capacity floor so
                // switching Claude 4.6 to adaptive cannot shrink high-effort
                // requests to the provider baseline. The strict `+ 1` applies
                // only when budget_tokens is serialized in manual mode.
                let max_tokens = if anthropic_model_is_claude_4_6(model) {
                    self.max_tokens.max(params.budget_tokens)
                } else {
                    self.max_tokens
                };
                (
                    Some(1.0),
                    Some(NativeThinkingConfig::Adaptive {
                        display: ThinkingDisplay::Summarized,
                    }),
                    max_tokens,
                )
            }
            Some(params) => {
                ::zeroclaw_log::record!(
                    INFO,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                        .with_attrs(::serde_json::json!({"budget_tokens": params.budget_tokens})),
                    "Native extended thinking enabled; forcing temperature=1.0"
                );
                // API requires max_tokens > budget_tokens (strictly greater).
                let min_required = params.budget_tokens + 1;
                let max_tokens = self.max_tokens.max(min_required);
                (
                    Some(1.0),
                    Some(NativeThinkingConfig::Enabled {
                        budget_tokens: params.budget_tokens,
                    }),
                    max_tokens,
                )
            }
            None => (temperature, None, self.max_tokens),
        }
    }

    fn http_client(&self) -> Client {
        zeroclaw_config::schema::build_runtime_proxy_client_with_timeouts(
            "model_provider.anthropic",
            self.timeout_secs,
            10,
        )
    }

    fn streaming_http_client(&self) -> Client {
        // No total-request timeout: SSE bodies for long-form generations can
        // legitimately stay open for several minutes of active streaming.
        // `SSE_IDLE_TIMEOUT` (per-line, above) already bounds a genuinely
        // stalled connection. See #2404.
        zeroclaw_config::schema::build_runtime_proxy_streaming_client(
            "model_provider.anthropic",
            10,
        )
    }

    /// Build a streaming request body from a `NativeChatRequest`.
    fn build_streaming_request(request: &NativeChatRequest) -> anyhow::Result<serde_json::Value> {
        let mut body = serde_json::to_value(request)
            .context("Failed to serialize NativeChatRequest to JSON")?;
        body["stream"] = serde_json::Value::Bool(true);
        Ok(body)
    }

    /// Parse Anthropic SSE lines from `response` and send `StreamEvent`s to `tx`.
    async fn parse_anthropic_sse(
        response: reqwest::Response,
        tx: &tokio::sync::mpsc::Sender<StreamResult<StreamEvent>>,
    ) {
        use tokio_util::io::StreamReader;

        let byte_stream = response
            .bytes_stream()
            .map(|result| result.map_err(std::io::Error::other));
        let reader = StreamReader::new(byte_stream);
        Self::parse_anthropic_sse_from_reader(reader, tx).await;
    }

    /// Inner loop split out of `parse_anthropic_sse` so unit tests can feed a
    /// `Cursor<&[u8]>` directly without spinning up a mock HTTP server.
    async fn parse_anthropic_sse_from_reader<R>(
        reader: R,
        tx: &tokio::sync::mpsc::Sender<StreamResult<StreamEvent>>,
    ) where
        R: tokio::io::AsyncBufRead + Unpin,
    {
        use tokio::io::AsyncBufReadExt;

        let mut lines = reader.lines();

        let mut tool_id: Option<String> = None;
        let mut tool_name: Option<String> = None;
        let mut tool_input_json = String::new();
        let mut tool_index: Option<u64> = None;
        let mut reasoning_block: Option<StreamingReasoningBlock> = None;
        let mut text_block: Option<StreamingTextBlock> = None;
        let mut replay_blocks = Vec::new();
        let mut saw_replay_reasoning = false;

        let mut input_tokens: Option<u64> = None;
        let mut output_tokens: Option<u64> = None;
        let mut cached_input_tokens: Option<u64> = None;
        let mut cache_creation_input_tokens: Option<u64> = None;

        // Reason from `message_delta`. Not a completion signal — only
        // `message_stop` (or a clean Final emit) marks the stream complete.
        let mut pending_stop = None;

        loop {
            let line = match tokio::time::timeout(SSE_IDLE_TIMEOUT, lines.next_line()).await {
                Ok(Ok(Some(line))) => line,
                Ok(Ok(None)) => break,
                Ok(Err(err)) => {
                    ::zeroclaw_log::record!(
                        WARN,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                            .with_category(::zeroclaw_log::EventCategory::Provider)
                            .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                            .with_attrs(::serde_json::json!({
                                "error": format!("{err}"),
                            })),
                        "stream: SSE read error — aborting stream"
                    );
                    let _ = tx
                        .send(Err(StreamError::Http(format!("SSE read error: {err}"))))
                        .await;
                    return;
                }
                Err(_) => {
                    ::zeroclaw_log::record!(
                        WARN,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                            .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                            .with_attrs(::serde_json::json!({
                                "idle_secs": SSE_IDLE_TIMEOUT.as_secs(),
                            })),
                        "stream: SSE idle timeout — connection stalled, aborting stream"
                    );
                    let _ = tx
                        .send(Err(StreamError::Http(format!(
                            "SSE stream stalled: no data for {}s",
                            SSE_IDLE_TIMEOUT.as_secs()
                        ))))
                        .await;
                    return;
                }
            };
            let line = line.trim().to_string();
            if !line.starts_with("data: ") {
                continue;
            }
            let json_str = &line["data: ".len()..];

            let event: serde_json::Value = match serde_json::from_str(json_str) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let event_type = event
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or_default();

            match event_type {
                "message_start" => {
                    let model = event
                        .get("message")
                        .and_then(|m| m.get("model"))
                        .and_then(|m| m.as_str())
                        .unwrap_or("unknown");
                    let usage = event.get("message").and_then(|m| m.get("usage"));
                    let observed_input = usage
                        .and_then(|u| u.get("input_tokens"))
                        .and_then(|t| t.as_u64());
                    let observed_cached = usage
                        .and_then(|u| u.get("cache_read_input_tokens"))
                        .and_then(|t| t.as_u64());
                    let observed_cache_create = usage
                        .and_then(|u| u.get("cache_creation_input_tokens"))
                        .and_then(|t| t.as_u64());
                    if let Some(v) = observed_input {
                        input_tokens = Some(v);
                    }
                    if let Some(v) = observed_cached {
                        cached_input_tokens = Some(v);
                    }
                    if let Some(v) = observed_cache_create {
                        cache_creation_input_tokens = Some(v);
                    }
                    ::zeroclaw_log::record!(DEBUG, ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note).with_attrs(::serde_json::json!({"model": model, "input_tokens": observed_input, "cached_input_tokens": observed_cached, "cache_creation_input_tokens": observed_cache_create})), "stream: message_start");
                }
                "content_block_start" => {
                    if let Some(block) = event.get("content_block") {
                        let index = event
                            .get("index")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or_default();
                        let block_type = block
                            .get("type")
                            .and_then(|t| t.as_str())
                            .unwrap_or_default();
                        if let Some(block) = StreamingReasoningBlock::from_start(index, block) {
                            reasoning_block = Some(block);
                        } else if block_type == "text" {
                            let initial_text = block
                                .get("text")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or_default();
                            text_block = Some(StreamingTextBlock::new(index, initial_text));
                        } else if block_type == "tool_use" {
                            if let Some(id) = tool_id.take() {
                                let name = tool_name.take().unwrap_or_default();
                                let input = std::mem::take(&mut tool_input_json);
                                if let Some(index) = tool_index.take() {
                                    replay_blocks.push(AnthropicReplayBlock::ToolUse {
                                        index,
                                        id: id.clone(),
                                    });
                                }
                                let _ = tx
                                    .send(Ok(StreamEvent::ToolCall(ProviderToolCall {
                                        id,
                                        name,
                                        arguments: input,
                                        extra_content: None,
                                    })))
                                    .await;
                            }
                            tool_id = block
                                .get("id")
                                .and_then(|v| v.as_str())
                                .map(ToString::to_string);
                            tool_name = block
                                .get("name")
                                .and_then(|v| v.as_str())
                                .map(ToString::to_string);
                            tool_input_json.clear();
                            tool_index = Some(index);
                        }
                    }
                }
                "content_block_delta" => {
                    if let Some(delta) = event.get("delta") {
                        let index = event
                            .get("index")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or_default();
                        let delta_type = delta
                            .get("type")
                            .and_then(|t| t.as_str())
                            .unwrap_or_default();
                        match delta_type {
                            "text_delta" => {
                                if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                    if let Some(block) = text_block.as_mut()
                                        && block.index == index
                                    {
                                        block.text.push_str(text);
                                    }
                                    if !text.is_empty()
                                        && tx
                                            .send(Ok(StreamEvent::TextDelta(StreamChunk::delta(
                                                text.to_string(),
                                            ))))
                                            .await
                                            .is_err()
                                    {
                                        return;
                                    }
                                }
                            }
                            "input_json_delta" => {
                                if let Some(json) =
                                    delta.get("partial_json").and_then(|j| j.as_str())
                                {
                                    tool_input_json.push_str(json);
                                }
                            }
                            "thinking_delta" => {
                                if let Some(thinking) =
                                    delta.get("thinking").and_then(|value| value.as_str())
                                {
                                    if let Some(StreamingReasoningBlock::Thinking(block)) =
                                        reasoning_block.as_mut()
                                        && block.index == index
                                    {
                                        block.thinking.push_str(thinking);
                                    }
                                    if !thinking.is_empty()
                                        && tx
                                            .send(Ok(StreamEvent::TextDelta(
                                                StreamChunk::reasoning(thinking.to_string()),
                                            )))
                                            .await
                                            .is_err()
                                    {
                                        return;
                                    }
                                }
                            }
                            "signature_delta" => {
                                if let Some(signature) =
                                    delta.get("signature").and_then(|value| value.as_str())
                                    && let Some(StreamingReasoningBlock::Thinking(block)) =
                                        reasoning_block.as_mut()
                                    && block.index == index
                                {
                                    block.signature.push_str(signature);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                "content_block_stop" => {
                    let index = event
                        .get("index")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or_default();
                    let mut replay_changed = false;
                    if reasoning_block
                        .as_ref()
                        .is_some_and(|block| block.index() == index)
                        && let Some(block) = reasoning_block
                            .take()
                            .and_then(StreamingReasoningBlock::replay_block)
                    {
                        replay_blocks.push(block);
                        replay_changed = true;
                    }
                    if text_block
                        .as_ref()
                        .is_some_and(|block| block.index == index)
                        && let Some(block) = text_block.take()
                    {
                        replay_blocks.push(block.replay_block());
                        replay_changed = true;
                    }
                    if tool_index == Some(index)
                        && let Some(id) = tool_id.take()
                    {
                        let name = tool_name.take().unwrap_or_default();
                        let input = std::mem::take(&mut tool_input_json);
                        tool_index = None;
                        replay_blocks.push(AnthropicReplayBlock::ToolUse {
                            index,
                            id: id.clone(),
                        });
                        replay_changed = true;
                        let _ = tx
                            .send(Ok(StreamEvent::ToolCall(ProviderToolCall {
                                id,
                                name,
                                arguments: input,
                                extra_content: None,
                            })))
                            .await;
                    }
                    let completed_reasoning =
                        replay_blocks.iter().any(AnthropicReplayBlock::is_reasoning);
                    if replay_changed && (saw_replay_reasoning || completed_reasoning) {
                        saw_replay_reasoning = true;
                        if let Some(content) =
                            AnthropicReplayEnvelope::encode(std::mem::take(&mut replay_blocks))
                        {
                            let _ = tx.send(Ok(StreamEvent::ReasoningContent(content))).await;
                        }
                    }
                }
                "message_delta" => {
                    let stop_reason = event
                        .get("delta")
                        .and_then(|d| d.get("stop_reason"))
                        .and_then(|s| s.as_str())
                        .unwrap_or("none");
                    if stop_reason != "none" {
                        pending_stop = Some(StopReason::from_provider_token(stop_reason));
                    }
                    // Anthropic's running-total: each `message_delta`
                    // supersedes the previous one, so we always overwrite.
                    let observed_output = event
                        .get("usage")
                        .and_then(|u| u.get("output_tokens"))
                        .and_then(|t| t.as_u64());
                    if let Some(v) = observed_output {
                        output_tokens = Some(v);
                    }
                    if StopReason::from_provider_token(stop_reason) == StopReason::OutputTruncated {
                        ::zeroclaw_log::record!(
                            WARN,
                            ::zeroclaw_log::Event::new(
                                module_path!(),
                                ::zeroclaw_log::Action::Note
                            )
                            .with_outcome(::zeroclaw_log::EventOutcome::Unknown)
                            .with_attrs(::serde_json::json!({"output_tokens": observed_output})),
                            "response truncated: hit max_tokens limit. Increase provider_max_tokens in config."
                        );
                    } else {
                        ::zeroclaw_log::record!(DEBUG, ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note).with_attrs(::serde_json::json!({"stop_reason": stop_reason, "output_tokens": observed_output})), "stream: message_delta");
                    }
                }
                "message_stop" => {
                    ::zeroclaw_log::record!(
                        DEBUG,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note),
                        "stream: message_stop"
                    );
                    if input_tokens.is_some()
                        || output_tokens.is_some()
                        || cached_input_tokens.is_some()
                        || cache_creation_input_tokens.is_some()
                    {
                        let uncached = input_tokens.unwrap_or(0);
                        let cache_read = cached_input_tokens.unwrap_or(0);
                        let cache_create = cache_creation_input_tokens.unwrap_or(0);
                        let normalized_input = Some(
                            uncached
                                .saturating_add(cache_read)
                                .saturating_add(cache_create),
                        );
                        let _ = tx
                            .send(Ok(StreamEvent::Usage(TokenUsage {
                                input_tokens: normalized_input,
                                output_tokens,
                                cached_input_tokens,
                            })))
                            .await;
                    }
                    let _ = tx
                        .send(Ok(StreamEvent::Final {
                            stop: pending_stop.take().unwrap_or(StopReason::Unspecified),
                        }))
                        .await;
                    return;
                }
                "error" => {
                    let msg = event
                        .get("error")
                        .and_then(|e| e.get("message"))
                        .and_then(|m| m.as_str())
                        .unwrap_or("unknown streaming error");
                    let _ = tx
                        .send(Err(StreamError::ModelProvider(msg.to_string())))
                        .await;
                    return;
                }
                _ => {}
            }
        }

        crate::stream_guard::finish_sse_stream(SseFinish {
            tx,
            stop: None,
            completion_signal: "message_stop",
        })
        .await;
    }
}

#[async_trait]
impl ModelProvider for AnthropicModelProvider {
    fn default_temperature(&self) -> f64 {
        TEMPERATURE_DEFAULT
    }

    fn default_base_url(&self) -> Option<&str> {
        Some(BASE_URL)
    }

    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        temperature: Option<f64>,
    ) -> anyhow::Result<String> {
        let credential = self.credential.as_ref().ok_or_else(|| {
            ::zeroclaw_log::record!(
                ERROR,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(::serde_json::json!({"missing": "credentials"})),
                "anthropic: no credentials configured"
            );
            anyhow::Error::msg(
                "Anthropic credentials not set. Set ANTHROPIC_API_KEY or ANTHROPIC_OAUTH_TOKEN (setup-token).",
            )
        })?;

        let system = system_prompt.map(|s| SystemPrompt::String(s.to_string()));
        let system = if Self::is_setup_token(credential) {
            Self::apply_oauth_system_prompt(system)
        } else {
            system
        };

        ::zeroclaw_log::record!(
            DEBUG,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                .with_attrs(::serde_json::json!({"max_tokens": self.max_tokens, "model": model})),
            "API request"
        );
        let request = NativeChatRequest {
            model: model.to_string(),
            max_tokens: self.max_tokens,
            system,
            messages: vec![NativeMessage {
                role: "user".to_string(),
                content: vec![NativeContentOut::Text {
                    text: message.to_string(),
                    cache_control: None,
                }],
            }],
            temperature,
            tools: None,
            tool_choice: None,
            stream: None,
            thinking: None,
            output_config: self
                .reasoning_effort
                .clone()
                .map(|effort| OutputConfig { effort }),
        };

        let mut request = self
            .http_client()
            .post(format!("{}/v1/messages", self.base_url))
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request);

        request = self.apply_auth(request, credential);
        request = crate::extra_headers::apply_extra_headers(request, &self.extra_headers);

        let response = request.send().await?;

        if !response.status().is_success() {
            return Err(super::api_error("Anthropic", response).await);
        }

        let chat_response: NativeChatResponse = response.json().await?;
        let parsed = Self::parse_native_response(chat_response);
        parsed.text.ok_or_else(|| {
            ::zeroclaw_log::record!(
                ERROR,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure),
                "anthropic: empty text in response"
            );
            anyhow::Error::msg("No response from Anthropic")
        })
    }

    async fn chat(
        &self,
        request: ProviderChatRequest<'_>,
        model: &str,
        temperature: Option<f64>,
    ) -> anyhow::Result<ProviderChatResponse> {
        let credential = self.credential.as_ref().ok_or_else(|| {
            ::zeroclaw_log::record!(
                ERROR,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(::serde_json::json!({"missing": "credentials"})),
                "anthropic: no credentials configured"
            );
            anyhow::Error::msg(
                "Anthropic credentials not set. Set ANTHROPIC_API_KEY or ANTHROPIC_OAUTH_TOKEN (setup-token).",
            )
        })?;

        let (system_prompt, mut messages) = Self::convert_messages(request.messages);

        // Cache the latest conversation tail when useful. MemBox stable
        // boundaries keep the request prefix cached; tool-loop tails can still
        // receive an incremental breakpoint when they are separate messages.
        Self::apply_conversation_cache_control(request.messages, &mut messages);

        // Check for tool_choice override from the agent loop (e.g. "any"
        // to force tool use for hardware requests).
        let tool_choice_override = zeroclaw_api::TOOL_CHOICE_OVERRIDE
            .try_with(Clone::clone)
            .ok()
            .flatten();
        let native_tools = Self::convert_tools(request.tools);
        let tools_count = native_tools.as_ref().map_or(0, Vec::len);
        let tool_choice = if native_tools.is_some() {
            tool_choice_override.map(|tc| serde_json::json!({ "type": tc }))
        } else {
            None
        };

        // For OAuth tokens, prepend Claude Code identity to system prompt
        let system_prompt = if Self::is_setup_token(credential) {
            Self::apply_oauth_system_prompt(system_prompt)
        } else {
            system_prompt
        };
        Self::enforce_max_explicit_cache_breakpoints(
            system_prompt.as_ref(),
            native_tools.as_deref(),
            &mut messages,
        );

        let (effective_temperature, thinking_config, effective_max_tokens) =
            self.resolve_thinking(request.thinking, temperature, model);

        if ::zeroclaw_log::debug_enabled() {
            ::zeroclaw_log::record!(
                DEBUG,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                    .with_attrs(::serde_json::json!({
                        "provider": "anthropic",
                        "alias": &self.alias,
                        "request_api": "messages",
                        "model": model,
                        "stream": false,
                        "max_tokens": effective_max_tokens,
                        "tools_count": tools_count,
                        "tool_choice": tool_choice.as_ref().and_then(|value| value.get("type")).and_then(|value| value.as_str()),
                        "thinking_enabled": thinking_config.is_some(),
                    })),
                "anthropic provider request prepared"
            );
        }
        let native_request = NativeChatRequest {
            model: model.to_string(),
            max_tokens: effective_max_tokens,
            system: system_prompt,
            messages,
            temperature: effective_temperature,
            tools: native_tools,
            tool_choice,
            stream: None,
            thinking: thinking_config,
            output_config: self
                .reasoning_effort
                .clone()
                .map(|effort| OutputConfig { effort }),
        };

        let req = self
            .http_client()
            .post(format!("{}/v1/messages", self.base_url))
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&native_request);

        let req = crate::extra_headers::apply_extra_headers(
            self.apply_auth(req, credential),
            &self.extra_headers,
        );
        let response = req.send().await?;
        if !response.status().is_success() {
            return Err(super::api_error("Anthropic", response).await);
        }

        let native_response: NativeChatResponse = response.json().await?;
        Ok(Self::parse_native_response(native_response))
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            native_tool_calling: true,
            vision: true,
            prompt_caching: true,
            extended_thinking: true,
        }
    }

    fn supports_native_tools(&self) -> bool {
        true
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
        model: &str,
        temperature: Option<f64>,
    ) -> anyhow::Result<ProviderChatResponse> {
        // Convert OpenAI-format tool JSON to ToolSpec so we can reuse the
        // existing `chat()` method which handles full message history,
        // system prompt extraction, caching, and Anthropic native formatting.
        let tool_specs: Vec<ToolSpec> = tools
            .iter()
            .filter_map(|t| {
                let func = t.get("function").or_else(|| {
                    ::zeroclaw_log::record!(
                        WARN,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                            .with_outcome(::zeroclaw_log::EventOutcome::Unknown),
                        "Skipping malformed tool definition (missing 'function' key)"
                    );
                    None
                })?;
                let name = func.get("name").and_then(|n| n.as_str()).or_else(|| {
                    ::zeroclaw_log::record!(
                        WARN,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                            .with_outcome(::zeroclaw_log::EventOutcome::Unknown),
                        "Skipping tool with missing or non-string 'name'"
                    );
                    None
                })?;
                Some(ToolSpec::new(
                    name.to_string(),
                    func.get("description")
                        .and_then(|d| d.as_str())
                        .unwrap_or("")
                        .to_string(),
                    func.get("parameters")
                        .cloned()
                        .unwrap_or(serde_json::json!({"type": "object"})),
                ))
            })
            .collect();

        let request = ProviderChatRequest {
            messages,
            tools: if tool_specs.is_empty() {
                None
            } else {
                Some(&tool_specs)
            },
            thinking: None,
        };
        self.chat(request, model, temperature).await
    }

    async fn warmup(&self) -> anyhow::Result<()> {
        if let Some(credential) = self.credential.as_ref() {
            let mut request = self
                .http_client()
                .post(format!("{}/v1/messages", self.base_url))
                .header("anthropic-version", "2023-06-01");
            request = self.apply_auth(request, credential);
            request = crate::extra_headers::apply_extra_headers(request, &self.extra_headers);
            // Send a minimal request; the goal is TLS + HTTP/2 setup, not a valid response.
            // Anthropic has no lightweight GET endpoint, so we accept any non-network error.
            let _ = request.send().await?;
        }
        Ok(())
    }

    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        // Anthropic's /v1/models requires a credential. Onboard pulls the
        // catalog from models.dev before the user has entered a key.
        crate::models_dev::list_models_for("anthropic").await
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_streaming_tool_events(&self) -> bool {
        true
    }

    fn stream_chat(
        &self,
        request: ProviderChatRequest<'_>,
        model: &str,
        temperature: Option<f64>,
        options: StreamOptions,
    ) -> stream::BoxStream<'static, StreamResult<StreamEvent>> {
        if !options.enabled {
            return stream::once(async { Ok(StreamEvent::unspecified_final()) }).boxed();
        }

        let credential = match self.credential.as_ref() {
            Some(c) => c.clone(),
            None => {
                return stream::once(async {
                    Err(StreamError::ModelProvider(
                        "Anthropic credentials not set".to_string(),
                    ))
                })
                .boxed();
            }
        };

        let (system_prompt, mut messages) = Self::convert_messages(request.messages);
        Self::apply_conversation_cache_control(request.messages, &mut messages);

        let tool_choice_override = zeroclaw_api::TOOL_CHOICE_OVERRIDE
            .try_with(Clone::clone)
            .ok()
            .flatten();
        let native_tools = Self::convert_tools(request.tools);
        let tools_count = native_tools.as_ref().map_or(0, Vec::len);
        let tool_choice = if native_tools.is_some() {
            tool_choice_override.map(|tc| serde_json::json!({ "type": tc }))
        } else {
            None
        };

        let system_prompt = if Self::is_setup_token(&credential) {
            Self::apply_oauth_system_prompt(system_prompt)
        } else {
            system_prompt
        };
        Self::enforce_max_explicit_cache_breakpoints(
            system_prompt.as_ref(),
            native_tools.as_deref(),
            &mut messages,
        );

        let (effective_temperature, thinking_config, effective_max_tokens) =
            self.resolve_thinking(request.thinking, temperature, model);

        if ::zeroclaw_log::debug_enabled() {
            ::zeroclaw_log::record!(
                DEBUG,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                    .with_attrs(::serde_json::json!({
                        "provider": "anthropic",
                        "alias": &self.alias,
                        "request_api": "messages",
                        "model": model,
                        "stream": true,
                        "max_tokens": effective_max_tokens,
                        "tools_count": tools_count,
                        "tool_choice": tool_choice.as_ref().and_then(|value| value.get("type")).and_then(|value| value.as_str()),
                        "thinking_enabled": thinking_config.is_some(),
                    })),
                "anthropic streaming provider request prepared"
            );
        }
        let native_request = NativeChatRequest {
            model: model.to_string(),
            max_tokens: effective_max_tokens,
            system: system_prompt,
            messages,
            temperature: effective_temperature,
            tools: native_tools,
            tool_choice,
            stream: Some(true),
            thinking: thinking_config,
            output_config: self
                .reasoning_effort
                .clone()
                .map(|effort| OutputConfig { effort }),
        };

        let body = match Self::build_streaming_request(&native_request) {
            Ok(body) => body,
            Err(e) => {
                return stream::once(async move { Err(StreamError::ModelProvider(e.to_string())) })
                    .boxed();
            }
        };
        let client = self.streaming_http_client();
        let url = format!("{}/v1/messages", self.base_url);
        let is_oauth = Self::is_setup_token(&credential);
        let extra_headers = self.extra_headers.clone();

        let (tx, rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(64);

        ::zeroclaw_log::record!(
            DEBUG,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Spawn)
                .with_category(::zeroclaw_log::EventCategory::Provider)
                .with_attrs(::serde_json::json!({
                    "idle_timeout_secs": SSE_IDLE_TIMEOUT.as_secs(),
                    "channel_capacity": 64,
                })),
            "stream: spawning detached Anthropic SSE parser task"
        );

        let parser_handle = ::zeroclaw_spawn::spawn!(async move {
            let mut req = client
                .post(&url)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&body);

            if is_oauth {
                req = req
                    .header("Authorization", format!("Bearer {credential}"))
                    .header(
                        "anthropic-beta",
                        "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14",
                    )
                    .header("anthropic-dangerous-direct-browser-access", "true");
            } else {
                req = req.header("x-api-key", &credential);
            }
            req = crate::extra_headers::apply_extra_headers(req, &extra_headers);

            let response = match req.send().await {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx
                        .send(Err(StreamError::Http(super::format_error_chain(&e))))
                        .await;
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let error = response
                    .text()
                    .await
                    .unwrap_or_else(|_| format!("HTTP error: {status}"));
                let _ = tx
                    .send(Err(StreamError::ModelProvider(format!(
                        "{status}: {error}"
                    ))))
                    .await;
                return;
            }

            Self::parse_anthropic_sse(response, &tx).await;
        });

        // The guard travels inside the unfold state so it is dropped at the
        // exact moment the consumer drops the stream — turning a turn cancel
        // (or normal completion) into an immediate parser-task abort instead
        // of a leaked socket that lingers until SSE_IDLE_TIMEOUT.
        let guard = AbortOnDrop::new(parser_handle.abort_handle());
        stream::unfold((rx, guard), |(mut rx, guard)| async move {
            rx.recv().await.map(|event| (event, (rx, guard)))
        })
        .boxed()
    }
}

impl ::zeroclaw_api::attribution::Attributable for AnthropicModelProvider {
    fn role(&self) -> ::zeroclaw_api::attribution::Role {
        ::zeroclaw_api::attribution::Role::Provider(
            ::zeroclaw_api::attribution::ProviderKind::Model(
                ::zeroclaw_api::attribution::ModelProviderKind::Anthropic,
            ),
        )
    }
    fn alias(&self) -> &str {
        &self.alias
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::anthropic_token::{AnthropicAuthKind, detect_auth_kind};

    fn replay_blocks(reasoning_content: &str) -> Vec<serde_json::Value> {
        AnthropicReplayEnvelope::decode(reasoning_content)
            .expect("valid replay envelope")
            .into_iter()
            .map(|block| serde_json::to_value(block).unwrap())
            .collect()
    }

    fn fake_anthropic_sse() -> &'static [u8] {
        b"event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-5\",\"usage\":{\"input_tokens\":314,\"cache_read_input_tokens\":42,\"cache_creation_input_tokens\":100}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":27}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n"
    }

    fn fake_anthropic_thinking_sse() -> &'static [u8] {
        b"event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Checking sources\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\" in detail\"}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n"
    }

    fn fake_signed_thinking_tool_sse() -> &'static [u8] {
        b"event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Checking sources\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_abc\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tool_1\",\"name\":\"search\",\"input\":{}}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\\\"rust\\\"}\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":1}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":12}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n"
    }

    fn fake_redacted_thinking_tool_sse() -> &'static [u8] {
        b"event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\"opaque_redacted_data\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tool_1\",\"name\":\"search\",\"input\":{}}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\\\"rust\\\"}\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":1}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":12}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n"
    }

    fn fake_interleaved_reasoning_tool_sse() -> &'static [u8] {
        b"event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"first\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_1\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\" I'll search. \"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":1}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu_1\",\"name\":\"search\",\"input\":{}}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\\\"one\\\"}\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":2}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":3,\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\"opaque_redacted_data\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":3}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":4,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":4,\"delta\":{\"type\":\"text_delta\",\"text\":\"\\u7b2c\\u4e8c\\u4e2a\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":4,\"delta\":{\"type\":\"text_delta\",\"text\":\"\\u6765\\u6e90\\u3002\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":4}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":5,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu_2\",\"name\":\"search\",\"input\":{}}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":5,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\\\"two\\\"}\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":5}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n"
    }

    #[tokio::test]
    async fn streaming_thinking_delta_is_emitted_as_reasoning() {
        use std::io::Cursor;

        let reader = tokio::io::BufReader::new(Cursor::new(fake_anthropic_thinking_sse()));
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(4);

        AnthropicModelProvider::parse_anthropic_sse_from_reader(reader, &tx).await;

        let first = rx
            .recv()
            .await
            .expect("first reasoning event")
            .expect("valid event");
        assert!(matches!(
            first,
            StreamEvent::TextDelta(StreamChunk {
                delta,
                reasoning: Some(reasoning),
                ..
            }) if delta.is_empty() && reasoning == "Checking sources"
        ));

        let second = rx
            .recv()
            .await
            .expect("second reasoning event")
            .expect("valid event");
        assert!(matches!(
            second,
            StreamEvent::TextDelta(StreamChunk {
                delta,
                reasoning: Some(reasoning),
                ..
            }) if delta.is_empty() && reasoning == " in detail"
        ));
    }

    #[tokio::test]
    async fn streaming_signed_thinking_is_replayable_without_exposing_signature() {
        use std::io::Cursor;

        let reader = tokio::io::BufReader::new(Cursor::new(fake_signed_thinking_tool_sse()));
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(16);

        AnthropicModelProvider::parse_anthropic_sse_from_reader(reader, &tx).await;

        let visible = rx
            .recv()
            .await
            .expect("thinking delta")
            .expect("valid delta");
        assert!(matches!(
            visible,
            StreamEvent::TextDelta(StreamChunk { reasoning: Some(reasoning), .. })
                if reasoning == "Checking sources"
        ));

        let mut replay = Vec::new();
        let mut tool = None;
        while let Ok(event) = rx.try_recv() {
            match event.expect("valid stream event") {
                StreamEvent::ReasoningContent(content) => replay.push(content),
                StreamEvent::ToolCall(call) => tool = Some(call),
                _ => {}
            }
        }
        let blocks = replay_blocks(&replay.join("\n"));
        assert_eq!(blocks[0]["thinking"], "Checking sources");
        assert_eq!(blocks[0]["signature"], "sig_abc");
        let tool = tool.expect("tool call");
        assert_eq!(tool.id, "tool_1");
        assert_eq!(tool.name, "search");
        assert_eq!(tool.arguments, r#"{"q":"rust"}"#);
        assert!(tool.extra_content.is_none());
    }

    #[tokio::test]
    async fn streaming_replay_snapshot_survives_disconnect_after_tool_block() {
        use std::io::Cursor;

        let stream = std::str::from_utf8(fake_signed_thinking_tool_sse()).unwrap();
        let partial = stream
            .split_once("event: message_delta")
            .expect("fixture contains message delta")
            .0;
        let reader = tokio::io::BufReader::new(Cursor::new(partial.as_bytes()));
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(16);

        AnthropicModelProvider::parse_anthropic_sse_from_reader(reader, &tx).await;

        let mut replay = Vec::new();
        let mut tool_calls = Vec::new();
        while let Ok(event) = rx.try_recv() {
            match event {
                Ok(StreamEvent::ReasoningContent(content)) => replay.push(content),
                Ok(StreamEvent::ToolCall(call)) => tool_calls.push(call),
                _ => {}
            }
        }
        let assistant = serde_json::json!({
            "content": null,
            "reasoning_content": replay.join("\n"),
            "tool_calls": tool_calls,
        })
        .to_string();
        let blocks = AnthropicModelProvider::parse_assistant_tool_call_message(&assistant)
            .expect("completed blocks remain replayable after disconnect");
        let replayed = serde_json::to_value(blocks).unwrap();

        assert_eq!(replayed[0]["type"], "thinking");
        assert_eq!(replayed[0]["signature"], "sig_abc");
        assert_eq!(replayed[1]["type"], "tool_use");
        assert_eq!(replayed[1]["id"], "tool_1");
    }

    #[tokio::test]
    async fn streaming_redacted_thinking_is_replayable_without_visible_reasoning() {
        use std::io::Cursor;

        let reader = tokio::io::BufReader::new(Cursor::new(fake_redacted_thinking_tool_sse()));
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(16);

        AnthropicModelProvider::parse_anthropic_sse_from_reader(reader, &tx).await;

        let mut replay = Vec::new();
        let mut tool = None;
        while let Ok(event) = rx.try_recv() {
            match event.expect("valid stream event") {
                StreamEvent::ReasoningContent(content) => replay.push(content),
                StreamEvent::ToolCall(call) => tool = Some(call),
                StreamEvent::TextDelta(chunk) => {
                    assert!(chunk.reasoning.is_none(), "redacted data must stay hidden")
                }
                _ => {}
            }
        }
        let blocks = replay_blocks(&replay.join("\n"));
        assert_eq!(blocks[0]["type"], "redacted_thinking");
        assert_eq!(blocks[0]["data"], "opaque_redacted_data");
        let tool = tool.expect("tool call");
        assert_eq!(tool.id, "tool_1");
        assert_eq!(tool.name, "search");
        assert_eq!(tool.arguments, r#"{"q":"rust"}"#);
        assert!(tool.extra_content.is_none());
    }

    #[tokio::test]
    async fn streaming_interleaved_reasoning_keeps_original_tool_order_on_replay() {
        use std::io::Cursor;

        let reader = tokio::io::BufReader::new(Cursor::new(fake_interleaved_reasoning_tool_sse()));
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(16);
        AnthropicModelProvider::parse_anthropic_sse_from_reader(reader, &tx).await;

        let mut reasoning = Vec::new();
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        while let Ok(event) = rx.try_recv() {
            match event.expect("valid stream event") {
                StreamEvent::ReasoningContent(content) => reasoning.push(content),
                StreamEvent::TextDelta(chunk) => text.push_str(&chunk.delta),
                StreamEvent::ToolCall(call) => tool_calls.push(call),
                _ => {}
            }
        }
        let assistant = serde_json::json!({
            "content": text.trim(),
            "reasoning_content": reasoning.join("\n"),
            "tool_calls": tool_calls,
        })
        .to_string();

        let blocks = AnthropicModelProvider::parse_assistant_tool_call_message(&assistant)
            .expect("assistant message should parse");
        let replayed = serde_json::to_value(blocks).unwrap();
        assert_eq!(replayed[0]["type"], "thinking");
        assert_eq!(replayed[1]["type"], "text");
        assert_eq!(replayed[1]["text"], " I'll search. ");
        assert_eq!(replayed[2]["id"], "tu_1");
        assert_eq!(replayed[3]["type"], "redacted_thinking");
        assert_eq!(replayed[4]["type"], "text");
        assert_eq!(replayed[4]["text"], "第二个来源。");
        assert_eq!(replayed[5]["id"], "tu_2");
    }

    #[tokio::test]
    async fn streaming_usage_emitted_before_final() {
        // The originallive repro was Anthropic streaming; before this
        // PR the message_start / message_delta usage frames were only logged
        // at DEBUG and never surfaced as `StreamEvent::Usage`. Now they are.
        use std::io::Cursor;

        let bytes = fake_anthropic_sse();
        let reader = tokio::io::BufReader::new(Cursor::new(bytes));
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(64);
        AnthropicModelProvider::parse_anthropic_sse_from_reader(reader, &tx).await;

        let mut events = Vec::new();
        while let Ok(Some(ev)) =
            tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv()).await
        {
            events.push(ev);
        }

        let states: Vec<&str> = events
            .iter()
            .map(|e| match e.as_ref() {
                Ok(StreamEvent::TextDelta(_)) => "text",
                Ok(StreamEvent::ReasoningContent(_)) => "reasoning_content",
                Ok(StreamEvent::ToolCall(_)) => "tool_call",
                Ok(StreamEvent::PreExecutedToolCall { .. }) => "pre_tool_call",
                Ok(StreamEvent::PreExecutedToolResult { .. }) => "pre_tool_result",
                Ok(StreamEvent::Usage(_)) => "usage",
                Ok(StreamEvent::Final { .. }) => "final",
                Err(_) => "err",
            })
            .collect();

        // Required ordering: usage event must appear before Final so the
        // gateway accumulator can capture it within the same turn boundary.
        let usage_pos = states
            .iter()
            .position(|s| *s == "usage")
            .unwrap_or_else(|| panic!("expected Usage event in stream, got {states:?}"));
        let final_pos = states
            .iter()
            .position(|s| *s == "final")
            .unwrap_or_else(|| panic!("expected Final event in stream, got {states:?}"));
        assert!(
            usage_pos < final_pos,
            "Usage must come before Final, got {states:?}"
        );

        // The Usage payload must carry both input + output token counts plus
        // the cached-input prompt-cache reads from message_start.
        let usage = events
            .iter()
            .find_map(|e| match e.as_ref() {
                Ok(StreamEvent::Usage(u)) => Some(u.clone()),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            usage.input_tokens,
            Some(456),
            "input_tokens must be the total of all three Anthropic buckets \
             (after-breakpoint 314 + cache_read 42 + cache_creation 100) \
             per the documented prompt-caching formula"
        );
        assert_eq!(
            usage.output_tokens,
            Some(27),
            "output_tokens from message_delta usage frame"
        );
        assert_eq!(
            usage.cached_input_tokens,
            Some(42),
            "cache_read_input_tokens from message_start"
        );
    }

    #[tokio::test]
    async fn max_tokens_stop_reason_emits_output_truncated() {
        use std::io::Cursor;

        let bytes = b"event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude\",\"usage\":{\"input_tokens\":10}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"},\"usage\":{\"output_tokens\":8}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";
        let reader = tokio::io::BufReader::new(Cursor::new(bytes.as_slice()));
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(16);
        AnthropicModelProvider::parse_anthropic_sse_from_reader(reader, &tx).await;

        let mut last_final = None;
        while let Ok(Some(ev)) =
            tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv()).await
        {
            if let Ok(StreamEvent::Final { stop }) = ev {
                last_final = Some(stop);
            }
        }
        assert_eq!(last_final, Some(StopReason::OutputTruncated));
    }

    #[tokio::test]
    async fn model_context_window_exceeded_emits_output_truncated() {
        use std::io::Cursor;

        let bytes = b"event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude\",\"usage\":{\"input_tokens\":10}}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"model_context_window_exceeded\"},\"usage\":{\"output_tokens\":8}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";
        let reader = tokio::io::BufReader::new(Cursor::new(bytes.as_slice()));
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(16);
        AnthropicModelProvider::parse_anthropic_sse_from_reader(reader, &tx).await;

        let mut last_final = None;
        while let Ok(Some(ev)) =
            tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv()).await
        {
            if let Ok(StreamEvent::Final { stop }) = ev {
                last_final = Some(stop);
            }
        }
        assert_eq!(last_final, Some(StopReason::OutputTruncated));
    }

    /// A reader that yields one buffer of bytes, then parks forever — models
    /// an SSE connection that delivers `message_start` and then goes silent
    /// with the socket still open. Without the idle timeout this hangs the
    /// parser indefinitely.
    struct StallAfterReader {
        data: std::io::Cursor<Vec<u8>>,
        drained: bool,
    }

    impl tokio::io::AsyncRead for StallAfterReader {
        fn poll_read(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
            buf: &mut tokio::io::ReadBuf<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            if self.drained {
                // Park without self-waking; the surrounding timeout's timer
                // provides the wake. Self-waking here would busy-spin under
                // paused time and starve the timer.
                return std::task::Poll::Pending;
            }
            let before = buf.filled().len();
            let inner = std::pin::Pin::new(&mut self.data);
            let res = inner.poll_read(cx, buf);
            // Once the seed buffer is exhausted, stall on the *next* read
            // rather than reporting EOF (0 bytes) — EOF would end the stream
            // cleanly and never exercise the idle timeout.
            if buf.filled().len() == before {
                self.drained = true;
                return std::task::Poll::Pending;
            }
            res
        }
    }

    #[tokio::test(start_paused = true)]
    async fn stalled_stream_times_out_instead_of_hanging() {
        // Repro: connection delivers message_start then goes silent. The
        // parser must surface a retryable StreamError rather than parking on
        // next_line() forever (the "stuck on working" hang).
        let start = b"event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude\",\"usage\":{\"input_tokens\":1}}}\n\n"
            .to_vec();
        let reader = tokio::io::BufReader::new(StallAfterReader {
            data: std::io::Cursor::new(start),
            drained: false,
        });
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(64);

        let parser = ::zeroclaw_spawn::spawn!(async move {
            AnthropicModelProvider::parse_anthropic_sse_from_reader(reader, &tx).await;
        });

        // Let the parser run until it parks on the stalled read before we
        // jump virtual time forward.
        tokio::task::yield_now().await;
        // Advance virtual time past the idle bound; the parser should wake,
        // emit an error, and return — closing the channel.
        tokio::time::advance(SSE_IDLE_TIMEOUT + std::time::Duration::from_secs(1)).await;

        let mut last_err = None;
        while let Some(ev) = rx.recv().await {
            if let Err(e) = ev {
                last_err = Some(e);
            }
        }
        parser.await.expect("parser task must finish, not hang");

        let err = last_err.expect("a StreamError must be emitted on stall");
        assert!(
            matches!(err, StreamError::Http(ref m) if m.contains("stalled")),
            "expected stalled-stream Http error, got: {err:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn dropping_guard_aborts_parser_without_idle_wait() {
        // The full-measure fix: dropping the consumer stream must abort the
        // detached parser immediately (turn cancel), not leak the socket until
        // SSE_IDLE_TIMEOUT. We model the stream's lifetime with AbortOnDrop and
        // assert the task is aborted the instant the guard drops.
        let start = b"event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude\",\"usage\":{\"input_tokens\":1}}}\n\n"
            .to_vec();
        let reader = tokio::io::BufReader::new(StallAfterReader {
            data: std::io::Cursor::new(start),
            drained: false,
        });
        let (tx, _rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(64);

        let handle = ::zeroclaw_spawn::spawn!(async move {
            AnthropicModelProvider::parse_anthropic_sse_from_reader(reader, &tx).await;
        });
        let probe = handle.abort_handle();
        let guard = AbortOnDrop::new(handle.abort_handle());

        // Let the parser park on the stalled read.
        tokio::task::yield_now().await;
        assert!(
            !probe.is_finished(),
            "parser must still be running (parked on the stalled read) before drop"
        );

        // Dropping the guard must abort the parser — no SSE_IDLE_TIMEOUT wait.
        drop(guard);
        tokio::task::yield_now().await;
        assert!(
            probe.is_finished(),
            "guard drop must abort the parser task immediately, not wait out the idle timeout"
        );
    }

    /// A reader that yields one buffer of valid SSE bytes, then reports a
    /// transport read error — models a connection killed mid-body (e.g.
    /// reqwest's total-request timeout firing while bytes are still being
    /// drained).
    struct ReadErrorAfterReader {
        data: std::io::Cursor<Vec<u8>>,
        drained: bool,
    }

    impl tokio::io::AsyncRead for ReadErrorAfterReader {
        fn poll_read(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
            buf: &mut tokio::io::ReadBuf<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            if self.drained {
                return std::task::Poll::Ready(Err(std::io::Error::other(
                    "connection reset by peer",
                )));
            }
            let before = buf.filled().len();
            let inner = std::pin::Pin::new(&mut self.data);
            let res = inner.poll_read(cx, buf);
            if buf.filled().len() == before {
                self.drained = true;
                return std::task::Poll::Ready(Err(std::io::Error::other(
                    "connection reset by peer",
                )));
            }
            res
        }
    }

    #[tokio::test]
    async fn read_error_after_partial_text_propagates_instead_of_final() {
        // Repro for #2404: Claude emits a short intro sentence, then the
        // connection is killed mid-body (reqwest's total-request timeout, a
        // proxy hiccup, etc). The parser must surface a StreamError so the
        // caller retries — not silently emit `Final` and let the partial
        // reply be recorded as a successful turn.
        let partial = b"event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"I'll write that now.\"}}\n\n"
            .to_vec();
        let reader = tokio::io::BufReader::new(ReadErrorAfterReader {
            data: std::io::Cursor::new(partial),
            drained: false,
        });
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(64);

        AnthropicModelProvider::parse_anthropic_sse_from_reader(reader, &tx).await;
        drop(tx);

        let mut events = Vec::new();
        while let Some(ev) = rx.recv().await {
            events.push(ev);
        }

        assert!(
            matches!(events.first(), Some(Ok(StreamEvent::TextDelta(_)))),
            "expected the partial text delta first, got {events:?}"
        );
        assert!(
            matches!(events.last(), Some(Err(StreamError::Http(_)))),
            "read error must propagate as StreamError::Http, got {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Final { .. }))),
            "must never emit Final when the stream was cut short by a read error, got {events:?}"
        );
    }

    #[tokio::test]
    async fn eof_before_message_stop_surfaces_error_not_final() {
        use std::io::Cursor;

        let bytes = b"event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude\",\"usage\":{\"input_tokens\":10}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n";
        let reader = tokio::io::BufReader::new(Cursor::new(bytes.as_slice()));
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(64);
        AnthropicModelProvider::parse_anthropic_sse_from_reader(reader, &tx).await;

        let mut saw_final = false;
        let mut last_err = None;
        while let Ok(Some(ev)) =
            tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv()).await
        {
            match ev {
                Ok(StreamEvent::Final { .. }) => saw_final = true,
                Err(e) => last_err = Some(e),
                Ok(_) => {}
            }
        }
        assert!(!saw_final, "truncated stream must not emit Final");
        let err = last_err.expect("truncated stream must emit a StreamError");
        assert!(
            matches!(err, StreamError::Http(ref m) if m.contains("truncated")),
            "expected truncation error, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn eof_after_message_delta_without_message_stop_is_truncation() {
        use std::io::Cursor;

        let bytes = b"event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude\",\"usage\":{\"input_tokens\":10}}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":4}}\n\n";
        let reader = tokio::io::BufReader::new(Cursor::new(bytes.as_slice()));
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(64);
        AnthropicModelProvider::parse_anthropic_sse_from_reader(reader, &tx).await;

        let mut saw_final = false;
        let mut last_err = None;
        while let Ok(Some(ev)) =
            tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv()).await
        {
            match ev {
                Ok(StreamEvent::Final { .. }) => saw_final = true,
                Err(e) => last_err = Some(e),
                Ok(_) => {}
            }
        }
        assert!(
            !saw_final,
            "message_delta is not a completion signal; must not emit Final"
        );
        let err = last_err.expect("truncated stream must emit a StreamError");
        assert!(
            matches!(err, StreamError::Http(ref m) if m.contains("truncated")),
            "expected truncation error, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn streaming_usage_omitted_when_provider_does_not_send_usage() {
        // Backward-compat: a stream that never emits a usage frame must not
        // synthesize a zero-valued Usage event. Consumers should treat
        // absence as "usage unavailable" rather than "usage was zero."
        use std::io::Cursor;

        let bytes = b"event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude\"}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";
        let reader = tokio::io::BufReader::new(Cursor::new(bytes.as_slice()));
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(64);
        AnthropicModelProvider::parse_anthropic_sse_from_reader(reader, &tx).await;

        let mut saw_usage = false;
        while let Ok(Some(ev)) =
            tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv()).await
        {
            if matches!(ev, Ok(StreamEvent::Usage(_))) {
                saw_usage = true;
            }
        }
        assert!(
            !saw_usage,
            "must not emit Usage when provider sent no usage frames"
        );
    }

    #[test]
    fn creates_with_key() {
        let p = AnthropicModelProvider::builder("test")
            .credential(Some("anthropic-test-credential"))
            .build();
        assert!(p.credential.is_some());
        assert_eq!(p.credential.as_deref(), Some("anthropic-test-credential"));
        assert_eq!(p.base_url, "https://api.anthropic.com");
    }

    #[test]
    fn creates_without_key() {
        let p = AnthropicModelProvider::builder("test").build();
        assert!(p.credential.is_none());
        assert_eq!(p.base_url, "https://api.anthropic.com");
    }

    #[test]
    fn creates_with_empty_key() {
        let p = AnthropicModelProvider::builder("test")
            .credential(Some(""))
            .build();
        assert!(p.credential.is_none());
    }

    #[test]
    fn creates_with_whitespace_key() {
        let p = AnthropicModelProvider::builder("test")
            .credential(Some("  anthropic-test-credential  "))
            .build();
        assert!(p.credential.is_some());
        assert_eq!(p.credential.as_deref(), Some("anthropic-test-credential"));
    }

    #[test]
    fn creates_with_custom_base_url() {
        let p = AnthropicModelProvider::builder("test")
            .credential(Some("anthropic-credential"))
            .base_url("https://api.example.com")
            .build();
        assert_eq!(p.base_url, "https://api.example.com");
        assert_eq!(p.credential.as_deref(), Some("anthropic-credential"));
    }

    #[test]
    fn custom_base_url_trims_trailing_slash() {
        let p = AnthropicModelProvider::builder("test")
            .base_url("https://api.example.com/")
            .build();
        assert_eq!(p.base_url, "https://api.example.com");
    }

    #[test]
    fn no_base_url_uses_published_endpoint() {
        let p = AnthropicModelProvider::builder("test").build();
        assert_eq!(p.base_url, "https://api.anthropic.com");
    }

    #[tokio::test]
    async fn chat_fails_without_key() {
        let p = AnthropicModelProvider::builder("test").build();
        let result = p
            .chat_with_system(None, "hello", "claude-3-opus", Some(0.7))
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("credentials not set"),
            "Expected key error, got: {err}"
        );
    }

    #[test]
    fn setup_token_detection_works() {
        assert!(AnthropicModelProvider::is_setup_token(
            "sk-ant-oat01-abcdef"
        ));
        assert!(!AnthropicModelProvider::is_setup_token("sk-ant-api-key"));
    }

    #[test]
    fn apply_auth_uses_bearer_and_beta_for_setup_tokens() {
        let model_provider = AnthropicModelProvider::builder("test").build();
        let request = model_provider
            .apply_auth(
                model_provider
                    .http_client()
                    .get("https://api.anthropic.com/v1/models"),
                "sk-ant-oat01-test-token",
            )
            .build()
            .expect("request should build");

        assert_eq!(
            request
                .headers()
                .get("authorization")
                .and_then(|v| v.to_str().ok()),
            Some("Bearer sk-ant-oat01-test-token")
        );
        assert_eq!(
            request
                .headers()
                .get("anthropic-beta")
                .and_then(|v| v.to_str().ok()),
            Some("claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14")
        );
        assert_eq!(
            request
                .headers()
                .get("anthropic-dangerous-direct-browser-access")
                .and_then(|v| v.to_str().ok()),
            Some("true")
        );
        assert!(request.headers().get("x-api-key").is_none());
    }

    #[test]
    fn apply_auth_uses_x_api_key_for_regular_tokens() {
        let model_provider = AnthropicModelProvider::builder("test").build();
        let request = model_provider
            .apply_auth(
                model_provider
                    .http_client()
                    .get("https://api.anthropic.com/v1/models"),
                "sk-ant-api-key",
            )
            .build()
            .expect("request should build");

        assert_eq!(
            request
                .headers()
                .get("x-api-key")
                .and_then(|v| v.to_str().ok()),
            Some("sk-ant-api-key")
        );
        assert!(request.headers().get("authorization").is_none());
        assert!(request.headers().get("anthropic-beta").is_none());
    }

    #[test]
    fn builder_extra_headers_ride_every_request_but_never_replace_the_credential() {
        let mut headers = std::collections::HashMap::new();
        headers.insert("x-membox-turn-id".to_string(), "membox-round-1".to_string());
        headers.insert("x-membox-call-kind".to_string(), "main".to_string());
        headers.insert("x-api-key".to_string(), "stolen".to_string());
        let model_provider = AnthropicModelProvider::builder("test")
            .extra_headers(headers)
            .build();
        let request = crate::extra_headers::apply_extra_headers(
            model_provider.apply_auth(
                model_provider
                    .http_client()
                    .post("https://api.anthropic.com/v1/messages"),
                "sk-ant-api-key",
            ),
            &model_provider.extra_headers,
        )
        .build()
        .expect("request should build");

        let header = |name: &str| {
            request
                .headers()
                .get_all(name)
                .iter()
                .map(|v| v.to_str().unwrap().to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(header("x-membox-turn-id"), vec!["membox-round-1"]);
        assert_eq!(header("x-membox-call-kind"), vec!["main"]);
        assert_eq!(
            header("x-api-key"),
            vec!["sk-ant-api-key"],
            "a reserved name in extra_headers is dropped, not appended"
        );
    }

    #[test]
    fn builder_without_extra_headers_stamps_nothing() {
        let model_provider = AnthropicModelProvider::builder("test").build();
        assert!(model_provider.extra_headers.is_empty());
    }

    #[tokio::test]
    async fn chat_with_system_fails_without_key() {
        let p = AnthropicModelProvider::builder("test").build();
        let result = p
            .chat_with_system(
                Some("You are ZeroClaw"),
                "hello",
                "claude-3-opus",
                Some(0.7),
            )
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn chat_request_serializes_without_system() {
        let req = ChatRequest {
            model: "claude-3-opus".to_string(),
            max_tokens: 4096,
            system: None,
            messages: vec![Message {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            temperature: Some(0.7),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(
            !json.contains("system"),
            "system field should be skipped when None"
        );
        assert!(json.contains("claude-3-opus"));
        assert!(json.contains("hello"));
    }

    #[test]
    fn chat_request_serializes_with_system() {
        let req = ChatRequest {
            model: "claude-3-opus".to_string(),
            max_tokens: 4096,
            system: Some("You are ZeroClaw".to_string()),
            messages: vec![Message {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            temperature: Some(0.7),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"system\":\"You are ZeroClaw\""));
    }

    #[test]
    fn chat_response_deserializes() {
        let json = r#"{"content":[{"type":"text","text":"Hello there!"}]}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.content.len(), 1);
        assert_eq!(resp.content[0].kind, "text");
        assert_eq!(resp.content[0].text.as_deref(), Some("Hello there!"));
    }

    #[test]
    fn chat_response_empty_content() {
        let json = r#"{"content":[]}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert!(resp.content.is_empty());
    }

    #[test]
    fn chat_response_multiple_blocks() {
        let json =
            r#"{"content":[{"type":"text","text":"First"},{"type":"text","text":"Second"}]}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.content.len(), 2);
        assert_eq!(resp.content[0].text.as_deref(), Some("First"));
        assert_eq!(resp.content[1].text.as_deref(), Some("Second"));
    }

    #[test]
    fn temperature_range_serializes() {
        for temp in [0.0, 0.5, 1.0, 2.0] {
            let req = ChatRequest {
                model: "claude-3-opus".to_string(),
                max_tokens: 4096,
                system: None,
                messages: vec![],
                temperature: Some(temp),
            };
            let json = serde_json::to_string(&req).unwrap();
            assert!(json.contains(&format!("{temp}")));
        }
    }

    #[test]
    fn anthropic_thinking_mode_adaptive_from_claude_4_6() {
        for model in [
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-7-20260101",
        ] {
            assert!(
                anthropic_model_uses_adaptive_thinking(model),
                "{model} should use adaptive thinking"
            );
        }
    }

    #[test]
    fn anthropic_thinking_mode_enabled_for_legacy_models() {
        // Legacy families keep the fixed-budget `{type:"enabled"}` shape.
        // Claude 4.1 is listed by Anthropic among the earlier Claude 4 models
        // and is manual-only, so both its dated and undated IDs are legacy.
        for id in [
            "claude-haiku-4-5",
            "claude-opus-4-1",
            "claude-opus-4-1-20250805",
            "claude-3-opus",
        ] {
            assert!(
                !anthropic_model_uses_adaptive_thinking(id),
                "{id} is a legacy (manual-thinking) model and must not be adaptive"
            );
        }
    }

    #[test]
    fn anthropic_thinking_mode_enabled_for_bare_claude_4_aliases() {
        // The original Claude 4 generation (no minor version) predates adaptive
        // thinking and must use the fixed-budget `{type:"enabled"}` shape. The
        // canonical dated ID is the one used in the deployment example.
        for id in [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-sonnet-4",
            "claude-opus-4",
        ] {
            assert!(
                !anthropic_model_uses_adaptive_thinking(id),
                "{id} is a bare Claude 4 model and must use enabled thinking"
            );
        }
        // Policy boundary: Claude 4.6 supports adaptive thinking and is where
        // ZeroClaw stops sending the deprecated fixed-budget shape.
        for id in ["claude-opus-4-6", "claude-opus-4-7-20260101"] {
            assert!(
                anthropic_model_uses_adaptive_thinking(id),
                "{id} is Claude 4.6+ and should use adaptive thinking"
            );
        }
    }

    #[test]
    fn resolve_thinking_emits_adaptive_from_claude_4_6() {
        let provider = AnthropicModelProvider::builder("test")
            .credential(Some("test-key"))
            .build();
        let params = zeroclaw_api::model_provider::NativeThinkingParams {
            budget_tokens: 10_000,
        };
        for (model, expected_max_tokens) in [
            ("claude-sonnet-4-6", 10_000),
            ("claude-opus-4-6", 10_000),
            ("claude-opus-4-7", provider.max_tokens),
        ] {
            let (temp, config, max_tokens) =
                provider.resolve_thinking(Some(params), Some(0.7_f64), model);
            let config = config.expect("adaptive model should emit a thinking config");
            let json = serde_json::to_string(&config).unwrap();
            assert!(json.contains(r#""type":"adaptive""#), "{model}: {json}");
            assert!(
                json.contains(r#""display":"summarized""#),
                "{model}: adaptive thinking should request summaries: {json}"
            );
            assert!(
                !json.contains("budget_tokens"),
                "{model}: adaptive must not carry budget_tokens: {json}"
            );
            assert!((temp.unwrap() - 1.0_f64).abs() < f64::EPSILON);
            assert_eq!(max_tokens, expected_max_tokens);
        }
    }

    #[test]
    fn resolve_thinking_emits_enabled_for_legacy_models() {
        let provider = AnthropicModelProvider::builder("test")
            .credential(Some("test-key"))
            .build();
        let params = zeroclaw_api::model_provider::NativeThinkingParams {
            budget_tokens: 10_000,
        };
        // Every manual-thinking model emits `{type:"enabled", budget_tokens}`,
        // including Claude 4.1 (regression: it must not flip to adaptive).
        for model in ["claude-sonnet-4-5", "claude-opus-4-1-20250805"] {
            let (temp, config, _) = provider.resolve_thinking(Some(params), Some(0.7_f64), model);
            let config = config.expect("legacy model should emit a thinking config");
            let json = serde_json::to_string(&config).unwrap();
            assert!(json.contains(r#""type":"enabled""#), "{model}: got {json}");
            assert!(
                json.contains(r#""budget_tokens":10000"#),
                "{model}: got {json}"
            );
            // Forced to 1.0 per Anthropic native-thinking contract.
            assert!((temp.unwrap() - 1.0_f64).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn output_config_serializes_with_effort_when_supported() {
        let req = NativeChatRequest {
            model: "claude-opus-5".to_string(),
            max_tokens: 4096,
            system: None,
            messages: vec![],
            temperature: None,
            tools: None,
            tool_choice: None,
            stream: None,
            thinking: None,
            output_config: Some(OutputConfig {
                effort: "high".to_string(),
            }),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(
            json.contains(r#""output_config":{"effort":"high"}"#),
            "expected output_config.effort, got: {json}"
        );
    }

    #[test]
    fn output_config_omitted_when_none() {
        let req = NativeChatRequest {
            model: "claude-opus-5".to_string(),
            max_tokens: 4096,
            system: None,
            messages: vec![],
            temperature: None,
            tools: None,
            tool_choice: None,
            stream: None,
            thinking: None,
            output_config: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(
            !json.contains("output_config"),
            "output_config should be omitted when None, got: {json}"
        );
    }

    #[test]
    fn resolve_thinking_emits_adaptive_for_fable5_no_budget() {
        let provider = AnthropicModelProvider::builder("test")
            .credential(Some("test-key"))
            .build();
        let params = zeroclaw_api::model_provider::NativeThinkingParams {
            budget_tokens: 10_000,
        };
        let (temp, config, max_tokens) =
            provider.resolve_thinking(Some(params), Some(0.7_f64), "claude-fable-5");
        let config = config.expect("fable-5 should emit an adaptive thinking config");
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains(r#""type":"adaptive""#), "got: {json}");
        assert!(
            json.contains(r#""display":"summarized""#),
            "adaptive thinking must opt into visible summaries, got: {json}"
        );
        assert!(
            !json.contains("budget_tokens"),
            "adaptive must not carry budget_tokens, got: {json}"
        );
        assert!((temp.unwrap() - 1.0_f64).abs() < f64::EPSILON);
        assert_eq!(max_tokens, provider.max_tokens);
    }

    #[test]
    fn adaptive_regression_for_modern_models() {
        // Previously the denylist only excluded opus-4-7, so these modern
        // adaptive-only models wrongly received `{type:"enabled"}` and 400'd.
        for id in ["claude-fable-5", "claude-opus-5", "claude-opus-4-8"] {
            assert!(
                anthropic_model_uses_adaptive_thinking(id),
                "{id} should require adaptive thinking"
            );
        }
    }

    #[tokio::test]
    async fn output_config_reaches_wire_via_chat() {
        use axum::{Json, Router, routing::post};
        use parking_lot::Mutex;
        use std::sync::Arc;
        use tokio::net::TcpListener;

        let captured: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));
        let captured_clone = captured.clone();

        let app = Router::new().route(
            "/v1/messages",
            post(move |Json(body): Json<serde_json::Value>| {
                let cap = captured_clone.clone();
                async move {
                    *cap.lock() = Some(body);
                    Json(serde_json::json!({
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "ok"}],
                        "model": "claude-opus-5",
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 10, "output_tokens": 1}
                    }))
                }
            }),
        );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server_handle = zeroclaw_spawn::spawn!(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let model_provider = AnthropicModelProvider::builder("test")
            .credential(Some("test-key"))
            .base_url(&format!("http://{addr}"))
            .reasoning_effort(Some("high".to_string()))
            .build();

        let result = model_provider
            .chat_with_system(None, "hi", "claude-opus-5", None)
            .await;
        assert!(
            result.is_ok(),
            "chat_with_system failed: {:?}",
            result.err()
        );

        let body = captured.lock().take().expect("No request captured");
        assert_eq!(
            body["output_config"]["effort"], "high",
            "output_config.effort should reach the wire, got: {}",
            body["output_config"]
        );

        server_handle.abort();
    }

    #[tokio::test]
    async fn claude_4_6_adaptive_max_budget_reaches_wire() {
        use axum::{Json, Router, routing::post};
        use parking_lot::Mutex;
        use std::sync::Arc;
        use tokio::net::TcpListener;

        let captured: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));
        let captured_clone = captured.clone();
        let app = Router::new().route(
            "/v1/messages",
            post(move |Json(body): Json<serde_json::Value>| {
                let captured = captured_clone.clone();
                async move {
                    *captured.lock() = Some(body);
                    Json(serde_json::json!({
                        "content": [{"type": "text", "text": "ok"}],
                        "usage": {"input_tokens": 1, "output_tokens": 1}
                    }))
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server_handle = zeroclaw_spawn::spawn!(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let provider = AnthropicModelProvider::builder("test")
            .credential(Some("test-key"))
            .base_url(&format!("http://{address}"))
            .build();
        let messages = [ChatMessage::user("hi")];
        let request = ProviderChatRequest {
            messages: &messages,
            tools: None,
            thinking: Some(zeroclaw_api::model_provider::NativeThinkingParams {
                budget_tokens: zeroclaw_api::model_provider::MAX_BUDGET_TOKENS,
            }),
        };

        provider
            .chat(request, "claude-sonnet-4-6", Some(0.7))
            .await
            .expect("Claude 4.6 chat should succeed");

        let body = captured.lock().take().expect("request should be captured");
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["thinking"]["display"], "summarized");
        assert!(body["thinking"].get("budget_tokens").is_none());
        assert_eq!(body["temperature"], 1.0);
        assert_eq!(
            body["max_tokens"],
            zeroclaw_api::model_provider::MAX_BUDGET_TOKENS
        );
        server_handle.abort();
    }

    #[test]
    fn native_chat_request_serializes_without_temperature_when_none() {
        let req = NativeChatRequest {
            model: "claude-opus-4-7".to_string(),
            max_tokens: 4096,
            system: None,
            messages: vec![],
            temperature: None,
            tools: None,
            tool_choice: None,
            stream: None,
            thinking: None,
            output_config: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("max_tokens"));
        assert!(
            !json.contains("temperature"),
            "expected temperature to be omitted, got: {json}"
        );
    }

    #[test]
    fn native_chat_request_serializes_with_temperature_when_some() {
        let req = NativeChatRequest {
            model: "claude-sonnet-4-6".to_string(),
            max_tokens: 4096,
            system: None,
            messages: vec![],
            temperature: Some(0.7),
            tools: None,
            tool_choice: None,
            stream: None,
            thinking: None,
            output_config: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(
            json.contains("\"temperature\":0.7"),
            "expected temperature to be present, got: {json}"
        );
    }

    #[test]
    fn detects_auth_from_jwt_shape() {
        let kind = detect_auth_kind("a.b.c", None);
        assert_eq!(kind, AnthropicAuthKind::Authorization);
    }

    #[test]
    fn cache_control_serializes_correctly() {
        let cache = CacheControl::ephemeral();
        let json = serde_json::to_string(&cache).unwrap();
        assert_eq!(json, r#"{"type":"ephemeral"}"#);
    }

    #[test]
    fn system_prompt_string_variant_serializes() {
        let prompt = SystemPrompt::String("You are a helpful assistant".to_string());
        let json = serde_json::to_string(&prompt).unwrap();
        assert_eq!(json, r#""You are a helpful assistant""#);
    }

    #[test]
    fn system_prompt_blocks_variant_serializes() {
        let prompt = SystemPrompt::Blocks(vec![SystemBlock {
            block_type: "text".to_string(),
            text: "You are a helpful assistant".to_string(),
            cache_control: Some(CacheControl::ephemeral()),
        }]);
        let json = serde_json::to_string(&prompt).unwrap();
        assert!(json.contains(r#""type":"text""#));
        assert!(json.contains("You are a helpful assistant"));
        assert!(json.contains(r#""type":"ephemeral""#));
    }

    #[test]
    fn system_prompt_blocks_without_cache_control() {
        let prompt = SystemPrompt::Blocks(vec![SystemBlock {
            block_type: "text".to_string(),
            text: "Short prompt".to_string(),
            cache_control: None,
        }]);
        let json = serde_json::to_string(&prompt).unwrap();
        assert!(json.contains("Short prompt"));
        assert!(!json.contains("cache_control"));
    }

    #[test]
    fn native_content_text_without_cache_control() {
        let content = NativeContentOut::Text {
            text: "Hello".to_string(),
            cache_control: None,
        };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains(r#""type":"text""#));
        assert!(json.contains("Hello"));
        assert!(!json.contains("cache_control"));
    }

    #[test]
    fn native_content_text_with_cache_control() {
        let content = NativeContentOut::Text {
            text: "Hello".to_string(),
            cache_control: Some(CacheControl::ephemeral()),
        };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains(r#""type":"text""#));
        assert!(json.contains("Hello"));
        assert!(json.contains(r#""cache_control":{"type":"ephemeral"}"#));
    }

    #[test]
    fn native_content_tool_use_without_cache_control() {
        let content = NativeContentOut::ToolUse {
            id: "tool_123".to_string(),
            name: "get_weather".to_string(),
            input: serde_json::json!({"location": "San Francisco"}),
            cache_control: None,
        };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains(r#""type":"tool_use""#));
        assert!(json.contains("tool_123"));
        assert!(json.contains("get_weather"));
        assert!(!json.contains("cache_control"));
    }

    #[test]
    fn native_content_tool_result_with_cache_control() {
        let content = NativeContentOut::ToolResult {
            tool_use_id: "tool_123".to_string(),
            content: "Result data".to_string(),
            cache_control: Some(CacheControl::ephemeral()),
        };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains(r#""type":"tool_result""#));
        assert!(json.contains("tool_123"));
        assert!(json.contains("Result data"));
        assert!(json.contains(r#""cache_control":{"type":"ephemeral"}"#));
    }

    #[test]
    fn native_tool_spec_without_cache_control() {
        let schema = serde_json::json!({"type": "object"});
        let tool = NativeToolSpec {
            name: "get_weather".to_string(),
            description: "Get weather info".to_string(),
            input_schema: schema.into(),
            cache_control: None,
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("get_weather"));
        assert!(!json.contains("cache_control"));
    }

    #[test]
    fn native_tool_spec_with_cache_control() {
        let schema = serde_json::json!({"type": "object"});
        let tool = NativeToolSpec {
            name: "get_weather".to_string(),
            description: "Get weather info".to_string(),
            input_schema: schema.into(),
            cache_control: Some(CacheControl::ephemeral()),
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("get_weather"));
        assert!(json.contains(r#""cache_control":{"type":"ephemeral"}"#));
    }

    #[test]
    fn should_cache_conversation_short() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "System prompt".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
        ];
        // Only 1 non-system message — should not cache
        assert!(!AnthropicModelProvider::should_cache_conversation(
            &messages
        ));
    }

    #[test]
    fn should_cache_conversation_long() {
        let mut messages = vec![ChatMessage {
            role: "system".to_string(),
            content: "System prompt".to_string(),
        }];
        // Add 3 non-system messages
        for i in 0..3 {
            messages.push(ChatMessage {
                role: if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
                content: format!("Message {i}"),
            });
        }
        assert!(AnthropicModelProvider::should_cache_conversation(&messages));
    }

    #[test]
    fn should_cache_conversation_boundary() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];
        // Exactly 1 non-system message — should not cache
        assert!(!AnthropicModelProvider::should_cache_conversation(
            &messages
        ));

        // Add one more to cross boundary (>1)
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "Hi".to_string(),
            },
        ];
        assert!(AnthropicModelProvider::should_cache_conversation(&messages));
    }

    #[test]
    fn convert_messages_applies_cache_breakpoint_on_membox_boundary() {
        let messages = vec![
            ChatMessage::user(format!(
                "{}{}",
                zeroclaw_api::MEMBOX_PROMPT_CACHE_BOUNDARY_PREFIX,
                "<current_user_request>\nfirst\n</current_user_request>"
            )),
            ChatMessage::assistant("answer"),
            ChatMessage::user(format!(
                "{}{}",
                zeroclaw_api::MEMBOX_PROMPT_CACHE_BOUNDARY_PREFIX,
                "<current_user_request>\nsecond\n</current_user_request>"
            )),
            ChatMessage::user("<memorybox_context>\nvolatile\n</memorybox_context>"),
        ];
        let (_, native) = AnthropicModelProvider::convert_messages(&messages);
        assert_eq!(native.len(), 3, "adjacent user turns merge for Anthropic");
        let merged_tail = native.last().expect("merged tail");
        assert_eq!(merged_tail.content.len(), 2);
        match &merged_tail.content[0] {
            NativeContentOut::Text {
                cache_control,
                text,
                ..
            } => {
                assert!(cache_control.is_some(), "latest boundary should be cached");
                assert!(text.contains("second"));
            }
            other => panic!("expected cached boundary block, got {other:?}"),
        }
        match &merged_tail.content[1] {
            NativeContentOut::Text {
                cache_control,
                text,
                ..
            } => {
                assert!(cache_control.is_none(), "volatile tail must stay uncached");
                assert!(text.contains("volatile"));
            }
            other => panic!("expected volatile block, got {other:?}"),
        }
        let (_, mut native_for_apply) = AnthropicModelProvider::convert_messages(&messages);
        AnthropicModelProvider::apply_conversation_cache_control(&messages, &mut native_for_apply);
        match native_for_apply.last().unwrap().content.last() {
            Some(NativeContentOut::Text { cache_control, .. }) => {
                assert!(
                    cache_control.is_none(),
                    "volatile MemoryBox context sibling must stay uncached"
                );
            }
            other => panic!("expected volatile tail after apply, got {other:?}"),
        }
    }

    #[test]
    fn no_membox_marker_retains_last_message_cache() {
        let messages = vec![
            ChatMessage::user("first"),
            ChatMessage::assistant("answer"),
            ChatMessage::user("second"),
        ];
        let (_, mut native) = AnthropicModelProvider::convert_messages(&messages);
        AnthropicModelProvider::apply_conversation_cache_control(&messages, &mut native);
        match native.last().unwrap().content.last() {
            Some(NativeContentOut::Text { cache_control, .. }) => {
                assert!(
                    cache_control.is_some(),
                    "unmarked conversations should keep last-message cache"
                );
            }
            other => panic!("expected cached last user text, got {other:?}"),
        }
    }

    #[test]
    fn membox_boundary_still_allows_tool_loop_tail_cache() {
        let messages = vec![
            ChatMessage::user(format!(
                "{}{}",
                zeroclaw_api::MEMBOX_PROMPT_CACHE_BOUNDARY_PREFIX,
                "<current_user_request>\nask\n</current_user_request>"
            )),
            ChatMessage::user("<memorybox_context>\nvolatile\n</memorybox_context>"),
            ChatMessage::assistant(
                r#"{"content":"","tool_calls":[{"id":"call_1","name":"search","arguments":"{}"}]}"#,
            ),
            ChatMessage::tool(r#"{"tool_call_id":"call_1","content":"result"}"#),
        ];
        let (_, mut native) = AnthropicModelProvider::convert_messages(&messages);
        AnthropicModelProvider::apply_conversation_cache_control(&messages, &mut native);
        let last = native.last().expect("tool result message");
        match last.content.last() {
            Some(NativeContentOut::ToolResult { cache_control, .. }) => {
                assert!(
                    cache_control.is_some(),
                    "tool-loop tail should keep an incremental breakpoint"
                );
            }
            other => panic!("expected cached tool_result tail, got {other:?}"),
        }
    }

    #[test]
    fn oauth_identity_does_not_consume_extra_breakpoint_when_system_follows() {
        let system = Some(SystemPrompt::String("MemBox system".to_string()));
        let with_oauth =
            AnthropicModelProvider::apply_oauth_system_prompt(system).expect("oauth system");
        match with_oauth {
            SystemPrompt::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert!(blocks[0].cache_control.is_none());
                assert!(blocks[1].cache_control.is_some());
            }
            other => panic!("expected blocks, got {other:?}"),
        }
    }

    #[test]
    fn membox_oauth_tool_request_keeps_at_most_four_breakpoints() {
        let messages = vec![
            ChatMessage::system("MemBox system"),
            ChatMessage::user(format!(
                "{}{}",
                zeroclaw_api::MEMBOX_PROMPT_CACHE_BOUNDARY_PREFIX,
                "<current_user_request>\nask\n</current_user_request>"
            )),
            ChatMessage::user("<memorybox_context>\nvolatile\n</memorybox_context>"),
            ChatMessage::assistant(
                r#"{"content":"","tool_calls":[{"id":"call_1","name":"search","arguments":"{}"}]}"#,
            ),
            ChatMessage::tool(r#"{"tool_call_id":"call_1","content":"result"}"#),
        ];
        let (system, mut native) = AnthropicModelProvider::convert_messages(&messages);
        AnthropicModelProvider::apply_conversation_cache_control(&messages, &mut native);
        let tools = AnthropicModelProvider::convert_tools(Some(&[ToolSpec::new(
            "search",
            "Search",
            serde_json::json!({"type": "object"}),
        )]));
        let system = AnthropicModelProvider::apply_oauth_system_prompt(system);
        AnthropicModelProvider::enforce_max_explicit_cache_breakpoints(
            system.as_ref(),
            tools.as_deref(),
            &mut native,
        );
        let count = AnthropicModelProvider::count_explicit_cache_breakpoints(
            system.as_ref(),
            tools.as_deref(),
            &native,
        );
        assert!(
            count <= AnthropicModelProvider::MAX_EXPLICIT_CACHE_BREAKPOINTS,
            "expected <=4 breakpoints, got {count}"
        );
        // Intended layout: tools + full system + MemBox boundary + tool_result tail.
        assert_eq!(count, 4, "expected full four-slot allocation, got {count}");
        assert!(
            AnthropicModelProvider::native_messages_have_stable_cache_boundary(&native),
            "MemBox boundary must remain"
        );
    }

    #[test]
    fn apply_cache_to_last_message_text() {
        let mut messages = vec![NativeMessage {
            role: "user".to_string(),
            content: vec![NativeContentOut::Text {
                text: "Hello".to_string(),
                cache_control: None,
            }],
        }];

        AnthropicModelProvider::apply_cache_to_last_message(&mut messages);

        match &messages[0].content[0] {
            NativeContentOut::Text { cache_control, .. } => {
                assert!(cache_control.is_some());
            }
            _ => panic!("Expected Text variant"),
        }
    }

    #[test]
    fn apply_cache_to_last_message_tool_result() {
        let mut messages = vec![NativeMessage {
            role: "user".to_string(),
            content: vec![NativeContentOut::ToolResult {
                tool_use_id: "tool_123".to_string(),
                content: "Result".to_string(),
                cache_control: None,
            }],
        }];

        AnthropicModelProvider::apply_cache_to_last_message(&mut messages);

        match &messages[0].content[0] {
            NativeContentOut::ToolResult { cache_control, .. } => {
                assert!(cache_control.is_some());
            }
            _ => panic!("Expected ToolResult variant"),
        }
    }

    #[test]
    fn apply_cache_to_last_message_does_not_affect_tool_use() {
        let mut messages = vec![NativeMessage {
            role: "assistant".to_string(),
            content: vec![NativeContentOut::ToolUse {
                id: "tool_123".to_string(),
                name: "get_weather".to_string(),
                input: serde_json::json!({}),
                cache_control: None,
            }],
        }];

        AnthropicModelProvider::apply_cache_to_last_message(&mut messages);

        // ToolUse should not be affected
        match &messages[0].content[0] {
            NativeContentOut::ToolUse { cache_control, .. } => {
                assert!(cache_control.is_none());
            }
            _ => panic!("Expected ToolUse variant"),
        }
    }

    #[test]
    fn apply_cache_empty_messages() {
        let mut messages = vec![];
        AnthropicModelProvider::apply_cache_to_last_message(&mut messages);
        // Should not panic
        assert!(messages.is_empty());
    }

    #[test]
    fn convert_tools_adds_cache_to_last_tool() {
        let tools = vec![
            ToolSpec::new("tool1", "First tool", serde_json::json!({"type": "object"})),
            ToolSpec::new(
                "tool2",
                "Second tool",
                serde_json::json!({"type": "object"}),
            ),
        ];

        let native_tools = AnthropicModelProvider::convert_tools(Some(&tools)).unwrap();

        assert_eq!(native_tools.len(), 2);
        assert!(native_tools[0].cache_control.is_none());
        assert!(native_tools[1].cache_control.is_some());
    }

    #[test]
    fn convert_tools_single_tool_gets_cache() {
        let tools = vec![ToolSpec::new(
            "tool1",
            "Only tool",
            serde_json::json!({"type": "object"}),
        )];

        let native_tools = AnthropicModelProvider::convert_tools(Some(&tools)).unwrap();

        assert_eq!(native_tools.len(), 1);
        assert!(native_tools[0].cache_control.is_some());
    }

    #[test]
    fn convert_tools_cleans_ref_from_input_schema() {
        let tools = vec![ToolSpec::new(
            "query",
            "Search with a ref",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "filter": {
                        "$ref": "#/$defs/FilterSpec"
                    }
                },
                "$defs": {
                    "FilterSpec": {
                        "type": "object",
                        "properties": {
                            "field": { "type": "string" }
                        }
                    }
                }
            }),
        )];

        let native_tools = AnthropicModelProvider::convert_tools(Some(&tools)).unwrap();
        let schema = &native_tools[0].input_schema;

        let filter = &schema["properties"]["filter"];
        assert!(filter.get("$ref").is_none(), "$ref was not cleaned");
        assert_eq!(filter["type"], "object");
        assert_eq!(filter["properties"]["field"]["type"], "string");
        assert!(schema.get("$defs").is_none(), "$defs was not stripped");
    }

    #[test]
    fn convert_tools_cleans_definitions_from_input_schema() {
        let tools = vec![ToolSpec::new(
            "query",
            "Search with a definitions ref",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "filter": {
                        "$ref": "#/definitions/FilterSpec"
                    }
                },
                "definitions": {
                    "FilterSpec": {
                        "type": "object",
                        "properties": {
                            "field": { "type": "string" }
                        }
                    }
                }
            }),
        )];

        let native_tools = AnthropicModelProvider::convert_tools(Some(&tools)).unwrap();
        let schema = &native_tools[0].input_schema;

        let filter = &schema["properties"]["filter"];
        assert!(filter.get("$ref").is_none(), "$ref was not cleaned");
        assert_eq!(filter["type"], "object");
        assert!(
            schema.get("definitions").is_none(),
            "definitions was not stripped"
        );
    }

    #[test]
    fn convert_tools_empty_tools_returns_none() {
        let tools: Vec<ToolSpec> = vec![];
        let result = AnthropicModelProvider::convert_tools(Some(&tools));
        assert!(result.is_none());
    }

    #[test]
    fn convert_tools_none_returns_none() {
        let result: Option<Vec<NativeToolSpec>> = AnthropicModelProvider::convert_tools(None);
        assert!(result.is_none());
    }

    #[test]
    fn convert_messages_small_system_prompt_uses_blocks_with_cache() {
        let messages = vec![ChatMessage {
            role: "system".to_string(),
            content: "Short system prompt".to_string(),
        }];

        let (system_prompt, _) = AnthropicModelProvider::convert_messages(&messages);

        match system_prompt.unwrap() {
            SystemPrompt::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert_eq!(blocks[0].text, "Short system prompt");
                assert!(
                    blocks[0].cache_control.is_some(),
                    "Small system prompts should have cache_control"
                );
            }
            SystemPrompt::String(_) => {
                panic!("Expected Blocks variant with cache_control for small prompt")
            }
        }
    }

    #[test]
    fn convert_messages_large_system_prompt() {
        let large_content = "a".repeat(3073);
        let messages = vec![ChatMessage {
            role: "system".to_string(),
            content: large_content.clone(),
        }];

        let (system_prompt, _) = AnthropicModelProvider::convert_messages(&messages);

        match system_prompt.unwrap() {
            SystemPrompt::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert_eq!(blocks[0].text, large_content);
                assert!(blocks[0].cache_control.is_some());
            }
            SystemPrompt::String(_) => panic!("Expected Blocks variant for large prompt"),
        }
    }

    #[test]
    fn native_chat_request_with_blocks_system() {
        // System prompts now always use Blocks format with cache_control
        let req = NativeChatRequest {
            model: "claude-3-opus".to_string(),
            max_tokens: 4096,
            system: Some(SystemPrompt::Blocks(vec![SystemBlock {
                block_type: "text".to_string(),
                text: "System".to_string(),
                cache_control: Some(CacheControl::ephemeral()),
            }])),
            messages: vec![NativeMessage {
                role: "user".to_string(),
                content: vec![NativeContentOut::Text {
                    text: "Hello".to_string(),
                    cache_control: None,
                }],
            }],
            temperature: Some(0.7),
            tools: None,
            tool_choice: None,
            stream: None,
            thinking: None,
            output_config: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("System"));
        assert!(
            json.contains(r#""cache_control":{"type":"ephemeral"}"#),
            "System prompt should include cache_control"
        );
    }

    #[test]
    fn native_chat_request_omits_temperature_when_none() {
        let req = NativeChatRequest {
            model: "claude-opus-4-7".to_string(),
            max_tokens: 4096,
            system: None,
            messages: vec![NativeMessage {
                role: "user".to_string(),
                content: vec![NativeContentOut::Text {
                    text: "hi".to_string(),
                    cache_control: None,
                }],
            }],
            temperature: None,
            tools: None,
            tool_choice: None,
            stream: None,
            thinking: None,
            output_config: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(
            !json.contains("temperature"),
            "temperature should be omitted when None; got: {json}"
        );
    }

    #[tokio::test]
    async fn warmup_without_key_is_noop() {
        let model_provider = AnthropicModelProvider::builder("test").build();
        let result = model_provider.warmup().await;
        assert!(result.is_ok());
    }

    #[test]
    fn convert_messages_preserves_multi_turn_history() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "gen a 2 sum in golang".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "```go\nfunc twoSum(nums []int) {}\n```".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "what's meaning of make here?".to_string(),
            },
        ];

        let (system, native_msgs) = AnthropicModelProvider::convert_messages(&messages);

        // System prompt extracted
        assert!(system.is_some());
        // All 3 non-system messages preserved in order
        assert_eq!(native_msgs.len(), 3);
        assert_eq!(native_msgs[0].role, "user");
        assert_eq!(native_msgs[1].role, "assistant");
        assert_eq!(native_msgs[2].role, "user");
    }

    #[tokio::test]
    async fn chat_with_tools_sends_full_history_and_native_tools() {
        use axum::{Json, Router, routing::post};
        use std::sync::{Arc, Mutex};
        use tokio::net::TcpListener;

        // Captured request body for assertion
        let captured: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));
        let captured_clone = captured.clone();

        let app = Router::new().route(
            "/v1/messages",
            post(move |Json(body): Json<serde_json::Value>| {
                let cap = captured_clone.clone();
                async move {
                    *cap.lock().unwrap() = Some(body);
                    // Return a minimal valid Anthropic response
                    Json(serde_json::json!({
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "The make function creates a map."}],
                        "model": "claude-opus-4-6",
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 100, "output_tokens": 20}
                    }))
                }
            }),
        );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server_handle = zeroclaw_spawn::spawn!(async move {
            axum::serve(listener, app).await.unwrap();
        });

        // Create model_provider pointing at mock server
        let model_provider = AnthropicModelProvider {
            alias: "test".to_string(),
            credential: Some("test-key".to_string()),
            base_url: format!("http://{addr}"),
            max_tokens: 4096,
            timeout_secs: 120,
            reasoning_effort: None,
            extra_headers: Vec::new(),
        };

        // Multi-turn conversation: system → user (Go code) → assistant (code response) → user (follow-up)
        let messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("gen a 2 sum in golang"),
            ChatMessage::assistant(
                "```go\nfunc twoSum(nums []int, target int) []int {\n    m := make(map[int]int)\n    for i, n := range nums {\n        if j, ok := m[target-n]; ok {\n            return []int{j, i}\n        }\n        m[n] = i\n    }\n    return nil\n}\n```",
            ),
            ChatMessage::user("what's meaning of make here?"),
        ];

        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }
            }
        })];

        let result = model_provider
            .chat_with_tools(&messages, &tools, "claude-opus-4-6", Some(0.7))
            .await;
        assert!(result.is_ok(), "chat_with_tools failed: {:?}", result.err());

        let body = captured
            .lock()
            .unwrap()
            .take()
            .expect("No request captured");

        // Verify system prompt extracted to top-level field
        let system = &body["system"];
        assert!(
            system.to_string().contains("helpful assistant"),
            "System prompt missing: {system}"
        );

        // Verify ALL conversation turns present in messages array
        let msgs = body["messages"].as_array().expect("messages not an array");
        assert_eq!(
            msgs.len(),
            3,
            "Expected 3 messages (2 user + 1 assistant), got {}",
            msgs.len()
        );

        // Turn 1: user with Go request
        assert_eq!(msgs[0]["role"], "user");
        let turn1_text = msgs[0]["content"].to_string();
        assert!(
            turn1_text.contains("2 sum"),
            "Turn 1 missing Go request: {turn1_text}"
        );

        // Turn 2: assistant with Go code
        assert_eq!(msgs[1]["role"], "assistant");
        let turn2_text = msgs[1]["content"].to_string();
        assert!(
            turn2_text.contains("make(map[int]int)"),
            "Turn 2 missing Go code: {turn2_text}"
        );

        // Turn 3: user follow-up
        assert_eq!(msgs[2]["role"], "user");
        let turn3_text = msgs[2]["content"].to_string();
        assert!(
            turn3_text.contains("meaning of make"),
            "Turn 3 missing follow-up: {turn3_text}"
        );

        // Verify native tools are present
        let api_tools = body["tools"].as_array().expect("tools not an array");
        assert_eq!(api_tools.len(), 1);
        assert_eq!(api_tools[0]["name"], "shell");
        assert!(
            api_tools[0]["input_schema"].is_object(),
            "Missing input_schema"
        );

        server_handle.abort();
    }

    #[test]
    fn native_response_parses_usage() {
        let json = r#"{
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"input_tokens": 300, "output_tokens": 75}
        }"#;
        let resp: NativeChatResponse = serde_json::from_str(json).unwrap();
        let result = AnthropicModelProvider::parse_native_response(resp);
        let usage = result.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(300));
        assert_eq!(usage.output_tokens, Some(75));
    }

    #[test]
    fn native_response_sums_all_three_anthropic_input_buckets() {
        let json = r#"{
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": 1,
                "cache_read_input_tokens": 148539,
                "cache_creation_input_tokens": 4200,
                "output_tokens": 27
            }
        }"#;
        let resp: NativeChatResponse = serde_json::from_str(json).unwrap();
        let result = AnthropicModelProvider::parse_native_response(resp);
        let usage = result.usage.expect("usage should be Some");
        assert_eq!(
            usage.input_tokens,
            Some(152_740),
            "total = 1 (after-breakpoint) + 148539 (cache_read) + 4200 (cache_creation)"
        );
        assert_eq!(
            usage.cached_input_tokens,
            Some(148_539),
            "cached_input_tokens is the cache-read portion only \
             (the discount-billed subset of the total)"
        );
        assert_eq!(usage.output_tokens, Some(27));
    }

    #[test]
    fn native_response_parses_without_usage() {
        let json = r#"{"content": [{"type": "text", "text": "Hello"}]}"#;
        let resp: NativeChatResponse = serde_json::from_str(json).unwrap();
        let result = AnthropicModelProvider::parse_native_response(resp);
        assert!(result.usage.is_none());
    }

    #[test]
    fn native_response_preserves_thinking_text_byte_for_byte() {
        // Signatures on extended-thinking blocks are computed over the exact
        // bytes the model returned. Any mutation — including trim() — breaks
        // signature validation on replay in a multi-turn tool-use conversation.
        let json = r#"{
            "content": [
                {
                    "type": "thinking",
                    "thinking": "  \nStep 1: consider the request.\nStep 2: respond.\n  ",
                    "signature": "sig_abc123"
                },
                {"type": "text", "text": "ok"}
            ]
        }"#;
        let resp: NativeChatResponse = serde_json::from_str(json).unwrap();
        let result = AnthropicModelProvider::parse_native_response(resp);
        let reasoning = result.reasoning_content.expect("thinking preserved");
        let blocks = replay_blocks(&reasoning);
        assert_eq!(
            blocks[0].get("thinking").and_then(|v| v.as_str()),
            Some("  \nStep 1: consider the request.\nStep 2: respond.\n  ")
        );
        assert_eq!(
            blocks[0].get("signature").and_then(|v| v.as_str()),
            Some("sig_abc123")
        );
    }

    #[test]
    fn native_response_preserves_signed_empty_thinking_blocks() {
        // Adaptive models default to `display:"omitted"` → `thinking:""`
        // with an encrypted `signature`. The signature must be preserved for
        // tool-result replay, or the continuation request 400s.
        let json = r#"{
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig_xyz"},
                {"type": "text", "text": "hello"}
            ]
        }"#;
        let resp: NativeChatResponse = serde_json::from_str(json).unwrap();
        let result = AnthropicModelProvider::parse_native_response(resp);
        let reasoning = result
            .reasoning_content
            .expect("omitted thinking block must be preserved with its signature");
        assert!(
            reasoning.contains(r#""signature":"sig_xyz""#),
            "signature missing from reasoning_content: {reasoning}"
        );
        assert!(
            reasoning.contains(r#""thinking":""#),
            "empty thinking text missing from reasoning_content: {reasoning}"
        );
    }

    #[test]
    fn signed_empty_thinking_round_trips_into_assistant_history() {
        // Full continuation path: a tool-use turn whose thinking was omitted
        // (empty text, signature present) must survive response → stored
        // reasoning_content → assistant-history rebuild, so the next request
        // replays the signed thinking block before the tool result.
        let response_json = r#"{
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig_omitted"},
                {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Paris"}}
            ]
        }"#;
        let resp: NativeChatResponse = serde_json::from_str(response_json).unwrap();
        let parsed = AnthropicModelProvider::parse_native_response(resp);
        let reasoning = parsed
            .reasoning_content
            .expect("omitted thinking must be preserved for replay");

        // Rebuild the stored assistant message and feed it back through the
        // history parser.
        let assistant = serde_json::json!({
            "content": null,
            "reasoning_content": reasoning,
            "tool_calls": parsed.tool_calls,
        })
        .to_string();
        let blocks = AnthropicModelProvider::parse_assistant_tool_call_message(&assistant)
            .expect("assistant message should parse");
        // The signed thinking block must come first, ahead of the tool_use,
        // with empty text and the signature intact.
        match blocks.first() {
            Some(NativeContentOut::Thinking {
                thinking,
                signature,
            }) => {
                assert!(
                    thinking.is_empty(),
                    "round-tripped thinking must stay empty"
                );
                assert_eq!(
                    signature.as_deref(),
                    Some("sig_omitted"),
                    "signature must survive the round trip"
                );
            }
            other => panic!("expected signed Thinking block first, got {other:?}"),
        }
    }

    #[test]
    fn redacted_thinking_round_trips_into_assistant_history() {
        let response_json = r#"{
            "content": [
                {"type": "redacted_thinking", "data": "opaque_redacted_data"},
                {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Paris"}}
            ]
        }"#;
        let response: NativeChatResponse = serde_json::from_str(response_json).unwrap();
        let parsed = AnthropicModelProvider::parse_native_response(response);
        let reasoning = parsed
            .reasoning_content
            .expect("redacted thinking must be preserved for replay");
        let preserved = replay_blocks(&reasoning);
        assert_eq!(preserved[0]["type"], "redacted_thinking");
        assert_eq!(preserved[0]["data"], "opaque_redacted_data");

        let assistant = serde_json::json!({
            "content": null,
            "reasoning_content": reasoning,
            "tool_calls": parsed.tool_calls,
        })
        .to_string();
        let blocks = AnthropicModelProvider::parse_assistant_tool_call_message(&assistant)
            .expect("assistant message should parse");
        let replayed = serde_json::to_value(blocks.first().expect("redacted block first")).unwrap();
        assert_eq!(replayed["type"], "redacted_thinking");
        assert_eq!(replayed["data"], "opaque_redacted_data");
    }

    #[test]
    fn interleaved_reasoning_keeps_original_tool_order_on_replay() {
        let response_json = r#"{
            "content": [
                {"type": "thinking", "thinking": "first", "signature": "sig_1"},
                {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "one"}},
                {"type": "redacted_thinking", "data": "opaque_redacted_data"},
                {"type": "tool_use", "id": "tu_2", "name": "search", "input": {"q": "two"}}
            ]
        }"#;
        let response: NativeChatResponse = serde_json::from_str(response_json).unwrap();
        let parsed = AnthropicModelProvider::parse_native_response(response);
        assert!(
            parsed
                .tool_calls
                .iter()
                .all(|call| call.extra_content.is_none()),
            "replay ordering must not depend on tool-call extension persistence"
        );
        let assistant = serde_json::json!({
            "content": null,
            "reasoning_content": parsed.reasoning_content,
            "tool_calls": parsed.tool_calls,
        })
        .to_string();

        let blocks = AnthropicModelProvider::parse_assistant_tool_call_message(&assistant)
            .expect("assistant message should parse");
        let replayed = serde_json::to_value(blocks).unwrap();

        assert_eq!(replayed[0]["type"], "thinking");
        assert_eq!(replayed[1]["type"], "tool_use");
        assert_eq!(replayed[1]["id"], "tu_1");
        assert_eq!(replayed[2]["type"], "redacted_thinking");
        assert_eq!(replayed[3]["type"], "tool_use");
        assert_eq!(replayed[3]["id"], "tu_2");
        assert!(replayed.as_array().unwrap().iter().all(|block| {
            block.get("index").is_none()
                && block.get("provider").is_none()
                && block.get("kind").is_none()
                && block.get(ANTHROPIC_BLOCK_INDEX_KEY).is_none()
                && block.get("extra_content").is_none()
        }));
    }

    #[test]
    fn interleaved_reasoning_with_text_keeps_complete_block_order_on_replay() {
        let response_json = r#"{
            "content": [
                {"type": "thinking", "thinking": "first", "signature": "sig_1"},
                {"type": "text", "text": "I'll search."},
                {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "one"}},
                {"type": "redacted_thinking", "data": "opaque_redacted_data"},
                {"type": "text", "text": "第二个来源。"},
                {"type": "tool_use", "id": "tu_2", "name": "search", "input": {"q": "two"}}
            ]
        }"#;
        let response: NativeChatResponse = serde_json::from_str(response_json).unwrap();
        let parsed = AnthropicModelProvider::parse_native_response(response);
        let assistant = serde_json::json!({
            "content": parsed.text,
            "reasoning_content": parsed.reasoning_content,
            "tool_calls": parsed.tool_calls,
        })
        .to_string();

        let blocks = AnthropicModelProvider::parse_assistant_tool_call_message(&assistant)
            .expect("assistant message should parse");
        let replayed = serde_json::to_value(blocks).unwrap();

        assert_eq!(replayed[0]["type"], "thinking");
        assert_eq!(replayed[1]["text"], "I'll search.");
        assert_eq!(replayed[2]["id"], "tu_1");
        assert_eq!(replayed[3]["type"], "redacted_thinking");
        assert_eq!(replayed[4]["text"], "第二个来源。");
        assert_eq!(replayed[5]["id"], "tu_2");
        assert!(replayed.as_array().unwrap().iter().all(|block| {
            block.get("index").is_none()
                && block.get("provider").is_none()
                && block.get("kind").is_none()
                && block.get(ANTHROPIC_BLOCK_INDEX_KEY).is_none()
                && block.get("extra_content").is_none()
        }));
    }

    #[test]
    fn invalid_text_span_falls_back_without_losing_assistant_blocks() {
        let damaged_reasoning = [
            serde_json::json!({
                "type": "thinking",
                "thinking": "first",
                "signature": "sig_1",
                ANTHROPIC_BLOCK_INDEX_KEY: 0,
            })
            .to_string(),
            serde_json::json!({
                "type": ANTHROPIC_TEXT_BLOCK_MARKER,
                ANTHROPIC_BLOCK_INDEX_KEY: 1,
                ANTHROPIC_TEXT_START_KEY: 0,
                ANTHROPIC_TEXT_END_KEY: usize::MAX,
            })
            .to_string(),
        ]
        .join("\n");
        let assistant = serde_json::json!({
            "content": "I'll search.",
            "reasoning_content": damaged_reasoning,
            "tool_calls": [{
                "id": "tu_1",
                "name": "search",
                "arguments": "{\"q\":\"one\"}",
                "extra_content": {
                    ANTHROPIC_EXTRA_CONTENT_KEY: {ANTHROPIC_BLOCK_INDEX_KEY: 2},
                },
            }],
        })
        .to_string();

        let blocks = AnthropicModelProvider::parse_assistant_tool_call_message(&assistant)
            .expect("invalid span should use the compatible fallback");
        let replayed = serde_json::to_value(blocks).unwrap();

        assert_eq!(replayed.as_array().unwrap().len(), 3);
        assert_eq!(replayed[0]["type"], "thinking");
        assert_eq!(replayed[0]["thinking"], "first");
        assert_eq!(replayed[1]["type"], "text");
        assert_eq!(replayed[1]["text"], "I'll search.");
        assert_eq!(replayed[2]["type"], "tool_use");
        assert_eq!(replayed[2]["id"], "tu_1");
    }

    #[test]
    fn legacy_indexed_history_keeps_exact_block_order() {
        let reasoning = [
            serde_json::json!({
                "type": "thinking",
                "thinking": "first",
                "signature": "sig_1",
                ANTHROPIC_BLOCK_INDEX_KEY: 0,
            })
            .to_string(),
            serde_json::json!({
                "type": ANTHROPIC_TEXT_BLOCK_MARKER,
                ANTHROPIC_BLOCK_INDEX_KEY: 1,
                ANTHROPIC_TEXT_START_KEY: 0,
                ANTHROPIC_TEXT_END_KEY: 12,
            })
            .to_string(),
        ]
        .join("\n");
        let assistant = serde_json::json!({
            "content": "I'll search.",
            "reasoning_content": reasoning,
            "tool_calls": [{
                "id": "tu_1",
                "name": "search",
                "arguments": "{}",
                "extra_content": {
                    ANTHROPIC_EXTRA_CONTENT_KEY: {ANTHROPIC_BLOCK_INDEX_KEY: 2},
                },
            }],
        })
        .to_string();

        let blocks = AnthropicModelProvider::parse_assistant_tool_call_message(&assistant)
            .expect("legacy assistant message should parse");
        let replayed = serde_json::to_value(blocks).unwrap();

        assert_eq!(replayed[0]["type"], "thinking");
        assert_eq!(replayed[1]["type"], "text");
        assert_eq!(replayed[2]["type"], "tool_use");
    }

    #[test]
    fn foreign_provider_replay_envelope_is_not_sent_as_thinking() {
        let foreign = serde_json::json!({
            "provider": "openai_codex",
            "kind": "responses_output_items",
            "items": [{"type": "reasoning", "id": "rs_1"}],
        });
        let assistant = serde_json::json!({
            "content": "Previous provider answer",
            "reasoning_content": foreign.to_string(),
            "tool_calls": [{
                "id": "tu_1",
                "name": "search",
                "arguments": "{}",
            }],
        })
        .to_string();

        let blocks = AnthropicModelProvider::parse_assistant_tool_call_message(&assistant)
            .expect("assistant message should parse");
        let replayed = serde_json::to_value(blocks).unwrap();

        assert_eq!(replayed.as_array().unwrap().len(), 2);
        assert_eq!(replayed[0]["type"], "text");
        assert_eq!(replayed[1]["type"], "tool_use");
    }

    #[test]
    fn legacy_unindexed_reasoning_history_keeps_compatible_replay_order() {
        let reasoning = [
            serde_json::json!({"thinking": "legacy", "signature": "sig"}).to_string(),
            serde_json::json!({"type": "redacted_thinking", "data": "opaque"}).to_string(),
        ]
        .join("\n");
        let assistant = serde_json::json!({
            "content": null,
            "reasoning_content": reasoning,
            "tool_calls": [{
                "id": "tu_1",
                "name": "search",
                "arguments": "{}",
            }],
        })
        .to_string();

        let blocks = AnthropicModelProvider::parse_assistant_tool_call_message(&assistant)
            .expect("legacy assistant message should parse");
        let replayed = serde_json::to_value(blocks).unwrap();
        assert_eq!(replayed[0]["type"], "thinking");
        assert_eq!(replayed[1]["type"], "redacted_thinking");
        assert_eq!(replayed[2]["id"], "tu_1");
    }

    #[test]
    fn capabilities_returns_vision_and_native_tools() {
        let model_provider = AnthropicModelProvider::builder("test")
            .credential(Some("test-key"))
            .build();
        let caps = model_provider.capabilities();
        assert!(
            caps.native_tool_calling,
            "Anthropic should support native tool calling"
        );
        assert!(caps.vision, "Anthropic should support vision");
    }

    #[test]
    fn convert_messages_with_image_marker_data_uri() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Check this image: [IMAGE:data:image/jpeg;base64,/9j/4AAQ] What do you see?"
                .to_string(),
        }];

        let (_, native_msgs) = AnthropicModelProvider::convert_messages(&messages);

        assert_eq!(native_msgs.len(), 1);
        assert_eq!(native_msgs[0].role, "user");
        // Should have 2 content blocks: image + text
        assert_eq!(native_msgs[0].content.len(), 2);

        // First block should be image
        match &native_msgs[0].content[0] {
            NativeContentOut::Image { source } => {
                assert_eq!(source.source_type, "base64");
                assert_eq!(source.media_type, "image/jpeg");
                assert_eq!(source.data, "/9j/4AAQ");
            }
            _ => panic!("Expected Image content block"),
        }

        // Second block should be text (parse_image_markers may leave extra spaces)
        match &native_msgs[0].content[1] {
            NativeContentOut::Text { text, .. } => {
                // The text may have extra spaces where the marker was removed
                assert!(
                    text.contains("Check this image:") && text.contains("What do you see?"),
                    "Expected text to contain 'Check this image:' and 'What do you see?', got: {}",
                    text
                );
            }
            _ => panic!("Expected Text content block"),
        }
    }

    #[test]
    fn convert_messages_with_only_image_marker() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "[IMAGE:data:image/png;base64,iVBORw0KGgo]".to_string(),
        }];

        let (_, native_msgs) = AnthropicModelProvider::convert_messages(&messages);

        assert_eq!(native_msgs.len(), 1);
        assert_eq!(native_msgs[0].content.len(), 2);

        // First block should be image
        match &native_msgs[0].content[0] {
            NativeContentOut::Image { source } => {
                assert_eq!(source.media_type, "image/png");
            }
            _ => panic!("Expected Image content block"),
        }

        // Second block should be placeholder text
        match &native_msgs[0].content[1] {
            NativeContentOut::Text { text, .. } => {
                assert_eq!(text, "[image]");
            }
            _ => panic!("Expected Text content block with [image] placeholder"),
        }
    }

    #[test]
    fn convert_messages_without_image_marker() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello, how are you?".to_string(),
        }];

        let (_, native_msgs) = AnthropicModelProvider::convert_messages(&messages);

        assert_eq!(native_msgs.len(), 1);
        assert_eq!(native_msgs[0].content.len(), 1);

        match &native_msgs[0].content[0] {
            NativeContentOut::Text { text, .. } => {
                assert_eq!(text, "Hello, how are you?");
            }
            _ => panic!("Expected Text content block"),
        }
    }

    #[test]
    fn image_content_serializes_correctly() {
        let content = NativeContentOut::Image {
            source: ImageSource {
                source_type: "base64".to_string(),
                media_type: "image/jpeg".to_string(),
                data: "testdata".to_string(),
            },
        };
        let json = serde_json::to_string(&content).unwrap();
        // The outer "type" is the enum tag, inner "type" (source_type) is renamed
        assert!(json.contains(r#""type":"image""#), "JSON: {}", json);
        assert!(json.contains(r#""type":"base64""#), "JSON: {}", json); // source_type is serialized as "type"
        assert!(
            json.contains(r#""media_type":"image/jpeg""#),
            "JSON: {}",
            json
        );
        assert!(json.contains(r#""data":"testdata""#), "JSON: {}", json);
    }

    #[test]
    fn convert_messages_merges_consecutive_tool_results() {
        // Simulate a multi-tool-call turn: assistant with two tool_use blocks
        // followed by two separate tool result messages.
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Do two things.".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: serde_json::json!({
                    "content": "",
                    "tool_calls": [
                        {"id": "call_1", "name": "shell", "arguments": "{\"command\":\"ls\"}"},
                        {"id": "call_2", "name": "shell", "arguments": "{\"command\":\"pwd\"}"}
                    ]
                })
                .to_string(),
            },
            ChatMessage {
                role: "tool".to_string(),
                content: serde_json::json!({
                    "tool_call_id": "call_1",
                    "content": "file1.txt\nfile2.txt"
                })
                .to_string(),
            },
            ChatMessage {
                role: "tool".to_string(),
                content: serde_json::json!({
                    "tool_call_id": "call_2",
                    "content": "/home/user"
                })
                .to_string(),
            },
        ];

        let (system, native_msgs) = AnthropicModelProvider::convert_messages(&messages);

        assert!(system.is_some());
        // Should be: user, assistant, user (merged tool results)
        // NOT: user, assistant, user, user (which Anthropic rejects)
        assert_eq!(
            native_msgs.len(),
            3,
            "Expected 3 messages (user, assistant, merged tool results), got {}.\nRoles: {:?}",
            native_msgs.len(),
            native_msgs.iter().map(|m| &m.role).collect::<Vec<_>>()
        );
        assert_eq!(native_msgs[0].role, "user");
        assert_eq!(native_msgs[1].role, "assistant");
        assert_eq!(native_msgs[2].role, "user");
        // The merged user message should contain both tool results
        assert_eq!(
            native_msgs[2].content.len(),
            2,
            "Expected 2 tool_result blocks in merged message"
        );
    }

    #[test]
    fn convert_messages_degrades_leading_orphaned_tool_result() {
        let messages = vec![ChatMessage::tool(
            serde_json::json!({
                "tool_call_id": "toolu_orphan",
                "content": "created /tmp/image.png"
            })
            .to_string(),
        )];

        let (_, native_messages) = AnthropicModelProvider::convert_messages(&messages);

        assert_eq!(native_messages.len(), 1);
        assert!(matches!(
            native_messages[0].content.first(),
            Some(NativeContentOut::Text { text, .. })
                if text.contains("toolu_orphan") && text.contains("created /tmp/image.png")
        ));
        assert!(
            !native_messages[0]
                .content
                .iter()
                .any(|block| matches!(block, NativeContentOut::ToolResult { .. }))
        );
    }

    #[test]
    fn convert_messages_preserves_valid_result_and_degrades_second_round_orphan() {
        let messages = vec![
            ChatMessage::user("Create an image."),
            ChatMessage::assistant(
                serde_json::json!({
                    "content": "",
                    "tool_calls": [
                        {"id": "toolu_search", "name": "search_tools", "arguments": "{}"}
                    ]
                })
                .to_string(),
            ),
            ChatMessage::tool(
                serde_json::json!({
                    "tool_call_id": "toolu_search",
                    "content": "create_image activated"
                })
                .to_string(),
            ),
            ChatMessage::tool(
                serde_json::json!({
                    "tool_call_id": "toolu_create_image",
                    "content": "created /tmp/image.png"
                })
                .to_string(),
            ),
        ];

        let (_, native_messages) = AnthropicModelProvider::convert_messages(&messages);
        let result_message = native_messages.last().expect("result message present");

        assert!(matches!(
            result_message.content.first(),
            Some(NativeContentOut::ToolResult { tool_use_id, .. })
                if tool_use_id == "toolu_search"
        ));
        assert!(matches!(
            result_message.content.get(1),
            Some(NativeContentOut::Text { text, .. })
                if text.contains("toolu_create_image") && text.contains("created /tmp/image.png")
        ));
    }

    #[test]
    fn convert_messages_serializes_two_live_tool_rounds_with_valid_adjacency() {
        let messages = vec![
            ChatMessage::user("Create an image."),
            ChatMessage::assistant(
                serde_json::json!({
                    "content": "",
                    "tool_calls": [
                        {"id": "toolu_search", "name": "search_tools", "arguments": "{}"}
                    ]
                })
                .to_string(),
            ),
            ChatMessage::tool(
                serde_json::json!({
                    "tool_call_id": "toolu_search",
                    "content": "create_image activated"
                })
                .to_string(),
            ),
            ChatMessage::assistant(
                serde_json::json!({
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "toolu_create_image",
                            "name": "create_image",
                            "arguments": "{\"prompt\":\"lighthouse\"}"
                        }
                    ]
                })
                .to_string(),
            ),
            ChatMessage::tool(
                serde_json::json!({
                    "tool_call_id": "toolu_create_image",
                    "content": "created /tmp/lighthouse.png"
                })
                .to_string(),
            ),
        ];

        let (_, native_messages) = AnthropicModelProvider::convert_messages(&messages);
        let wire_messages = serde_json::to_value(&native_messages).expect("messages serialize");

        assert_eq!(wire_messages[1]["content"][0]["id"], "toolu_search");
        assert_eq!(
            wire_messages[2]["content"][0]["tool_use_id"],
            "toolu_search"
        );
        assert_eq!(wire_messages[3]["content"][0]["id"], "toolu_create_image");
        assert_eq!(
            wire_messages[4]["content"][0]["tool_use_id"],
            "toolu_create_image"
        );
    }

    #[test]
    fn convert_messages_backfills_orphaned_tool_use() {
        // A turn interrupted mid-flight: assistant emitted a tool_use but the
        // matching tool_result was never persisted, and a new user message
        // follows. Sending this raw is a hard 400. The converter must
        // synthesize a stub tool_result so the history stays well-formed.
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "Do a thing.".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: serde_json::json!({
                    "content": "",
                    "tool_calls": [
                        {"id": "orphan_1", "name": "shell", "arguments": "{\"command\":\"ls\"}"}
                    ]
                })
                .to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Actually, never mind.".to_string(),
            },
        ];

        let (_, native_msgs) = AnthropicModelProvider::convert_messages(&messages);

        let assistant_idx = native_msgs
            .iter()
            .position(|m| m.role == "assistant")
            .expect("assistant message present");
        let next = native_msgs
            .get(assistant_idx + 1)
            .expect("a message must follow the tool_use");

        let has_stub = next.content.iter().any(|block| {
            matches!(
                block,
                NativeContentOut::ToolResult { tool_use_id, .. } if tool_use_id == "orphan_1"
            )
        });
        assert!(
            has_stub,
            "orphaned tool_use should be answered by a synthesized tool_result"
        );

        assert!(
            matches!(
                next.content.first(),
                Some(NativeContentOut::ToolResult { .. })
            ),
            "tool_result must precede any text in the user message"
        );
    }

    #[test]
    fn convert_messages_backfills_trailing_orphaned_tool_use() {
        // The interrupted tool_use is the very last thing in history with no
        // following message at all. A tool_result message must be appended.
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "Do a thing.".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: serde_json::json!({
                    "content": "",
                    "tool_calls": [
                        {"id": "trailing_1", "name": "shell", "arguments": "{}"}
                    ]
                })
                .to_string(),
            },
        ];

        let (_, native_msgs) = AnthropicModelProvider::convert_messages(&messages);

        let last = native_msgs.last().expect("messages present");
        assert_eq!(last.role, "user");
        assert!(
            last.content.iter().any(|block| matches!(
                block,
                NativeContentOut::ToolResult { tool_use_id, .. } if tool_use_id == "trailing_1"
            )),
            "trailing orphaned tool_use should get an appended tool_result message"
        );
    }

    #[test]
    fn convert_messages_no_adjacent_same_role() {
        // Verify that convert_messages never produces adjacent messages with the
        // same role, regardless of input ordering.
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: serde_json::json!({
                    "content": "I'll run a command",
                    "tool_calls": [
                        {"id": "tc1", "name": "shell", "arguments": "{\"command\":\"echo hi\"}"}
                    ]
                })
                .to_string(),
            },
            ChatMessage {
                role: "tool".to_string(),
                content: serde_json::json!({
                    "tool_call_id": "tc1",
                    "content": "hi"
                })
                .to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Thanks!".to_string(),
            },
        ];

        let (_system, native_msgs) = AnthropicModelProvider::convert_messages(&messages);

        for window in native_msgs.windows(2) {
            assert_ne!(
                window[0].role, window[1].role,
                "Adjacent messages must not share the same role: found two '{}' messages in a row",
                window[0].role
            );
        }
    }

    #[tokio::test]
    async fn anthropic_factory_forwards_timeout_to_native_provider() {
        use crate::ModelProviderRuntimeOptions;
        use crate::factory::FamilyProviderFactory;
        use axum::{Json, Router, routing::post};
        use serde_json::json;
        use tokio::time::{Duration, Instant};
        use zeroclaw_config::schema::AnthropicModelProviderConfig;

        async fn slow_messages() -> Json<serde_json::Value> {
            tokio::time::sleep(Duration::from_secs(3)).await;
            Json(json!({
                "id": "msg_late",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "too late"}],
                "model": "claude-sonnet-4-5",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1}
            }))
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let addr = listener.local_addr().expect("test server addr");
        let app = Router::new().route("/v1/messages", post(slow_messages));
        let server = zeroclaw_spawn::spawn!(async move {
            axum::serve(listener, app).await.expect("serve test server");
        });

        let opts = ModelProviderRuntimeOptions {
            provider_timeout_secs: Some(1),
            ..Default::default()
        };
        let provider = AnthropicModelProviderConfig::default()
            .create_provider(
                "native",
                Some("test-key"),
                Some(&format!("http://{addr}")),
                &opts,
            )
            .expect("anthropic provider should build");

        let started = Instant::now();
        let result = provider
            .chat_with_system(None, "hello", "claude-sonnet-4-5", Some(0.7))
            .await;
        let elapsed = started.elapsed();

        server.abort();

        assert!(
            result.is_err(),
            "slow response should time out when factory forwards provider_timeout_secs"
        );
        assert!(
            elapsed < Duration::from_secs(3),
            "request waited for the server response instead of using configured timeout: {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn anthropic_factory_forwards_reasoning_effort_to_output_config() {
        use crate::ModelProviderRuntimeOptions;
        use crate::factory::FamilyProviderFactory;
        use axum::{Json, Router, routing::post};
        use parking_lot::Mutex;
        use serde_json::json;
        use std::sync::Arc;
        use zeroclaw_config::schema::AnthropicModelProviderConfig;

        // Capture the outbound request body so we can prove the factory handoff
        // (ModelProviderRuntimeOptions.reasoning_effort -> output_config.effort)
        // reaches the wire. The builder-level path is covered by
        // `output_config_reaches_wire_via_chat`; this exercises create_provider.
        let captured: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));
        let captured_clone = captured.clone();
        let app = Router::new().route(
            "/v1/messages",
            post(move |Json(body): Json<serde_json::Value>| {
                let cap = captured_clone.clone();
                async move {
                    *cap.lock() = Some(body);
                    Json(json!({
                        "id": "msg_effort",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "ok"}],
                        "model": "claude-opus-5",
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 1, "output_tokens": 1}
                    }))
                }
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let addr = listener.local_addr().expect("test server addr");
        let server = zeroclaw_spawn::spawn!(async move {
            axum::serve(listener, app).await.expect("serve test server");
        });

        let opts = ModelProviderRuntimeOptions {
            reasoning_effort: Some("high".to_string()),
            ..Default::default()
        };
        let provider = AnthropicModelProviderConfig::default()
            .create_provider(
                "native",
                Some("test-key"),
                Some(&format!("http://{addr}")),
                &opts,
            )
            .expect("anthropic provider should build");

        let result = provider
            .chat_with_system(None, "hi", "claude-opus-5", None)
            .await;
        server.abort();
        assert!(
            result.is_ok(),
            "chat_with_system failed: {:?}",
            result.err()
        );

        let body = captured.lock().take().expect("no request captured");
        assert_eq!(
            body["output_config"]["effort"], "high",
            "factory should forward reasoning_effort as output_config.effort, got: {}",
            body["output_config"]
        );
    }
}
