//! Provider-neutral reason a model stream stopped.
//!
//! Providers spell this as `finish_reason`, `stop_reason`, Responses `status`,
//! or omit it. This enum is the single source of truth for runtime events.

use serde::{Deserialize, Serialize};

/// Why a provider stream ended. Truncation is a successful stream with this
/// reason, not a transport error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// Natural end: OpenAI `stop`, Anthropic `end_turn` / `stop_sequence`.
    Complete,
    /// Model wants tools: OpenAI `tool_calls`, Anthropic `tool_use`.
    ToolUse,
    /// Output budget exhausted: `length`, `max_tokens`, `MAX_TOKENS`.
    OutputTruncated,
    /// Safety / policy: `content_filter`, `refusal`, `recitation`.
    ContentFiltered,
    /// Provider omitted the field.
    Unspecified,
    /// Unmapped spelling; keep the raw token.
    Unknown(String),
}

impl Default for StopReason {
    fn default() -> Self {
        Self::Unspecified
    }
}

impl StopReason {
    /// Map a provider finish/stop token. Case-insensitive. Empty → `Unspecified`.
    pub fn from_provider_token(token: &str) -> Self {
        let normalized = token.trim();
        if normalized.is_empty() {
            return Self::Unspecified;
        }
        match normalized.to_ascii_lowercase().as_str() {
            "stop" | "end_turn" | "stop_sequence" | "completed" => Self::Complete,
            "tool_calls" | "tool_use" | "function_call" => Self::ToolUse,
            "length" | "max_tokens" | "max_output_tokens" => Self::OutputTruncated,
            "content_filter" | "refusal" | "recitation" | "safety" => Self::ContentFiltered,
            other => Self::Unknown(other.to_string()),
        }
    }

    /// ACP / RPC session stop string. Unknown and unspecified stay `end_turn`
    /// so consumers that only know the old product codes do not break.
    pub fn as_session_stop(&self) -> &'static str {
        match self {
            Self::OutputTruncated => "max_tokens",
            Self::ContentFiltered => "refusal",
            Self::Complete | Self::ToolUse | Self::Unspecified | Self::Unknown(_) => "end_turn",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StopReason;

    #[test]
    fn maps_openai_and_anthropic_complete_spellings() {
        assert_eq!(
            StopReason::from_provider_token("stop"),
            StopReason::Complete
        );
        assert_eq!(
            StopReason::from_provider_token("end_turn"),
            StopReason::Complete
        );
        assert_eq!(
            StopReason::from_provider_token("stop_sequence"),
            StopReason::Complete
        );
        assert_eq!(
            StopReason::from_provider_token("completed"),
            StopReason::Complete
        );
    }

    #[test]
    fn maps_tool_use_spellings() {
        assert_eq!(
            StopReason::from_provider_token("tool_calls"),
            StopReason::ToolUse
        );
        assert_eq!(
            StopReason::from_provider_token("tool_use"),
            StopReason::ToolUse
        );
        assert_eq!(
            StopReason::from_provider_token("function_call"),
            StopReason::ToolUse
        );
    }

    #[test]
    fn maps_truncation_spellings_case_insensitively() {
        assert_eq!(
            StopReason::from_provider_token("length"),
            StopReason::OutputTruncated
        );
        assert_eq!(
            StopReason::from_provider_token("max_tokens"),
            StopReason::OutputTruncated
        );
        assert_eq!(
            StopReason::from_provider_token("MAX_TOKENS"),
            StopReason::OutputTruncated
        );
        assert_eq!(
            StopReason::from_provider_token("max_output_tokens"),
            StopReason::OutputTruncated
        );
    }

    #[test]
    fn maps_content_filter_spellings() {
        assert_eq!(
            StopReason::from_provider_token("content_filter"),
            StopReason::ContentFiltered
        );
        assert_eq!(
            StopReason::from_provider_token("refusal"),
            StopReason::ContentFiltered
        );
    }

    #[test]
    fn empty_token_is_unspecified() {
        assert_eq!(StopReason::from_provider_token(""), StopReason::Unspecified);
        assert_eq!(
            StopReason::from_provider_token("  "),
            StopReason::Unspecified
        );
    }

    #[test]
    fn unknown_spelling_preserves_normalized_raw() {
        assert_eq!(
            StopReason::from_provider_token("weird_reason"),
            StopReason::Unknown("weird_reason".to_string())
        );
    }

    #[test]
    fn session_stop_keeps_legacy_end_turn_for_unspecified() {
        assert_eq!(StopReason::Unspecified.as_session_stop(), "end_turn");
        assert_eq!(StopReason::Complete.as_session_stop(), "end_turn");
        assert_eq!(StopReason::OutputTruncated.as_session_stop(), "max_tokens");
        assert_eq!(StopReason::ContentFiltered.as_session_stop(), "refusal");
    }
}
