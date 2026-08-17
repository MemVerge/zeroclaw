//! Provider-neutral reason a model stream stopped.
//!
//! Providers spell this as `finish_reason`, `stop_reason`, Responses `status`,
//! or omit it. This enum is the single source of truth for runtime events.
//!
//! Wire JSON is adjacent-tagged snake_case so downstream clients (MemBox
//! PR-08) can lock the shape now:
//! `{"type":"output_truncated"}`, `{"type":"unknown","value":"pause_turn"}`.

use serde::{Deserialize, Serialize};

/// Why a provider stream ended. Truncation is a successful stream with this
/// reason, not a transport error.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum StopReason {
    /// Natural end: OpenAI `stop`, Anthropic `end_turn` / `stop_sequence`.
    Complete,
    /// Model wants tools: OpenAI `tool_calls`, Anthropic `tool_use`.
    /// This is a mid-turn handoff, not user-turn completion.
    ToolUse,
    /// Output budget exhausted: `length`, `max_tokens`, `MAX_TOKENS`,
    /// Responses `incomplete` without a more specific reason.
    OutputTruncated,
    /// Safety / policy: `content_filter`, `refusal`, `recitation`.
    ContentFiltered,
    /// Provider omitted the field.
    #[default]
    Unspecified,
    /// Unmapped spelling; keep the raw token.
    Unknown(String),
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
            "length"
            | "max_tokens"
            | "max_output_tokens"
            | "incomplete"
            | "model_context_window_exceeded" => Self::OutputTruncated,
            "content_filter" | "refusal" | "recitation" | "safety" => Self::ContentFiltered,
            other => Self::Unknown(other.to_string()),
        }
    }

    /// True when this LLM call ended the model's own turn.
    ///
    /// [`Self::ToolUse`] is a mid-turn handoff: tools still run, and more LLM
    /// calls may follow. Wire surfaces (gateway WS, ACP session result) must
    /// not treat that as user-turn completion.
    pub fn ends_model_turn(&self) -> bool {
        !matches!(self, Self::ToolUse)
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
        assert!(!StopReason::ToolUse.ends_model_turn());
        assert!(StopReason::Complete.ends_model_turn());
        assert!(StopReason::OutputTruncated.ends_model_turn());
        assert!(StopReason::Unspecified.ends_model_turn());
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
        assert_eq!(
            StopReason::from_provider_token("incomplete"),
            StopReason::OutputTruncated
        );
        assert_eq!(
            StopReason::from_provider_token("model_context_window_exceeded"),
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
    fn wire_json_is_adjacent_snake_case() {
        assert_eq!(
            serde_json::to_value(StopReason::OutputTruncated).unwrap(),
            serde_json::json!({"type": "output_truncated"})
        );
        assert_eq!(
            serde_json::to_value(StopReason::Unknown("pause_turn".into())).unwrap(),
            serde_json::json!({"type": "unknown", "value": "pause_turn"})
        );
        let parsed: StopReason =
            serde_json::from_value(serde_json::json!({"type": "complete"})).unwrap();
        assert_eq!(parsed, StopReason::Complete);
        let unknown: StopReason = serde_json::from_value(serde_json::json!({
            "type": "unknown",
            "value": "pause_turn"
        }))
        .unwrap();
        assert_eq!(unknown, StopReason::Unknown("pause_turn".into()));
    }
}
