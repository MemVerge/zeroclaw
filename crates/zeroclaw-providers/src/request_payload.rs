pub(crate) fn non_empty_string_field(value: &serde_json::Value, field: &str) -> Option<String> {
    value
        .get(field)
        .and_then(serde_json::Value::as_str)
        .filter(|content| !content.trim().is_empty())
        .map(ToString::to_string)
}

/// Return provider-visible reasoning while keeping provider-owned replay
/// envelopes inside ZeroClaw history. Envelopes are opaque continuation data,
/// not portable reasoning text, and must never be forwarded to another API.
pub(crate) fn reasoning_content_for_wire(value: &serde_json::Value, field: &str) -> Option<String> {
    value
        .get(field)
        .and_then(serde_json::Value::as_str)
        .filter(|content| !is_provider_replay_envelope(content))
        .map(ToString::to_string)
}

fn is_provider_replay_envelope(content: &str) -> bool {
    content.lines().any(|line| {
        serde_json::from_str::<serde_json::Value>(line)
            .ok()
            .is_some_and(|value| {
                value
                    .get("provider")
                    .and_then(serde_json::Value::as_str)
                    .is_some()
                    && value
                        .get("kind")
                        .and_then(serde_json::Value::as_str)
                        .is_some()
            })
    })
}

#[cfg(test)]
mod tests {
    use super::reasoning_content_for_wire;

    #[test]
    fn reasoning_content_for_wire_preserves_plain_reasoning() {
        let value = serde_json::json!({"reasoning_content": "inspect the files"});

        assert_eq!(
            reasoning_content_for_wire(&value, "reasoning_content").as_deref(),
            Some("inspect the files")
        );
    }

    #[test]
    fn reasoning_content_for_wire_drops_provider_replay_envelope() {
        let first = serde_json::json!({
            "provider": "anthropic",
            "kind": "content_blocks",
            "version": 1,
            "blocks": [],
        });
        let second = serde_json::json!({
            "provider": "anthropic",
            "kind": "content_blocks",
            "version": 1,
            "blocks": [{"type": "redacted_thinking", "data": "opaque"}],
        });
        let value = serde_json::json!({
            "reasoning_content": format!("{first}\n{second}"),
        });

        assert!(reasoning_content_for_wire(&value, "reasoning_content").is_none());
    }
}
