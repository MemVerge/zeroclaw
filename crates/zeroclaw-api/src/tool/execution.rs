/// Borrowed identities from the active dispatcher. No payload or durable state
/// is copied into this context. `call.id()` is the value used for the pending
/// and terminal TurnEvents, including the dispatcher's id-less-call fallback.
///
/// Correlation is scoped to `turn_id`; a provider may repeat a call ID. A host
/// requiring uniqueness must reject ambiguity rather than infer a new identity
/// from tool names, arguments, timestamps or completion order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ToolExecutionContext<'a> {
    pub turn_id: &'a str,
    pub call: ToolCallCorrelation<'a>,
}

/// A synthesized UI/event correlation must not masquerade as a provider's
/// native protocol identity. Both variants correlate dispatcher events.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolCallCorrelation<'a> {
    Native(&'a str),
    EventOnly(&'a str),
}

impl<'a> ToolCallCorrelation<'a> {
    pub fn id(self) -> &'a str {
        match self {
            Self::Native(id) | Self::EventOnly(id) => id,
        }
    }

    pub fn native_id(self) -> Option<&'a str> {
        match self {
            Self::Native(id) => Some(id),
            Self::EventOnly(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{Tool, ToolResult};
    use serde_json::{Value, json};

    struct ExistingTool;
    crate::mock_tool_attribution!(ExistingTool);

    #[async_trait::async_trait]
    impl Tool for ExistingTool {
        fn name(&self) -> &str {
            "existing"
        }
        fn description(&self) -> &str {
            "Synthetic compatibility fixture"
        }
        fn parameters_schema(&self) -> Value {
            json!({"type":"object"})
        }
        async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
            if args["fail"] == true {
                anyhow::bail!("synthetic existing failure");
            }
            Ok(ToolResult::ok(args.to_string()))
        }
    }

    fn context() -> ToolExecutionContext<'static> {
        ToolExecutionContext {
            turn_id: "synthetic-turn",
            call: ToolCallCorrelation::Native("synthetic-call"),
        }
    }

    #[tokio::test]
    async fn default_context_dispatch_preserves_existing_arguments_and_result() {
        let tool: &dyn Tool = &ExistingTool;
        let args = json!({"value":"原文🙂"});
        let result = tool
            .execute_with_context(args.clone(), context())
            .await
            .unwrap();
        assert_eq!(result.output, args.to_string());
        assert!(result.success);
    }

    #[tokio::test]
    async fn default_context_dispatch_does_not_swallow_existing_errors() {
        let result = ExistingTool
            .execute_with_context(json!({"fail":true}), context())
            .await;
        assert_eq!(
            result.unwrap_err().to_string(),
            "synthetic existing failure"
        );
    }

    #[test]
    fn event_correlation_does_not_invent_a_native_protocol_id() {
        let call = ToolCallCorrelation::EventOnly("event-id");
        assert_eq!(call.id(), "event-id");
        assert_eq!(call.native_id(), None);
    }

    #[test]
    fn native_correlation_retains_the_exact_protocol_id() {
        let call = ToolCallCorrelation::Native("provider-call");
        assert_eq!(call.id(), "provider-call");
        assert_eq!(call.native_id(), Some("provider-call"));
    }
}
