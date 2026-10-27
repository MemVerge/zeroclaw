use super::*;
use crate::observability::NoopObserver;
use serde_json::{Value, json};
use std::sync::Mutex;
use zeroclaw_api::tool::{ToolExecutionContext, ToolResult};

#[derive(Clone, Debug, Eq, PartialEq)]
struct CapturedCall {
    turn: String,
    call: String,
    native: Option<String>,
    args: Value,
}

struct ContextTool(Arc<Mutex<Vec<CapturedCall>>>);
zeroclaw_api::mock_tool_attribution!(ContextTool);

#[async_trait::async_trait]
impl Tool for ContextTool {
    fn name(&self) -> &str {
        "synthetic_context"
    }
    fn description(&self) -> &str {
        "Synthetic dispatcher context fixture"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type":"object"})
    }
    async fn execute(&self, _args: Value) -> Result<ToolResult> {
        anyhow::bail!("dispatcher bypassed context-aware execution")
    }
    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext<'_>,
    ) -> Result<ToolResult> {
        tokio::task::yield_now().await;
        self.0.lock().unwrap().push(CapturedCall {
            turn: context.turn_id.into(),
            call: context.call.id().into(),
            native: context.call.native_id().map(str::to_string),
            args,
        });
        Ok(ToolResult::ok("synthetic result"))
    }
}

fn meta() -> TurnMeta<'static> {
    TurnMeta {
        parent_agent_alias: None,
        agent_alias: None,
        turn_id: "synthetic-turn",
        channel_name: "test",
    }
}

fn dispatch(tools: &[Box<dyn Tool>]) -> ToolDispatchContext<'_> {
    ToolDispatchContext {
        tools_registry: tools,
        activated_tools: None,
        excluded_tools: &[],
        model_switch_callback: None,
    }
}

struct Invocation<'a> {
    call: Option<&'a str>,
    dispatch: ToolDispatchContext<'a>,
    events: Option<&'a Sender<TurnEvent>>,
    cancel: Option<&'a CancellationToken>,
}

async fn invoke(input: Invocation<'_>) -> Result<ToolExecutionOutcome> {
    execute_one_tool(
        "synthetic_context",
        json!({"value":"same"}),
        input.call,
        input.dispatch,
        &meta(),
        &NoopObserver,
        input.cancel,
        None,
        input.events,
    )
    .await
}

struct Fixture {
    tools: Vec<Box<dyn Tool>>,
    calls: Arc<Mutex<Vec<CapturedCall>>>,
}

fn fixture() -> Fixture {
    let calls = Arc::new(Mutex::new(Vec::new()));
    Fixture {
        tools: vec![Box::new(ContextTool(Arc::clone(&calls)))],
        calls,
    }
}

#[tokio::test]
async fn native_context_matches_pending_and_terminal_event_ids() {
    let Fixture { tools, calls } = fixture();
    let (sender, mut receiver) = tokio::sync::mpsc::channel(4);
    let result = invoke(Invocation {
        call: Some("native-call"),
        dispatch: dispatch(&tools),
        events: Some(&sender),
        cancel: None,
    })
    .await
    .unwrap();
    assert!(result.success);
    assert_eq!(calls.lock().unwrap()[0].call, "native-call");
    assert_eq!(calls.lock().unwrap()[0].turn, "synthetic-turn");
    assert_eq!(
        calls.lock().unwrap()[0].native.as_deref(),
        Some("native-call")
    );
    assert_events(&mut receiver, "native-call");
}

fn assert_events(receiver: &mut tokio::sync::mpsc::Receiver<TurnEvent>, expected: &str) {
    assert!(
        matches!(receiver.try_recv().unwrap(), TurnEvent::ToolCall { id, .. } if id == expected)
    );
    assert!(
        matches!(receiver.try_recv().unwrap(), TurnEvent::ToolResult { id, .. } if id == expected)
    );
}

#[tokio::test]
async fn idless_context_reuses_the_existing_event_fallback_not_another_id() {
    let Fixture { tools, calls } = fixture();
    let (sender, mut receiver) = tokio::sync::mpsc::channel(4);
    invoke(Invocation {
        call: None,
        dispatch: dispatch(&tools),
        events: Some(&sender),
        cancel: None,
    })
    .await
    .unwrap();
    let id = calls.lock().unwrap()[0].call.clone();
    assert!(!id.is_empty());
    assert_eq!(calls.lock().unwrap()[0].native, None);
    assert_events(&mut receiver, &id);
}

#[tokio::test]
async fn concurrent_same_tool_and_arguments_do_not_share_a_context_identity() {
    let Fixture { tools, calls } = fixture();
    let first = invoke(Invocation {
        call: Some("call-a"),
        dispatch: dispatch(&tools),
        events: None,
        cancel: None,
    });
    let second = invoke(Invocation {
        call: Some("call-b"),
        dispatch: dispatch(&tools),
        events: None,
        cancel: None,
    });
    let (first, second) = tokio::join!(first, second);
    assert!(first.unwrap().success && second.unwrap().success);
    let mut calls = calls.lock().unwrap().clone();
    calls.sort_by(|left, right| left.call.cmp(&right.call));
    assert_eq!(
        calls
            .iter()
            .map(|call| call.call.as_str())
            .collect::<Vec<_>>(),
        ["call-a", "call-b"]
    );
    assert_eq!(calls[0].args, calls[1].args);
}

#[tokio::test]
async fn cancelled_dispatch_does_not_mint_a_successful_tool_execution() {
    let Fixture { tools, calls } = fixture();
    let cancel = CancellationToken::new();
    cancel.cancel();
    let outcome = invoke(Invocation {
        call: Some("cancelled"),
        dispatch: dispatch(&tools),
        events: None,
        cancel: Some(&cancel),
    })
    .await;
    assert!(outcome.is_err());
    assert!(calls.lock().unwrap().is_empty());
}
