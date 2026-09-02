//! The max-iteration / host-requested exit: ask the LLM for a tools-free
//! final summary (with step timeout + cancel select) and return it appended
//! to the accumulated display text, or bail.

use super::knobs::{LoopKnobs, MaxIterationBehavior};
use super::outcome::ToolLoopCancelled;
use super::protocol_detect::detect_internal_protocol_without_tools;
use anyhow::Result;
use std::time::Duration;
use tokio::sync::mpsc::Sender;
use tokio_util::sync::CancellationToken;
use zeroclaw_api::GracefulStopReason;
use zeroclaw_api::agent::TurnEvent;
use zeroclaw_config::schema::PacingConfig;
use zeroclaw_providers::{ChatMessage, ChatResponse, ModelProvider};

/// ~8k tokens of latin text; truncates pathological finalizer dumps.
const FINALIZER_OUTPUT_CHAR_CAP: usize = 32_000;

pub(crate) struct GracefulFinishInput<'a> {
    pub model_provider: &'a dyn ModelProvider,
    pub history: &'a mut Vec<ChatMessage>,
    pub provider_name: &'a str,
    pub model: &'a str,
    pub temperature: Option<f64>,
    pub pacing: &'a PacingConfig,
    pub cancellation_token: Option<&'a CancellationToken>,
    pub max_iterations: usize,
    pub accumulated_display_text: String,
    pub turn_id: &'a str,
    pub knobs: &'a LoopKnobs,
    pub new_messages_out: Option<&'a mut Vec<ChatMessage>>,
    pub reason: GracefulStopReason,
    pub event_tx: Option<&'a Sender<TurnEvent>>,
}

pub(crate) fn requested_graceful_stop() -> Option<GracefulStopReason> {
    zeroclaw_api::GRACEFUL_STOP
        .try_with(|slot| slot.as_ref().and_then(|signal| signal.requested()))
        .ok()
        .flatten()
}

pub(crate) async fn finish_after_max_iterations(
    mut input: GracefulFinishInput<'_>,
) -> Result<String> {
    if input
        .cancellation_token
        .is_some_and(CancellationToken::is_cancelled)
    {
        return Err(ToolLoopCancelled.into());
    }
    if should_error_at_cap(input.knobs, input.reason) {
        anyhow::bail!(
            "{}",
            finalizer_failure_message(input.reason, input.max_iterations)
        );
    }
    log_finalizer_start(&input);
    sanitize_orphaned_tool_history(input.history);
    let prompt = ChatMessage::user(finalizer_prompt(input.reason));
    let prompt_mirror = prompt.clone();
    input.history.push(prompt);
    match await_finalizer_response(&input).await {
        SummaryCall::Cancelled => {
            input.history.pop();
            Err(ToolLoopCancelled.into())
        }
        SummaryCall::TimedOut(step_secs) => {
            input.history.pop();
            anyhow::bail!("Final summary LLM call timed out after {step_secs}s (step_timeout_secs)")
        }
        SummaryCall::Done(result) => {
            // `biased select!` does not re-check cancel if `chat()` cancelled
            // the token during its own poll and then returned Ready.
            abort_if_cancelled(input.cancellation_token, input.history)?;
            match result {
                Err(error) => fail_finalizer_provider_error(&mut input, error),
                Ok(resp) => commit_finalizer_response(input, prompt_mirror, resp).await,
            }
        }
    }
}

fn abort_if_cancelled(
    token: Option<&CancellationToken>,
    history: &mut Vec<ChatMessage>,
) -> Result<()> {
    if token.is_some_and(CancellationToken::is_cancelled) {
        history.pop();
        return Err(ToolLoopCancelled.into());
    }
    Ok(())
}

fn should_error_at_cap(knobs: &LoopKnobs, reason: GracefulStopReason) -> bool {
    reason == GracefulStopReason::MaxIterations
        && knobs.max_iteration_behavior == MaxIterationBehavior::ErrorAtCap
}

fn finalizer_failure_message(reason: GracefulStopReason, max_iterations: usize) -> String {
    match reason {
        GracefulStopReason::MaxIterations => {
            format!("Agent exceeded maximum tool iterations ({max_iterations})")
        }
        GracefulStopReason::NoProgress => "Agent stopped: retrieval made no progress".to_string(),
    }
}

fn log_finalizer_start(input: &GracefulFinishInput<'_>) {
    let attrs = ::serde_json::json!({
        "model": input.model,
        "max_iterations": input.max_iterations,
        "trace_id": input.turn_id,
        "reason": format!("{:?}", input.reason),
    });
    match input.reason {
        GracefulStopReason::MaxIterations => {
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                    .with_category(::zeroclaw_log::EventCategory::Agent)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(attrs.clone()),
                "tool_loop_exhausted"
            );
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                    .with_category(::zeroclaw_log::EventCategory::Agent)
                    .with_outcome(::zeroclaw_log::EventOutcome::Unknown)
                    .with_attrs(attrs),
                "Max iterations reached, requesting final summary"
            );
        }
        GracefulStopReason::NoProgress => {
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                    .with_category(::zeroclaw_log::EventCategory::Agent)
                    .with_outcome(::zeroclaw_log::EventOutcome::Unknown)
                    .with_attrs(attrs),
                "No progress detected, requesting final summary"
            );
        }
    }
}

fn sanitize_orphaned_tool_history(history: &mut Vec<ChatMessage>) {
    let tool_calls_stripped =
        crate::agent::history_pruner::strip_orphaned_tool_calls_from_assistants(history);
    let tool_messages_removed =
        crate::agent::history_pruner::remove_orphaned_tool_messages(history).removed;
    if tool_calls_stripped == 0 && tool_messages_removed == 0 {
        return;
    }
    ::zeroclaw_log::record!(
        WARN,
        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
            .with_outcome(::zeroclaw_log::EventOutcome::Unknown)
            .with_attrs(::serde_json::json!({
                "tool_calls_stripped": tool_calls_stripped,
                "tool_messages_removed": tool_messages_removed,
            })),
        "Sanitised orphaned tool_use/tool_result pairing before graceful shutdown"
    );
}

fn finalizer_prompt(reason: GracefulStopReason) -> String {
    match reason {
        GracefulStopReason::MaxIterations => {
            crate::i18n::get_required_cli_string("turn-max-iterations-finalizer-prompt")
        }
        GracefulStopReason::NoProgress => {
            crate::i18n::get_required_cli_string("turn-no-progress-finalizer-prompt")
        }
    }
}

enum SummaryCall {
    Cancelled,
    TimedOut(u64),
    Done(Result<ChatResponse>),
}

async fn await_finalizer_response(input: &GracefulFinishInput<'_>) -> SummaryCall {
    let summary_request = zeroclaw_providers::ChatRequest {
        messages: input.history,
        tools: None,
        thinking: zeroclaw_api::NATIVE_THINKING_OVERRIDE
            .try_with(Clone::clone)
            .ok()
            .flatten(),
    };
    let access = crate::agent::turn::execution::ResolvedModelAccess {
        model_provider: input.model_provider,
        provider_name: input.provider_name,
        model: input.model,
        temperature: input.temperature,
    };
    let summary_future = access.run_model_query(summary_request);
    match input.pacing.step_timeout_secs {
        Some(step_secs) if step_secs > 0 => {
            await_with_timeout(summary_future, step_secs, input.cancellation_token).await
        }
        _ => await_without_timeout(summary_future, input.cancellation_token).await,
    }
}

async fn await_with_timeout<F>(
    summary_future: F,
    step_secs: u64,
    cancellation_token: Option<&CancellationToken>,
) -> SummaryCall
where
    F: std::future::Future<Output = Result<ChatResponse>>,
{
    let step_timeout = Duration::from_secs(step_secs);
    if let Some(token) = cancellation_token {
        tokio::select! {
            biased;
            () = token.cancelled() => SummaryCall::Cancelled,
            result = tokio::time::timeout(step_timeout, summary_future) => match result {
                Ok(inner) => SummaryCall::Done(inner),
                Err(_) => SummaryCall::TimedOut(step_secs),
            },
        }
    } else {
        match tokio::time::timeout(step_timeout, summary_future).await {
            Ok(inner) => SummaryCall::Done(inner),
            Err(_) => SummaryCall::TimedOut(step_secs),
        }
    }
}

async fn await_without_timeout<F>(
    summary_future: F,
    cancellation_token: Option<&CancellationToken>,
) -> SummaryCall
where
    F: std::future::Future<Output = Result<ChatResponse>>,
{
    if let Some(token) = cancellation_token {
        tokio::select! {
            biased;
            () = token.cancelled() => SummaryCall::Cancelled,
            result = summary_future => SummaryCall::Done(result),
        }
    } else {
        SummaryCall::Done(summary_future.await)
    }
}

fn is_budget_exceeded(error: &anyhow::Error) -> bool {
    error.to_string().contains("Budget exceeded:")
}

fn fail_finalizer_provider_error(
    input: &mut GracefulFinishInput<'_>,
    error: anyhow::Error,
) -> Result<String> {
    input.history.pop();
    if is_budget_exceeded(&error) {
        return Err(error);
    }
    log_finalizer_provider_error(input, &error);
    anyhow::bail!(
        "{}",
        finalizer_failure_message(input.reason, input.max_iterations)
    )
}

fn log_finalizer_provider_error(input: &GracefulFinishInput<'_>, error: &anyhow::Error) {
    ::zeroclaw_log::record!(
        ERROR,
        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
            .with_category(::zeroclaw_log::EventCategory::Provider)
            .with_outcome(::zeroclaw_log::EventOutcome::Failure)
            .with_attrs(::serde_json::json!({
                "model": input.model,
                "provider": input.provider_name,
                "max_iterations": input.max_iterations,
                "trace_id": input.turn_id,
                "error": format!("{error}"),
            })),
        "final summary LLM call failed after iteration exhaustion; bailing"
    );
}

fn finalizer_text_is_usable(knobs: &LoopKnobs, resp: &ChatResponse) -> bool {
    if !resp.tool_calls.is_empty() {
        return false;
    }
    let text = resp.text.as_deref().unwrap_or_default();
    if text.trim().is_empty() {
        return false;
    }
    if knobs.detect_protocol_without_tools && detect_internal_protocol_without_tools(text).is_some()
    {
        return false;
    }
    true
}

fn cap_finalizer_text(text: String) -> String {
    if text.chars().count() <= FINALIZER_OUTPUT_CHAR_CAP {
        return text;
    }
    let mut capped: String = text.chars().take(FINALIZER_OUTPUT_CHAR_CAP).collect();
    capped.push('…');
    capped
}

async fn commit_finalizer_response(
    input: GracefulFinishInput<'_>,
    prompt_mirror: ChatMessage,
    resp: ChatResponse,
) -> Result<String> {
    if !finalizer_text_is_usable(input.knobs, &resp) {
        input.history.pop();
        anyhow::bail!(
            "{}",
            finalizer_failure_message(input.reason, input.max_iterations)
        );
    }
    let text = cap_finalizer_text(resp.text.unwrap_or_default());
    let summary_msg = ChatMessage::assistant(text.clone());
    if let Some(out) = input.new_messages_out {
        out.push(prompt_mirror);
        out.push(summary_msg.clone());
    }
    input.history.push(summary_msg);
    super::events::emit_posthoc_turn_chunk(input.event_tx, &text).await;
    Ok(compose_finalizer_display(
        input.accumulated_display_text,
        &text,
        input.reason,
        input.max_iterations,
    ))
}

fn compose_finalizer_display(
    mut accumulated: String,
    text: &str,
    reason: GracefulStopReason,
    max_iterations: usize,
) -> String {
    accumulated.push_str(text);
    if reason != GracefulStopReason::MaxIterations {
        return accumulated;
    }
    accumulated.push_str("\n\n");
    accumulated.push_str(&crate::i18n::get_required_cli_string_with_args(
        "turn-max-iterations-reached",
        &[("max_iterations", &max_iterations.to_string())],
    ));
    accumulated
}

#[cfg(test)]
mod graceful_summary_metering_tests {
    use super::{GracefulFinishInput, finish_after_max_iterations};
    use crate::agent::cost::{TOOL_LOOP_COST_TRACKING_CONTEXT, ToolLoopCostTrackingContext};
    use crate::agent::turn::LoopKnobs;
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use zeroclaw_api::GracefulStopReason;
    use zeroclaw_api::attribution::{Attributable, ModelProviderKind, ProviderKind, Role};
    use zeroclaw_api::model_provider::{ChatRequest, ChatResponse};
    use zeroclaw_config::schema::{CostConfig, PacingConfig};
    use zeroclaw_providers::traits::TokenUsage;
    use zeroclaw_providers::{ChatMessage, ModelProvider};

    /// Provider stub that counts calls and returns a summary WITH token usage.
    struct CountingUsageProvider {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ModelProvider for CountingUsageProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<String> {
            Ok("wrap-up summary".to_string())
        }

        async fn chat(
            &self,
            _request: ChatRequest<'_>,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<ChatResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ChatResponse {
                text: Some("wrap-up summary".to_string()),
                tool_calls: Vec::new(),
                usage: Some(TokenUsage {
                    input_tokens: Some(100),
                    output_tokens: Some(20),
                    cached_input_tokens: None,
                }),
                reasoning_content: None,
            })
        }
    }

    impl Attributable for CountingUsageProvider {
        fn role(&self) -> Role {
            Role::Provider(ProviderKind::Model(ModelProviderKind::Custom))
        }
        fn alias(&self) -> &str {
            "counting-usage-provider"
        }
    }

    async fn run_summary(provider: &CountingUsageProvider) -> anyhow::Result<String> {
        let mut history = vec![ChatMessage::user("do the work")];
        let pacing = PacingConfig::default();
        let knobs = LoopKnobs::default(); // GracefulSummary
        finish_after_max_iterations(GracefulFinishInput {
            model_provider: provider,
            history: &mut history,
            provider_name: "custom",
            model: "test-model",
            temperature: None,
            pacing: &pacing,
            cancellation_token: None,
            max_iterations: 2,
            accumulated_display_text: String::new(),
            turn_id: "trace-req-test",
            knobs: &knobs,
            new_messages_out: None,
            reason: GracefulStopReason::MaxIterations,
            event_tx: None,
        })
        .await
    }

    // The graceful summary now routes through the metered provider seam: under a
    // cost-tracking scope its token usage is recorded (before this change the
    // summary recorded nothing).
    #[tokio::test]
    async fn graceful_summary_records_usage_through_the_metered_seam() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = CountingUsageProvider {
            calls: Arc::clone(&calls),
        };
        let ctx = ToolLoopCostTrackingContext::usage_only();
        let turn_usage = Arc::clone(&ctx.turn_usage);

        let out = TOOL_LOOP_COST_TRACKING_CONTEXT
            .scope(Some(ctx), async { run_summary(&provider).await })
            .await
            .expect("graceful summary should succeed");

        assert!(out.contains("wrap-up summary"), "unexpected summary: {out}");
        // The returned display text must carry both the summary and the visible
        // stop reason — deleting the stop-reason append would leave this green
        // on `wrap-up summary` alone, so the stop-reason assertion pins the
        // user-observed contract.
        assert!(
            out.contains("Turn stopped: reached maximum tool iterations (2)"),
            "stop reason with iteration count must reach returned output: {out}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1, "provider called once");
        let recorded = *turn_usage.lock();
        assert_eq!(recorded.input_tokens, 100);
        assert_eq!(recorded.output_tokens, 20);
    }

    // The graceful summary now fails closed on budget exhaustion: it was the one
    // tool-loop provider call that skipped the budget check. A tripped budget
    // (negative limit) makes the seam bail BEFORE spending, so the provider is
    // never called and the cap is surfaced as an error.
    #[tokio::test]
    async fn graceful_summary_is_budget_gated_and_skips_the_provider_when_over_budget() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = CountingUsageProvider {
            calls: Arc::clone(&calls),
        };
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = CostConfig {
            enabled: true,
            daily_limit_usd: -1.0,
            monthly_limit_usd: -1.0,
            ..CostConfig::default()
        };
        let tracker = Arc::new(crate::cost::CostTracker::new(cfg, tmp.path()).unwrap());
        let ctx = ToolLoopCostTrackingContext::new(tracker, Arc::new(HashMap::new()));

        let result = TOOL_LOOP_COST_TRACKING_CONTEXT
            .scope(Some(ctx), async { run_summary(&provider).await })
            .await;

        let err = result.expect_err("over-budget summary must bail, not spend");
        let err = err.to_string();
        assert!(
            err.contains("Budget exceeded"),
            "budget errors must reach the host unchanged: {err}"
        );
        assert!(
            !err.contains("maximum tool iterations"),
            "must not rewrite a budget failure as max-iter: {err}"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "budget gate must fire before the provider call"
        );
    }

    /// Provider stub that records the exact messages it was dispatched, so a
    /// test can assert on what actually reached the provider.
    struct CapturingProvider {
        seen: Arc<std::sync::Mutex<Vec<String>>>,
        tools_none: Arc<std::sync::atomic::AtomicBool>,
    }

    #[async_trait]
    impl ModelProvider for CapturingProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<String> {
            Ok(String::new())
        }

        async fn chat(
            &self,
            request: ChatRequest<'_>,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<ChatResponse> {
            self.tools_none
                .store(request.tools.is_none(), Ordering::SeqCst);
            let joined = request
                .messages
                .iter()
                .map(|m| m.content.clone())
                .collect::<Vec<_>>()
                .join("\n");
            self.seen.lock().unwrap().push(joined);
            Ok(ChatResponse {
                text: Some("wrap-up summary".to_string()),
                tool_calls: Vec::new(),
                usage: None,
                reasoning_content: None,
            })
        }
    }

    impl Attributable for CapturingProvider {
        fn role(&self) -> Role {
            Role::Provider(ProviderKind::Model(ModelProviderKind::Custom))
        }
        fn alias(&self) -> &str {
            "capturing-provider"
        }
    }

    // The graceful-summary path dispatches the accumulated history directly
    // through `run_model_query`, which does NOT run
    // `prepare_messages_for_provider`. A tool-result `[AUDIO:/path]` in that
    // history must be stripped before it reaches the provider, or the raw
    // filesystem path leaks and is hallucinated over on the max-iteration exit.
    #[tokio::test]
    async fn graceful_summary_strips_tool_audio_marker_before_dispatch() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let tools_none = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let provider = CapturingProvider {
            seen: Arc::clone(&seen),
            tools_none: Arc::clone(&tools_none),
        };
        // A properly paired assistant tool_call + native tool-result JSON blob,
        // so the orphaned-tool-message sweep in finish_after_max_iterations keeps
        // the exchange intact and the audio marker survives to dispatch. This
        // also exercises stripping a marker embedded inside a tool-result JSON
        // object (the native-dispatcher shape), not just plain text.
        let mut history = vec![
            ChatMessage::user("call the tool and tell me what you hear"),
            ChatMessage::assistant(r#"{"tool_calls":[{"id":"toolu_1"}]}"#),
            ChatMessage::tool(
                r#"{"content":"[AUDIO:/tmp/clip.wav] recorded 3:00 PM","tool_call_id":"toolu_1"}"#,
            ),
        ];
        let pacing = PacingConfig::default();
        let knobs = LoopKnobs::default();

        let out = finish_after_max_iterations(GracefulFinishInput {
            model_provider: &provider,
            history: &mut history,
            provider_name: "custom",
            model: "test-model",
            temperature: None,
            pacing: &pacing,
            cancellation_token: None,
            max_iterations: 2,
            accumulated_display_text: String::new(),
            turn_id: "trace-req-audio",
            knobs: &knobs,
            new_messages_out: None,
            reason: GracefulStopReason::MaxIterations,
            event_tx: None,
        })
        .await
        .expect("graceful summary should succeed");

        assert!(out.contains("wrap-up summary"), "unexpected summary: {out}");
        let captured = seen.lock().unwrap().join("\n");
        assert!(
            !captured.contains("/tmp/clip.wav"),
            "raw audio path reached the provider on the max-iteration path: {captured}"
        );
        assert!(
            captured.contains("[media attachment]"),
            "audio marker should be replaced with a placeholder: {captured}"
        );
        assert!(
            tools_none.load(Ordering::SeqCst),
            "finalizer must dispatch with tools=None"
        );
    }

    struct ScriptedProvider {
        text: Option<String>,
        tool_calls: Vec<zeroclaw_api::model_provider::ToolCall>,
        seen_tools_none: Arc<AtomicBool>,
        seen_prompt: Arc<std::sync::Mutex<String>>,
    }

    #[async_trait]
    impl ModelProvider for ScriptedProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<String> {
            Ok(String::new())
        }

        async fn chat(
            &self,
            request: ChatRequest<'_>,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<ChatResponse> {
            self.seen_tools_none
                .store(request.tools.is_none(), Ordering::SeqCst);
            let last = request
                .messages
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            *self.seen_prompt.lock().unwrap() = last;
            Ok(ChatResponse {
                text: self.text.clone(),
                tool_calls: self.tool_calls.clone(),
                usage: None,
                reasoning_content: None,
            })
        }
    }

    impl Attributable for ScriptedProvider {
        fn role(&self) -> Role {
            Role::Provider(ProviderKind::Model(ModelProviderKind::Custom))
        }
        fn alias(&self) -> &str {
            "scripted-provider"
        }
    }

    async fn run_scripted(
        provider: &ScriptedProvider,
        reason: GracefulStopReason,
        cancel: Option<&tokio_util::sync::CancellationToken>,
    ) -> anyhow::Result<String> {
        let mut history = vec![ChatMessage::user("do the work")];
        let pacing = PacingConfig::default();
        let knobs = LoopKnobs::default();
        finish_after_max_iterations(GracefulFinishInput {
            model_provider: provider,
            history: &mut history,
            provider_name: "custom",
            model: "test-model",
            temperature: None,
            pacing: &pacing,
            cancellation_token: cancel,
            max_iterations: 2,
            accumulated_display_text: String::new(),
            turn_id: "trace-scripted",
            knobs: &knobs,
            new_messages_out: None,
            reason,
            event_tx: None,
        })
        .await
    }

    #[tokio::test]
    async fn no_progress_finalizer_uses_prompt_and_tools_none() {
        let provider = ScriptedProvider {
            text: Some("here is the answer".to_string()),
            tool_calls: Vec::new(),
            seen_tools_none: Arc::new(AtomicBool::new(false)),
            seen_prompt: Arc::new(std::sync::Mutex::new(String::new())),
        };
        let tools_none = Arc::clone(&provider.seen_tools_none);
        let prompt = Arc::clone(&provider.seen_prompt);
        let out = run_scripted(&provider, GracefulStopReason::NoProgress, None)
            .await
            .expect("no-progress finalizer should succeed");
        assert!(out.contains("here is the answer"), "{out}");
        assert!(
            !out.contains("maximum tool iterations"),
            "no-progress must not append the max-iter footer: {out}"
        );
        assert!(tools_none.load(Ordering::SeqCst));
        let prompt = prompt.lock().unwrap().clone();
        assert!(
            prompt.contains("repeated searches") || prompt.contains("evidence already gathered"),
            "expected no-progress prompt, got: {prompt}"
        );
    }

    #[tokio::test]
    async fn empty_finalizer_text_returns_error() {
        let provider = ScriptedProvider {
            text: Some("   ".to_string()),
            tool_calls: Vec::new(),
            seen_tools_none: Arc::new(AtomicBool::new(false)),
            seen_prompt: Arc::new(std::sync::Mutex::new(String::new())),
        };
        let result = run_scripted(&provider, GracefulStopReason::NoProgress, None).await;
        let err = result.expect_err("whitespace-only summary must fail");
        let err = err.to_string();
        assert!(
            err.contains("no progress"),
            "NoProgress failure must keep its reason: {err}"
        );
        assert!(
            !err.contains("maximum tool iterations"),
            "NoProgress must not reuse the max-iter error: {err}"
        );
    }

    #[tokio::test]
    async fn protocol_shaped_finalizer_text_returns_error() {
        let provider = ScriptedProvider {
            text: Some("<tool_call>{\"name\":\"search_user_knowledge\"}</tool_call>".to_string()),
            tool_calls: Vec::new(),
            seen_tools_none: Arc::new(AtomicBool::new(false)),
            seen_prompt: Arc::new(std::sync::Mutex::new(String::new())),
        };
        let result = run_scripted(&provider, GracefulStopReason::NoProgress, None).await;
        assert!(result.is_err(), "tool-protocol envelope must fail");
    }

    #[tokio::test]
    async fn native_tool_calls_on_finalizer_return_error() {
        let provider = ScriptedProvider {
            text: Some("calling a tool".to_string()),
            tool_calls: vec![zeroclaw_api::model_provider::ToolCall {
                id: "c1".to_string(),
                name: "search_user_knowledge".to_string(),
                arguments: "{}".to_string(),
                extra_content: None,
            }],
            seen_tools_none: Arc::new(AtomicBool::new(false)),
            seen_prompt: Arc::new(std::sync::Mutex::new(String::new())),
        };
        let result = run_scripted(&provider, GracefulStopReason::NoProgress, None).await;
        assert!(
            result.is_err(),
            "native tool_calls with tools=None must fail"
        );
    }

    #[tokio::test]
    async fn cancel_during_finalizer_returns_cancelled() {
        let provider = ScriptedProvider {
            text: Some("late answer".to_string()),
            tool_calls: Vec::new(),
            seen_tools_none: Arc::new(AtomicBool::new(false)),
            seen_prompt: Arc::new(std::sync::Mutex::new(String::new())),
        };
        let token = tokio_util::sync::CancellationToken::new();
        token.cancel();
        let result = run_scripted(&provider, GracefulStopReason::NoProgress, Some(&token)).await;
        assert!(result.is_err());
        assert!(
            crate::agent::turn::is_tool_loop_cancelled(&result.unwrap_err()),
            "pre-cancelled token must surface ToolLoopCancelled"
        );
    }

    /// Provider that cancels the host token during its own `chat()` poll, then
    /// still returns `Ok(text)`. `biased select!` can take that Ready result
    /// without re-checking cancel; the post-await check must still abort.
    struct CancelThenAnswerProvider {
        token: tokio_util::sync::CancellationToken,
    }

    #[async_trait]
    impl ModelProvider for CancelThenAnswerProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<String> {
            Ok(String::new())
        }

        async fn chat(
            &self,
            _request: ChatRequest<'_>,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<ChatResponse> {
            self.token.cancel();
            Ok(ChatResponse {
                text: Some("answer after cancellation".to_string()),
                tool_calls: Vec::new(),
                usage: None,
                reasoning_content: None,
            })
        }
    }

    impl Attributable for CancelThenAnswerProvider {
        fn role(&self) -> Role {
            Role::Provider(ProviderKind::Model(ModelProviderKind::Custom))
        }
        fn alias(&self) -> &str {
            "cancel-then-answer"
        }
    }

    #[tokio::test]
    async fn cancel_during_provider_return_does_not_commit_answer() {
        let token = tokio_util::sync::CancellationToken::new();
        let provider = CancelThenAnswerProvider {
            token: token.clone(),
        };
        let mut history = vec![ChatMessage::user("do the work")];
        let pacing = PacingConfig::default();
        let knobs = LoopKnobs::default();
        let result = finish_after_max_iterations(GracefulFinishInput {
            model_provider: &provider,
            history: &mut history,
            provider_name: "custom",
            model: "test-model",
            temperature: None,
            pacing: &pacing,
            cancellation_token: Some(&token),
            max_iterations: 2,
            accumulated_display_text: String::new(),
            turn_id: "trace-cancel-during-return",
            knobs: &knobs,
            new_messages_out: None,
            reason: GracefulStopReason::NoProgress,
            event_tx: None,
        })
        .await;
        let err = result.expect_err("cancelled finalizer must not keep the late answer");
        assert!(
            crate::agent::turn::is_tool_loop_cancelled(&err),
            "cancel during provider return must surface ToolLoopCancelled, got: {err}"
        );
        assert!(
            !history
                .iter()
                .any(|m| m.content.contains("answer after cancellation")),
            "late answer must not be committed to history: {history:?}"
        );
    }

    #[test]
    fn requested_graceful_stop_sees_already_set_signal() {
        let signal = std::sync::Arc::new(zeroclaw_api::GracefulStopSignal::new());
        signal.request(GracefulStopReason::NoProgress);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(zeroclaw_api::GRACEFUL_STOP.scope(Some(signal), async {
            assert_eq!(
                super::requested_graceful_stop(),
                Some(GracefulStopReason::NoProgress)
            );
        }));
    }
}

#[cfg(test)]
mod graceful_stop_loop_tests {
    use crate::agent::turn::{
        LoopKnobs, ResolvedAgentExecution, ResolvedModelAccess, ToolLoop, run_tool_call_loop,
    };
    use crate::observability::NoopObserver;
    use crate::tools::{Tool, ToolResult};
    use async_trait::async_trait;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::mpsc;
    use zeroclaw_api::agent::TurnEvent;
    use zeroclaw_api::attribution::{Attributable, ModelProviderKind, ProviderKind, Role};
    use zeroclaw_api::ingress::IngressContext;
    use zeroclaw_api::model_provider::{ChatRequest, ChatResponse, ProviderCapabilities, ToolCall};
    use zeroclaw_api::{GracefulStopReason, GracefulStopSignal};
    use zeroclaw_providers::{ChatMessage, ModelProvider};

    const FINALIZER_TEXT: &str = "gathered answer";

    struct WaveThenFinalizerProvider {
        tools_none: Arc<Mutex<Vec<bool>>>,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ModelProvider for WaveThenFinalizerProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                native_tool_calling: true,
                ..ProviderCapabilities::default()
            }
        }

        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<String> {
            anyhow::bail!("chat_with_system unused");
        }

        async fn chat(
            &self,
            request: ChatRequest<'_>,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<ChatResponse> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            self.tools_none
                .lock()
                .expect("tools_none lock")
                .push(request.tools.is_none());
            match n {
                0 => Ok(ChatResponse {
                    text: None,
                    tool_calls: vec![native_call("a", "search_a"), native_call("b", "search_b")],
                    usage: None,
                    reasoning_content: None,
                }),
                1 => Ok(ChatResponse {
                    text: Some(FINALIZER_TEXT.to_string()),
                    tool_calls: Vec::new(),
                    usage: None,
                    reasoning_content: None,
                }),
                _ => anyhow::bail!("unexpected extra provider call {n}"),
            }
        }
    }

    impl Attributable for WaveThenFinalizerProvider {
        fn role(&self) -> Role {
            Role::Provider(ProviderKind::Model(ModelProviderKind::Custom))
        }
        fn alias(&self) -> &str {
            "wave-then-finalizer"
        }
    }

    fn native_call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: "{}".to_string(),
            extra_content: None,
        }
    }

    struct SignalingTool {
        name: &'static str,
        calls: Arc<AtomicUsize>,
        signal: Arc<GracefulStopSignal>,
    }

    zeroclaw_api::tool_attribution!(SignalingTool, ::zeroclaw_api::attribution::ToolKind::Plugin);

    #[async_trait]
    impl Tool for SignalingTool {
        fn name(&self) -> &str {
            self.name
        }
        fn description(&self) -> &str {
            self.name
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<ToolResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.signal.request(GracefulStopReason::NoProgress);
            Ok(ToolResult {
                success: true,
                output: format!("{}-out", self.name).into(),
                error: None,
            })
        }
    }

    struct CountingTool {
        name: &'static str,
        calls: Arc<AtomicUsize>,
    }

    zeroclaw_api::tool_attribution!(CountingTool, ::zeroclaw_api::attribution::ToolKind::Plugin);

    #[async_trait]
    impl Tool for CountingTool {
        fn name(&self) -> &str {
            self.name
        }
        fn description(&self) -> &str {
            self.name
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<ToolResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ToolResult {
                success: true,
                output: format!("{}-out", self.name).into(),
                error: None,
            })
        }
    }

    #[tokio::test]
    async fn no_progress_signal_finishes_current_wave_then_tools_none() {
        let tools_none = Arc::new(Mutex::new(Vec::new()));
        let provider_calls = Arc::new(AtomicUsize::new(0));
        let provider = WaveThenFinalizerProvider {
            tools_none: Arc::clone(&tools_none),
            calls: Arc::clone(&provider_calls),
        };
        let signal = Arc::new(GracefulStopSignal::new());
        let search_a_calls = Arc::new(AtomicUsize::new(0));
        let search_b_calls = Arc::new(AtomicUsize::new(0));
        let tools_registry: Vec<Box<dyn Tool>> = vec![
            Box::new(SignalingTool {
                name: "search_a",
                calls: Arc::clone(&search_a_calls),
                signal: Arc::clone(&signal),
            }),
            Box::new(CountingTool {
                name: "search_b",
                calls: Arc::clone(&search_b_calls),
            }),
        ];
        let mut history = vec![ChatMessage::user("research this")];
        let observer = NoopObserver;
        let turn_id = "graceful-loop-boundary";
        let (event_tx, mut event_rx) = mpsc::channel(16);
        let approval_cfg = zeroclaw_config::schema::RiskProfileConfig {
            level: crate::security::AutonomyLevel::Full,
            ..zeroclaw_config::schema::RiskProfileConfig::default()
        };
        let approval_mgr = crate::approval::ApprovalManager::from_risk_profile(&approval_cfg);

        let result = zeroclaw_api::GRACEFUL_STOP
            .scope(
                Some(signal),
                run_tool_call_loop(ToolLoop {
                    parent_agent_alias: None,
                    sop_reassembly: None,
                    exec: ResolvedAgentExecution {
                        model_access: ResolvedModelAccess {
                            model_provider: &provider,
                            provider_name: "mock-provider",
                            model: "mock-model",
                            temperature: Some(0.0),
                        },
                        tools_registry: &tools_registry,
                        observer: &observer,
                        silent: true,
                        approval: Some(&approval_mgr),
                        multimodal_config: &zeroclaw_config::schema::MultimodalConfig::default(),
                        config: None,
                        max_tool_iterations: 8,
                        hooks: None,
                        excluded_tools: &[],
                        dedup_exempt_tools: &[],
                        activated_tools: None,
                        model_switch_callback: None,
                        pacing: &zeroclaw_config::schema::PacingConfig::default(),
                        strict_tool_parsing: false,
                        parallel_tools: true,
                        max_tool_result_chars: 0,
                        context_token_budget: 0,
                        receipt_generator: None,
                        knobs: &LoopKnobs::default(),
                    },
                    history: &mut history,
                    channel_name: "cli",
                    channel_reply_target: None,
                    cancellation_token: None,
                    on_delta: None,
                    shared_budget: None,
                    channel: None,
                    collected_receipts: None,
                    event_tx: Some(event_tx),
                    steering: None,
                    new_messages_out: None,
                    image_cache: None,
                    memory: None,
                    ingress: IngressContext::sub_turn(),
                    agent_alias: None,
                    turn_id,
                }),
            )
            .await
            .expect("graceful no-progress loop should return a finalizer answer");

        assert_eq!(search_a_calls.load(Ordering::SeqCst), 1);
        assert_eq!(search_b_calls.load(Ordering::SeqCst), 1);
        assert_eq!(provider_calls.load(Ordering::SeqCst), 2);
        let flags = tools_none.lock().expect("tools_none lock").clone();
        assert_eq!(flags, vec![false, true], "wave then tools=None: {flags:?}");
        assert!(
            result.contains(FINALIZER_TEXT),
            "returned text must include the finalizer answer: {result}"
        );
        assert!(
            !result.contains("maximum tool iterations"),
            "NoProgress must not append the max-iter footer: {result}"
        );
        let mut emitted = String::new();
        while let Ok(event) = event_rx.try_recv() {
            if let TurnEvent::Chunk { delta } = event {
                emitted.push_str(&delta);
            }
        }
        assert_eq!(emitted, FINALIZER_TEXT);
    }
}

#[cfg(test)]
mod i18n_message_tests {
    /// The graceful max-iteration shutdown must include the iteration count in
    /// the user-visible message so the operator knows why the agent stopped.
    #[test]
    fn max_iterations_message_includes_count() {
        let msg = crate::i18n::get_required_cli_string_with_args(
            "turn-max-iterations-reached",
            &[("max_iterations", "42")],
        );
        assert!(
            msg.contains("42"),
            "message should contain iteration count: {msg}"
        );
        assert!(
            msg.contains("maximum tool iterations"),
            "message should describe the limit: {msg}"
        );
    }
}
