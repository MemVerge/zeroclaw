use super::*;
use zeroclaw_api::model_provider::submission::SubmissionDenied;

struct DenyingProvider(Arc<AtomicUsize>);
impl zeroclaw_api::attribution::Attributable for DenyingProvider {
    fn role(&self) -> zeroclaw_api::attribution::Role {
        use zeroclaw_api::attribution::*;
        Role::Provider(ProviderKind::Model(ModelProviderKind::Custom))
    }
    fn alias(&self) -> &str {
        "local-denial-fixture"
    }
}
#[async_trait]
impl ModelProvider for DenyingProvider {
    async fn chat_with_system(
        &self,
        _: Option<&str>,
        _: &str,
        _: &str,
        _: Option<f64>,
    ) -> anyhow::Result<String> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err(SubmissionDenied::PermitUnavailable.into())
    }
}

enum CallPath {
    System,
    History,
    Tools,
    Structured,
}

async fn assert_no_retry_or_fallback(path: CallPath) {
    let primary = Arc::new(AtomicUsize::new(0));
    let fallback = Arc::new(AtomicUsize::new(0));
    let provider = ReliableModelProvider::new(
        "fixture",
        vec![
            ("primary".into(), Box::new(DenyingProvider(primary.clone()))),
            (
                "fallback".into(),
                Box::new(DenyingProvider(fallback.clone())),
            ),
        ],
        2,
        50,
    )
    .with_model_fallbacks(HashMap::from([(
        "fixture".into(),
        vec!["alternate".into()],
    )]));
    let messages = [ChatMessage::user("synthetic question")];
    let result = match path {
        CallPath::System => provider
            .chat_with_system(None, "synthetic question", "fixture", None)
            .await
            .map(|_| ()),
        CallPath::History => provider
            .chat_with_history(&messages, "fixture", None)
            .await
            .map(|_| ()),
        CallPath::Tools => provider
            .chat_with_tools(&messages, &[], "fixture", None)
            .await
            .map(|_| ()),
        CallPath::Structured => provider
            .chat(
                ChatRequest {
                    messages: &messages,
                    tools: None,
                    thinking: None,
                },
                "fixture",
                None,
            )
            .await
            .map(|_| ()),
    };
    assert!(is_local_submission_denied(&result.unwrap_err()));
    assert_eq!(primary.load(Ordering::SeqCst), 1);
    assert_eq!(fallback.load(Ordering::SeqCst), 0);
}

#[tokio::test(start_paused = true)]
async fn local_denial_in_system_call_never_retries_or_switches_models() {
    assert_no_retry_or_fallback(CallPath::System).await;
}
#[tokio::test(start_paused = true)]
async fn local_denial_in_history_call_never_retries_or_switches_models() {
    assert_no_retry_or_fallback(CallPath::History).await;
}
#[tokio::test(start_paused = true)]
async fn local_denial_in_tools_call_never_retries_or_switches_models() {
    assert_no_retry_or_fallback(CallPath::Tools).await;
}
#[tokio::test(start_paused = true)]
async fn local_denial_in_structured_call_never_retries_or_switches_models() {
    assert_no_retry_or_fallback(CallPath::Structured).await;
}

#[test]
fn a_local_denial_cannot_be_overridden_by_a_broad_retry_extension() {
    let strategy = StructuredStreamRetryStrategy {
        extension: Some(Arc::new(|_: &StreamError| true)),
        ..Default::default()
    };
    assert!(!strategy.should_retry(&StreamError::SubmissionDenied(
        SubmissionDenied::PermitUnavailable
    )));
}

#[test]
fn local_denial_does_not_trigger_context_truncation_or_transport_recovery() {
    let error = StreamError::SubmissionDenied(SubmissionDenied::BindingChanged);
    assert!(!is_context_window_stream_error(&error));
    assert!(!StructuredStreamRetryPolicy::BroadTransient.should_retry(&error));
    assert!(
        !StructuredStreamRetryPolicy::ConnectionErrorsAndRetryableStatuses.should_retry(&error)
    );
}

#[test]
fn typed_local_denial_survives_an_anyhow_stream_error_wrapper() {
    let error = anyhow::Error::new(StreamError::SubmissionDenied(SubmissionDenied::WorkLimit))
        .context("outer caller");
    assert!(is_local_submission_denied(&error));
    assert!(is_non_retryable(&error));
}
