//! Ties a spawned streaming-parser task's lifetime to the stream the consumer
//! holds, so dropping the stream (turn cancel, timeout, client disconnect)
//! aborts the task and releases its socket instead of leaking it.

use zeroclaw_api::StopReason;
use zeroclaw_api::model_provider::{StreamError, StreamEvent, StreamResult};

/// Aborts the wrapped task when dropped. Carry it inside the returned stream's
/// `unfold` state so the abort fires exactly when the consumer drops the
/// stream. `AbortHandle::abort` is a no-op once the task has finished, so the
/// happy path is unaffected.
pub(crate) struct AbortOnDrop(tokio::task::AbortHandle);

impl AbortOnDrop {
    pub(crate) fn new(handle: tokio::task::AbortHandle) -> Self {
        Self(handle)
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if self.0.is_finished() {
            return;
        }
        self.0.abort();
        ::zeroclaw_log::record!(
            DEBUG,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Kill)
                .with_category(::zeroclaw_log::EventCategory::Provider)
                .with_outcome(::zeroclaw_log::EventOutcome::Success),
            "stream: consumer dropped — aborting detached parser task to release socket"
        );
    }
}

pub(crate) struct SseFinish<'a> {
    pub tx: &'a tokio::sync::mpsc::Sender<StreamResult<StreamEvent>>,
    pub stop: Option<StopReason>,
    pub completion_signal: &'a str,
}

pub(crate) async fn finish_sse_stream(finish: SseFinish<'_>) {
    let Some(stop) = finish.stop else {
        emit_truncation_error(finish.tx, finish.completion_signal).await;
        return;
    };
    emit_final(finish.tx, stop).await;
}

async fn emit_final(tx: &tokio::sync::mpsc::Sender<StreamResult<StreamEvent>>, stop: StopReason) {
    ::zeroclaw_log::record!(
        DEBUG,
        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Complete)
            .with_category(::zeroclaw_log::EventCategory::Provider)
            .with_outcome(::zeroclaw_log::EventOutcome::Success),
        "stream: SSE parser reached end of stream, emitting Final"
    );
    let _ = tx.send(Ok(StreamEvent::Final { stop })).await;
}

async fn emit_truncation_error(
    tx: &tokio::sync::mpsc::Sender<StreamResult<StreamEvent>>,
    completion_signal: &str,
) {
    ::zeroclaw_log::record!(
        WARN,
        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
            .with_category(::zeroclaw_log::EventCategory::Provider)
            .with_outcome(::zeroclaw_log::EventOutcome::Failure)
            .with_attrs(::serde_json::json!({
                "completion_signal": completion_signal,
            })),
        "stream: SSE connection closed before completion signal — truncated response, surfacing error"
    );
    let _ = tx
        .send(Err(StreamError::Http(format!(
            "SSE stream closed before {completion_signal}: response truncated"
        ))))
        .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn finish_emits_final_when_stop_known() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(4);
        super::finish_sse_stream(SseFinish {
            tx: &tx,
            stop: Some(StopReason::Complete),
            completion_signal: "message_stop",
        })
        .await;
        assert!(matches!(
            rx.recv().await,
            Some(Ok(StreamEvent::Final {
                stop: StopReason::Complete
            }))
        ));
    }

    #[tokio::test]
    async fn finish_emits_truncation_error_without_stop() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(4);
        super::finish_sse_stream(SseFinish {
            tx: &tx,
            stop: None,
            completion_signal: "message_stop",
        })
        .await;
        match rx.recv().await {
            Some(Err(StreamError::Http(msg))) => {
                assert!(msg.contains("truncated"), "got: {msg}");
                assert!(msg.contains("message_stop"), "got: {msg}");
            }
            other => panic!("expected truncation StreamError, got {other:?}"),
        }
    }
}
