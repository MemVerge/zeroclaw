//! Host-requested graceful wrap-up of an in-flight tool loop.
//!
//! Distinct from [`tokio_util::sync::CancellationToken`]: cancel aborts the
//! loop immediately, while this signal asks the runtime to finish the current
//! tool wave and then issue one `tools=None` completion.

use std::sync::Mutex;

/// Why the tool loop stopped taking tools.
///
/// A host request and the runtime's own iteration cap share this value. The
/// cap records [`GracefulStopReason::MaxIterations`] with no wrap-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GracefulStopReason {
    /// Iteration / step ceiling.
    MaxIterations = 1,
    /// Repeated retrieval made no new progress.
    NoProgress = 2,
}

/// One winning stop. The reason and its wrap-up are stored together.
#[derive(Debug)]
struct PublishedStop {
    reason: GracefulStopReason,
    wrap_up: Option<String>,
}

/// Request-once flag the host sets and the tool loop polls at wave boundaries.
///
/// The winning reason and its optional wrap-up are one mutex value, so a
/// reader that observes the reason also observes that request's instruction.
#[derive(Debug, Default)]
pub struct GracefulStopSignal {
    state: Mutex<Option<PublishedStop>>,
}

impl GracefulStopSignal {
    pub fn new() -> Self {
        Self::default()
    }

    /// First caller wins. Later requests are ignored so a hard-stop cannot be
    /// overwritten by a stale no-progress write, and vice versa.
    pub fn request(&self, reason: GracefulStopReason) {
        self.publish(reason, None);
    }

    /// Same once-only claim as [`Self::request`], plus the caller's wrap-up text.
    pub fn request_with_wrap_up(&self, reason: GracefulStopReason, instruction: impl Into<String>) {
        self.publish(reason, Some(instruction.into()));
    }

    pub fn requested(&self) -> Option<GracefulStopReason> {
        self.lock().as_ref().map(|stop| stop.reason)
    }

    pub fn wrap_up_instruction(&self) -> Option<String> {
        self.lock().as_ref().and_then(|stop| stop.wrap_up.clone())
    }

    fn publish(&self, reason: GracefulStopReason, wrap_up: Option<String>) {
        let mut state = self.lock();
        if state.is_some() {
            return;
        }
        *state = Some(PublishedStop { reason, wrap_up });
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<PublishedStop>> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::{GracefulStopReason, GracefulStopSignal};
    use std::sync::{Arc, Barrier};
    use std::thread;

    #[test]
    fn first_request_wins_and_is_visible() {
        let signal = GracefulStopSignal::new();
        assert_eq!(signal.requested(), None);
        signal.request(GracefulStopReason::NoProgress);
        signal.request(GracefulStopReason::MaxIterations);
        assert_eq!(signal.requested(), Some(GracefulStopReason::NoProgress));
    }

    #[test]
    fn already_set_signal_is_not_missed() {
        let signal = GracefulStopSignal::new();
        signal.request(GracefulStopReason::MaxIterations);
        assert_eq!(signal.requested(), Some(GracefulStopReason::MaxIterations));
        assert_eq!(signal.requested(), Some(GracefulStopReason::MaxIterations));
    }

    #[test]
    fn wrap_up_is_kept_only_for_the_winning_request() {
        let signal = GracefulStopSignal::new();
        signal.request_with_wrap_up(GracefulStopReason::NoProgress, "answer from evidence");
        signal.request_with_wrap_up(GracefulStopReason::MaxIterations, "summarize the work");
        assert_eq!(signal.requested(), Some(GracefulStopReason::NoProgress));
        assert_eq!(
            signal.wrap_up_instruction().as_deref(),
            Some("answer from evidence")
        );
    }

    #[test]
    fn plain_request_stores_no_wrap_up() {
        let signal = GracefulStopSignal::new();
        signal.request(GracefulStopReason::MaxIterations);
        assert_eq!(signal.wrap_up_instruction(), None);
    }

    #[test]
    fn wrap_up_is_visible_when_the_reason_is() {
        let signal = Arc::new(GracefulStopSignal::new());
        let start = Arc::new(Barrier::new(2));
        let reader = spawn_wrap_up_reader(Arc::clone(&signal), Arc::clone(&start));
        start.wait();
        signal.request_with_wrap_up(GracefulStopReason::NoProgress, "answer from evidence");
        let instruction = reader.join().expect("reader");
        assert_eq!(instruction.as_deref(), Some("answer from evidence"));
    }

    fn spawn_wrap_up_reader(
        signal: Arc<GracefulStopSignal>,
        start: Arc<Barrier>,
    ) -> thread::JoinHandle<Option<String>> {
        thread::spawn(move || {
            start.wait();
            loop {
                if signal.requested() == Some(GracefulStopReason::NoProgress) {
                    return signal.wrap_up_instruction();
                }
                thread::yield_now();
            }
        })
    }
}
