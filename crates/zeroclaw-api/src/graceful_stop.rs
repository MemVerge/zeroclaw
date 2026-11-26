//! Host-requested graceful wrap-up of an in-flight tool loop.
//!
//! Distinct from [`tokio_util::sync::CancellationToken`]: cancel aborts the
//! loop immediately, while this signal asks the runtime to finish the current
//! tool wave and then issue one `tools=None` completion.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};

/// Why the host asked the tool loop to stop taking tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GracefulStopReason {
    /// Iteration / step ceiling.
    MaxIterations = 1,
    /// Repeated retrieval made no new progress.
    NoProgress = 2,
}

impl GracefulStopReason {
    fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::MaxIterations),
            2 => Some(Self::NoProgress),
            _ => None,
        }
    }
}

/// Request-once flag the host sets and the tool loop polls at wave boundaries.
///
/// An optional wrap-up instruction is stored with the winning request. The
/// finalizer appends it only when the caller supplied one.
#[derive(Debug, Default)]
pub struct GracefulStopSignal {
    code: AtomicU8,
    wrap_up: Mutex<Option<String>>,
}

impl GracefulStopSignal {
    pub fn new() -> Self {
        Self::default()
    }

    /// First caller wins. Later requests are ignored so a hard-stop cannot be
    /// overwritten by a stale no-progress write, and vice versa.
    pub fn request(&self, reason: GracefulStopReason) {
        let _ = self.claim(reason);
    }

    /// Same once-only claim as [`Self::request`], plus the caller's wrap-up text.
    pub fn request_with_wrap_up(&self, reason: GracefulStopReason, instruction: impl Into<String>) {
        if self.claim(reason) {
            *lock_wrap_up(&self.wrap_up) = Some(instruction.into());
        }
    }

    pub fn requested(&self) -> Option<GracefulStopReason> {
        GracefulStopReason::from_code(self.code.load(Ordering::Acquire))
    }

    pub fn wrap_up_instruction(&self) -> Option<String> {
        lock_wrap_up(&self.wrap_up).clone()
    }

    fn claim(&self, reason: GracefulStopReason) -> bool {
        self.code
            .compare_exchange(0, reason as u8, Ordering::Release, Ordering::Relaxed)
            .is_ok()
    }
}

fn lock_wrap_up(wrap_up: &Mutex<Option<String>>) -> std::sync::MutexGuard<'_, Option<String>> {
    wrap_up
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::{GracefulStopReason, GracefulStopSignal};

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
}
