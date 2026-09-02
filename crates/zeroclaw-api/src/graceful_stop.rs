//! Host-requested graceful wrap-up of an in-flight tool loop.
//!
//! Distinct from [`tokio_util::sync::CancellationToken`]: cancel aborts the
//! loop immediately, while this signal asks the runtime to finish the current
//! tool wave and then issue one `tools=None` completion.

use std::sync::atomic::{AtomicU8, Ordering};

/// Why the host asked the tool loop to stop taking tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GracefulStopReason {
    /// Iteration / step ceiling. Existing max-iteration summary prompt.
    MaxIterations = 1,
    /// Repeated retrieval made no new progress. No-progress summary prompt.
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
#[derive(Debug, Default)]
pub struct GracefulStopSignal {
    code: AtomicU8,
}

impl GracefulStopSignal {
    pub fn new() -> Self {
        Self::default()
    }

    /// First caller wins. Later requests are ignored so a hard-stop cannot be
    /// overwritten by a stale no-progress write, and vice versa.
    pub fn request(&self, reason: GracefulStopReason) {
        let _ = self
            .code
            .compare_exchange(0, reason as u8, Ordering::Release, Ordering::Relaxed);
    }

    pub fn requested(&self) -> Option<GracefulStopReason> {
        GracefulStopReason::from_code(self.code.load(Ordering::Acquire))
    }
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
}
