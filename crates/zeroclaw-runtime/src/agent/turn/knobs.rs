//! Per-caller loop behaviour knobsconsolidation).

/// How to handle max-tool-iteration exhaustion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MaxIterationBehavior {
    /// Ask the LLM for a tools-free final summary (channel/CLI behaviour).
    #[default]
    GracefulSummary,
    /// Bail with "exceeded maximum tool iterations" (embedder control signal).
    ErrorAtCap,
}

/// How to handle a provider stream failure that is eligible for retry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamFailureBehavior {
    /// Retry the same request through the provider's non-streaming chat API.
    #[default]
    FallbackToNonStreaming,
    /// Return the stream error without issuing a second provider request.
    ReturnError,
}

/// Ownership of lossy provider-overflow recovery in the shared turn loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContextOverflowBehavior {
    /// Preserve existing runtime recovery by discarding oldest whole turns.
    #[default]
    DropOldestTurns,
    /// Return the provider error without mutating history or retrying a
    /// smaller request. The caller owns context projection and safe recovery.
    FailClosed,
}

/// Explicit knobs for per-caller loop behaviour.
#[derive(Debug, Clone)]
pub struct LoopKnobs {
    pub dedup_enabled: bool,
    pub max_iteration_behavior: MaxIterationBehavior,
    pub stream_failure_behavior: StreamFailureBehavior,
    pub context_overflow_behavior: ContextOverflowBehavior,
    /// When `true` (channel paths), a response that resembles the internal
    /// tool protocol while no tools are enabled is classified as a parse
    /// issue (malformed-protocol retry, then i18n fallback) so raw protocol
    /// text never reaches end users. Embedder wrappers set `false`: their
    /// contract is to return the model text verbatim and let the embedder
    /// do its own post-processing.
    pub detect_protocol_without_tools: bool,
}

impl Default for LoopKnobs {
    fn default() -> Self {
        Self {
            dedup_enabled: true,
            max_iteration_behavior: MaxIterationBehavior::GracefulSummary,
            stream_failure_behavior: StreamFailureBehavior::FallbackToNonStreaming,
            context_overflow_behavior: ContextOverflowBehavior::DropOldestTurns,
            detect_protocol_without_tools: true,
        }
    }
}
