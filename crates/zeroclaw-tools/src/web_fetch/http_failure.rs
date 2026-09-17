use std::error::Error;

const HTTP_FAILURE_DETAIL_LIMIT: usize = 1_024;

pub(super) struct HttpFailureDiagnostic {
    category: &'static str,
    root_cause: String,
    cause_chain: String,
}

impl HttpFailureDiagnostic {
    pub(super) fn request(error: &reqwest::Error) -> Self {
        Self::new(classify_reqwest_error(error), error)
    }

    pub(super) fn response_body(error: &anyhow::Error) -> Self {
        let source = error.as_ref();
        let category = find_reqwest_error(source)
            .map(classify_reqwest_error)
            .unwrap_or("response body failed");
        Self::new(category, source)
    }

    fn new(category: &'static str, error: &(dyn Error + 'static)) -> Self {
        Self {
            category,
            root_cause: root_cause(error),
            cause_chain: cause_chain(error),
        }
    }

    pub(super) fn detail(&self) -> String {
        format!("{}: {}", self.category, self.root_cause)
    }
}

fn classify_reqwest_error(error: &reqwest::Error) -> &'static str {
    match (error.is_timeout(), error.is_connect()) {
        (true, true) => "connection timed out",
        (true, false) => "request timed out",
        (false, true) => "connection failed",
        _ if error.is_redirect() => "redirect failed",
        _ if error.is_body() => "response body failed",
        _ if error.is_decode() => "response decoding failed",
        _ => "request failed",
    }
}

fn find_reqwest_error<'a>(mut error: &'a (dyn Error + 'static)) -> Option<&'a reqwest::Error> {
    loop {
        if let Some(reqwest_error) = error.downcast_ref::<reqwest::Error>() {
            return Some(reqwest_error);
        }
        error = error.source()?;
    }
}

fn root_cause(mut error: &(dyn Error + 'static)) -> String {
    while let Some(source) = error.source() {
        error = source;
    }
    truncate_http_failure_detail(&error.to_string())
}

fn cause_chain(error: &(dyn Error + 'static)) -> String {
    // Start at the nested cause so the top-level reqwest display, which can include the URL, is not logged.
    let mut messages = Vec::new();
    let mut current = error.source();
    while let Some(source) = current {
        messages.push(source.to_string());
        current = source.source();
    }
    let detail = if messages.is_empty() {
        error.to_string()
    } else {
        messages.join(": ")
    };
    truncate_http_failure_detail(&detail)
}

fn truncate_http_failure_detail(detail: &str) -> String {
    if detail.chars().count() <= HTTP_FAILURE_DETAIL_LIMIT {
        return detail.to_string();
    }
    let mut truncated = detail
        .chars()
        .take(HTTP_FAILURE_DETAIL_LIMIT)
        .collect::<String>();
    truncated.push_str("...");
    truncated
}

pub(super) fn log_http_failure(error_key: &str, diagnostic: &HttpFailureDiagnostic) {
    ::zeroclaw_log::record!(
        ERROR,
        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
            .with_outcome(::zeroclaw_log::EventOutcome::Failure)
            .with_attrs(::serde_json::json!({
                "error_key": error_key,
                "category": diagnostic.category,
                "cause_chain": diagnostic.cause_chain,
            })),
        "web_fetch: HTTP transport failed"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_bounds_model_and_log_details() {
        let error = anyhow::Error::msg("x".repeat(HTTP_FAILURE_DETAIL_LIMIT + 10));

        let diagnostic = HttpFailureDiagnostic::response_body(&error);

        assert_eq!(
            diagnostic.root_cause.chars().count(),
            HTTP_FAILURE_DETAIL_LIMIT + 3
        );
        assert_eq!(
            diagnostic.cause_chain.chars().count(),
            HTTP_FAILURE_DETAIL_LIMIT + 3
        );
        assert!(diagnostic.root_cause.ends_with("..."));
        assert!(diagnostic.cause_chain.ends_with("..."));
    }
}
