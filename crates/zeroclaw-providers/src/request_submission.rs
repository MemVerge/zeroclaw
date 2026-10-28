//! Final immutable-request authorization, separate from diagnostic observation.
//! No request body/header/source is logged, mutated or retained by this module.
pub(crate) mod receipt;
use reqwest::{
    Request,
    header::{HeaderMap, HeaderName},
};
use zeroclaw_api::model_provider::submission::*;
use zeroclaw_api::model_provider::{ProjectedMessageSource, StreamError};

const MAX_SUBMISSION_BODY_BYTES: usize = 16 * 1024 * 1024;
const MAX_SUBMISSION_SOURCE_SPANS: usize = 16_384;
const MAX_REQUIRED_HEADERS: usize = 32;
const MAX_REQUIRED_HEADER_NAME_BYTES: usize = 128;

#[derive(Debug, thiserror::Error)]
pub(crate) enum SubmissionSendError {
    #[error("{0}")]
    Denied(#[from] SubmissionDenied),
    #[error("{0}")]
    Http(#[from] reqwest::Error),
}

impl SubmissionSendError {
    pub(crate) fn into_stream_error(self) -> StreamError {
        match self {
            Self::Denied(reason) => StreamError::SubmissionDenied(reason),
            Self::Http(error) => StreamError::Http(crate::format_error_chain(&error)),
        }
    }
}

pub(crate) fn preparation_stream_error(error: anyhow::Error) -> StreamError {
    match error
        .chain()
        .find_map(|cause| cause.downcast_ref::<SubmissionDenied>())
    {
        Some(reason) => StreamError::SubmissionDenied(*reason),
        None => StreamError::ModelProvider(error.to_string()),
    }
}

pub(crate) struct SubmissionHeaderAccess<'a> {
    pub(crate) headers: &'a HeaderMap,
    pub(crate) credential_header: &'a str,
    pub(crate) allowed: &'a [&'a str],
}

impl SubmissionHeaders for SubmissionHeaderAccess<'_> {
    fn single_value(&self, name: &str) -> Result<Option<&[u8]>, SubmissionDenied> {
        let name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|_| SubmissionDenied::InvalidHeaders)?;
        if !self
            .allowed
            .iter()
            .any(|allowed| name.as_str().eq_ignore_ascii_case(allowed))
        {
            return Err(SubmissionDenied::InvalidHeaders);
        }
        if sensitive_header(name.as_str(), self.credential_header) {
            return Err(SubmissionDenied::InvalidHeaders);
        }
        let values = self.headers.get_all(name);
        let mut values = values.iter();
        let first = values.next();
        if values.next().is_some() {
            return Err(SubmissionDenied::InvalidHeaders);
        }
        Ok(first.map(|value| value.as_bytes()))
    }
}

fn sensitive_header(name: &str, credential: &str) -> bool {
    name.eq_ignore_ascii_case(credential)
        || matches!(
            name,
            "authorization"
                | "proxy-authorization"
                | "cookie"
                | "set-cookie"
                | "x-api-key"
                | "api-key"
        )
}

pub(crate) struct SubmissionAuthorization<'a> {
    pub(crate) request: &'a Request,
    pub(crate) sources: &'a [ProjectedMessageSource],
    pub(crate) credential_header: &'a str,
}

pub(crate) fn authorize(
    authorizer: &dyn RequestSubmissionAuthorizer,
    input: SubmissionAuthorization<'_>,
) -> Result<SubmissionLease, SubmissionDenied> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        authorize_bounded(authorizer, input)
    }))
    .map_err(|_| SubmissionDenied::HostUnavailable)?
}

fn authorize_bounded(
    authorizer: &dyn RequestSubmissionAuthorizer,
    input: SubmissionAuthorization<'_>,
) -> Result<SubmissionLease, SubmissionDenied> {
    let limits = authorizer.limits();
    let body = input
        .request
        .body()
        .and_then(reqwest::Body::as_bytes)
        .ok_or(SubmissionDenied::UnsupportedBody)?;
    require_limits(limits, body.len(), input.sources.len())?;
    require_destination(input.request)?;
    let allowed = authorizer.required_headers();
    require_header_policy(allowed, input.credential_header)?;
    let headers = SubmissionHeaderAccess {
        headers: input.request.headers(),
        credential_header: input.credential_header,
        allowed,
    };
    authorizer.authorize(SerializedSubmission {
        method: input.request.method().as_str(),
        destination: input.request.url().as_str(),
        body,
        sources: input.sources,
        headers: &headers,
    })
}

fn require_header_policy(names: &[&str], credential: &str) -> Result<(), SubmissionDenied> {
    if names.len() > MAX_REQUIRED_HEADERS {
        return Err(SubmissionDenied::InvalidHeaders);
    }
    let mut seen = std::collections::BTreeSet::new();
    for name in names {
        if name.len() > MAX_REQUIRED_HEADER_NAME_BYTES {
            return Err(SubmissionDenied::InvalidHeaders);
        }
        let parsed = HeaderName::from_bytes(name.as_bytes())
            .map_err(|_| SubmissionDenied::InvalidHeaders)?;
        if sensitive_header(parsed.as_str(), credential) || !seen.insert(parsed.as_str().to_owned())
        {
            return Err(SubmissionDenied::InvalidHeaders);
        }
    }
    Ok(())
}

fn require_limits(
    limits: SubmissionLimits,
    body_bytes: usize,
    sources: usize,
) -> Result<(), SubmissionDenied> {
    if limits.max_body_bytes == 0
        || limits.max_body_bytes > MAX_SUBMISSION_BODY_BYTES
        || limits.max_source_spans > MAX_SUBMISSION_SOURCE_SPANS
        || body_bytes > limits.max_body_bytes
        || sources > limits.max_source_spans
    {
        return Err(SubmissionDenied::WorkLimit);
    }
    Ok(())
}

fn require_destination(request: &Request) -> Result<(), SubmissionDenied> {
    let url = request.url();
    if request.method() != reqwest::Method::POST
        || !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(SubmissionDenied::UnsupportedTransport);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
