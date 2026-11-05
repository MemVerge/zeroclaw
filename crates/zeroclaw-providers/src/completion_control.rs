//! Transport readers shared by native adapters; no routing or context policy.
use anyhow::Context;
use std::num::NonZeroUsize;
use zeroclaw_api::model_provider::CompletionControl;

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("completion response exceeded decoded body limit ({maximum} bytes)")]
pub struct CompletionBodyLimitExceeded {
    pub maximum: usize,
}

pub(crate) async fn read_bounded(
    mut response: reqwest::Response,
    limit: NonZeroUsize,
) -> anyhow::Result<Vec<u8>> {
    let maximum = limit.get();
    if response
        .content_length()
        .is_some_and(|length| length > maximum as u64)
    {
        return Err(CompletionBodyLimitExceeded { maximum }.into());
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .context("read bounded completion chunk")?
    {
        append_bounded(&mut body, &chunk, maximum)?;
    }
    Ok(body)
}

fn append_bounded(
    body: &mut Vec<u8>,
    chunk: &[u8],
    maximum: usize,
) -> Result<(), CompletionBodyLimitExceeded> {
    if chunk.len() > maximum.saturating_sub(body.len()) {
        return Err(CompletionBodyLimitExceeded { maximum });
    }
    body.extend_from_slice(chunk);
    Ok(())
}

pub(crate) fn client(
    builder: reqwest::ClientBuilder,
    service: &str,
) -> anyhow::Result<reqwest::Client> {
    let proxy = zeroclaw_config::schema::runtime_proxy_config();
    proxy
        .validate()
        .context("validate controlled completion proxy")?;
    proxy
        .apply_to_reqwest_builder(
            builder
                .redirect(reqwest::redirect::Policy::none())
                .retry(reqwest::retry::never()),
            service,
        )
        .build()
        .context("build controlled completion client")
}

pub(crate) async fn text(
    response: reqwest::Response,
    control: Option<CompletionControl>,
) -> anyhow::Result<String> {
    match control {
        None => response.text().await.context("read completion body"),
        Some(control) => {
            let bytes = read_bounded(response, control.maximum_body_bytes()).await?;
            String::from_utf8(bytes).context("decode bounded completion UTF-8")
        }
    }
}

pub(crate) async fn json<T: serde::de::DeserializeOwned>(
    response: reqwest::Response,
    control: Option<CompletionControl>,
) -> anyhow::Result<T> {
    match control {
        None => response.json().await.context("read completion JSON"),
        Some(_) => serde_json::from_str(&text(response, control).await?)
            .context("parse bounded completion JSON"),
    }
}

pub(crate) async fn api_error(
    provider: &str,
    response: reqwest::Response,
    control: Option<CompletionControl>,
) -> anyhow::Error {
    if control.is_none() {
        return super::api_error(provider, response).await;
    }
    let status = response.status();
    match text(response, control).await {
        Ok(body) => anyhow::Error::msg(format!(
            "{provider} API error ({status}): {}",
            super::sanitize_api_error(&body)
        )),
        Err(error) => error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_boundary_is_kept_without_truncation() {
        let mut body = vec![1, 2];
        append_bounded(&mut body, &[3, 4], 4).unwrap();
        assert_eq!(body, vec![1, 2, 3, 4]);
    }

    #[test]
    fn one_byte_over_limit_does_not_extend_the_buffer() {
        let mut body = vec![1, 2];
        assert_eq!(
            append_bounded(&mut body, &[3, 4, 5], 4),
            Err(CompletionBodyLimitExceeded { maximum: 4 })
        );
        assert_eq!(body, vec![1, 2]);
    }

    #[test]
    fn empty_chunks_do_not_spend_capacity() {
        let mut body = vec![1];
        append_bounded(&mut body, &[], 1).unwrap();
        assert_eq!(body, vec![1]);
    }
}
