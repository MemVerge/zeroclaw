use super::OpenAiCompatibleModelProvider;
use anyhow::Context;
use std::num::NonZeroUsize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("compatible completion response exceeded decoded body limit ({maximum} bytes)")]
pub struct CompletionBodyLimitExceeded {
    pub maximum: usize,
}

impl OpenAiCompatibleModelProvider {
    pub(super) async fn completion_text(
        &self,
        response: reqwest::Response,
    ) -> anyhow::Result<String> {
        let Some(limit) = self.completion_body_limit else {
            return Ok(response.text().await?);
        };
        let bytes = read_bounded(response, limit).await?;
        String::from_utf8(bytes).context("decode bounded compatible completion as UTF-8")
    }

    pub(super) async fn completion_json<T: serde::de::DeserializeOwned>(
        &self,
        response: reqwest::Response,
    ) -> anyhow::Result<T> {
        let Some(limit) = self.completion_body_limit else {
            return Ok(response.json().await?);
        };
        let bytes = read_bounded(response, limit).await?;
        serde_json::from_slice(&bytes).context("parse bounded compatible completion JSON")
    }

    pub(super) async fn completion_error(&self, response: reqwest::Response) -> anyhow::Error {
        if self.completion_body_limit.is_none() {
            return super::super::api_error(&self.name, response).await;
        }
        let status = response.status();
        match self.completion_text(response).await {
            Ok(text) => self.bounded_api_error(status, &text),
            Err(error) => error,
        }
    }

    fn bounded_api_error(&self, status: reqwest::StatusCode, text: &str) -> anyhow::Error {
        ::zeroclaw_log::record!(
            ERROR,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                .with_attrs(serde_json::json!({"http_status": status.as_u16()})),
            "bounded compatible completion HTTP error"
        );
        anyhow::Error::msg(format!(
            "{} API error ({status}): {}",
            self.name,
            super::super::sanitize_api_error(text)
        ))
    }
}

async fn read_bounded(
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
        .context("read bounded compatible completion chunk")?
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
