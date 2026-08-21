use anyhow::bail;
use encoding_rs::{CoderResult, Encoding, UTF_8, UTF_16BE, UTF_16LE, WINDOWS_1252, X_USER_DEFINED};

const HTML_META_CHARSET_SCAN_LIMIT: usize = 1024;

pub(super) struct DecodedResponseBody {
    pub(super) text: String,
    pub(super) completeness: ResponseCompleteness,
    pub(super) had_decode_errors: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ResponseCompleteness {
    Complete,
    Truncated,
}

impl ResponseCompleteness {
    pub(super) fn is_complete(self) -> bool {
        self == Self::Complete
    }
}

pub(super) fn decode_response_body(
    bytes: &[u8],
    content_type: &str,
    completeness: ResponseCompleteness,
) -> anyhow::Result<DecodedResponseBody> {
    let (encoding, bom_length) = select_encoding(bytes, content_type);
    let (text, had_decode_errors) = decode_bytes(encoding, &bytes[bom_length..], completeness)?;
    Ok(DecodedResponseBody {
        text,
        completeness,
        had_decode_errors,
    })
}

fn select_encoding(bytes: &[u8], content_type: &str) -> (&'static Encoding, usize) {
    if let Some((encoding, length)) = Encoding::for_bom(bytes) {
        return (encoding, length);
    }
    let encoding = encoding_from_content_type(content_type)
        .or_else(|| {
            is_html_content_type(content_type)
                .then(|| encoding_from_html_meta(bytes))
                .flatten()
        })
        .unwrap_or(UTF_8);
    (encoding, 0)
}

fn decode_bytes(
    encoding: &'static Encoding,
    bytes: &[u8],
    completeness: ResponseCompleteness,
) -> anyhow::Result<(String, bool)> {
    let mut decoder = encoding.new_decoder_without_bom_handling();
    let capacity = decoder
        .max_utf8_buffer_length(bytes.len())
        .ok_or_else(|| anyhow::Error::msg("response body is too large to decode"))?;
    let mut text = String::with_capacity(capacity);
    let (result, bytes_read, had_errors) =
        decoder.decode_to_string(bytes, &mut text, completeness.is_complete());
    if result == CoderResult::OutputFull || bytes_read != bytes.len() {
        bail!("response body decoder output capacity was insufficient");
    }
    Ok((text, had_errors))
}

fn encoding_from_content_type(content_type: &str) -> Option<&'static Encoding> {
    content_type.split(';').skip(1).find_map(|parameter| {
        let (name, value) = parameter.split_once('=')?;
        if !name.trim().eq_ignore_ascii_case("charset") {
            return None;
        }
        let label = value
            .trim()
            .trim_matches(|character| character == '"' || character == '\'');
        Encoding::for_label(label.as_bytes())
    })
}

fn is_html_content_type(content_type: &str) -> bool {
    content_type.is_empty()
        || find_ascii_case_insensitive(content_type.as_bytes(), b"text/html").is_some()
}

fn encoding_from_html_meta(bytes: &[u8]) -> Option<&'static Encoding> {
    let sample = &bytes[..bytes.len().min(HTML_META_CHARSET_SCAN_LIMIT)];
    let mut position = 0;
    while position < sample.len() {
        if sample[position..].starts_with(b"<!--") {
            position = position_after_comment(sample, position + 4);
        } else if is_meta_tag_start(sample, position) {
            let end = position_after_tag(sample, position + 5);
            if let Some(encoding) = encoding_from_meta_attributes(&sample[position + 5..end]) {
                return Some(encoding);
            }
            position = end.saturating_add(1);
        } else if is_tag_start(sample, position) {
            position = position_after_tag(sample, position + 1).saturating_add(1);
        } else {
            position += 1;
        }
    }
    None
}

fn position_after_comment(bytes: &[u8], start: usize) -> usize {
    find_bytes(&bytes[start..], b"-->")
        .map(|relative| start + relative + 3)
        .unwrap_or(bytes.len())
}

fn is_meta_tag_start(bytes: &[u8], position: usize) -> bool {
    let Some(tail) = bytes.get(position..) else {
        return false;
    };
    starts_ascii_case_insensitive(tail, b"<meta")
        && tail
            .get(5)
            .is_some_and(|byte| is_html_space(*byte) || *byte == b'/')
}

fn is_tag_start(bytes: &[u8], position: usize) -> bool {
    bytes.get(position) == Some(&b'<')
        && bytes
            .get(position + 1)
            .is_some_and(|byte| byte.is_ascii_alphabetic() || matches!(byte, b'!' | b'/' | b'?'))
}

fn position_after_tag(bytes: &[u8], start: usize) -> usize {
    let mut quote = None;
    for (offset, byte) in bytes[start..].iter().enumerate() {
        match (quote, byte) {
            (None, b'"' | b'\'') => quote = Some(*byte),
            (Some(opening), closing) if opening == *closing => quote = None,
            (None, b'>') => return start + offset,
            _ => {}
        }
    }
    bytes.len()
}

#[derive(Clone, Copy)]
struct Attribute<'a> {
    name: &'a [u8],
    value: &'a [u8],
}

#[derive(Default)]
struct MetaAttributes<'a> {
    charset: Option<&'a [u8]>,
    charset_seen: bool,
    content: Option<&'a [u8]>,
    http_equiv: Option<&'a [u8]>,
}

fn encoding_from_meta_attributes(tag: &[u8]) -> Option<&'static Encoding> {
    let mut position = 0;
    let mut values = MetaAttributes::default();
    while let Some((attribute, next)) = next_attribute(tag, position) {
        position = next;
        if eq_ascii_case_insensitive(attribute.name, b"charset") && !values.charset_seen {
            values.charset_seen = true;
            values.charset = Some(attribute.value);
        } else if eq_ascii_case_insensitive(attribute.name, b"content") && values.content.is_none()
        {
            values.content = Some(attribute.value);
        } else if eq_ascii_case_insensitive(attribute.name, b"http-equiv")
            && values.http_equiv.is_none()
        {
            values.http_equiv = Some(attribute.value);
        }
    }
    encoding_from_meta_values(values)
}

fn encoding_from_meta_values(values: MetaAttributes<'_>) -> Option<&'static Encoding> {
    if values.charset_seen {
        return values
            .charset
            .and_then(|value| Encoding::for_label(trim_html_space(value)))
            .map(normalize_meta_encoding);
    }
    let has_pragma = values
        .http_equiv
        .is_some_and(|value| eq_ascii_case_insensitive(trim_html_space(value), b"content-type"));
    has_pragma
        .then(|| values.content.and_then(encoding_from_content_attribute))
        .flatten()
        .map(normalize_meta_encoding)
}

fn next_attribute(tag: &[u8], mut position: usize) -> Option<(Attribute<'_>, usize)> {
    while tag
        .get(position)
        .is_some_and(|byte| is_html_space(*byte) || *byte == b'/')
    {
        position += 1;
    }
    let name_start = position;
    while tag
        .get(position)
        .is_some_and(|byte| !is_html_space(*byte) && !matches!(byte, b'/' | b'=' | b'>'))
    {
        position += 1;
    }
    if name_start == position {
        return None;
    }
    let name = &tag[name_start..position];
    position = skip_html_space(tag, position);
    let (value, next) = attribute_value(tag, position);
    Some((Attribute { name, value }, next))
}

fn attribute_value(tag: &[u8], position: usize) -> (&[u8], usize) {
    if tag.get(position) != Some(&b'=') {
        return (&[], position);
    }
    let start = skip_html_space(tag, position + 1);
    match tag.get(start) {
        Some(quote @ (b'"' | b'\'')) => quoted_attribute_value(tag, start + 1, *quote),
        Some(_) => unquoted_attribute_value(tag, start),
        None => (&[], start),
    }
}

fn quoted_attribute_value(tag: &[u8], start: usize, quote: u8) -> (&[u8], usize) {
    let length = tag[start..]
        .iter()
        .position(|byte| *byte == quote)
        .unwrap_or(tag.len() - start);
    let end = start + length;
    (&tag[start..end], end.saturating_add(1).min(tag.len()))
}

fn unquoted_attribute_value(tag: &[u8], start: usize) -> (&[u8], usize) {
    let length = tag[start..]
        .iter()
        .position(|byte| is_html_space(*byte) || *byte == b'>')
        .unwrap_or(tag.len() - start);
    let end = start + length;
    (&tag[start..end], end)
}

fn skip_html_space(bytes: &[u8], mut position: usize) -> usize {
    while bytes.get(position).is_some_and(|byte| is_html_space(*byte)) {
        position += 1;
    }
    position
}

fn encoding_from_content_attribute(content: &[u8]) -> Option<&'static Encoding> {
    let mut position = 0;
    loop {
        let relative = find_ascii_case_insensitive(&content[position..], b"charset")?;
        position += relative + b"charset".len();
        let tail = trim_html_space_start(&content[position..]);
        if let Some(label) = charset_label(tail) {
            return Encoding::for_label(label);
        }
    }
}

fn charset_label(bytes: &[u8]) -> Option<&[u8]> {
    if bytes.first() != Some(&b'=') {
        return None;
    }
    let bytes = trim_html_space_start(&bytes[1..]);
    if let Some(quote @ (b'"' | b'\'')) = bytes.first() {
        let value = &bytes[1..];
        let length = value.iter().position(|byte| byte == quote)?;
        return (length > 0).then_some(&value[..length]);
    }
    let length = bytes
        .iter()
        .position(|byte| is_html_space(*byte) || matches!(byte, b';' | b'"' | b'\''))
        .unwrap_or(bytes.len());
    (length > 0).then_some(&bytes[..length])
}

/// Applies the HTML prescan overrides without changing HTTP `charset` semantics.
fn normalize_meta_encoding(encoding: &'static Encoding) -> &'static Encoding {
    if std::ptr::eq(encoding, UTF_16BE) || std::ptr::eq(encoding, UTF_16LE) {
        UTF_8
    } else if std::ptr::eq(encoding, X_USER_DEFINED) {
        WINDOWS_1252
    } else {
        encoding
    }
}

fn is_html_space(byte: u8) -> bool {
    matches!(byte, b'\t' | b'\n' | 0x0C | b'\r' | b' ')
}

fn trim_html_space(mut bytes: &[u8]) -> &[u8] {
    bytes = trim_html_space_start(bytes);
    while bytes.last().is_some_and(|byte| is_html_space(*byte)) {
        bytes = &bytes[..bytes.len() - 1];
    }
    bytes
}

fn trim_html_space_start(mut bytes: &[u8]) -> &[u8] {
    while bytes.first().is_some_and(|byte| is_html_space(*byte)) {
        bytes = &bytes[1..];
    }
    bytes
}

fn find_ascii_case_insensitive(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|candidate| eq_ascii_case_insensitive(candidate, needle))
}

fn starts_ascii_case_insensitive(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .get(..needle.len())
        .is_some_and(|candidate| eq_ascii_case_insensitive(candidate, needle))
}

fn eq_ascii_case_insensitive(left: &[u8], right: &[u8]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.eq_ignore_ascii_case(right))
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|value| value == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(bytes: &[u8], content_type: &str) -> DecodedResponseBody {
        decode_response_body(bytes, content_type, ResponseCompleteness::Complete)
            .expect("response should decode")
    }

    #[test]
    fn declared_gbk_charset_decodes_chinese() {
        let original = "8月20日收盘：上证指数报3903.72点";
        let (encoded, _, had_errors) = encoding_rs::GBK.encode(original);
        assert!(!had_errors);

        let decoded = decode(&encoded, "text/html; charset=gb2312");

        assert_eq!(decoded.text, original);
        assert!(!decoded.had_decode_errors);
    }

    #[test]
    fn html_meta_charset_decodes_chinese() {
        let original = "<meta charset=\"gbk\"><p>医药领涨，医药股涨停潮</p>";
        let (encoded, _, had_errors) = encoding_rs::GBK.encode(original);
        assert!(!had_errors);

        let decoded = decode(&encoded, "text/html");

        assert_eq!(decoded.text, original);
    }

    #[test]
    fn http_charset_takes_precedence_over_html_meta() {
        let original = "<meta charset=gbk><p>上证指数</p>";

        let decoded = decode(original.as_bytes(), "text/html; charset=utf-8");

        assert_eq!(decoded.text, original);
    }

    #[test]
    fn non_html_content_does_not_scan_meta_charset() {
        let original = "<meta charset=gbk>上证指数";

        let decoded = decode(original.as_bytes(), "text/plain");

        assert_eq!(decoded.text, original);
    }

    #[test]
    fn legacy_content_type_meta_decodes_chinese() {
        let original =
            "<meta http-equiv=\"content-type\" content=\"text/html; charset=gbk\"><p>医药领涨</p>";
        let (encoded, _, had_errors) = encoding_rs::GBK.encode(original);
        assert!(!had_errors);

        let decoded = decode(&encoded, "text/html");

        assert_eq!(decoded.text, original);
    }

    #[test]
    fn legacy_meta_accepts_content_before_http_equiv() {
        let original =
            "<meta content=\"text/html; charset=gbk\" http-equiv=\"content-type\"><p>医药领涨</p>";
        let (encoded, _, had_errors) = encoding_rs::GBK.encode(original);
        assert!(!had_errors);

        let decoded = decode(&encoded, "text/html");

        assert_eq!(decoded.text, original);
    }

    #[test]
    fn legacy_meta_accepts_quoted_charset_inside_content() {
        let original = "<meta http-equiv=\"content-type\" content=\"text/html; charset='gbk'\"><p>上证指数</p>";
        let (encoded, _, had_errors) = encoding_rs::GBK.encode(original);
        assert!(!had_errors);

        let decoded = decode(&encoded, "text/html");

        assert_eq!(decoded.text, original);
    }

    #[test]
    fn commented_meta_does_not_change_encoding() {
        let original = "<!-- <meta charset=gbk> --><p>创业板指涨0.64%</p>";

        let decoded = decode(original.as_bytes(), "text/html");

        assert_eq!(decoded.text, original);
    }

    #[test]
    fn metadata_element_does_not_count_as_meta() {
        let original = "<metadata charset=gbk></metadata><p>科创50指</p>";

        let decoded = decode(original.as_bytes(), "text/html");

        assert_eq!(decoded.text, original);
    }

    #[test]
    fn refresh_meta_does_not_count_as_content_type_pragma() {
        let original =
            "<meta http-equiv=\"refresh\" content=\"text/html; charset=gbk\"><p>上证指数</p>";

        let decoded = decode(original.as_bytes(), "text/html");

        assert_eq!(decoded.text, original);
    }

    #[test]
    fn unrelated_meta_content_does_not_declare_charset() {
        let original = "<meta name=\"description\" content=\"charset=gbk\"><p>创业板指</p>";

        let decoded = decode(original.as_bytes(), "text/html");

        assert_eq!(decoded.text, original);
    }

    #[test]
    fn legacy_content_finds_declaration_after_similar_text() {
        let original =
            "<meta http-equiv=\"content-type\" content=\"xcharset; charset=gbk\"><p>科创50指</p>";
        let (encoded, _, had_errors) = encoding_rs::GBK.encode(original);
        assert!(!had_errors);

        let decoded = decode(&encoded, "text/html");

        assert_eq!(decoded.text, original);
    }

    #[test]
    fn meta_text_inside_another_tag_attribute_is_ignored() {
        let original = "<div data-value=\"<meta charset=gbk>\"><p>上证指数</p></div>";

        let decoded = decode(original.as_bytes(), "text/html");

        assert_eq!(decoded.text, original);
    }

    #[test]
    fn meta_utf16_label_is_normalized_to_utf8() {
        let original = "<meta charset=\"utf-16le\"><p>深证成指</p>";

        let decoded = decode(original.as_bytes(), "text/html");

        assert_eq!(decoded.text, original);
    }

    #[test]
    fn meta_x_user_defined_is_normalized_to_windows_1252() {
        let bytes = b"<meta charset=x-user-defined><p>\x80</p>";

        let decoded = decode(bytes, "text/html");

        assert_eq!(decoded.text, "<meta charset=x-user-defined><p>€</p>");
    }

    #[test]
    fn bom_overrides_declared_charset() {
        let original = "创业板指涨0.64%";
        let mut encoded = b"\xEF\xBB\xBF".to_vec();
        encoded.extend_from_slice(original.as_bytes());

        let decoded = decode(&encoded, "text/plain; charset=gbk");

        assert_eq!(decoded.text, original);
    }

    #[test]
    fn truncated_multibyte_tail_is_not_replaced() {
        let (encoded, _, had_errors) = encoding_rs::GBK.encode("中文");
        assert!(!had_errors);

        let decoded = decode_response_body(
            &encoded[..3],
            "text/plain; charset=gbk",
            ResponseCompleteness::Truncated,
        )
        .expect("truncated response should decode");

        assert_eq!(decoded.text, "中");
        assert!(!decoded.text.contains('\u{FFFD}'));
        assert!(!decoded.had_decode_errors);
        assert!(!decoded.completeness.is_complete());
    }

    #[test]
    fn malformed_complete_input_reports_decode_errors() {
        let decoded = decode(&[0xFF], "text/plain; charset=utf-8");

        assert!(decoded.had_decode_errors);
        assert_eq!(decoded.text, "�");
    }
}
