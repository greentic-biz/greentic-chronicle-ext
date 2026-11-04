//! Citation metadata (`title`, `source_url`, `section`) a document carries.
//!
//! Every field is optional end to end. A value that fails its rule is
//! dropped, never repaired and never a reason to refuse the batch: a hit
//! without a link is fine, a `javascript:` link is not.

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use url::Url;

/// Longest `title` / `section` kept, in characters.
pub const MAX_TEXT_CHARS: usize = 300;
/// Longest `source_url` kept, in bytes.
pub const MAX_URL_LEN: usize = 2048;

pub const KEY_TITLE: &str = "title";
pub const KEY_SOURCE_URL: &str = "source_url";
pub const KEY_SECTION: &str = "section";

/// `Some(text)` when `raw` is a non-blank value of at most
/// [`MAX_TEXT_CHARS`] characters with no control character. Over-long text is
/// dropped rather than truncated: a half title reads as a wrong one.
pub fn checked_text(raw: Option<&str>) -> Option<String> {
    let raw = raw?.trim();
    let ok = !raw.is_empty()
        && raw.chars().count() <= MAX_TEXT_CHARS
        && !raw.chars().any(char::is_control);
    ok.then(|| raw.to_string())
}

/// `Some(link)` when `raw` is an absolute `https` URL with a host, no
/// credentials, no control character and at most [`MAX_URL_LEN`] bytes.
pub fn checked_url(raw: Option<&str>) -> Option<String> {
    let raw = raw?.trim();
    if raw.is_empty() || raw.len() > MAX_URL_LEN || raw.chars().any(char::is_control) {
        return None;
    }
    let url = Url::parse(raw).ok()?;
    let ok = url.scheme() == "https"
        && url.host_str().is_some()
        && url.username().is_empty()
        && url.password().is_none();
    ok.then(|| url.to_string())
}

/// The validated metadata of one document: the document-level fields plus the
/// section of each chunk, in chunk order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DocumentCitation {
    pub title: Option<String>,
    pub source_url: Option<String>,
    /// `(chunk_index, section)` for the chunks that carry one, sorted.
    pub sections: Vec<(i64, String)>,
}

impl DocumentCitation {
    /// True when the upsert carried no usable citation field at all. Such a
    /// document (an old client's) is judged by its content hash alone.
    pub fn is_empty(&self) -> bool {
        self.title.is_none() && self.source_url.is_none() && self.sections.is_empty()
    }

    /// Stable digest of the metadata, stored beside the content hash so a
    /// metadata-only change is seen as a change. Fields are length-prefixed,
    /// so no value can masquerade as another.
    pub fn fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        for part in [&self.title, &self.source_url] {
            absorb(&mut hasher, part.as_deref().unwrap_or(""));
        }
        let mut sections = self.sections.clone();
        sections.sort();
        for (index, section) in &sections {
            absorb(&mut hasher, &index.to_string());
            absorb(&mut hasher, section);
        }
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// Attributes to stamp on the node of chunk `index`.
    pub fn chunk_attributes(&self, index: i64) -> Map<String, Value> {
        let mut map = Map::new();
        if let Some(title) = &self.title {
            map.insert(KEY_TITLE.into(), Value::String(title.clone()));
        }
        if let Some(url) = &self.source_url {
            map.insert(KEY_SOURCE_URL.into(), Value::String(url.clone()));
        }
        if let Some((_, section)) = self.sections.iter().find(|(i, _)| *i == index) {
            map.insert(KEY_SECTION.into(), Value::String(section.clone()));
        }
        map
    }
}

fn absorb(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

/// Reads `key` from a chunk node's attributes, re-checking it with `check` so
/// a value stored by an older build is held to today's rule.
pub fn attribute(
    attrs: &Map<String, Value>,
    key: &str,
    check: fn(Option<&str>) -> Option<String>,
) -> Option<String> {
    check(attrs.get(key).and_then(Value::as_str))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_a_plain_https_link_and_drops_everything_else() {
        assert_eq!(
            checked_url(Some("https://drive.google.com/file/d/abc/view")).as_deref(),
            Some("https://drive.google.com/file/d/abc/view")
        );
        for bad in [
            "",
            "http://example.com/a",
            "javascript:alert(1)",
            "data:text/html,x",
            "//example.com/a",
            "/relative",
            "https://user:pw@example.com/a",
            "https://example.com/a\nb",
        ] {
            assert_eq!(checked_url(Some(bad)), None, "{bad}");
        }
        assert_eq!(
            checked_url(Some(&format!("https://e.com/{}", "a".repeat(3000)))),
            None
        );
        assert_eq!(checked_url(None), None);
    }

    #[test]
    fn text_is_trimmed_bounded_and_free_of_control_characters() {
        assert_eq!(
            checked_text(Some("  Refund policy ")).as_deref(),
            Some("Refund policy")
        );
        assert_eq!(checked_text(Some("   ")), None);
        assert_eq!(checked_text(Some("a\u{0}b")), None);
        assert_eq!(
            checked_text(Some(&"x".repeat(MAX_TEXT_CHARS))).map(|s| s.len()),
            Some(300)
        );
        assert_eq!(checked_text(Some(&"x".repeat(MAX_TEXT_CHARS + 1))), None);
    }

    #[test]
    fn the_fingerprint_moves_with_every_field_and_ignores_section_order() {
        let base = DocumentCitation {
            title: Some("T".into()),
            source_url: Some("https://e.com/".into()),
            sections: vec![(0, "A".into()), (1, "B".into())],
        };
        let reordered = DocumentCitation {
            sections: vec![(1, "B".into()), (0, "A".into())],
            ..base.clone()
        };
        assert_eq!(base.fingerprint(), reordered.fingerprint());
        let mut other = base.clone();
        other.title = Some("T2".into());
        assert_ne!(base.fingerprint(), other.fingerprint());
        other = base.clone();
        other.source_url = None;
        assert_ne!(base.fingerprint(), other.fingerprint());
        other = base.clone();
        other.sections[1].1 = "C".into();
        assert_ne!(base.fingerprint(), other.fingerprint());
        // "ab"+"" must not equal "a"+"b".
        let a = DocumentCitation {
            title: Some("ab".into()),
            ..Default::default()
        };
        let b = DocumentCitation {
            title: Some("a".into()),
            source_url: Some("b".into()),
            ..Default::default()
        };
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn chunk_attributes_carry_the_document_fields_and_that_chunks_section() {
        let c = DocumentCitation {
            title: Some("T".into()),
            source_url: None,
            sections: vec![(2, "S".into())],
        };
        let hit = c.chunk_attributes(2);
        assert_eq!(hit.get(KEY_TITLE).and_then(Value::as_str), Some("T"));
        assert_eq!(hit.get(KEY_SECTION).and_then(Value::as_str), Some("S"));
        assert!(!hit.contains_key(KEY_SOURCE_URL));
        assert!(!c.chunk_attributes(0).contains_key(KEY_SECTION));
    }
}
