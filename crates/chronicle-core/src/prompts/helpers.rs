// Ported from graphiti_core/prompts/prompt_helpers.py @ 34f56e65 (v0.29.1)

/// Appended to every system prompt.
/// Upstream constant: DO_NOT_ESCAPE_UNICODE = '\nDo not escape unicode characters.\n'
pub const DO_NOT_ESCAPE_UNICODE: &str = "\nDo not escape unicode characters.\n";

/// Upstream `to_prompt_json` is `json.dumps(data, ensure_ascii=False, indent=None)`.
///
/// Two byte-level details must be matched for verbatim prompt fidelity:
/// - unicode preserved (`ensure_ascii=False`): serde_json preserves non-ASCII by default.
/// - separators: Python's `json.dumps` with `indent=None` uses the DEFAULT separators
///   `(', ', ': ')` — a space after every `,` and `:`. serde_json's `to_string` uses
///   `(',', ':')` (no spaces). To render byte-identical prompts we use a custom
///   `Formatter` that reproduces Python's default spacing.
pub fn to_prompt_json<T: serde::Serialize>(value: &T) -> String {
    let mut buf = Vec::new();
    let formatter = PythonDefaultFormatter;
    let mut serializer = serde_json::Serializer::with_formatter(&mut buf, formatter);
    if let Err(err) = value.serialize(&mut serializer) {
        tracing::error!(error = %err, "to_prompt_json serialization failed; emitting null");
        return "null".to_string();
    }
    String::from_utf8(buf).unwrap_or_else(|err| {
        tracing::error!(error = %err, "to_prompt_json produced invalid utf-8; emitting null");
        "null".to_string()
    })
}

/// Reproduces Python `json.dumps` default separators `(', ', ': ')` so prompt JSON
/// interpolations are byte-identical to upstream graphiti.
struct PythonDefaultFormatter;

impl serde_json::ser::Formatter for PythonDefaultFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        writer.write_all(b": ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_prompt_json_matches_python_json_dumps_separators() {
        // Python: json.dumps([{"entity_type_id": 0, "entity_type_name": "Person"}],
        //                    ensure_ascii=False)
        //   -> '[{"entity_type_id": 0, "entity_type_name": "Person"}]'
        let value = serde_json::json!([{"entity_type_id": 0, "entity_type_name": "Person"}]);
        assert_eq!(
            to_prompt_json(&value),
            r#"[{"entity_type_id": 0, "entity_type_name": "Person"}]"#
        );
    }

    #[test]
    fn to_prompt_json_preserves_unicode() {
        // ensure_ascii=False: non-ASCII characters preserved verbatim, not \uXXXX-escaped.
        let value = serde_json::json!({"name": "東京"});
        assert_eq!(to_prompt_json(&value), r#"{"name": "東京"}"#);
    }
}
