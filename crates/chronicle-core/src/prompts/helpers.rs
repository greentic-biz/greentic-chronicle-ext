// Ported from graphiti_core/prompts/prompt_helpers.py @ 34f56e65 (v0.29.1)

/// Appended to every system prompt.
/// Upstream constant: DO_NOT_ESCAPE_UNICODE = '\nDo not escape unicode characters.\n'
pub const DO_NOT_ESCAPE_UNICODE: &str = "\nDo not escape unicode characters.\n";

/// Upstream to_prompt_json: minified JSON, unicode preserved (ensure_ascii=False, indent=None).
/// serde_json::to_string already preserves non-ASCII characters by default (no escaping),
/// matching the Python json.dumps(data, ensure_ascii=False, indent=None) behaviour.
pub fn to_prompt_json<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
}
