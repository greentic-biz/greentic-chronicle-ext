// Ported from graphiti_core/llm_client/config.py @ 34f56e65 (v0.29.1)
//
// Upstream DEFAULT_MAX_TOKENS = 16384 (config.py line 19).
// Upstream has two DEFAULT_TEMPERATURE values: config.py defines 1 but
// client.py (base client) defines DEFAULT_TEMPERATURE = 0 and uses it as
// the actual default for LLMClient. We follow client.py (0) here since
// that is the runtime default applied to all LLM calls.

/// Maximum tokens for LLM generation (upstream `DEFAULT_MAX_TOKENS`).
pub const DEFAULT_MAX_TOKENS: u32 = 16384;

/// Default sampling temperature (upstream base-client `DEFAULT_TEMPERATURE = 0`).
pub const DEFAULT_TEMPERATURE: f32 = 0.0;

#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub small_model: Option<String>,
    pub base_url: Option<String>,
    pub temperature: f32,
    pub max_tokens: u32,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            model: None,
            small_model: None,
            base_url: None,
            temperature: DEFAULT_TEMPERATURE,
            max_tokens: DEFAULT_MAX_TOKENS,
        }
    }
}
