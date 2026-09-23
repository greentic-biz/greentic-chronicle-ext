use std::fmt::Display;

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

/// Every error leaves as `{"error":{"code","message"}}`. The designer acts on
/// `code` only; `message` is for logs and never carries a key, a document
/// body or a vector.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    pub fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }

    pub fn unauthorized() -> Self {
        Self::new(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "missing or unknown api key",
        )
    }

    pub fn forbidden() -> Self {
        Self::new(
            StatusCode::FORBIDDEN,
            "unauthorized",
            "the api key does not cover this tenant or team",
        )
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "bad_request", message)
    }

    pub fn index_not_found() -> Self {
        Self::new(StatusCode::NOT_FOUND, "index_not_found", "no such index")
    }

    pub fn model_mismatch() -> Self {
        Self::new(
            StatusCode::CONFLICT,
            "model_mismatch",
            "the index was created with another embedding model",
        )
    }

    pub fn dim_mismatch(status: StatusCode) -> Self {
        Self::new(
            status,
            "dim_mismatch",
            "vector dimension does not match the index",
        )
    }

    /// Logs the cause and answers a generic 500: the cause may name paths or
    /// store internals a caller has no business reading.
    pub fn internal(cause: impl Display) -> Self {
        tracing::error!(error = %cause, "internal error");
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            "internal error",
        )
    }

    pub fn status(&self) -> StatusCode {
        self.status
    }

    pub fn code(&self) -> &'static str {
        self.code
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({"error": {"code": self.code, "message": self.message}});
        (self.status, Json(body)).into_response()
    }
}

impl From<chronicle_core::ChronicleError> for ApiError {
    fn from(err: chronicle_core::ChronicleError) -> Self {
        match err {
            chronicle_core::ChronicleError::InvalidInput(msg) => Self::bad_request(msg),
            other => Self::internal(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    #[tokio::test]
    async fn the_body_is_the_error_envelope_the_designer_parses() {
        let resp = ApiError::model_mismatch().into_response();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let bytes = to_bytes(resp.into_body(), 1024).await.expect("body");
        let json: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        assert_eq!(json["error"]["code"], "model_mismatch");
    }

    #[tokio::test]
    async fn an_internal_error_never_echoes_its_cause() {
        let resp = ApiError::internal("disk at /secret/path failed").into_response();
        let bytes = to_bytes(resp.into_body(), 1024).await.expect("body");
        assert!(!String::from_utf8_lossy(&bytes).contains("/secret/path"));
    }
}
