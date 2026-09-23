use axum::extract::{Request, State};
use axum::http::HeaderMap;
use axum::middleware::Next;
use axum::response::Response;
use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use rand::RngCore as _;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq as _;

use crate::error::ApiError;
use crate::meta::KeyRecord;
use crate::state::AppState;
use crate::wire::{valid_index_id, valid_slug};

pub const KEY_PREFIX: &str = "cix_";
pub const DEFAULT_TEAM: &str = "general";
pub const ALL_TEAMS: &str = "*";

pub fn generate_api_key() -> String {
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    format!("{KEY_PREFIX}{}", URL_SAFE_NO_PAD.encode(bytes))
}

pub fn hash_key(key: &str) -> String {
    format!("{:x}", Sha256::digest(key.as_bytes()))
}

fn bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(axum::http::header::AUTHORIZATION)?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
        .map(str::trim)
        .filter(|k| !k.is_empty())
}

fn header<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)?
        .to_str()
        .ok()
        .map(str::trim)
        .filter(|v| !v.is_empty())
}

/// The tenant and team a request is scoped to, established by
/// [`require_tenant_key`] and read by every `/v1` handler.
#[derive(Debug, Clone)]
pub struct Scope {
    pub tenant: String,
    pub team: String,
}

impl Scope {
    pub fn group_id(&self, index_id: &str) -> Result<String, ApiError> {
        if !valid_index_id(index_id) {
            return Err(ApiError::bad_request("invalid index id"));
        }
        Ok(format!("idx:{}:{}:{index_id}", self.tenant, self.team))
    }
}

fn covers(key: &KeyRecord, tenant: &str, team: &str) -> bool {
    key.tenant_slug == tenant && key.teams.iter().any(|t| t == ALL_TEAMS || t == team)
}

pub async fn require_tenant_key(
    State(state): State<AppState>,
    mut req: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let key = bearer(req.headers())
        .ok_or_else(ApiError::unauthorized)?
        .to_string();
    let tenant = header(req.headers(), "x-greentic-tenant")
        .filter(|t| valid_slug(t))
        .ok_or_else(|| ApiError::bad_request("X-Greentic-Tenant is missing or invalid"))?
        .to_string();
    let team = match header(req.headers(), "x-greentic-team") {
        Some(team) if valid_slug(team) => team.to_string(),
        Some(_) => return Err(ApiError::bad_request("X-Greentic-Team is invalid")),
        None => DEFAULT_TEAM.to_string(),
    };
    let record = state
        .meta
        .key_by_hash(&hash_key(&key))
        .await?
        .ok_or_else(ApiError::unauthorized)?;
    if !covers(&record, &tenant, &team) {
        return Err(ApiError::forbidden());
    }
    req.extensions_mut().insert(Scope { tenant, team });
    Ok(next.run(req).await)
}

pub async fn require_bootstrap(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let presented = bearer(req.headers()).ok_or_else(ApiError::unauthorized)?;
    let presented: [u8; 32] = Sha256::digest(presented.as_bytes()).into();
    if !bool::from(presented.ct_eq(state.bootstrap_hash())) {
        return Err(ApiError::unauthorized());
    }
    Ok(next.run(req).await)
}
