// Ported from graphiti_core/graphiti.py @ 34f56e65 (v0.29.1):
//   - _process_episode_data saga block (~743-781)
//   - _get_or_create_saga (~346-392)
//   - _saga_get_previous_episode_uuid (~394-420)
//   - summarize_saga (~438-568)
//
// Saga narrative-thread association for the single-episode add path, plus the
// incremental saga summarizer. The bulk path has its own (multi-episode)
// association helper in pipeline::bulk; this module covers the single-episode
// threading and the summarizer.

use chrono::{DateTime, Utc};

use crate::errors::ChronicleError;
use crate::helpers::utc_now;
use crate::llm::{LlmRequest, generate_typed};
use crate::pipeline::clients::Clients;
use crate::pipeline::community_ops::MAX_SUMMARY_CHARS;
use crate::prompts::models::SagaSummary;
use crate::prompts::summarize_sagas::{
    SummarizeSagaContext, summarize_saga as summarize_saga_prompt,
};
use crate::types::{EpisodicNode, HasEpisodeEdge, NextEpisodeEdge, SagaNode};

/// Thread a saved primary episode into a saga (single-episode add path).
///
/// Port of upstream `_process_episode_data` saga block (graphiti.py ~743-781):
/// get-or-create the saga by `(name, group_id)` (anchoring a fresh saga's
/// `created_at` to the episode's `valid_at`), resolve the previous episode (the
/// caller-supplied `saga_previous_episode_uuid` if any, else the driver's latest
/// HAS_EPISODE-by-valid_at query), save a `NEXT_EPISODE(prev → current)` edge when
/// a previous episode exists, save the `HAS_EPISODE(saga → current)` edge, then
/// advance the saga's first/last episode pointers and persist it.
pub async fn associate_episode_with_saga(
    clients: &Clients,
    saga_name: &str,
    group_id: &str,
    primary_episode: &EpisodicNode,
    saga_previous_episode_uuid: Option<&str>,
    now: DateTime<Utc>,
) -> Result<SagaNode, ChronicleError> {
    // Get-or-create the saga node. A fresh saga inherits the originating
    // episode's reference time (valid_at) as its created_at (upstream).
    let mut saga_node = match clients.driver.get_saga_by_name(saga_name, group_id).await? {
        Some(existing) => existing,
        None => {
            let saga = SagaNode::new(
                saga_name.to_string(),
                group_id.to_string(),
                primary_episode.valid_at,
            );
            clients.driver.save_saga_node(&saga).await?;
            saga
        }
    };

    // Resolve the previous episode: caller-supplied wins, else query.
    let previous_episode_uuid: Option<String> = match saga_previous_episode_uuid {
        Some(uuid) => Some(uuid.to_string()),
        None => {
            clients
                .driver
                .saga_previous_episode_uuid(&saga_node.uuid, &primary_episode.uuid)
                .await?
        }
    };

    // Chain NEXT_EPISODE from the previous episode to the new one.
    if let Some(prev_uuid) = &previous_episode_uuid {
        let next_edge = NextEpisodeEdge::new(
            prev_uuid.clone(),
            primary_episode.uuid.clone(),
            group_id.to_string(),
            now,
        );
        clients.driver.save_next_episode_edge(&next_edge).await?;
    }

    // HAS_EPISODE from the saga to the new episode.
    let has_edge = HasEpisodeEdge::new(
        saga_node.uuid.clone(),
        primary_episode.uuid.clone(),
        group_id.to_string(),
        now,
    );
    clients.driver.save_has_episode_edge(&has_edge).await?;

    // Advance first/last episode pointers.
    if saga_node.first_episode_uuid.is_none() {
        saga_node.first_episode_uuid = Some(primary_episode.uuid.clone());
    }
    saga_node.last_episode_uuid = Some(primary_episode.uuid.clone());
    clients.driver.save_saga_node(&saga_node).await?;

    Ok(saga_node)
}

/// Incrementally summarize a saga using only episodes added since the last run.
///
/// Port of upstream `Graphiti.summarize_saga` (graphiti.py ~438-568, plan R9).
/// Two watermarks are maintained on the saga node with deliberately different
/// semantics:
/// - `last_summarized_at` (wall-clock) is the *filter* watermark: the next run
///   picks up any episode whose `created_at` is greater than this value, so a
///   backfilled episode (past `valid_at`, `created_at = now`) is still picked up.
/// - `last_summarized_episode_valid_at` (episode-time) is the maximum `valid_at`
///   across the episodes covered by the current summary; it only advances
///   forward and never regresses.
///
/// If no new episodes are found, the saga is returned unchanged (no LLM call,
/// no watermark update).
pub async fn summarize_saga(
    clients: &Clients,
    saga_uuid: &str,
) -> Result<SagaNode, ChronicleError> {
    let mut saga = clients
        .driver
        .get_saga_by_uuid(saga_uuid)
        .await?
        .ok_or_else(|| ChronicleError::NodeNotFound {
            uuid: saga_uuid.to_string(),
        })?;

    // Fetch only episodes added since the last summary (or all if never
    // summarized). The driver applies the created_at > since filter and returns
    // chronological (content, valid_at) rows.
    const MAX_EPISODES: usize = 200;
    let episodes_data = clients
        .driver
        .saga_episode_contents(saga_uuid, saga.last_summarized_at, MAX_EPISODES)
        .await?;

    if episodes_data.is_empty() {
        tracing::info!(
            saga_uuid,
            "no new episodes found for saga, skipping summary"
        );
        return Ok(saga);
    }

    let episode_contents: Vec<String> = episodes_data
        .iter()
        .map(|(content, _)| content.clone())
        .collect();
    let valid_ats: Vec<DateTime<Utc>> = episodes_data.iter().map(|(_, v)| *v).collect();

    let ctx = SummarizeSagaContext {
        saga_name: &saga.name,
        existing_summary: &saga.summary,
        episodes: &episode_contents,
    };
    let request =
        LlmRequest::new(summarize_saga_prompt(&ctx)).named("summarize_sagas.summarize_saga");
    let response: SagaSummary = generate_typed(clients.llm.as_ref(), request).await?;

    let mut summary = response.summary;
    if summary.len() > MAX_SUMMARY_CHARS {
        // Truncate on a char boundary at or before MAX_SUMMARY_CHARS (upstream
        // slices by code units; we clamp to the nearest char boundary so the
        // byte slice is always valid UTF-8).
        let mut end = MAX_SUMMARY_CHARS;
        while end > 0 && !summary.is_char_boundary(end) {
            end -= 1;
        }
        summary.truncate(end);
    }
    saga.summary = summary;

    // Wall-clock filter watermark: keeps backfilled episodes reachable next run.
    saga.last_summarized_at = Some(utc_now());

    // Episode-time watermark: advance forward only to the latest reference time
    // just summarized; leave unchanged if no episode carried a valid_at.
    if let Some(new_watermark) = valid_ats.into_iter().max()
        && saga
            .last_summarized_episode_valid_at
            .is_none_or(|prev| new_watermark > prev)
    {
        saga.last_summarized_episode_valid_at = Some(new_watermark);
    }

    clients.driver.save_saga_node(&saga).await?;
    tracing::info!(saga_uuid, "updated summary for saga");

    Ok(saga)
}
