// Ported from graphiti_core/utils/maintenance/edge_operations.py @ 34f56e65 (v0.29.1).
//
// Edge extraction + the per-edge resolution / bi-temporal invalidation pipeline:
//   1. `extract_edges`               — LLM fact-triple extraction (upstream ~117-322),
//   2. `resolve_extracted_edges`     — candidate retrieval + per-edge resolution
//                                       fan-out (upstream ~325-535),
//   3. `resolve_extracted_edge`      — single-edge dedupe / contradiction resolution
//                                       (upstream ~623-847),
//   4. `hydrate_node_summaries`      — Phase-1 node summary refresh (analog of
//                                       node_operations.py:extract_attributes_from_nodes).
//
// FIDELITY NOTES (see also the project fidelity ledger):
//
// * ModelSize: `extract_edges.edge` is called WITHOUT model_size (Medium, the
//   upstream default). `extract_timestamps`, `dedupe_edges.resolve_edge`, and the
//   summary prompt all use `ModelSize::Small` (edge_operations.py:602/729 and
//   node_operations.py:807/979). We mirror that with `.small()`.
//
// * Custom entity-type / edge-type registries (entity_types / edge_types) are a
//   Phase-2 concern. The single-episode core loop here passes neither, so:
//     - `extract_edges` emits no <FACT_TYPES> block (edge_types = None);
//     - `resolve_extracted_edge` extracts no structured edge attributes (the
//       `edge_type_candidates` path, edge_operations.py:655-678 / 780-809, is a
//       HOOK comment, not ported);
//     - `hydrate_node_summaries` refreshes the prose summary only.
//
// * DOCUMENTED DEVIATION — candidate re-ranking (resolve_extracted_edges):
//   Upstream re-ranks the node-pair candidate pool via a hybrid edge search
//   filtered to the pool's UUIDs (edge_operations.py:392-405). Phase-1 uses the
//   node-pair pool from `get_edges_between_nodes` DIRECTLY, capped at
//   RELEVANT_SCHEMA_LIMIT. Same candidate SET; ordering may differ. The
//   invalidation-candidate set still goes through `edge_search` exactly as
//   upstream, minus any UUID already present in the related (pair) pool.
//
// * DOCUMENTED DEVIATION — dedupe_edges context serialization:
//   Upstream interpolates the Python list-of-dicts `[{'idx': i, 'fact': ...}]`
//   into the prompt via the f-string, which calls `str()` and yields a Python
//   `repr` (single-quoted keys/values). We reproduce that `repr` shape via
//   `python_repr_edge_context` so the rendered prompt is byte-faithful.
//
// * DOCUMENTED DEVIATION — candidate-input gathering sequential vs upstream
//   semaphore_gather (resolve_extracted_edges):
//   Upstream collects (related_edges, existing_edges) pairs inside an asyncio
//   gather under the shared semaphore, executing all per-edge lookups concurrently
//   (edge_operations.py:406-430). Phase-1 gathers the pairs in a sequential loop
//   before the tokio::spawn fan-out. The resulting candidate SET is identical;
//   only wall-clock throughput differs (sequential I/O vs concurrent). Flagged
//   for fidelity ledger — no correctness impact, perf concern only.
//
// * DOCUMENTED DEVIATION — persist is 4 sequential driver calls vs upstream single
//   transaction (add_episode.rs persist step):
//   Upstream graphiti_core wraps the final graph writes (upsert nodes, upsert
//   edges, invalidate edges, save episode) in a single Neo4j transaction, giving
//   atomic all-or-nothing semantics. Phase-1 issues 4 sequential driver calls
//   without an enclosing transaction. Partial failures leave the graph in an
//   inconsistent state (e.g., nodes written but edges missing). Real Neo4j drivers
//   (Task 15) MUST consider transactional batching via an explicit session
//   transaction or a multi-statement Cypher batch. Flagged in the fidelity ledger;
//   atomicity guarantee is deferred to the neo4j driver implementation.

use std::sync::Arc;

use serde_json::{Value, json};

use crate::errors::ChronicleError;
use crate::helpers::{RELEVANT_SCHEMA_LIMIT, isoformat, normalize_string_exact, utc_now};
use crate::llm::{LlmRequest, generate_typed};
use crate::pipeline::clients::Clients;
use crate::pipeline::node_ops::previous_episodes_context;
use crate::pipeline::temporal::{expire_new_edge_against_candidates, resolve_edge_contradictions};
use crate::prompts::dedupe_edges::{ResolveEdgeContext, resolve_edge as resolve_edge_prompt};
use crate::prompts::extract_edges::{
    EdgeContext, ExtractTimestampsContext, edge as edge_prompt,
    extract_timestamps as extract_timestamps_prompt,
};
use crate::prompts::models::{EdgeDuplicate, EdgeTimestamps, ExtractedEdges, Summary};
use crate::prompts::summarize_nodes::{SummarizeContext, summarize_context};
use crate::search::{edge_hybrid_search_rrf, edge_search};
use crate::types::{EntityEdge, EntityNode, EpisodicNode};

/// Outcome of [`resolve_extracted_edges`].
///
/// Mirrors the `(resolved_edges, invalidated_edges)` portion of upstream's
/// `(resolved_edges, invalidated_edges, new_edges)` tuple. The `new_edges`
/// sub-list (extracted edges that resolved to themselves) is recoverable from
/// `resolved_edges` by the caller if needed; the Phase-1 core loop only needs
/// resolved + invalidated for persistence.
pub struct EdgeResolutionOutcome {
    pub resolved_edges: Vec<EntityEdge>,
    pub invalidated_edges: Vec<EntityEdge>,
}

// ---------------------------------------------------------------------------
// extract_edges
// ---------------------------------------------------------------------------

/// Extract fact-triple edges from a single episode.
///
/// Port of upstream `extract_edges` (edge_operations.py:117-322), single-episode
/// path. The multi-episode list path, edge-type signatures, and episode
/// attribution suffix are Phase-2 and not ported.
pub async fn extract_edges(
    clients: &Clients,
    episode: &EpisodicNode,
    nodes: &[EntityNode],
    previous_episodes: &[EpisodicNode],
    group_id: &str,
    custom_extraction_instructions: Option<&str>,
) -> Result<Vec<EntityEdge>, ChronicleError> {
    if nodes.is_empty() {
        return Ok(Vec::new());
    }

    // Upstream entities context: [{'name': node.name, 'entity_types': node.labels}].
    let nodes_context: Value = Value::Array(
        nodes
            .iter()
            .map(|n| json!({"name": n.name, "entity_types": n.labels}))
            .collect(),
    );
    let prev = previous_episodes_context(previous_episodes);
    // Reference time = latest episode's valid_at (single-episode → this episode).
    let reference_time = isoformat(episode.valid_at);
    let custom = custom_extraction_instructions.unwrap_or("");

    let ctx = EdgeContext {
        previous_episodes: &prev,
        episode_content: &episode.content,
        nodes: &nodes_context,
        reference_time: &reference_time,
        edge_types: None,
        custom_extraction_instructions: custom,
    };
    let request = LlmRequest::new(edge_prompt(&ctx)).named("extract_edges.edge");
    let extracted: ExtractedEdges = generate_typed(clients.llm.as_ref(), request).await?;

    // Build exact-name → node lookup (upstream `name_to_node`, case-sensitive).
    let mut name_to_node: std::collections::HashMap<&str, &EntityNode> =
        std::collections::HashMap::new();
    for node in nodes {
        name_to_node.insert(node.name.as_str(), node);
    }

    let mut edges = Vec::new();
    for edge_data in extracted.edges {
        // Validate entity names exist in the node list (upstream 217-230).
        let Some(source_node) = name_to_node.get(edge_data.source_entity_name.as_str()) else {
            tracing::warn!(
                relation_type = %edge_data.relation_type,
                "source entity not found in nodes for edge relation"
            );
            continue;
        };
        let Some(target_node) = name_to_node.get(edge_data.target_entity_name.as_str()) else {
            tracing::warn!(
                relation_type = %edge_data.relation_type,
                "target entity not found in nodes for edge relation"
            );
            continue;
        };

        // Drop self-edges (upstream 232-240).
        if source_node.uuid == target_node.uuid {
            tracing::info!(
                node = %source_node.uuid,
                "dropping self-edge (source and target resolve to same node)"
            );
            continue;
        }

        // Filter out empty facts (upstream 260-261).
        if edge_data.fact.trim().is_empty() {
            continue;
        }

        let valid_at = edge_data
            .valid_at
            .as_deref()
            .and_then(|s| parse_iso_or_warn(s, "valid_at"));
        let invalid_at = edge_data
            .invalid_at
            .as_deref()
            .and_then(|s| parse_iso_or_warn(s, "invalid_at"));

        let mut edge = EntityEdge::new(
            source_node.uuid.clone(),
            target_node.uuid.clone(),
            edge_data.relation_type,
            edge_data.fact,
            group_id.to_string(),
        );
        edge.episodes = vec![episode.uuid.clone()];
        edge.created_at = utc_now();
        edge.valid_at = valid_at;
        edge.invalid_at = invalid_at;
        edges.push(edge);
    }

    Ok(edges)
}

/// Parse `datetime.fromisoformat(s.replace('Z', '+00:00'))` with upstream
/// tolerance: on parse failure, log a warning and treat as absent (`None`).
fn parse_iso_or_warn(raw: &str, field: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    let normalized = raw.replace('Z', "+00:00");
    match chrono::DateTime::parse_from_rfc3339(&normalized) {
        Ok(dt) => Some(dt.with_timezone(&chrono::Utc)),
        Err(err) => {
            tracing::warn!(field, input = raw, error = %err, "error parsing edge date, skipping");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// resolve_extracted_edges
// ---------------------------------------------------------------------------

/// Resolve extracted edges against existing graph context.
///
/// Port of upstream `resolve_extracted_edges` (edge_operations.py:325-535) for
/// the Phase-1 candidate-retrieval shape (see the DOCUMENTED DEVIATION at the
/// top of this module re: pair-pool re-ranking).
pub async fn resolve_extracted_edges(
    clients: &Clients,
    extracted_edges: Vec<EntityEdge>,
    episode: &EpisodicNode,
    _nodes: &[EntityNode],
) -> Result<EdgeResolutionOutcome, ChronicleError> {
    // Fast path: dedup exact matches within the extracted edges before resolving
    // (upstream 344-358). Key = (source, target, normalize_string_exact(fact)).
    let mut seen: std::collections::HashSet<(String, String, String)> =
        std::collections::HashSet::new();
    let mut deduplicated: Vec<EntityEdge> = Vec::new();
    for edge in extracted_edges {
        let key = (
            edge.source_node_uuid.clone(),
            edge.target_node_uuid.clone(),
            normalize_string_exact(&edge.fact),
        );
        if seen.insert(key) {
            deduplicated.push(edge);
        }
    }
    let mut extracted_edges = deduplicated;

    if extracted_edges.is_empty() {
        return Ok(EdgeResolutionOutcome {
            resolved_edges: Vec::new(),
            invalidated_edges: Vec::new(),
        });
    }

    // (a) Embed extracted edge facts first (upstream create_entity_edge_embeddings,
    // line 363) — required for the cosine leg of the invalidation-candidate search.
    let facts: Vec<String> = extracted_edges.iter().map(|e| e.fact.clone()).collect();
    let fact_embeddings = clients.embedder.create_batch(&facts).await?;
    for (edge, emb) in extracted_edges.iter_mut().zip(fact_embeddings) {
        edge.fact_embedding = Some(emb);
    }

    // (b) related_edges per extracted edge = the node-pair pool, capped at
    // RELEVANT_SCHEMA_LIMIT (DOCUMENTED DEVIATION: no hybrid re-rank).
    // (d) existing_edges (invalidation candidates) per edge = edge_search over the
    // fact, minus any uuid already in the related pool (upstream 392-430).
    let mut per_edge_inputs: Vec<(Vec<EntityEdge>, Vec<EntityEdge>)> =
        Vec::with_capacity(extracted_edges.len());
    for edge in &extracted_edges {
        let mut related = clients
            .driver
            .get_edges_between_nodes(&edge.source_node_uuid, &edge.target_node_uuid)
            .await?;
        related.truncate(RELEVANT_SCHEMA_LIMIT);
        let related_uuids: std::collections::HashSet<String> =
            related.iter().map(|e| e.uuid.clone()).collect();

        let invalidation_hits = edge_search(
            clients.driver.as_ref(),
            clients.embedder.as_ref(),
            &edge.fact,
            std::slice::from_ref(&edge.group_id),
            &edge_hybrid_search_rrf(),
        )
        .await?;
        let existing: Vec<EntityEdge> = invalidation_hits
            .into_iter()
            .filter(|e| !related_uuids.contains(&e.uuid))
            .collect();

        per_edge_inputs.push((related, existing));
    }

    // Per-edge resolution, fanned out under the shared semaphore (upstream 489-509).
    let mut handles = Vec::with_capacity(extracted_edges.len());
    for (edge, (related, existing)) in extracted_edges.iter().cloned().zip(per_edge_inputs) {
        let clients = clients.clone();
        let episode = episode.clone();
        let semaphore = Arc::clone(&clients.semaphore);
        handles.push(tokio::spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .map_err(|e| ChronicleError::InvalidInput(format!("semaphore closed: {e}")))?;
            resolve_extracted_edge(&clients, edge, related, existing, &episode).await
        }));
    }

    let mut resolved_edges: Vec<EntityEdge> = Vec::with_capacity(handles.len());
    let mut invalidated_edges: Vec<EntityEdge> = Vec::new();
    for handle in handles {
        let (resolved, invalidated) = handle.await.map_err(|e| {
            ChronicleError::InvalidInput(format!("edge resolution task panicked: {e}"))
        })??;
        resolved_edges.push(resolved);
        invalidated_edges.extend(invalidated);
    }

    Ok(EdgeResolutionOutcome {
        resolved_edges,
        invalidated_edges,
    })
}

// ---------------------------------------------------------------------------
// resolve_extracted_edge
// ---------------------------------------------------------------------------

/// Resolve a single extracted edge against its related (duplicate-candidate) and
/// existing (invalidation-candidate) pools.
///
/// Port of upstream `resolve_extracted_edge` (edge_operations.py:623-847),
/// without the custom edge-type attribute extraction (Phase-2 HOOK).
pub async fn resolve_extracted_edge(
    clients: &Clients,
    extracted_edge: EntityEdge,
    related_edges: Vec<EntityEdge>,
    existing_edges: Vec<EntityEdge>,
    episode: &EpisodicNode,
) -> Result<(EntityEdge, Vec<EntityEdge>), ChronicleError> {
    let now = utc_now();

    // Fast path: both pools empty → only extract timestamps for the new edge.
    if related_edges.is_empty() && existing_edges.is_empty() {
        let mut edge = extracted_edge;
        extract_edge_timestamps(clients, &mut edge, episode).await?;
        // expire-new stamping (upstream 822-823 applies inside the shared block;
        // with no candidates only the invalid_at→expired_at stamp can fire).
        if edge.invalid_at.is_some() && edge.expired_at.is_none() {
            edge.expired_at = Some(now);
        }
        return Ok((edge, Vec::new()));
    }

    // Fast path: exact normalized fact match (same endpoints) in related_edges
    // (upstream 684-695). Reuse the existing edge, appending this episode's uuid.
    let normalized_fact = normalize_string_exact(&extracted_edge.fact);
    for edge in &related_edges {
        if edge.source_node_uuid == extracted_edge.source_node_uuid
            && edge.target_node_uuid == extracted_edge.target_node_uuid
            && normalize_string_exact(&edge.fact) == normalized_fact
        {
            let mut resolved = edge.clone();
            if !resolved.episodes.contains(&episode.uuid) {
                resolved.episodes.push(episode.uuid.clone());
            }
            return Ok((resolved, Vec::new()));
        }
    }

    // LLM dedupe path (upstream 699-776). Build the continuous-index context:
    // EXISTING FACTS = related (idx 0..n), INVALIDATION CANDIDATES = existing
    // (idx n..n+m).
    let related_context = python_repr_edge_context(&related_edges, 0);
    let invalidation_offset = related_edges.len();
    let invalidation_context = python_repr_edge_context(&existing_edges, invalidation_offset);

    let ctx = ResolveEdgeContext {
        existing_edges: &related_context,
        edge_invalidation_candidates: &invalidation_context,
        new_edge: &extracted_edge.fact,
    };
    let request = LlmRequest::new(resolve_edge_prompt(&ctx))
        .small()
        .named("dedupe_edges.resolve_edge");
    let response: EdgeDuplicate = generate_typed(clients.llm.as_ref(), request).await?;

    // duplicate_facts: only valid indices into related_edges (upstream 736-749).
    let mut resolved_edge = extracted_edge.clone();
    let mut is_duplicate = false;
    for &idx in &response.duplicate_facts {
        if idx >= 0 && (idx as usize) < related_edges.len() {
            resolved_edge = related_edges[idx as usize].clone();
            is_duplicate = true;
            break;
        }
        tracing::warn!(
            idx,
            valid_max = related_edges.len() as i64 - 1,
            "LLM returned invalid duplicate_facts idx (EXISTING FACTS range)"
        );
    }
    if is_duplicate && !resolved_edge.episodes.contains(&episode.uuid) {
        resolved_edge.episodes.push(episode.uuid.clone());
    }

    // contradicted_facts: map indices over the combined range to candidates
    // (upstream 754-776).
    let max_valid_idx = related_edges.len() + existing_edges.len();
    let mut invalidation_candidates: Vec<EntityEdge> = Vec::new();
    for &idx in &response.contradicted_facts {
        if idx < 0 || (idx as usize) >= max_valid_idx {
            tracing::warn!(
                idx,
                valid_max = max_valid_idx as i64 - 1,
                "LLM returned invalid contradicted_facts idx"
            );
            continue;
        }
        let idx = idx as usize;
        if idx < related_edges.len() {
            invalidation_candidates.push(related_edges[idx].clone());
        } else {
            invalidation_candidates.push(existing_edges[idx - invalidation_offset].clone());
        }
    }

    // Timestamps for NEW edges only (resolved == extracted), upstream 811-813.
    if resolved_edge.uuid == extracted_edge.uuid {
        extract_edge_timestamps(clients, &mut resolved_edge, episode).await?;
    }

    // Expire new edge against candidates (upstream 820-839), then determine which
    // contradictory candidates the new edge invalidates (upstream 842-844).
    expire_new_edge_against_candidates(&mut resolved_edge, &invalidation_candidates, now);
    let invalidated_edges =
        resolve_edge_contradictions(&resolved_edge, invalidation_candidates, now);

    Ok((resolved_edge, invalidated_edges))
}

/// Extract `valid_at` / `invalid_at` for an edge via the lightweight timestamps
/// prompt. Skips when timestamps are already set (upstream `_extract_edge_timestamps`,
/// edge_operations.py:576-621). Mutates the edge in place.
async fn extract_edge_timestamps(
    clients: &Clients,
    edge: &mut EntityEdge,
    episode: &EpisodicNode,
) -> Result<(), ChronicleError> {
    if edge.valid_at.is_some() || edge.invalid_at.is_some() {
        return Ok(());
    }

    let reference_time = isoformat(episode.valid_at);
    let ctx = ExtractTimestampsContext {
        fact: &edge.fact,
        reference_time: &reference_time,
    };
    let request = LlmRequest::new(extract_timestamps_prompt(&ctx))
        .small()
        .named("extract_edges.extract_timestamps");
    // Upstream swallows extraction errors (logs warning, leaves timestamps unset).
    let timestamps: EdgeTimestamps = match generate_typed(clients.llm.as_ref(), request).await {
        Ok(t) => t,
        Err(err) => {
            tracing::warn!(edge = %edge.uuid, error = %err, "failed to extract timestamps for edge");
            return Ok(());
        }
    };

    if let Some(valid_at) = timestamps.valid_at.as_deref() {
        edge.valid_at = parse_iso_or_warn(valid_at, "valid_at");
    }
    if let Some(invalid_at) = timestamps.invalid_at.as_deref() {
        edge.invalid_at = parse_iso_or_warn(invalid_at, "invalid_at");
    }
    Ok(())
}

/// Reproduce Python's `str([{'idx': i, 'fact': edge.fact}, ...])` repr for the
/// dedupe_edges prompt context. Python `repr` of a str uses single quotes (and
/// switches to double quotes only when the string itself contains a single quote
/// but no double quote). We replicate the common single-quote case and escape
/// backslashes / single quotes so the output is valid Python-list syntax.
fn python_repr_edge_context(edges: &[EntityEdge], offset: usize) -> String {
    let items: Vec<String> = edges
        .iter()
        .enumerate()
        .map(|(i, edge)| {
            format!(
                "{{'idx': {}, 'fact': {}}}",
                offset + i,
                python_repr_str(&edge.fact)
            )
        })
        .collect();
    format!("[{}]", items.join(", "))
}

/// Reproduce CPython's `unicode_repr()` for a string value.
///
/// Quote selection mirrors CPython (Objects/unicodeobject.c `unicode_repr`):
/// - default: single-quoted `'...'`
/// - if the string contains `'` but not `"`: double-quoted `"..."`
/// - if both are present: single-quoted with `'` escaped as `\'`
///
/// Character escaping order (CPython priority):
/// 1. `\` → `\\`
/// 2. active quote char → `\'` or `\"`
/// 3. `\n` → `\n`, `\r` → `\r`, `\t` → `\t`
/// 4. C0 controls (< 0x20) and DEL (0x7f) → `\xNN` (lowercase 2-digit hex)
/// 5. C1 range 0x80–0xa0 (inclusive) → `\xNN`  (CPython treats these as
///    non-printable; `unicodedata.category` returns Cc/Cf/Zs for the range;
///    verified: `repr('\xa0')` → `'\\xa0'`, `repr('\xa1')` → `'¡'`)
/// 6. All other chars (printable Unicode): pass through verbatim.
fn python_repr_str(s: &str) -> String {
    let has_single = s.contains('\'');
    let has_double = s.contains('"');
    let (quote, escape_quote) = if has_single && !has_double {
        ('"', '"')
    } else {
        ('\'', '\'')
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            c if c == escape_quote => {
                out.push('\\');
                out.push(c);
            }
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c < '\x20') || c == '\x7f' => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c if ('\u{0080}'..='\u{00a0}').contains(&c) => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

// ---------------------------------------------------------------------------
// Node summary hydration (Phase-1 analog of extract_attributes_from_nodes)
// ---------------------------------------------------------------------------

/// Refresh each node's prose summary from the current episode via the
/// `summarize_nodes.summarize_context` prompt (ModelSize Small).
///
/// Phase-1 scope of upstream `extract_attributes_from_nodes`
/// (node_operations.py:726): summary only. Custom-attribute extraction driven by
/// an entity-types registry is Phase-2 (HOOK).
///
/// Runs one LLM call per node, fanned out under the shared semaphore. Nodes are
/// returned in input order with their `summary` updated.
pub async fn hydrate_node_summaries(
    clients: &Clients,
    nodes: Vec<EntityNode>,
    episode: &EpisodicNode,
    previous_episodes: &[EpisodicNode],
) -> Result<Vec<EntityNode>, ChronicleError> {
    if nodes.is_empty() {
        return Ok(nodes);
    }

    let prev = previous_episodes_context(previous_episodes);
    let episode_content = Value::String(episode.content.clone());

    let mut handles = Vec::with_capacity(nodes.len());
    for node in nodes {
        let clients = clients.clone();
        let prev = prev.clone();
        let episode_content = episode_content.clone();
        let semaphore = Arc::clone(&clients.semaphore);
        handles.push(tokio::spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .map_err(|e| ChronicleError::InvalidInput(format!("semaphore closed: {e}")))?;
            let attributes = Value::Object(node.attributes.clone());
            let ctx = SummarizeContext {
                previous_episodes: &prev,
                episode_content: &episode_content,
                node_name: &node.name,
                node_summary: &node.summary,
                attributes: &attributes,
            };
            let request = LlmRequest::new(summarize_context(&ctx))
                .small()
                .named("summarize_nodes.summarize_context");
            let summary: Summary = generate_typed(clients.llm.as_ref(), request).await?;
            let mut node = node;
            node.summary = summary.summary;
            Ok::<EntityNode, ChronicleError>(node)
        }));
    }

    let mut out = Vec::with_capacity(handles.len());
    for handle in handles {
        let node = handle.await.map_err(|e| {
            ChronicleError::InvalidInput(format!("summary hydration task panicked: {e}"))
        })??;
        out.push(node);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_iso_handles_z_suffix() {
        let dt = parse_iso_or_warn("2025-04-30T00:00:00Z", "valid_at").unwrap();
        assert_eq!(dt.to_rfc3339(), "2025-04-30T00:00:00+00:00");
    }

    #[test]
    fn parse_iso_returns_none_on_garbage() {
        assert!(parse_iso_or_warn("not-a-date", "valid_at").is_none());
    }

    #[test]
    fn python_repr_str_single_quotes_by_default() {
        assert_eq!(
            python_repr_str("Alice works at Acme"),
            "'Alice works at Acme'"
        );
    }

    #[test]
    fn python_repr_str_switches_to_double_quotes_with_apostrophe() {
        assert_eq!(python_repr_str("Bob's job"), "\"Bob's job\"");
    }

    #[test]
    fn python_repr_str_escapes_when_both_quotes_present() {
        // Contains both ' and " → single-quote with the ' escaped (CPython behaviour).
        assert_eq!(python_repr_str("a'b\"c"), "'a\\'b\"c'");
    }

    // --- control-character escaping (CPython repr fidelity) ---

    #[test]
    fn python_repr_str_escapes_newline() {
        // python3: repr('a\nb') == "'a\\nb'"
        assert_eq!(python_repr_str("a\nb"), "'a\\nb'");
    }

    #[test]
    fn python_repr_str_escapes_tab() {
        // python3: repr('a\tb') == "'a\\tb'"
        assert_eq!(python_repr_str("a\tb"), "'a\\tb'");
    }

    #[test]
    fn python_repr_str_escapes_carriage_return() {
        // python3: repr('a\rb') == "'a\\rb'"
        assert_eq!(python_repr_str("a\rb"), "'a\\rb'");
    }

    #[test]
    fn python_repr_str_escapes_nul() {
        // python3: repr('a\x00b') == "'a\\x00b'"
        assert_eq!(python_repr_str("a\x00b"), "'a\\x00b'");
    }

    #[test]
    fn python_repr_str_escapes_del() {
        // python3: repr('a\x7fb') == "'a\\x7fb'"
        assert_eq!(python_repr_str("a\x7fb"), "'a\\x7fb'");
    }

    #[test]
    fn python_repr_str_escapes_0x85_nel() {
        // python3: repr('a\x85b') == "'a\\x85b'"
        assert_eq!(python_repr_str("a\u{0085}b"), "'a\\x85b'");
    }

    #[test]
    fn python_repr_str_escapes_0xa0_nbsp() {
        // python3: repr('a\xa0b') == "'a\\xa0b'"
        assert_eq!(python_repr_str("a\u{00a0}b"), "'a\\xa0b'");
    }

    #[test]
    fn python_repr_str_passes_through_0xa1_printable() {
        // python3: repr('\xa1') == "'¡'"  (printable, not escaped)
        assert_eq!(python_repr_str("\u{00a1}"), "'\u{00a1}'");
    }

    #[test]
    fn python_repr_str_mixed_newline_and_apostrophe() {
        // python3: repr("a\nb'c") == '"a\\nb\'c"'
        // Has single-quote but no double-quote → double-quoted outer.
        assert_eq!(python_repr_str("a\nb'c"), "\"a\\nb'c\"");
    }

    #[test]
    fn python_repr_edge_context_offsets_indices() {
        let mut e0 = EntityEdge::new(
            "s".into(),
            "t".into(),
            "R".into(),
            "fact zero".into(),
            "g".into(),
        );
        e0.uuid = "e0".into();
        let mut e1 = EntityEdge::new(
            "s".into(),
            "t".into(),
            "R".into(),
            "fact one".into(),
            "g".into(),
        );
        e1.uuid = "e1".into();
        let rendered = python_repr_edge_context(&[e0, e1], 3);
        assert_eq!(
            rendered,
            "[{'idx': 3, 'fact': 'fact zero'}, {'idx': 4, 'fact': 'fact one'}]"
        );
    }
}
