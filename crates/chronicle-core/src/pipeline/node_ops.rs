// Ported from graphiti_core/utils/maintenance/node_operations.py @ 34f56e65 (v0.29.1)
//
// Node extraction + the three-stage deduplication / resolution pipeline:
//   1. semantic candidate retrieval (per-node cosine search, fanned out under
//      the shared semaphore),
//   2. deterministic resolution (`_resolve_with_similarity`: exact normalized
//      name, then entropy-gated MinHash/LSH fuzzy match),
//   3. LLM escalation for whatever remains unresolved (`_resolve_with_llm`).
//
// FIDELITY NOTES (see also the project fidelity ledger):
//
// * ModelSize: upstream extraction (`_call_extraction_llm`) and dedupe
//   (`_resolve_with_llm`) both call `generate_response` WITHOUT a `model_size`
//   argument, so both default to `ModelSize::Medium`. (The `ModelSize.small`
//   call sites at node_operations.py:807/979 are summary/attribute prompts —
//   Task 14, not here.) We therefore leave `LlmRequest::model_size` at its
//   `Medium` default and do NOT call `.small()`.
//
// * Reflexion: there is NO reflexion loop in v0.29.1 — `grep -rn reflexion
//   graphiti_core/` returns nothing. The whole "extract → reflect → re-extract"
//   machinery was removed before this tag, so there is nothing to port and no
//   `MAX_REFLEXION_ITERATIONS` constant. Extraction is a single LLM call.
//
// * Attributes / summary: upstream constructs each `EntityNode` here with
//   `summary=''` and no attributes; `extract_attributes_from_nodes`
//   (node_operations.py:726) runs LATER as a separate hydration step (Task 14).
//   We mirror that: extraction leaves `summary` empty and `attributes` empty.
//   See the HOOK comment at the end of `extract_nodes`.
//
// * `previous_episodes` context shape: upstream builds
//   `[{'content': ep.content, 'timestamp': ep.valid_at.isoformat()} ...]` and
//   renders the whole list via `to_prompt_json`. We reproduce this exactly via
//   [`previous_episodes_context`], which the prompt fns now consume as a
//   `&serde_json::Value` (a JSON array of `{content, timestamp}` objects). The
//   `timestamp` matches Python `datetime.isoformat()` byte-for-byte (see
//   [`crate::helpers::isoformat`]). This was previously a `&[String]`
//   content-only divergence; it is now byte-identical to upstream.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::errors::ChronicleError;
use crate::helpers::{isoformat, utc_now};
use crate::llm::{LlmRequest, generate_typed};
use crate::pipeline::clients::Clients;
use crate::pipeline::dedup_helpers::{
    DedupCandidateIndexes, DedupResolutionState, build_candidate_indexes, promote_resolved_node,
};
use crate::prompts::dedupe_nodes::{NodesContext, nodes as dedupe_nodes_prompt};
use crate::prompts::extract_nodes::{
    ExtractJsonContext, ExtractMessageContext, ExtractTextContext, extract_json, extract_message,
    extract_text,
};
use crate::prompts::models::{ExtractedEntities, NodeResolutions};
use crate::types::{EntityNode, EpisodeType, EpisodicNode};

// Upstream node_operations.py:63-65.
pub const MAX_NODES: usize = 30;
pub const NODE_DEDUP_CANDIDATE_LIMIT: usize = 15;
pub const NODE_DEDUP_COSINE_MIN_SCORE: f32 = 0.6;

/// Outcome of [`resolve_extracted_nodes`].
///
/// Mirrors upstream's `(resolved_nodes, uuid_map, duplicate_pairs)` tuple.
///
/// `uuid_map` maps every extracted-node UUID to its canonical UUID, INCLUDING
/// self-mappings: upstream's final loop (node_operations.py:694-697) inserts
/// `node.uuid -> node.uuid` for any node that resolved to itself, so kept-as-new
/// nodes appear as identity entries. `duplicates` only contains genuine
/// `(extracted, canonical)` pairs where the UUIDs differ.
pub struct NodeResolutionOutcome {
    pub nodes: Vec<EntityNode>,
    pub uuid_map: HashMap<String, String>,
    pub duplicates: Vec<(EntityNode, EntityNode)>,
}

/// Build the entity-types context list upstream `_build_entity_types_context`
/// produces (node_operations.py:152-181).
///
/// The first element is always the default `Entity` type with id 0 and the
/// verbatim description; user-provided types follow with ids `1..=n`.
///
/// `entity_types` is expected to be a JSON object mapping `type_name -> {..,
/// "description": ".."}` (the user's type registry). For each key we emit
/// `{entity_type_id, entity_type_name, entity_type_description}` where the
/// description is the type's docstring/`description` field (upstream reads
/// `type_model.__doc__`). Unknown/missing descriptions render as JSON null,
/// matching upstream's `type_model.__doc__` being `None`.
pub fn build_entity_types_context(entity_types: Option<&Value>) -> Value {
    let mut context = vec![json!({
        "entity_type_id": 0,
        "entity_type_name": "Entity",
        "entity_type_description":
            "A specific, identifiable entity that does not fit any of the other listed \
    types. Must still be a concrete, meaningful thing — specific enough to be \
    uniquely identifiable. GOOD: a named entity not covered by the other types. \
    BAD: \"luck\", \"ideas\", \"tomorrow\", \"things\", \"them\", \"everybody\", \
    \"a sense of wonder\", \"great times\". \
    When in doubt, do not extract the entity.",
    })];

    if let Some(Value::Object(map)) = entity_types {
        for (i, (type_name, type_model)) in map.iter().enumerate() {
            // Upstream uses `type_model.__doc__`; for our JSON registry we look
            // for a "description" field, falling back to null (== Python None).
            let description = type_model
                .get("description")
                .cloned()
                .unwrap_or(Value::Null);
            context.push(json!({
                "entity_type_id": i + 1,
                "entity_type_name": type_name,
                "entity_type_description": description,
            }));
        }
    }

    Value::Array(context)
}

/// Map an `entity_type_id` to its name using the entity-types context, mirroring
/// upstream `_create_entity_nodes` (node_operations.py:301-306): valid ids index
/// into the context; anything out of range falls back to `"Entity"`.
fn entity_type_name_for_id(entity_types_context: &Value, type_id: i64) -> String {
    if type_id >= 0
        && let Some(arr) = entity_types_context.as_array()
        && let Some(entry) = arr.get(type_id as usize)
        && let Some(name) = entry.get("entity_type_name").and_then(|v| v.as_str())
    {
        return name.to_string();
    }
    "Entity".to_string()
}

/// Compute the label set for an extracted entity, mirroring upstream
/// `list({'Entity', str(entity_type_name)})` (node_operations.py:313).
///
/// Upstream uses a Python set, so label ORDER is non-deterministic there; we
/// emit a deterministic `["Entity"]` for the generic case and
/// `["Entity", name]` otherwise. Order is not load-bearing for resolution
/// (`promote_resolved_node` and the deterministic passes treat labels as a set
/// modulo the literal `"Entity"`).
fn labels_for(entity_type_name: &str) -> Vec<String> {
    if entity_type_name == "Entity" {
        vec!["Entity".to_string()]
    } else {
        vec!["Entity".to_string(), entity_type_name.to_string()]
    }
}

/// Build the `previous_episodes` context value the pipeline feeds to the prompt
/// fns, byte-identical to upstream. Upstream (node_operations.py / edge_operations.py /
/// combined_extraction.py) builds, at every prompt call site:
///
/// ```python
/// [
///     {'content': ep.content, 'timestamp': ep.valid_at.isoformat() if ep.valid_at else None}
///     for ep in previous_episodes
/// ]
/// ```
///
/// Key order (`content` then `timestamp`) and the `isoformat()` timestamp format
/// are reproduced exactly (see [`crate::helpers::isoformat`]). NOTE: our
/// `EpisodicNode::valid_at` is a non-optional `DateTime<Utc>`, so `timestamp` is
/// always a string here — the upstream `if ep.valid_at else None` branch is
/// unreachable given our type, which is strictly more information, never less.
pub fn previous_episodes_context(previous_episodes: &[EpisodicNode]) -> Value {
    Value::Array(
        previous_episodes
            .iter()
            .map(|ep| {
                json!({
                    "content": ep.content,
                    "timestamp": isoformat(ep.valid_at),
                })
            })
            .collect(),
    )
}

/// Extract entity nodes from a single episode.
///
/// Port of upstream `extract_nodes` (node_operations.py:70-149) for the
/// single-episode path (the multi-episode list path and its episode-attribution
/// instructions are not used by the Phase-1 core loop and are deferred).
pub async fn extract_nodes(
    clients: &Clients,
    episode: &EpisodicNode,
    previous_episodes: &[EpisodicNode],
    entity_types: Option<&Value>,
    custom_extraction_instructions: Option<&str>,
) -> Result<Vec<EntityNode>, ChronicleError> {
    let entity_types_context = build_entity_types_context(entity_types);
    let prev = previous_episodes_context(previous_episodes);
    // Upstream resolves `custom_extraction_instructions or ''` at the call site;
    // single-episode path adds no episode_attribution suffix.
    let custom = custom_extraction_instructions.unwrap_or("");

    // Choose the extraction prompt by episode source (node_operations.py:261-273;
    // the `else` fallback also routes to extract_text).
    let messages = match episode.source {
        EpisodeType::Message => {
            let ctx = ExtractMessageContext {
                entity_types: &entity_types_context,
                previous_episodes: &prev,
                episode_content: &episode.content,
                custom_extraction_instructions: custom,
            };
            extract_message(&ctx)
        }
        EpisodeType::Json => {
            let ctx = ExtractJsonContext {
                entity_types: &entity_types_context,
                source_description: &episode.source_description,
                episode_content: &episode.content,
                custom_extraction_instructions: custom,
            };
            extract_json(&ctx)
        }
        EpisodeType::Text => {
            let ctx = ExtractTextContext {
                entity_types: &entity_types_context,
                episode_content: &episode.content,
                custom_extraction_instructions: custom,
            };
            extract_text(&ctx)
        }
    };

    let request = LlmRequest::new(messages).named(match episode.source {
        EpisodeType::Message => "extract_nodes.extract_message",
        EpisodeType::Json => "extract_nodes.extract_json",
        EpisodeType::Text => "extract_nodes.extract_text",
    });

    let extracted: ExtractedEntities = generate_typed(clients.llm.as_ref(), request).await?;

    // Filter empty names (upstream: `e.name.strip()` truthiness,
    // node_operations.py:135).
    let mut nodes = Vec::new();
    for entity in extracted.extracted_entities {
        if entity.name.trim().is_empty() {
            continue;
        }
        let type_name = entity_type_name_for_id(&entity_types_context, entity.entity_type_id);
        let labels = labels_for(&type_name);

        let mut node = EntityNode::new(entity.name, episode.group_id.clone(), utc_now());
        node.labels = labels;
        // summary stays "" and attributes stay empty — hydrated later.
        nodes.push(node);
    }

    // HOOK: upstream next runs `_collapse_exact_duplicate_extracted_nodes` and,
    // separately and later, `extract_attributes_from_nodes` for summaries/
    // attributes. Same-call exact-duplicate collapse and attribute hydration are
    // Task 14; we return the raw extracted nodes here.
    Ok(nodes)
}

/// Resolve extracted nodes against existing graph nodes via semantic retrieval,
/// deterministic heuristics, then LLM escalation.
///
/// Port of upstream `resolve_extracted_nodes` (node_operations.py:627-708).
pub async fn resolve_extracted_nodes(
    clients: &Clients,
    extracted_nodes: Vec<EntityNode>,
    episode: &EpisodicNode,
    previous_episodes: &[EpisodicNode],
) -> Result<NodeResolutionOutcome, ChronicleError> {
    // --- (a) per-node semantic candidate search ---------------------------
    let candidate_nodes_by_extracted = collect_candidate_nodes(clients, &extracted_nodes).await?;

    // --- resolution bookkeeping (upstream DedupResolutionState) -----------
    let mut resolved_nodes: Vec<Option<EntityNode>> = vec![None; extracted_nodes.len()];
    let mut uuid_map: HashMap<String, String> = HashMap::new();
    let mut duplicate_pairs: Vec<(EntityNode, EntityNode)> = Vec::new();
    let mut unresolved_indices: Vec<usize> = Vec::new();

    // --- (c) deterministic pass, per extracted node -----------------------
    for (idx, (node, candidates)) in extracted_nodes
        .iter()
        .zip(candidate_nodes_by_extracted.iter())
        .enumerate()
    {
        if candidates.is_empty() {
            continue;
        }

        let indexes = build_candidate_indexes(candidates.clone());
        let mut local_state = DedupResolutionState {
            resolved_nodes: vec![None],
            uuid_map: HashMap::new(),
            unresolved_indices: Vec::new(),
            duplicate_pairs: Vec::new(),
        };
        resolve_with_similarity(std::slice::from_ref(node), &indexes, &mut local_state);

        if let Some(resolved) = local_state.resolved_nodes[0].take() {
            // commit (upstream `_commit_resolution`)
            resolved_nodes[idx] = Some(resolved);
            uuid_map.extend(local_state.uuid_map);
            duplicate_pairs.extend(local_state.duplicate_pairs);
            continue;
        }

        unresolved_indices.push(idx);
    }

    // --- (d) LLM escalation for the unresolved -----------------------------
    if !unresolved_indices.is_empty() {
        // Union the candidate pools of unresolved nodes, dedup by uuid
        // preserving first-seen order (upstream `_merge_candidate_nodes`).
        let mut merged: Vec<EntityNode> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for &idx in &unresolved_indices {
            for cand in &candidate_nodes_by_extracted[idx] {
                if seen.insert(cand.uuid.clone()) {
                    merged.push(cand.clone());
                }
            }
        }
        let indexes = build_candidate_indexes(merged);

        resolve_with_llm(
            clients,
            &extracted_nodes,
            &indexes,
            &unresolved_indices,
            &mut resolved_nodes,
            &mut uuid_map,
            &mut duplicate_pairs,
            episode,
            previous_episodes,
        )
        .await?;
    }

    // --- (e) finalize: keep-as-new + self-mappings ------------------------
    // Upstream node_operations.py:694-697.
    for (idx, node) in extracted_nodes.iter().enumerate() {
        if resolved_nodes[idx].is_none() {
            resolved_nodes[idx] = Some(node.clone());
            uuid_map.insert(node.uuid.clone(), node.uuid.clone());
        }
    }

    let nodes: Vec<EntityNode> = resolved_nodes.into_iter().flatten().collect();

    Ok(NodeResolutionOutcome {
        nodes,
        uuid_map,
        duplicates: duplicate_pairs,
    })
}

/// Per extracted node, embed its name and run a direct cosine candidate search.
///
/// Port of upstream `_collect_candidate_nodes` / `_semantic_candidate_search`
/// (node_operations.py:407-449). Each search runs on its own task under the
/// shared semaphore, matching upstream's `semaphore_gather`.
async fn collect_candidate_nodes(
    clients: &Clients,
    extracted_nodes: &[EntityNode],
) -> Result<Vec<Vec<EntityNode>>, ChronicleError> {
    if extracted_nodes.is_empty() {
        return Ok(Vec::new());
    }

    // Upstream replaces newlines in the query name with spaces.
    let queries: Vec<String> = extracted_nodes
        .iter()
        .map(|n| n.name.replace('\n', " "))
        .collect();

    let query_vectors = clients.embedder.create_batch(&queries).await?;

    // Fan out one similarity search per extracted node under the semaphore.
    let mut handles = Vec::with_capacity(extracted_nodes.len());
    for (node, vector) in extracted_nodes.iter().zip(query_vectors) {
        let driver = Arc::clone(&clients.driver);
        let semaphore = Arc::clone(&clients.semaphore);
        let group_id = node.group_id.clone();
        handles.push(tokio::spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .map_err(|e| ChronicleError::InvalidInput(format!("semaphore closed: {e}")))?;
            driver
                .node_similarity_search(
                    &vector,
                    std::slice::from_ref(&group_id),
                    NODE_DEDUP_CANDIDATE_LIMIT,
                    NODE_DEDUP_COSINE_MIN_SCORE,
                )
                .await
                .map_err(ChronicleError::from)
        }));
    }

    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        let candidates = handle.await.map_err(|e| {
            ChronicleError::InvalidInput(format!("candidate search task panicked: {e}"))
        })??;
        results.push(candidates);
    }

    Ok(results)
}

/// Deterministic resolution pass: exact normalized-name match, then
/// entropy-gated MinHash/LSH fuzzy match.
///
/// Line-for-line port of upstream `_resolve_with_similarity`
/// (dedup_helpers.py:220-279). Lives here (rather than in `dedup_helpers.rs`)
/// because it is the resolution driver, not a pure heuristic primitive; it
/// composes the `dedup_helpers` primitives.
fn resolve_with_similarity(
    extracted_nodes: &[EntityNode],
    indexes: &DedupCandidateIndexes,
    state: &mut DedupResolutionState,
) {
    use crate::helpers::normalize_string_exact;
    use crate::pipeline::dedup_helpers::{
        FUZZY_JACCARD_THRESHOLD, has_high_entropy, jaccard_similarity, lsh_bands,
        minhash_signature, normalize_name_for_fuzzy, shingles,
    };

    for (idx, node) in extracted_nodes.iter().enumerate() {
        let normalized_exact = normalize_string_exact(&node.name);
        let normalized_fuzzy = normalize_name_for_fuzzy(&node.name);

        // --- exact-name matching (always attempted) ---
        let existing_matches = indexes
            .normalized_existing
            .get(&normalized_exact)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        if existing_matches.len() == 1 {
            let matched = promote_resolved_node(node, &existing_matches[0]);
            let matched_uuid = matched.uuid.clone();
            state.resolved_nodes[idx] = Some(matched.clone());
            state
                .uuid_map
                .insert(node.uuid.clone(), matched_uuid.clone());
            if matched_uuid != node.uuid {
                state.duplicate_pairs.push((node.clone(), matched));
            }
            continue;
        }
        if existing_matches.len() > 1 {
            // Ambiguous: escalate to LLM.
            state.unresolved_indices.push(idx);
            continue;
        }

        // --- entropy gate (protects fuzzy matching only) ---
        if !has_high_entropy(&normalized_fuzzy) {
            state.unresolved_indices.push(idx);
            continue;
        }

        // --- fuzzy matching via MinHash/LSH ---
        let node_shingles = shingles(&normalized_fuzzy);
        let signature = minhash_signature(&node_shingles);
        let mut candidate_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (band_index, band) in lsh_bands(&signature).into_iter().enumerate() {
            if let Some(ids) = indexes.lsh_buckets.get(&(band_index, band)) {
                candidate_ids.extend(ids.iter().cloned());
            }
        }

        let mut best_candidate: Option<&EntityNode> = None;
        let mut best_score = 0.0_f64;
        for candidate_id in &candidate_ids {
            let empty = std::collections::BTreeSet::new();
            let candidate_shingles = indexes
                .shingles_by_candidate
                .get(candidate_id)
                .unwrap_or(&empty);
            let score = jaccard_similarity(&node_shingles, candidate_shingles);
            if score > best_score {
                best_score = score;
                best_candidate = indexes.nodes_by_uuid.get(candidate_id);
            }
        }

        if let Some(candidate) = best_candidate
            && best_score >= FUZZY_JACCARD_THRESHOLD
        {
            let promoted = promote_resolved_node(node, candidate);
            let promoted_uuid = promoted.uuid.clone();
            state.resolved_nodes[idx] = Some(promoted.clone());
            state
                .uuid_map
                .insert(node.uuid.clone(), promoted_uuid.clone());
            if promoted_uuid != node.uuid {
                state.duplicate_pairs.push((node.clone(), promoted));
            }
            continue;
        }

        state.unresolved_indices.push(idx);
    }
}

/// LLM escalation pass for unresolved nodes.
///
/// Port of upstream `_resolve_with_llm` (node_operations.py:467-624). The
/// `id` field in the dedupe response is a RELATIVE index into
/// `unresolved_indices` (NOT the original extracted-node index); we guard
/// against out-of-range and duplicate relative ids exactly as upstream does,
/// and guard against out-of-range `duplicate_candidate_id` values defensively.
#[allow(clippy::too_many_arguments)]
async fn resolve_with_llm(
    clients: &Clients,
    extracted_nodes: &[EntityNode],
    indexes: &DedupCandidateIndexes,
    unresolved_indices: &[usize],
    resolved_nodes: &mut [Option<EntityNode>],
    uuid_map: &mut HashMap<String, String>,
    duplicate_pairs: &mut Vec<(EntityNode, EntityNode)>,
    episode: &EpisodicNode,
    previous_episodes: &[EpisodicNode],
) -> Result<(), ChronicleError> {
    if unresolved_indices.is_empty() {
        return Ok(());
    }

    let llm_extracted_nodes: Vec<&EntityNode> = unresolved_indices
        .iter()
        .map(|&i| &extracted_nodes[i])
        .collect();

    // extracted_nodes context: relative `id` 0..n-1 (upstream lines 488-496).
    let extracted_nodes_context: Vec<Value> = llm_extracted_nodes
        .iter()
        .enumerate()
        .map(|(i, node)| {
            json!({
                "id": i,
                "name": node.name,
                "entity_type": node.labels,
                "entity_type_description": get_entity_type_description(&node.labels),
            })
        })
        .collect();

    // existing_nodes context: spread attributes, then candidate_id/name/etc.
    // (upstream lines 519-528). Attributes are spread FIRST so the explicit keys
    // below win on collision, matching Python dict-unpacking order.
    let existing_nodes_context: Vec<Value> = indexes
        .existing_nodes
        .iter()
        .enumerate()
        .map(|(i, candidate)| {
            let mut obj = serde_json::Map::new();
            for (k, v) in &candidate.attributes {
                obj.insert(k.clone(), v.clone());
            }
            obj.insert("candidate_id".to_string(), json!(i));
            obj.insert("name".to_string(), json!(candidate.name));
            obj.insert("entity_types".to_string(), json!(candidate.labels));
            let summary: String = candidate.summary.chars().take(120).collect();
            obj.insert("summary".to_string(), json!(summary));
            Value::Object(obj)
        })
        .collect();

    let prev = previous_episodes_context(previous_episodes);
    let extracted_value = Value::Array(extracted_nodes_context);
    let existing_value = Value::Array(existing_nodes_context);

    let ctx = NodesContext {
        previous_episodes: &prev,
        episode_content: &episode.content,
        extracted_nodes: &extracted_value,
        existing_nodes: &existing_value,
    };
    let request = LlmRequest::new(dedupe_nodes_prompt(&ctx)).named("dedupe_nodes.nodes");

    let response: NodeResolutions = generate_typed(clients.llm.as_ref(), request).await?;

    let valid_relative_range = unresolved_indices.len();
    let mut processed_relative_ids: std::collections::HashSet<i64> =
        std::collections::HashSet::new();

    for resolution in response.entity_resolutions {
        let relative_id = resolution.id;
        let duplicate_candidate_id = resolution.duplicate_candidate_id;

        // Guard: relative id must be in 0..unresolved_indices.len().
        if relative_id < 0 || (relative_id as usize) >= valid_relative_range {
            tracing::warn!(
                relative_id,
                valid_max = valid_relative_range as i64 - 1,
                "skipping invalid LLM dedupe id"
            );
            continue;
        }
        if !processed_relative_ids.insert(relative_id) {
            tracing::warn!(relative_id, "duplicate LLM dedupe id received; ignoring");
            continue;
        }

        let original_index = unresolved_indices[relative_id as usize];
        let extracted_node = &extracted_nodes[original_index];

        let resolved_node = if duplicate_candidate_id < 0 {
            extracted_node.clone()
        } else if let Some(candidate) = indexes.existing_nodes.get(duplicate_candidate_id as usize)
        {
            promote_resolved_node(extracted_node, candidate)
        } else {
            // Out-of-range candidate id — defensively keep the extracted node.
            tracing::warn!(
                duplicate_candidate_id,
                node_uuid = %extracted_node.uuid,
                "invalid duplicate_candidate_id; treating as no duplicate"
            );
            extracted_node.clone()
        };

        let resolved_uuid = resolved_node.uuid.clone();
        resolved_nodes[original_index] = Some(resolved_node.clone());
        uuid_map.insert(extracted_node.uuid.clone(), resolved_uuid.clone());
        if resolved_uuid != extracted_node.uuid {
            duplicate_pairs.push((extracted_node.clone(), resolved_node));
        }
    }

    Ok(())
}

/// Upstream `_get_entity_type_description` (node_operations.py:184-189): the
/// first non-`"Entity"` label is the type name; with no entity-type registry
/// available here the description is always the default.
fn get_entity_type_description(labels: &[String]) -> &'static str {
    // We have no JSON type registry plumbed through resolution (upstream passes
    // `entity_types` here; the Phase-1 loop calls resolve without it), so the
    // docstring lookup always misses → upstream's `or 'Default Entity Type'`.
    let _type_name = labels.iter().find(|l| *l != "Entity");
    "Default Entity Type"
}
