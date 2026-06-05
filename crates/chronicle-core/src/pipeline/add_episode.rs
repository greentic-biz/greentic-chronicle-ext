// Ported from graphiti_core/graphiti.py::add_episode (~1067-1223),
// _extract_and_resolve_edges (~631-678), _process_episode_data (~680-781),
// and bulk_utils.py::resolve_edge_pointers (~627-634) @ 34f56e65 (v0.29.1).
//
// Single-episode core-loop orchestration. Community updates, sagas, excluded
// entity types, and the multi-episode bulk path are out of Phase-1 scope.

use std::collections::HashMap;

use crate::chronicle::{AddEpisodeRequest, AddEpisodeResults};
use crate::errors::ChronicleError;
use crate::helpers::{RELEVANT_SCHEMA_LIMIT, utc_now};
use crate::pipeline::clients::Clients;
use crate::pipeline::edge_ops::{extract_edges, hydrate_node_summaries, resolve_extracted_edges};
use crate::pipeline::node_ops::{extract_nodes, resolve_extracted_nodes};
use crate::types::{EntityEdge, EntityNode, EpisodicEdge, EpisodicNode};

/// Remap an edge's source/target node UUIDs through the node-resolution
/// `uuid_map` (extracted → canonical). Port of upstream `resolve_edge_pointers`
/// (bulk_utils.py:627-634); UUIDs absent from the map are left unchanged.
fn resolve_edge_pointers(edges: &mut [EntityEdge], uuid_map: &HashMap<String, String>) {
    for edge in edges {
        if let Some(canonical) = uuid_map.get(&edge.source_node_uuid) {
            edge.source_node_uuid = canonical.clone();
        }
        if let Some(canonical) = uuid_map.get(&edge.target_node_uuid) {
            edge.target_node_uuid = canonical.clone();
        }
    }
}

/// Add a single episode to the graph: extract + resolve nodes and edges, hydrate
/// node summaries, build MENTIONS edges, embed, and persist.
///
/// Port of upstream `Graphiti.add_episode` single-episode path.
pub async fn add_episode(
    clients: &Clients,
    req: AddEpisodeRequest,
) -> Result<AddEpisodeResults, ChronicleError> {
    let now = utc_now();
    let group_id = req.group_id.clone();

    // 1. Previous-episode context.
    let previous_episodes = match &req.previous_episode_uuids {
        Some(uuids) => clients.driver.get_episodes_by_uuids(uuids).await?,
        None => {
            clients
                .driver
                .retrieve_episodes(
                    req.reference_time,
                    RELEVANT_SCHEMA_LIMIT,
                    std::slice::from_ref(&group_id),
                    None,
                )
                .await?
        }
    };

    // 2. Get or create the episode.
    let episode = match &req.uuid {
        Some(uuid) => clients
            .driver
            .get_episode(uuid)
            .await?
            .ok_or_else(|| ChronicleError::EpisodeNotFound { uuid: uuid.clone() })?,
        None => EpisodicNode::new(
            req.name.clone(),
            group_id.clone(),
            req.source,
            req.source_description.clone(),
            req.episode_body.clone(),
            now,
            req.reference_time,
        ),
    };

    // 3. Extract + resolve nodes.
    let extracted_nodes = extract_nodes(
        clients,
        &episode,
        &previous_episodes,
        req.entity_types.as_ref(),
        req.custom_extraction_instructions.as_deref(),
    )
    .await?;

    let resolution = resolve_extracted_nodes(
        clients,
        extracted_nodes.clone(),
        &episode,
        &previous_episodes,
    )
    .await?;
    let nodes = resolution.nodes;
    let uuid_map = resolution.uuid_map;

    // 4. Extract edges over the EXTRACTED (pre-resolution) nodes, then remap
    // source/target through uuid_map before resolving (upstream 656-667).
    let mut extracted_edges = extract_edges(
        clients,
        &episode,
        &extracted_nodes,
        &previous_episodes,
        &group_id,
        req.custom_extraction_instructions.as_deref(),
    )
    .await?;
    resolve_edge_pointers(&mut extracted_edges, &uuid_map);

    // 5. Resolve edges → resolved + invalidated.
    let outcome = resolve_extracted_edges(clients, extracted_edges, &episode, &nodes).await?;
    let mut entity_edges: Vec<EntityEdge> = outcome.resolved_edges;
    entity_edges.extend(outcome.invalidated_edges);

    // 6. Hydrate node summaries from the current episode.
    let hydrated_nodes =
        hydrate_node_summaries(clients, nodes, &episode, &previous_episodes).await?;

    // 7. Build MENTIONS episodic edges (episode → each final node) and stamp the
    // episode's entity_edges list (upstream build_episodic_edges + 720-722).
    let episodic_edges: Vec<EpisodicEdge> = hydrated_nodes
        .iter()
        .map(|node| {
            EpisodicEdge::new(
                episode.uuid.clone(),
                node.uuid.clone(),
                group_id.clone(),
                now,
            )
        })
        .collect();
    let mut episode = episode;
    episode.entity_edges = entity_edges.iter().map(|e| e.uuid.clone()).collect();

    // 8. Embeddings.
    //  - node name embeddings for nodes still missing them;
    //  - edge fact embeddings for edges still missing them (extracted edges were
    //    embedded inside resolve_extracted_edges; this covers any duplicate/
    //    invalidated edge that lost its embedding on the round-trip).
    let mut hydrated_nodes = hydrated_nodes;
    embed_missing_node_names(clients, &mut hydrated_nodes).await?;
    embed_missing_edge_facts(clients, &mut entity_edges).await?;

    // 9. Persist: episode, nodes, entity edges, episodic edges (in that order).
    clients.driver.save_episode(&episode).await?;
    clients.driver.save_entity_nodes(&hydrated_nodes).await?;
    clients.driver.save_entity_edges(&entity_edges).await?;
    clients.driver.save_episodic_edges(&episodic_edges).await?;

    // 10. Result.
    Ok(AddEpisodeResults {
        episode,
        episodic_edges,
        nodes: hydrated_nodes,
        edges: entity_edges,
    })
}

/// Batch-embed `name` for any node missing a `name_embedding`.
async fn embed_missing_node_names(
    clients: &Clients,
    nodes: &mut [EntityNode],
) -> Result<(), ChronicleError> {
    let missing: Vec<usize> = nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.name_embedding.is_none())
        .map(|(i, _)| i)
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    let names: Vec<String> = missing.iter().map(|&i| nodes[i].name.clone()).collect();
    let embeddings = clients.embedder.create_batch(&names).await?;
    for (&i, emb) in missing.iter().zip(embeddings) {
        nodes[i].name_embedding = Some(emb);
    }
    Ok(())
}

/// Batch-embed `fact` for any edge missing a `fact_embedding`.
async fn embed_missing_edge_facts(
    clients: &Clients,
    edges: &mut [EntityEdge],
) -> Result<(), ChronicleError> {
    let missing: Vec<usize> = edges
        .iter()
        .enumerate()
        .filter(|(_, e)| e.fact_embedding.is_none())
        .map(|(i, _)| i)
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    let facts: Vec<String> = missing.iter().map(|&i| edges[i].fact.clone()).collect();
    let embeddings = clients.embedder.create_batch(&facts).await?;
    for (&i, emb) in missing.iter().zip(embeddings) {
        edges[i].fact_embedding = Some(emb);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_edge_pointers_remaps_known_uuids_only() {
        let mut map = HashMap::new();
        map.insert("extracted-src".to_string(), "canonical-src".to_string());
        let mut edge = EntityEdge::new(
            "extracted-src".into(),
            "unknown-tgt".into(),
            "R".into(),
            "f".into(),
            "g".into(),
        );
        resolve_edge_pointers(std::slice::from_mut(&mut edge), &map);
        assert_eq!(edge.source_node_uuid, "canonical-src");
        // Unknown UUID left unchanged.
        assert_eq!(edge.target_node_uuid, "unknown-tgt");
    }
}
