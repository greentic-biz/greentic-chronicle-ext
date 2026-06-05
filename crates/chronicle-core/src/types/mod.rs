// UUIDs stored as String to match upstream graphiti's str type and simplify driver integration; generated via uuid::Uuid::new_v4().

pub mod edge;
pub mod episode;
pub mod node;

pub use edge::{EntityEdge, EpisodicEdge};
pub use episode::{EpisodeType, EpisodicNode};
pub use node::EntityNode;
