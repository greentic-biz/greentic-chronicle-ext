#![forbid(unsafe_code)]

pub mod fake_driver;
pub mod metering;
pub mod mock_cross_encoder;
pub mod mock_embedder;
pub mod mock_llm;

pub use fake_driver::FakeDriver;
pub use metering::{
    CallRecord, CostReport, EmbeddingPricing, Meter, MeteringEmbedder, MeteringLlm,
    OperationSubtotal, Pricing, PricingTable, Totals,
};
pub use mock_cross_encoder::MockCrossEncoder;
pub use mock_embedder::MockEmbedder;
pub use mock_llm::MockLlm;
