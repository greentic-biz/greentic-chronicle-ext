# Build: docker build -f crates/chronicle-index-server/Dockerfile -t chronicle-index-server .
FROM rust:1.95-bookworm AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends clang libclang-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cargo build --release --locked -p chronicle-index-server

FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 chronicle \
 && mkdir -p /data && chown chronicle /data
COPY --from=build /src/target/release/chronicle-index-server /usr/local/bin/chronicle-index-server
USER chronicle
ENV CHRONICLE_INDEX_DATA_DIR=/data \
    CHRONICLE_INDEX_BIND=0.0.0.0:8088
EXPOSE 8088
VOLUME ["/data"]
ENTRYPOINT ["/usr/local/bin/chronicle-index-server"]
