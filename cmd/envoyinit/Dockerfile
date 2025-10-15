ARG ENVOY_IMAGE
ARG CARGO_ZBUILD_IMAGE=ghcr.io/rust-cross/cargo-zigbuild:0.19.8
# Stage 1: pull in all rust build dependency and cache it in docker build cache layer
FROM --platform=$BUILDPLATFORM ${CARGO_ZBUILD_IMAGE} AS rust_build_deps
WORKDIR /build
RUN apt update && apt install -y clang
ARG RUSTFORMATIONS_DIR=./rustformations
COPY ${RUSTFORMATIONS_DIR}/Cargo.toml ${RUSTFORMATIONS_DIR}/Cargo.lock ./
RUN mkdir rustformations transformations
COPY ${RUSTFORMATIONS_DIR}/rustformations/Cargo.toml ./rustformations
COPY ${RUSTFORMATIONS_DIR}/transformations/Cargo.toml ./transformations
RUN mkdir rustformations/src && echo "pub fn dummy() {}" > rustformations/src/lib.rs && \
    mkdir transformations/src && echo "pub fn dummy() {}" > transformations/src/lib.rs 
ARG RUST_BUILD_ARCH=x86_64 # options are x86_64 or aarch64
RUN cargo fetch && cargo zigbuild --target "${RUST_BUILD_ARCH}-unknown-linux-gnu"

# Stage 2: build envoy dynamic module
FROM --platform=$BUILDPLATFORM ${CARGO_ZBUILD_IMAGE} AS rust_builder
WORKDIR /build
RUN apt update && apt install -y clang

# Copy cached target and cargo dirs from build-deps
COPY --from=rust_build_deps /usr/local/cargo /usr/local/cargo
COPY --from=rust_build_deps /build/target /build/target

# Copy the source code and build the module.
ARG RUSTFORMATIONS_DIR=./rustformations
COPY ${RUSTFORMATIONS_DIR} .
ARG RUST_BUILD_ARCH=x86_64 # options are x86_64 or aarch64
RUN cargo zigbuild --target "${RUST_BUILD_ARCH}-unknown-linux-gnu"

RUN cp /build/target/${RUST_BUILD_ARCH}-unknown-linux-gnu/debug/librust_module.so /build/librust_module.so

# Stage 3: build the final envoy wrapper image
FROM --platform=$BUILDPLATFORM $ENVOY_IMAGE AS envoy
ENV DEBIAN_FRONTEND=noninteractive

# Update our deps to make cve toil lower
#install wget for our default probes
USER root
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install wget ca-certificates -y \
    && rm -rf  /var/log/*log /var/lib/apt/lists/* /var/log/apt/* /var/lib/dpkg/*-old /var/cache/debconf/*-old

ARG GOARCH=amd64
COPY envoyinit-linux-$GOARCH /usr/local/bin/envoyinit

ENV ENVOY_DYNAMIC_MODULES_SEARCH_PATH=/usr/local/lib
COPY --from=rust_builder /build/librust_module.so /usr/local/lib/librust_module.so


# SDS-specific setup, only used if ENVOY_SIDECAR=true
ARG ENTRYPOINT_SCRIPT=/docker-entrypoint.sh
COPY $ENTRYPOINT_SCRIPT /

USER 10101

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD []
