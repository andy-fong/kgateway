ARG ENVOY_IMAGE

FROM --platform=$BUILDPLATFORM ghcr.io/rust-cross/cargo-zigbuild:0.19.8 AS rust_builder

WORKDIR /build

RUN apt update && apt install -y clang



# Then copy the rest of the source code and build the library.
# Paramerized to allow for supporting local, make and goreleaser builds
ARG RUSTFORMATIONS_DIR=./rustformations
COPY ${RUSTFORMATIONS_DIR}/Cargo.toml ${RUSTFORMATIONS_DIR}/Cargo.lock ./
RUN mkdir src && echo "" > src/lib.rs
RUN cargo fetch
RUN rm -rf src

# Then copy the rest of the source code and build the library.
COPY ${RUSTFORMATIONS_DIR} .
# RUN cargo zigbuild --target x86_64-unknown-linux-gnu
# switch to this locally if you are on arm
RUN cargo zigbuild --target aarch64-unknown-linux-gnu

RUN cp /build/target/aarch64-unknown-linux-gnu/debug/librust_module.so /build/arm64_librust_module.so
# RUN cp /build/target/x86_64-unknown-linux-gnu/debug/librust_module.so /build/amd64_librust_module.so

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
COPY --from=rust_builder /build/${GOARCH}_librust_module.so /usr/local/lib/librust_module.so


# SDS-specific setup, only used if ENVOY_SIDECAR=true
ARG ENTRYPOINT_SCRIPT=/docker-entrypoint.sh
COPY $ENTRYPOINT_SCRIPT /

USER 10101

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD []
