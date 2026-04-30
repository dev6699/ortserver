FROM debian:bookworm-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG ORT_VERSION=1.25.1
ARG ORT_TARBALL=onnxruntime-linux-x64-1.25.1.tgz
ARG ORT_URL=https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_TARBALL}

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  cmake \
  curl \
  libgrpc++-dev \
  libprotobuf-dev \
  pkg-config \
  protobuf-compiler \
  protobuf-compiler-grpc \
  && rm -rf /var/lib/apt/lists/*

# CMake in this repo expects a /root/.local-style layout.
RUN mkdir -p /root/.local/bin /root/.local/lib \
  && rm -rf /root/.local/include \
  && ln -s /usr/include /root/.local/include \
  && ln -sfn /usr/bin/protoc /root/.local/bin/protoc \
  && ln -sfn /usr/bin/grpc_cpp_plugin /root/.local/bin/grpc_cpp_plugin \
  && ln -sfn /usr/lib/x86_64-linux-gnu/libprotobuf.so /root/.local/lib/libprotobuf.so

WORKDIR /opt/ort
RUN curl -fsSL -o "${ORT_TARBALL}" "${ORT_URL}" \
  && tar -xzf "${ORT_TARBALL}"

WORKDIR /src
COPY . /src

RUN set -eux; \
  cmake -S . -B build -DONNXRUNTIME_ROOT=/opt/ort/onnxruntime-linux-x64-1.25.1; \
  cmake --build build -j"$(nproc)" --target ortserver; \
  strip build/ortserver || true; \
  mkdir -p /out/bin /out/lib; \
  cp build/ortserver /out/bin/ortserver; \
  for lib in $(ldd build/ortserver | awk '/=> \// { print $3 }'); do cp -L "$lib" /out/lib/; done; \
  cp -L /opt/ort/onnxruntime-linux-x64-1.25.1/lib/libonnxruntime.so* /out/lib/; \
  cp -L /opt/ort/onnxruntime-linux-x64-1.25.1/lib/libonnxruntime_providers_shared.so /out/lib/


FROM debian:bookworm-slim AS runtime

COPY --from=builder /out/ /opt/ortserver/

ENV LD_LIBRARY_PATH=/opt/ortserver/lib
WORKDIR /models

EXPOSE 18500

ENTRYPOINT ["/opt/ortserver/bin/ortserver"]
CMD ["--model-root", "/models", "--host", "0.0.0.0", "--port", "18500"]
