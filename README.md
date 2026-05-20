# Ortserver

Ortserver is a small native C++ inference server for ONNX models. It exposes a
gRPC `Predict` API, runs models with ONNX Runtime, and exposes Prometheus
metrics over HTTP.

The server is designed for container deployment. Model files are mounted at
runtime instead of baked into the image, so the same image can serve different
model sets.

## Model Layout

Models are discovered from `ORTSERVER_MODEL_ROOT` using this directory layout:

```text
models/
  <model-name>/
    <version>/
      model.onnx
```

Example:

```text
models/
  classifier/
    1/
      model.onnx
    2/
      model.onnx
```

For each model family, Ortserver serves the highest numeric version currently
available on disk. If a higher version appears and reload is enabled, the server
loads it automatically.

## Run Published Image

The GitHub Actions workflow publishes Docker images to GitHub Container
Registry:

```bash
docker pull ghcr.io/dev6699/ortserver:latest
```

Run the published image:

```bash
docker run --rm \
  -p 18500:18500 \
  -p 9090:9090 \
  -v "$(pwd)/models:/models:ro" \
  ghcr.io/dev6699/ortserver:latest
```

The gRPC server listens on port `18500`. Metrics are exposed on
`http://localhost:9090/metrics`.

## Build Locally

Build the image from this repository:

```bash
docker build -t ortserver .
```

Run the local image:

```bash
docker run --rm \
  -p 18500:18500 \
  -p 9090:9090 \
  -v "$(pwd)/models:/models:ro" \
  ortserver
```

The image stays lightweight by keeping the model files out of the final
runtime layer and copying only the compiled server plus the runtime shared
libraries it needs.

## Docker Compose

Run with Docker Compose:

```bash
docker compose up --build
```

The compose file mounts `./models` into the container and sets common
environment variables for ports, threading, message size, reload interval, and
logging.

## Configuration

Useful flags:
- `--intra-threads` controls ONNX Runtime intra-op threading
  - default: auto, based on host CPU count
- `--inter-threads` controls ONNX Runtime inter-op threading
  - default: `1`
- `--max-message-mb` caps gRPC request/response size
  - default: `16`
- `--reload-interval-seconds` enables periodic model rescans and reloads
  newer versions automatically
  - default: `60`
- `--metrics-port` exposes Prometheus metrics on `/metrics`
  - default: `9090`
- `--metrics-host` controls the metrics listen address
  - default: `0.0.0.0`
- `--log-predict` enables per-request `Predict` log lines
  - default: off

Environment variables:
- `ORTSERVER_MODEL_ROOT`
- `ORTSERVER_HOST`
- `ORTSERVER_PORT`
- `ORTSERVER_METRICS_HOST`
- `ORTSERVER_METRICS_PORT`
- `ORTSERVER_WORKERS`
- `ORTSERVER_MAX_MESSAGE_MB`
- `ORTSERVER_INTRA_THREADS`
- `ORTSERVER_INTER_THREADS`
- `ORTSERVER_RELOAD_INTERVAL_SECONDS`
- `ORTSERVER_LOG_PREDICT`

## Reloading

When reloading is enabled, the server keeps the highest version currently on
disk for each model family. If that version disappears, it falls back to the
next-highest version still present. Version changes are logged to stdout. The
default reload interval is 60 seconds.

Actual reloads also log RSS memory before and after the reload:

```text
event=reload_rss rss_before_bytes="..." rss_after_bytes="..." rss_delta_bytes="..."
```

No-op reload checks are skipped without logging.

## Metrics

Metrics exposed:
- `ort_model_requests_total`
- `ort_model_request_duration_seconds_bucket`
- `ort_model_request_duration_seconds_sum`
- `ort_model_request_duration_seconds_count`
- `ort_model_request_errors_total`

## Go Bindings

To generate Go bindings from `ortserver.proto`:

```bash
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

protoc \
  --proto_path=. \
  --go_out=. \
  --go_opt=module=example.com/your/module \
  --go_opt=Mortserver.proto=example.com/your/module/ortserverpb \
  --go-grpc_out=. \
  --go-grpc_opt=module=example.com/your/module \
  --go-grpc_opt=Mortserver.proto=example.com/your/module/ortserverpb \
  ortserver.proto
```

Replace `example.com/your/module` with your real Go module path.
