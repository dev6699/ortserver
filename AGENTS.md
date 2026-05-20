# Repository Guidelines

## Project Structure & Module Organization

This repository contains a native C++ ONNX Runtime inference server. The main
server implementation is in `main.cc`, and the gRPC/protobuf API is defined in
`ortserver.proto`. Build configuration lives in `CMakeLists.txt`. Container
packaging is handled by `Dockerfile`, `docker-compose.yml`, and
`run-docker.sh`. GitHub Actions publishing is configured under
`.github/workflows/`.

Runtime model files are expected under `models/<model-name>/<version>/model.onnx`
and are intentionally ignored by Git. Keep large local artifacts such as
`models_old/`, generated build output, and ONNX binaries out of commits.

## Build, Test, and Development Commands

Build the Docker image locally:

```bash
docker build -t ortserver .
```

Run with Docker Compose:

```bash
docker compose up --build
```

Run the existing local image with mounted models:

```bash
./run-docker.sh
```

For a direct CMake build, set `ONNXRUNTIME_ROOT` to an extracted ONNX Runtime
release:

```bash
cmake -S . -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-<ver>
cmake --build build --target ortserver
```

## Coding Style & Naming Conventions

Use C++17 and follow the existing style in `main.cc`: two-space indentation,
`snake_case` for helper functions, `CamelCase` for classes and structs, and
short local variables where context is clear. Keep comments sparse and focused
on non-obvious behavior. Do not commit generated protobuf files or build
artifacts.

## Testing Guidelines

There is no dedicated test suite yet. Validate changes by building the Docker
image or CMake target, then run the server with a mounted `models/` directory.
For reload behavior, use numeric version directories and check stdout logs for
`model_version_changed` and `reload_rss`.

## Commit & Pull Request Guidelines

Recent commits use short, imperative messages such as `optimize model reloads
and log RSS deltas` or `add github actions docker publish workflow`. Keep
messages concise and focused on one change.

Pull requests should describe the behavior change, list validation commands,
and call out Docker, model layout, or GitHub Actions impacts. Do not include
large model files, local backups, or generated build output in the diff.

## Security & Configuration Tips

The published image is intended to keep models out of the runtime layer. Mount
models read-only with `-v "$(pwd)/models:/models:ro"`. Use GHCR package
visibility deliberately: a private repository can publish a public image, but
anything copied into the image is visible to image users.
