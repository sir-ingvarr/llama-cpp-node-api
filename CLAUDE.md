# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`llama-node` is a Node.js native addon (N-API) that wraps [llama.cpp](https://github.com/ggml-org/llama.cpp) for direct in-process LLM inference. It is built with `cmake-js` and targets Electron 35.

## Build Commands

The compiled addon is placed at `build/Release/llama_node.node`.

### Standard build (submodule — default)

The `vendor/llama.cpp` submodule is auto-detected by `CMakeLists.txt`. After cloning with `--recurse-submodules`, no flags are needed:

```bash
npm install        # installs deps and builds
npm run build      # incremental build
npm run rebuild    # clean rebuild
```

`GGML_NATIVE` is forced `OFF` in vendor mode because cmake-js may cross-compile (x86_64 host → arm64 target). Metal handles GPU acceleration.

### Build variants

```bash
npm run build:debug    # debug symbols
npm run build:cuda     # enable CUDA
npm run build:cpu      # disable Metal (CPU-only)
```

### Building against a pre-built sibling llama.cpp tree

Set `LLAMA_ROOT` as an environment variable (cmake-js's `--CD` flag does not reliably forward `-D` vars to cmake configure):

```bash
LLAMA_ROOT=/path/to/llama.cpp npm run build:root
```

The llama.cpp tree must be already built for the same architecture. On Apple Silicon, rebuild it if it was compiled under Rosetta:

```bash
cmake -B /path/to/llama.cpp/build \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_NATIVE=OFF \
  /path/to/llama.cpp
cmake --build /path/to/llama.cpp/build --target llama ggml -j$(sysctl -n hw.logicalcpu)
```

### Embedded mode

When llama-node is `add_subdirectory`'d inside a parent CMake project that already defines the `llama` target, neither `LLAMA_ROOT` nor the submodule is needed.

## Architecture

### Layer structure

```
js/index.js             ← JS wrapper: async generator + LlamaModelPool
js/index.d.ts           ← TypeScript declarations
src/addon.cpp           ← N-API entry: backend init, LlamaModel registration
src/llama_model.cpp/.h  ← LlamaModel: load model, manage context, dispatch worker
src/generate_worker.cpp/.h  ← AsyncProgressQueueWorker: llama_decode on libuv thread
vendor/llama.cpp/       ← git submodule (pinned commit)
```

### Key design points

- **`LlamaModel` (C++)** — `Napi::ObjectWrap` owning `llama_model*`, `llama_context*`, `llama_vocab*`. Context is created lazily on first `generate()` and reused unless `nCtx` changes or `resetContext: true`. Only one generation at a time (`generating_` atomic flag). Model loading dots are suppressed via a no-op `progress_callback`.

- **`GenerateWorker` (C++)** — `Napi::AsyncProgressQueueWorker<TokenChunk>` that runs on a libuv thread. Holds `ctx_mutex_` for the full decode loop. Abort is signalled via `cancel_` atomic (checked per-token and wired to `llama_context`'s `abort_callback`). Stop sequences are handled with a lookahead buffer: characters are held back when they could be the start of a stop sequence, flushed once they're proven not to match.

- **Sampler chain order** — penalties → top_k → top_p → min_p → temperature → grammar → dist. Mirrors llama.cpp's `common_sampler`.

- **`js/index.js`** — Bridges the native callback API to an async generator (queue + Promise resolver). Reads `grammarFile` from disk before the native call and passes its contents as `grammar`. Also exports `LlamaModelPool`.

- **`LlamaModelPool` (JS)** — Registers models by name, loads lazily on first `generate()` or `load()` call, supports `unload(name)` to free a single model's resources while keeping its registration.

### Threading model

- All JS calls happen on the Node.js main thread.
- `llama_decode` runs exclusively on a libuv worker thread, protected by `ctx_mutex_`.
- Token text is marshalled back to JS via `OnProgress` (main thread).
- `abort()` is lock-free: sets `cancel_` atomic.

## Known compatibility notes

- **node-addon-api 8.x**: `Napi::AddEnvironmentCleanupHook()` was removed; use `env.AddCleanupHook()` (`src/addon.cpp`).
- **cmake-js `--CD`**: does not forward `-D` variables to cmake configure. Use environment variables instead.
- **Cross-compilation**: cmake-js sets `CMAKE_OSX_ARCHITECTURES=arm64` while the host may report `x86_64`, making `-mcpu=native` invalid. `GGML_NATIVE=OFF` is forced in vendor mode to prevent this.

## Smoke test

```bash
node smoke.js /path/to/model.gguf
```

Tests baseline generation, minP, repeatPenalty, stop sequences, and `LlamaModelPool`.
