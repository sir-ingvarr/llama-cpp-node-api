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
js/index.js                   ← JS wrapper: async generator, AbortSignal bridge,
                                 LlamaModelPool, quantize()
js/index.d.ts                 ← TypeScript declarations
src/addon.cpp                 ← N-API entry: backend init, LlamaModel + quantize registration
src/llama_model.cpp/.h        ← LlamaModel: load model, request map, dispatch worker
src/generate_worker.cpp/.h    ← AsyncProgressQueueWorker: llama_decode on libuv thread
src/quantize_worker.cpp/.h    ← AsyncWorker: llama_model_quantize on libuv thread
vendor/llama.cpp/             ← git submodule (pinned commit)
```

### Key design points

- **`LlamaModel` (C++)** — `Napi::ObjectWrap` owning `llama_model*`, `llama_context*`, `llama_vocab*`. Context is created lazily on the worker thread (not the JS thread — see below) and reused unless `nCtx` changes or `resetContext: true`. Model loading dots are suppressed via a no-op `progress_callback`.

- **Concurrent generate()** — `LlamaModel` maintains a request map (`id → shared_ptr<RequestState>`) under `req_mutex_`. Each `generate()` call allocates a `RequestState{ id, cancel }`, registers it, and enqueues a worker. Multiple workers serialize on `ctx_mutex_` in FIFO order. On completion, `wrapped_done` removes the state from the map.

- **Per-request abort** — `llama_context::abort_callback_data` is set once (at ctx creation) to the `LlamaModel*`. The callback reads `active_request_->cancel`, where `active_request_` is set by the worker under `ctx_mutex_`. Because `llama_decode` runs on the same thread that set `active_request_`, no extra synchronisation is needed for that pointer. `model.abort()` flips every tracked request's cancel; `opts.signal` calls `abortRequest(id)` to flip one.

- **`GenerateWorker` (C++)** — `Napi::AsyncProgressQueueWorker<TokenChunk>` that runs on a libuv thread. Holds its own `shared_ptr<RequestState>`. After acquiring `ctx_mutex_`, calls `LlamaModel::PrepareContextLocked()` to build/reset the context, then runs the decode loop. Stop sequences use a lookahead buffer: characters that could start a stop sequence are held back, flushed once they're proven not to match.

- **`QuantizeWorker` (C++)** — thin wrapper over `llama_model_quantize`. Standalone (no `LlamaModel` instance needed); registered as module-level `quantize()` in `addon.cpp`. Ftype name lookup lives in `quantize_worker.cpp`.

- **Sampler chain order** — grammar → penalties → top_k → top_p → min_p → temperature → dist. Grammar must be first so it zeroes out non-grammatical tokens from the full distribution before probability filters can discard required ones.

- **`js/index.js`** — Bridges the native callback API to an async generator (queue + Promise resolver). Reads `grammarFile` from disk before the native call. Wires `opts.signal` (AbortSignal) to `native.abortRequest(id)` and throws `AbortError` when the signal fires. Exports `LlamaModel`, `LlamaModelPool`, `quantize`, `quantizeFtypes`.

- **`LlamaModelPool` (JS)** — Registers models by name, loads lazily on first `generate()` or `load()` call, supports `unload(name)` to free a single model's resources while keeping its registration.

### Threading model

- All JS calls happen on the Node.js main thread.
- **The JS main thread never acquires `ctx_mutex_`.** Context reset/creation is deferred to the worker (`PrepareContextLocked` under `ctx_mutex_`). Taking `ctx_mutex_` on the main thread would block Node's event loop while any other generation is decoding, stalling timers, I/O, and `AbortSignal` events — a real bug caught during development.
- `llama_decode` runs exclusively on a libuv worker thread, protected by `ctx_mutex_`. Multiple workers queue on the libuv thread pool and serialize on the mutex.
- Token text is marshalled back to JS via `OnProgress` (main thread).
- Abort is lock-free: `abort()` / `abortRequest(id)` flip atomic flags; `abort_callback` reads the active request's flag from inside `llama_decode`.
- `dispose()` sets all request cancel flags, then acquires `ctx_mutex_` to wait for the active worker, then frees `ctx_` / `model_`. Queued workers fast-reject when they eventually run (cancel already set, or `disposed()` true after acquiring the mutex).

## Known compatibility notes

- **node-addon-api 8.x**: `Napi::AddEnvironmentCleanupHook()` was removed; use `env.AddCleanupHook()` (`src/addon.cpp`).
- **cmake-js `--CD`**: does not forward `-D` variables to cmake configure. Use environment variables instead.
- **Cross-compilation**: cmake-js sets `CMAKE_OSX_ARCHITECTURES=arm64` while the host may report `x86_64`, making `-mcpu=native` invalid. `GGML_NATIVE=OFF` is forced in vendor mode to prevent this.

## Smoke tests

```bash
node smoke.js             /path/to/model.gguf    # sampling + pool
node smoke_concurrent.js  /path/to/model.gguf    # queue + abort semantics
```

`smoke.js` covers baseline generation, minP, repeatPenalty, stop sequences, and `LlamaModelPool`.

`smoke_concurrent.js` covers: two overlapping `generate()` calls on the same model; per-request cancellation via `opts.signal` (AbortSignal → `AbortError`); global `model.abort()` stopping everything in flight.

### Native addon loader

`js/index.js` prefers the platform-specific prebuilt package from `optionalDependencies` over the local `build/` — convenient for end users, but inconvenient during development of native code. If you want the local build to win, delete the prebuild:

```bash
rm -rf node_modules/@llama-cpp-node-api
```

A fresh `npm install` restores it.

## Quantize

`quantize(inputPath, outputPath, opts)` wraps `llama_model_quantize`. Only GGUF → GGUF requantization. Safetensors / HF → GGUF conversion is **not** exposed — it lives in llama.cpp's `convert_hf_to_gguf.py` and requires a Python environment (`torch`, `safetensors`, `gguf`). Bringing it in would force a ~1.5 GB dependency tree on consumers; keeping it out-of-process is the deliberate choice.
