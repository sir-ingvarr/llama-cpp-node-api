# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`llama-node` is a Node.js native addon (N-API) that wraps [llama.cpp](https://github.com/ggml-org/llama.cpp) for direct in-process LLM inference. It is built with `cmake-js` and targets Electron 35.

## Changelog

`CHANGELOG.md` (Keep a Changelog format) must be updated alongside any user-visible change — new options, behaviour changes, deprecations, upstream submodule bumps, fixes. Add entries under `## [Unreleased]`; on release, rename that section to the new version + date and start a fresh `[Unreleased]`. The sibling repo `node-stable-diffusion-cpp` follows the same convention.

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
                                 LlamaModelPool, chat(), parseChatResponse(),
                                 quantize(), inspect() (memoized),
                                 LlamaModel.load() (async), model.embed()
js/index.d.ts                 ← TypeScript declarations
src/addon.cpp                 ← N-API entry: backend init, AddonState, cleanup hook,
                                 LlamaModel + quantize + inspect + loadModel registration
src/addon_state.h             ← Per-env worker counter + shutdown flag + WorkerGuard RAII
src/llama_model.cpp/.h        ← LlamaModel: load model, request map, dispatch workers
src/generate_worker.cpp/.h    ← AsyncProgressQueueWorker: llama_decode on libuv thread
                                 (also computes per-token logprobs when requested)
src/quantize_worker.cpp/.h    ← AsyncWorker: llama_model_quantize on libuv thread
src/inspect_worker.cpp/.h     ← AsyncWorker: gguf_init_from_file (no_alloc) — header-only metadata
src/load_worker.cpp/.h        ← AsyncWorker: llama_model_load_from_file off the JS thread,
                                 hands the loaded model to JS via External<LoadHandle>
src/embed_worker.cpp/.h       ← AsyncWorker: llama_decode in embedding mode + Float32Array marshaling
src/state_worker.cpp/.h       ← AsyncWorkers: llama_state_save_file / _load_file under ctx_mutex_
                                 (also implements LlamaModel::SaveState/LoadState JS entry points)
src/chat_templates.cpp        ← libcommon Jinja renderer + chat-response parser
                                 (applyChatTemplateJinja, parseChatResponse)
vendor/llama.cpp/             ← git submodule (pinned commit)
```

Links against both `llama` and `llama-common` from the vendor tree. `LLAMA_BUILD_COMMON` is forced `ON` so libcommon (Jinja engine, `common_json`, PEG parser, chat format registry) is available. Since llama.cpp v0.6.0 libcommon no longer exposes nlohmann directly — `chat_templates.cpp` uses the `common_json` pimpl wrapper (`common/json.h`: `parse`, `at`, `get<T>`, no iterators).

### Key design points

- **`LlamaModel` (C++)** — `Napi::ObjectWrap` owning `llama_model*`, `llama_context*`, `llama_vocab*`. Context is created lazily on the worker thread (not the JS thread — see below) and reused unless `nCtx` changes or `resetContext: true`. Model loading dots are suppressed via a no-op `progress_callback`. KV cache quantization (`cacheTypeK` / `cacheTypeV`) is captured at construction and applied in `EnsureContext`; sentinel `-1` leaves llama.cpp's default (F16). The JS surface accepts string names from the same allowlist as `llama-cli` (`f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`); `normalizeModelOpts()` maps them to ggml_type ints before reaching native. Same instance-field+sentinel pattern carries `flashAttention` (string `'auto'|'on'|'off'` → `llama_flash_attn_type` int, sentinel `-1` = AUTO; **always** assigned because AUTO is itself a valid enum value), `nThreads`, and `nThreadsBatch` (sentinel `-1` skips the override). `useMmap` / `useMlock` keep their boolean JS surface but are mapped onto the `llama_load_mode` enum (which replaced the `use_mmap`/`use_mlock` bools in `llama_model_params` as of llama.cpp v0.6.0) via `ResolveLoadMode()` in `load_worker.h`; neither option supplied → `LLAMA_LOAD_MODE_AUTO`. Applied before `llama_model_load_from_file` in both the sync ctor and `LoadWorker::Execute`.

- **Concurrent generate()** — `LlamaModel` maintains a request map (`id → shared_ptr<RequestState>`) under `req_mutex_`. Each `generate()` call allocates a `RequestState{ id, cancel }`, registers it, and enqueues a worker. Multiple workers serialize on `ctx_mutex_` in FIFO order. On completion, `wrapped_done` removes the state from the map.

- **Per-request abort** — `llama_context::abort_callback_data` is set once (at ctx creation) to the `LlamaModel*`. The callback reads `active_request_->cancel`, where `active_request_` is set by the worker under `ctx_mutex_`. Because `llama_decode` runs on the same thread that set `active_request_`, no extra synchronisation is needed for that pointer. `model.abort()` flips every tracked request's cancel; `opts.signal` calls `abortRequest(id)` to flip one.

- **`GenerateWorker` (C++)** — `Napi::AsyncProgressQueueWorker<TokenChunk>` that runs on a libuv thread. Holds its own `shared_ptr<RequestState>`. After acquiring `ctx_mutex_`, calls `LlamaModel::PrepareContextLocked()` to build/reset the context, then runs the decode loop. Stop sequences use a lookahead buffer: characters that could start a stop sequence are held back, flushed once they're proven not to match.

- **`QuantizeWorker` (C++)** — thin wrapper over `llama_model_quantize`. Standalone (no `LlamaModel` instance needed); registered as module-level `quantize()` in `addon.cpp`. Ftype name lookup lives in `quantize_worker.cpp`.

- **`InspectWorker` (C++)** — `AsyncWorker` calling `gguf_init_from_file` with `no_alloc=true`. Reads only the GGUF header, KV pairs, and tensor descriptors (no weights, no backend init, no `llama_model`). On the worker thread it snapshots everything into plain C++ types — the `gguf_context` is held in a `unique_ptr<gguf_context, &gguf_free>` so it's freed even on the throw path; the snapshot pass is wrapped in `try/catch (std::exception&)` because `NAPI_DISABLE_CPP_EXCEPTIONS` means a thrown `bad_alloc` would otherwise neutralise subsequent N-API calls. 64-bit ints and tensor offsets/sizes use `BigInt` to avoid 2^53 precision loss; 8/16/32-bit ints become `Number`. Ggml tensor types are returned as their canonical name strings (`"F16"`, `"Q4_K"`, …) via `ggml_type_name`. Hard cap of 16M elements per array (`INSPECT_MAX_ARRAY_LEN`) guards against malformed/malicious files. Scalar-array reads use `memcpy` into typed locals — `reinterpret_cast` through `vector<uint8_t>::data()` is technically strict-aliasing UB.

- **`LoadWorker` (C++)** — `AsyncWorker` running `llama_model_load_from_file` off the JS thread. Owns a heap-allocated `LoadHandle { llama_model*, n_ctx, n_gpu_layers, embeddings, pooling_type }`; on success the handle is wrapped in a `Napi::External<LoadHandle>` whose finalizer frees the model if no JS code ever picks it up (e.g. caller dropped the Promise). The `LlamaModel` constructor accepts that External as a 3rd arg, takes ownership of the model (sets `handle->model = nullptr` so the External finalizer doesn't double-free), and skips the sync load path. Two-phase API: `addon.loadModel(path, opts, done)` gives back the handle; `new LlamaModel(path, opts, handle)` consumes it. The `LlamaModel.load()` static factory bundles both. Sync constructor path remains for CLI / tests that don't care about blocking.

- **`EmbedWorker` (C++)** — `AsyncWorker` running `llama_decode` in embedding mode and copying the pooled (or last-token) embedding into a `Napi::Float32Array`. Acquires `LlamaModel::ctx_mutex_` so it serialises with concurrent `generate()` / `embed()` on the same model. Uses `PrepareContextLocked(n_ctx, /*reset=*/true)` so each embed call starts from a fresh KV. Pooling is honoured by llama.cpp via the context's `pooling_type`; if `LLAMA_POOLING_TYPE_NONE`, the worker reads the last-token embedding via `llama_get_embeddings_ith(ctx, n_tokens-1)`, otherwise `llama_get_embeddings_seq(ctx, 0)`. The `llama_batch` is held in a `unique_ptr` with a custom deleter (`llama_batch_free` then `delete`) so it's freed on every error path. Generative models can still `generate()` while in embedding mode — every decode populates the embedding buffer as a side effect — but for embedding-only encoders only `embed()` makes sense.

- **Logprobs in `GenerateWorker`** — when `opts.logprobs` (or `opts.topLogprobs > 0`) is set, the worker computes logprobs from the **raw** model logits (`llama_get_logits_ith(ctx, -1)`) **before** any sampler-stage filtering — so users see the model's actual distribution, not the post-grammar/post-penalty one. Logsumexp is computed once per token; chosen-token logprob = `logits[id] - lse`. For top-K, a `partial_sort` over the (logit, idx) pairs picks the top entries — O(N log K) per token, fine for vocab=128k / K=5. Logprobs ride on `TokenChunk` and are passed via `OnProgress` as extra args to the JS callback (`cb(text, logprob, topLogprobs[])`); JS bundles them into `{ text, logprob, topLogprobs }`. Stop-sequence lookahead caveat: when a flush chunk spans multiple sampled tokens, `logprob` is the most-recent token's value — documented contract. Recommended: avoid `stop` sequences when you need exact token-level alignment.

- **Lifecycle / shutdown safety** — `AddonState` (per-env, `addon_state.h`) holds an atomic worker counter and a `shutting_down` flag. Every worker's `Execute()` instantiates a `WorkerGuard` at the top, which inc/decrements the counter; the env cleanup hook (registered in `addon.cpp`) sets `shutting_down=true`, waits up to 5s for the counter to drain (`AddonState::wait_for_drain`), then calls `llama_backend_free`. `LlamaModel::AbortCallback` *also* reads `shutting_down`, so an in-flight `llama_decode` returns promptly during teardown rather than running to the next token boundary. Without this, `llama_backend_free` could race a libuv-thread worker mid-decode (UAF on Electron reload / `worker_threads` shutdown).

- **AsyncWorker resource RAII** — `GenerateWorker` owns its sampler chain via `unique_ptr<llama_sampler, &llama_sampler_free>`; new error branches in `Execute` no longer need to remember to call `llama_sampler_free`. Same pattern in `InspectWorker` for `gguf_context`. Manual `llama_sampler_free` calls have been removed.

- **`chat_templates.cpp`** — Jinja chat rendering + response parsing via libcommon. Exposes two `LlamaModel` methods:
  - `applyChatTemplateJinja(messages, opts)` → `{ prompt, format, parser?, grammar?, grammarLazy?, grammarTriggerPatterns?, grammarTriggerTokens?, preservedTokens?, additionalStops? }`. Auto-falls back to `llama_chat_apply_template` (legacy C API, resolves alias names like `mistral-v7-tekken`) when the embedded template isn't Jinja source; throws if `tools` / `jsonSchema` were supplied on that path (can't be honoured). `chatTemplateOverride` lets the caller supply full Jinja source when the embedded template is an alias.
  - `parseChatResponse(text, { format, parser, ... })` → `{ content, reasoningContent, toolCalls[] }`. Round-trips the opaque `parser` blob (from `common_peg_arena::save()`) and dispatches to the right per-format parser. `format === 'legacy'` bypasses parsing and returns `text` as content.
  - The `common_chat_templates *` is lazily initialised on first use and cached on the `LlamaModel` instance; freed in `~LlamaModel` / `Dispose()`.

- **Lazy grammar** — `GenerateWorker` branches on `grammarTriggerPatterns` / `grammarTriggerTokens`: if either is non-empty, uses `llama_sampler_init_grammar_lazy_patterns()` (grammar activates only after a trigger appears in the output); else uses eager `llama_sampler_init_grammar()`. Backward-compatible: existing callers passing only `grammar` see no behavioural change. Trigger conversion from libcommon's typed `common_grammar_trigger` mirrors `common/sampling.cpp` (regex_escape words, anchor PATTERN_FULL patterns, TOKEN triggers go to the separate token-array argument).

- **Sampler chain order** — reasoning_budget → logit_bias → grammar → penalties → dry → top_n_sigma → top_k → typical → top_p → min_p → xtc → temp/temp_ext → dist-or-adaptive_p (mirrors upstream `common_sampler`). When `mirostat` is 1/2 the filter stack is replaced: … → grammar → temp → mirostat. Grammar must come before the probability filters so it can zero out non-grammatical tokens before they're discarded for low probability. `logitBias` is added **before** grammar so a positive bias on a grammar-disallowed token cannot resurrect it (grammar masks to -inf afterward and has the final say). `seed` (default `-1` → `LLAMA_DEFAULT_SEED`) feeds dist/xtc/mirostat/adaptive-p. All generation options travel in a single `GenerateParams` struct (generate_worker.h) — don't grow the worker ctor again.

- **Reasoning budget** — `common_reasoning_budget_init` (libcommon) sampler added FIRST in the chain when `reasoningBudget >= 0` and both tags are set; it forces the end tag by masking all other logits, so combining it with an *eager* grammar is unsupported (lazy tool-call grammars are fine — they trigger after the block closes). Tags come from `common_chat_params.thinking_start_tag/thinking_end_tags`, surfaced by `applyChatTemplateJinja` as `thinkingStartTag`/`thinkingEndTags`; `chat()` wires them automatically and also passes `generationPrompt` as `reasoningBudgetPrefill` so a template that pre-opens `<think>` starts counting immediately (mirrors upstream's prefill accept, incl. the leading-space-token skip).

- **LoRA adapters** — `loras: [{path, scale}]` model-load option. Adapters load with the model (`llama_adapter_lora_init`; on the worker thread in `LoadWorker::Execute`, sync in the ctor) and are applied to every new context in `EnsureContext` via `llama_set_adapters_lora`. No manual free: llama.cpp frees adapters with the model.

- **Session state** — `saveState(path, {tokens})` / `loadState(path)` (state_worker.cpp) wrap `llama_state_save_file/_load_file` under `ctx_mutex_`. Save requires a live ctx; load creates one via `PrepareContextLocked(n_ctx_default, reset=false)`. The token list is llama.cpp's session-token convention (caller supplies via `tokenize()`, gets it back on load); files are version/model/ctx-config specific.

- **`js/index.js`** — Bridges the native callback API to an async generator. Reads `grammarFile` asynchronously (`fs.promises.readFile`) before the native call. Wires `opts.signal` (AbortSignal) to `native.abortRequest(id)`. Exports `LlamaModel`, `LlamaModelPool`, `quantize`, `quantizeFtypes`, `inspect`, `clearInspectCache`. `LlamaModel.load(path, opts)` is the async factory — calls `addon.loadModel`, then constructs the JS-side instance with the External handle so the C++ ctor skips the sync load. `LlamaModel.chat(opts)` is a one-shot helper that bundles `applyChatTemplateJinja` → `generate` → `parseChatResponse`, buffering tokens and returning `{ content, reasoningContent, toolCalls[], format, raw }`. No streaming variant yet — for streaming, use `generate()` directly. `model.embed(text)` returns `Promise<Float32Array>` and throws if not constructed with `embeddings: true`. JS-side `normalizeModelOpts()` converts string `poolingType` (`'mean'`, `'cls'`, …) to the `llama_pooling_type` enum int before reaching native.

- **Generator lifecycle (CRITICAL)** — `createChannel()` rejects via `next()` (not via a post-loop `ch.error` re-throw), so a producer error reaches the consumer's `for await` as a rejection. The generator's `finally` calls `native.abortRequest(reqId)` whenever the channel hasn't closed, so a consumer `break`/`return`/throw inside `for await` cancels the worker — without this, the native side would decode into a closed channel until natural completion. Error precedence: when both happen, **abort wins** over a channel error wins over a consumer-body error (the abort intent is what the user actually asked for). `signal.reason` is honoured per the Web AbortSignal contract — if the caller did `ac.abort(myError)`, the generator throws `myError`, not a generic `AbortError`. Native exits cleanly on abort (`ret==2` from `llama_decode` → `close(null)`), so the post-loop `if (aborted) throw` is required to surface the abort even though the channel closed without a JS-side error.

- **`#disposed` guard (JS)** — both `LlamaModel` and `LlamaModelPool` track a `#disposed` private flag. `dispose()` is idempotent. `#assertLive()` is called at the top of every JS-side method (`generate`, `tokenize`, `applyChatTemplate*`, etc.) so use-after-`dispose` throws `Error("LlamaModel: instance has been disposed")` instead of bottoming out in a cryptic native error.

- **Token text marshaling** — `GenerateWorker::OnProgress` emits `Napi::String` (not `Napi::Buffer`); the C++ side already enforces UTF-8 boundaries via `complete_utf8_boundary`, so V8 can short-string-optimise. After each callback, `OnProgress` checks `env.IsExceptionPending()` — with `NAPI_DISABLE_CPP_EXCEPTIONS` a thrown JS callback silently neutralises subsequent N-API calls; bailing out of the loop avoids cascading bad effects.

- **`inspect()` memoization (JS)** — module-level `Map`-backed LRU keyed by canonical realpath, validated by file `mtime + size` via `fs.stat`. Default bound 8 entries; the tokenizer-vocab array dominates per-entry memory. Override via env var `LLAMA_NODE_INSPECT_CACHE_MAX` (set to `0` to disable caching globally). On a hit the same `Promise` (and resolved object) is returned, so callers compare by reference. Rejections auto-evict so retries can recover. Opt-out per call: `inspect(path, { cache: false })`. Manual flush: `clearInspectCache()`. The cache lives on the JS side rather than C++ because invalidation policy is workload-specific and v8-side memory is easier for consumers to reason about.

- **`LlamaModelPool` (JS)** — Registers models by name, loads lazily on first `generate()` or `load()` call, supports `unload(name)` to free a single model's resources while keeping its registration.

### Threading model

- All JS calls happen on the Node.js main thread.
- **The JS main thread never acquires `ctx_mutex_`.** Context reset/creation is deferred to the worker (`PrepareContextLocked` under `ctx_mutex_`). Taking `ctx_mutex_` on the main thread would block Node's event loop while any other generation is decoding, stalling timers, I/O, and `AbortSignal` events — a real bug caught during development.
- `llama_decode` runs exclusively on a libuv worker thread, protected by `ctx_mutex_`. Multiple workers queue on the libuv thread pool and serialize on the mutex.
- Token text is marshalled back to JS via `OnProgress` (main thread).
- Abort is lock-free: `abort()` / `abortRequest(id)` flip atomic flags; `abort_callback` reads the active request's flag from inside `llama_decode`.
- `dispose()` sets all request cancel flags, then acquires `ctx_mutex_` to wait for the active worker, then frees `ctx_` / `model_` / nulls `vocab_`. Queued workers fast-reject when they eventually run (cancel already set, or `disposed()` true after acquiring the mutex).
- **Env teardown drain**: every worker's `Execute()` (`GenerateWorker`, `QuantizeWorker`, `InspectWorker`, `LoadWorker`, `EmbedWorker`) increments `AddonState::in_flight` via a top-of-body `WorkerGuard`; the env cleanup hook flips `shutting_down`, waits up to 5s for the counter to drain, then calls `llama_backend_free`. `LlamaModel::AbortCallback` reads `shutting_down` so an in-flight `llama_decode` returns promptly during shutdown rather than running to its next token boundary.
- `EmbedWorker` shares `ctx_mutex_` with `GenerateWorker`, so concurrent `generate()` and `embed()` calls on the same model serialise correctly. `LoadWorker` does **not** touch `ctx_mutex_` — it loads a fresh `llama_model*` independent of any existing instance.

## Known compatibility notes

- **node-addon-api 8.x**: `Napi::AddEnvironmentCleanupHook()` was removed; use `env.AddCleanupHook()` (`src/addon.cpp`).
- **cmake-js `--CD`**: does not forward `-D` variables to cmake configure. Use environment variables instead.
- **Cross-compilation**: cmake-js sets `CMAKE_OSX_ARCHITECTURES=arm64` while the host may report `x86_64`, making `-mcpu=native` invalid. `GGML_NATIVE=OFF` is forced in vendor mode to prevent this.

## Tests

`node:test`-based, zero dependencies. `npm test` runs everything via
`test/run.js`, which resolves the functional-test model path from
`LLAMA_NODE_MODEL_PATH` → `test/.model-path` cache → interactive prompt
(reads `/dev/tty`, so it works from the pre-commit hook; silently skips
functional when there is no terminal, e.g. CI).

```bash
npm run test:unit         # no model needed — runs in CI on every platform
                          # in the prebuild matrix, gating publish
npm run test:functional   # real model; text-quality + grammar + feature coverage
npm run hooks:install     # sets core.hooksPath=.githooks (pre-commit runs npm test)
```

Optional env: `LLAMA_NODE_LORA_PATH` enables the positive LoRA test.
Unit tests include a hand-crafted minimal GGUF fixture that exercises the
full `inspect()` pipeline without a model — keep it in sync if the GGUF
header layout ever changes upstream.

## Smoke tests

```bash
node smoke.js              /path/to/model.gguf    # sampling + pool
node smoke_concurrent.js   /path/to/model.gguf    # queue + abort semantics
node smoke_chat_jinja.js   /path/to/model.gguf    # Jinja templates + tools + chat()
node smoke_features.js     /path/to/model.gguf    # new samplers + state save/load +
                                                  # reasoning budget + loadMode + LoRA plumbing
```

`smoke.js` covers baseline generation, minP, repeatPenalty, stop sequences, and `LlamaModelPool`.

`smoke_concurrent.js` covers: two overlapping `generate()` calls on the same model; per-request cancellation via `opts.signal` (AbortSignal → `AbortError`); global `model.abort()` stopping everything in flight.

`smoke_chat_jinja.js` covers: `applyChatTemplateJinja` on content-only and tool-annotated prompts; end-to-end `chat()` extracting structured tool calls; `parseChatResponse` on free-form output; lazy grammar with a manual trigger. Requires a tool-trained model (e.g. Llama 3.1+) to exercise the full tools path; works on any Jinja-templated model for the content-only path.

### Native addon loader

`js/index.js` prefers the platform-specific prebuilt package from `optionalDependencies` over the local `build/` — convenient for end users, but inconvenient during development of native code. If you want the local build to win, delete the prebuild:

```bash
rm -rf node_modules/@llama-cpp-node-api
```

A fresh `npm install` restores it.

## Quantize

`quantize(inputPath, outputPath, opts)` wraps `llama_model_quantize`. Only GGUF → GGUF requantization. Safetensors / HF → GGUF conversion is **not** exposed — it lives in llama.cpp's `convert_hf_to_gguf.py` and requires a Python environment (`torch`, `safetensors`, `gguf`). Bringing it in would force a ~1.5 GB dependency tree on consumers; keeping it out-of-process is the deliberate choice.
