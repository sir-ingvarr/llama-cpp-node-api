# Changelog

All notable changes to `llama-cpp-node-api` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Changelog entries start at the llama.cpp v0.6.0 upgrade (2026-10-09); earlier
history lives in the git log.

## [Unreleased]

### Changed

- llama.cpp submodule updated to b11541 (`f2918ca`). Highlights inherited
  from upstream: Metal flash-attention kernels for more head-size pairs,
  few-row MMA matrix-vector kernels for most quant types, MoE expert
  caching (`moe_cache_size`, not yet exposed here), and the chat-parser
  refactor below.
- **Breaking:** saved session states from earlier releases no longer load.
  `saveState()` files are bound to the llama.cpp version that wrote them
  (as documented since 0.15.0), so this submodule bump invalidates existing
  snapshots — `loadState()` rejects them and the conversation must be
  re-processed (or re-saved) once. No code change is needed.
- **Breaking (edge case):** `applyChatTemplateJinja().parser` is now a
  compact handle string (`llama-node-parser:<id>`) instead of a serialized
  parser blob — upstream
  removed `common_peg_arena::save()/load()`. The contract is unchanged
  (treat it as an opaque string and round-trip it into
  `parseChatResponse`), but a handle is only valid on the model instance
  that produced it, while that instance is alive, for the 16 most recent
  `applyChatTemplateJinja` calls on it. Persisting the handle across
  processes or model reloads — which previously happened to work — no
  longer does; re-render the template to get a fresh one.

### Added

- **Vulkan GPU support in the Linux and Windows prebuilds.** The
  `linux-x64`, `linux-arm64` (new) and `win32-x64` platform packages are now
  built with `GGML_BACKEND_DL`: ggml backends ship as runtime-loadable
  libraries next to the addon instead of one statically linked CPU backend.
  At runtime the best CPU variant for the machine is picked by score
  (SSE4.2 → AVX-512/AMX on x86-64, armv8/armv9 on ARM — previously the
  prebuilds were capped at a fixed baseline), and the Vulkan backend
  (NVIDIA / AMD / Intel through the system Vulkan loader) is used when a
  loader + driver are present, falling back to CPU silently otherwise.
  Nothing to configure; `GGML_BACKEND_PATH` is honoured for out-of-tree
  backends. macOS prebuilds are unchanged (static, embedded Metal).
- **`linux-arm64` prebuilt platform package** — Graviton/Ampere servers and
  Docker on Apple Silicon no longer need a source build.
- **`LLAMA_NODE_BACKEND_DL=1` / `LLAMA_NODE_VULKAN=1`** environment
  variables for source builds (same mechanism as `LLAMA_ROOT`): enable the
  backend-DL layout and the ggml Vulkan backend respectively. Vulkan needs
  the Vulkan SDK (glslc + headers) at build time only.
- **`model.memoryUsage()`** — synchronous, non-blocking memory report:
  `{ modelBytes, nParams, stateBytes }`. `modelBytes` is the loaded weight
  footprint (`llama_model_size`), `stateBytes` the live context state size
  (`llama_state_get_size`: KV cache + RNG + logits). `stateBytes` is `null`
  before the first generation or while a decode holds the context — the JS
  thread only ever `try_lock`s `ctx_mutex_`, never blocks on it.

## [0.15.0] - 2026-10-09

### Added

- **LoRA adapters.** Pass `loras: ['adapter.gguf']` (or `{ path, scale }`
  objects) when loading a model to blend one or more LoRA adapters into
  generation. Works with both `new LlamaModel()` and the async
  `LlamaModel.load()`.
- **More sampling strategies** in `generate()`, all opt-in — existing calls
  behave exactly as before:
  - typical-p: `typicalP`
  - XTC: `xtcProbability`, `xtcThreshold`
  - top-n-sigma: `topNSigma`
  - DRY repetition penalty: `dryMultiplier`, `dryBase`, `dryAllowedLength`,
    `dryPenaltyLastN`, `drySequenceBreakers`
  - Mirostat v1/v2: `mirostat`, `mirostatTau`, `mirostatEta`
  - dynamic temperature: `dynatempRange`, `dynatempExponent`
  - adaptive-p: `adaptivePTarget`, `adaptivePDecay`
- **Reasoning budget for thinking models.** `chat({ reasoningBudget: 200 })`
  caps how many tokens the model may spend inside its thinking block and
  closes the block cleanly when the budget runs out (optionally injecting
  `reasoningBudgetMessage` first). The thinking tags are detected from the
  model's chat template automatically; `generate()` accepts the raw tag
  options for template-free setups, and `applyChatTemplateJinja` now reports
  the detected `thinkingStartTag` / `thinkingEndTags`.
- **Conversation snapshots.** `model.saveState(path, { tokens })` and
  `model.loadState(path)` save and restore the context state (KV cache), so
  a long conversation can resume later without re-processing the whole
  prompt. State files only load with the same model, context settings, and
  llama.cpp version that wrote them.
- **`loadMode` option** — finer control over how weights are loaded:
  `'auto' | 'none' | 'mmap' | 'mlock' | 'mmap+mlock' | 'directio'`. The
  existing `useMmap` / `useMlock` booleans keep working.
- `llamaVersion()` — the bundled llama.cpp version string, for diagnostics.
- `getModelInfo().suppressTokens` — tokens the model's metadata asks to
  suppress (empty for most text models; useful as `logitBias` bans).
- **Test suite.** `npm test` runs unit tests (no model required) and
  functional tests against a real model — set `LLAMA_NODE_MODEL_PATH` or
  answer the prompt. `npm run hooks:install` adds a pre-commit hook that
  runs the suite; unit tests also run on every platform in the release
  pipeline before anything is published. A `smoke_features.js` demo script
  covers the new features end to end.
- README now carries live Build / Tests status badges and links to this
  changelog, and GitHub release notes are generated from the matching
  changelog section automatically.

### Fixed

- **Stop sequences are no longer missed mid-token.** When a single sampled
  token contained a stop string in the middle (for example stopping on
  `"\n"` while the model emits a `"\n\n"` token), the stop could leak into
  the output. Output now ends exactly at the first stop occurrence.

## [0.14.0] - 2026-10-09

### Changed

- **Bumped `vendor/llama.cpp` from `74a80dd9` (June 2026) to the `v0.6.0` release
  (`d8123504`, 2026-10-05).** First upstream semver release; session/state
  formats move to `LLAMA_SESSION_VERSION 11` / `LLAMA_STATE_SEQ_VERSION 4`
  (not exposed by this wrapper, so no consumer impact).
- `useMmap` / `useMlock` keep their boolean JS surface but now map onto the new
  `llama_load_mode` enum (upstream replaced the `use_mmap`/`use_mlock` bools in
  `llama_model_params`). When neither option is supplied, the new
  `LLAMA_LOAD_MODE_AUTO` lets llama.cpp auto-detect based on device
  capabilities — previously the default was always mmap-on.
- `chat_templates.cpp` migrated from direct nlohmann usage to libcommon's new
  `common_json` wrapper (upstream no longer exposes nlohmann). No JS-visible
  change.
- Chat format names returned by `applyChatTemplateJinja` /accepted by
  `parseChatResponse` are now resolved by walking upstream's format registry
  instead of a hardcoded list, so new upstream formats (e.g. `peg-minimax-m3`)
  work without wrapper changes.
- `llama_sampler_init_penalties` call updated for the restored `n_vocab`
  parameter (internal; no behaviour change).

### Added

- `Q2_0` quantization ftype in `quantize()` / `quantizeFtypes()` (new in
  llama.cpp v0.6.0).

### Notes

- Verified on macOS ARM64 (Metal): `smoke.js`, `smoke_concurrent.js`
  (concurrency + abort), `smoke_chat_jinja.js` (Jinja templates, tool calls,
  lazy grammar), plus `inspect()`, async `LlamaModel.load()`, `embed()`.
