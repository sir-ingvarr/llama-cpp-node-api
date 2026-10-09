# Changelog

All notable changes to `llama-cpp-node-api` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Changelog entries start at the llama.cpp v0.6.0 upgrade (2026-10-09); earlier
history lives in the git log.

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
