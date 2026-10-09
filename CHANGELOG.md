# Changelog

All notable changes to `llama-cpp-node-api` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Changelog entries start at the llama.cpp v0.6.0 upgrade (2026-10-09); earlier
history lives in the git log.

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
