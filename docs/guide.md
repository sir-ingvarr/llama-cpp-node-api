# llama-node — Developer Guide

## Table of Contents

1. [Preparation & Build](#1-preparation--build)
   - 1.1 [Prerequisites](#11-prerequisites)
   - 1.2 [Repository layout](#12-repository-layout)
   - 1.3 [Build modes](#13-build-modes)
   - 1.4 [Build — standalone (with bundled llama.cpp)](#14-build--standalone-with-bundled-llamacpp)
   - 1.5 [Build — embedded inside the llama.cpp tree](#15-build--embedded-inside-the-llamacpp-tree)
   - 1.6 [Verifying the build](#16-verifying-the-build)
2. [Publishing & Consuming the Package](#2-publishing--consuming-the-package)
   - 2.1 [Publishing to npm](#21-publishing-to-npm)
   - 2.2 [Installing in a Node.js project](#22-installing-in-a-nodejs-project)
   - 2.3 [Installing in an Electron project](#23-installing-in-an-electron-project)
   - 2.4 [Consuming from a local path (monorepo / dev loop)](#24-consuming-from-a-local-path-monorepo--dev-loop)
3. [JavaScript API](#3-javascript-api)
   - 3.1 [`new LlamaModel(modelPath, opts?)`](#31-new-llamamodelmodelpath-opts)
   - 3.2 [`model.generate(prompt, opts?)`](#32-modelgenerateprompt-opts)
   - 3.3 [Cancellation: `model.abort()` and `opts.signal`](#33-cancellation)
   - 3.4 [`model.dispose()`](#34-modeldispose)
   - 3.5 [Model introspection: `contextLength`, `chatTemplate`, `getModelInfo`](#35-model-introspection)
   - 3.6 [Tokenization: `model.tokenize`, `model.detokenize`](#36-tokenization)
   - 3.7 [`model.applyChatTemplate(messages, opts?)` (legacy)](#37-modelapplychattemplatemessages-opts-legacy)
   - 3.8 [`model.applyChatTemplateJinja(messages, opts?)`](#38-modelapplychattemplatejinjamessages-opts)
   - 3.9 [`model.parseChatResponse(text, opts)`](#39-modelparsechatresponsetext-opts)
   - 3.10 [`model.chat(opts)`](#310-modelchatopts)
   - 3.11 [`LlamaModelPool`](#311-llamamodelpool)
   - 3.12 [`quantize` / `quantizeFtypes`](#312-quantize--quantizeftypes)
   - 3.13 [Explicit resource management (`using`)](#313-explicit-resource-management-using)
   - 3.14 [Error handling](#314-error-handling)
   - 3.15 [Full usage examples](#315-full-usage-examples)

---

## 1. Preparation & Build

### 1.1 Prerequisites

| Tool | Minimum version | Notes |
|------|----------------|-------|
| Node.js | 18 LTS | 22 LTS recommended |
| npm | 8 | ships with Node |
| CMake | 3.15 | `brew install cmake` on macOS |
| C++ compiler | C++17-capable | Xcode CLT on macOS, `build-essential` on Linux |
| Xcode Command Line Tools | any recent | macOS only — `xcode-select --install` |

For GPU builds only:

| Backend | Extra requirement |
|---------|-------------------|
| Metal (macOS) | Xcode 14+ (included with CLT) — enabled **by default** on macOS |
| CUDA | NVIDIA CUDA Toolkit ≥ 12.0, matching driver |

> **Note on the Electron target.** `package.json` declares `"runtime": "electron", "runtimeVersion": "35.0.0"` for cmake-js. This means the addon ABI is compiled against Electron 35 headers. If you intend to consume the package from plain Node.js (not Electron), change `cmake-js.runtime` to `"node"` and remove `runtimeVersion` before building.

### 1.2 Repository layout

```
llama-node/
├── CMakeLists.txt          # cmake-js build definition
├── package.json
├── src/
│   ├── addon.cpp           # N-API entry: backend init, LlamaModel registration
│   ├── llama_model.cpp/.h  # JS-facing class: load model, manage context
│   └── generate_worker.cpp/.h  # libuv async worker: runs llama_decode
├── js/
│   ├── index.js            # JS wrapper (async generator surface)
│   └── index.d.ts          # TypeScript declarations
├── build/                  # cmake-js output (git-ignored)
│   └── Release/
│       └── llama_node.node # compiled addon
└── vendor/                 # optional — llama.cpp as a git submodule
    └── llama.cpp/
```

### 1.3 Build modes

The project supports two mutually exclusive ways to supply the `llama` CMake target:

| Mode | When to use | How |
|------|-------------|-----|
| **Standalone** (`LLAMA_NODE_VENDOR=ON`) | Building this package in isolation, outside the llama.cpp source tree | Initialize the `vendor/llama.cpp` submodule; pass `-DLLAMA_NODE_VENDOR=ON` |
| **Embedded** (`LLAMA_NODE_VENDOR=OFF`, default) | llama-node lives inside or alongside a llama.cpp CMake project that already defines the `llama` target | The parent `CMakeLists.txt` adds this directory; no submodule needed |

### 1.4 Build — standalone (with bundled llama.cpp)

**Step 1 — clone with the submodule**

```bash
git clone --recurse-submodules <repo-url>
# or, if already cloned:
git submodule update --init --recursive
```

**Step 2 — install JS dependencies**

```bash
npm install
```

This installs `cmake-js` and `node-addon-api` into `node_modules/`.

> `npm install` also runs the `install` script (`cmake-js compile`), which will attempt to build the addon immediately. If you are not ready to build yet (e.g. submodule not initialized), run `npm install --ignore-scripts` and build manually in step 3.

**Step 3 — compile**

```bash
# Default: Release, Metal-accelerated on macOS
npm run build

# CPU-only (no Metal / no CUDA)
npm run build:cpu

# CUDA
npm run build:cuda

# Debug symbols
npm run build:debug

# Force a clean rebuild from scratch
npm run rebuild
```

All of the above accept extra CMake variables via `--`:

```bash
# Standalone + CUDA
npm run build -- -DLLAMA_NODE_VENDOR=ON -DGGML_CUDA=ON
```

The compiled `llama_node.node` is written to `build/Release/`.

### 1.5 Build — embedded inside the llama.cpp tree

When llama-node is a subdirectory of a larger CMake project that already defines the `llama` target, set `LLAMA_NODE_VENDOR=OFF` (the default). cmake-js must still be invoked from the `llama-node` directory to generate the `.node` binary:

```bash
cd llama-node
npm install --ignore-scripts   # only install JS deps, skip auto-build
cmake-js compile               # uses the parent-scope llama target
```

Or let the parent build drive cmake-js by invoking it as a CMake `execute_process` or ExternalProject step — the exact integration is project-specific.

### 1.6 Verifying the build

A quick smoke-test from the repository root (requires a `.gguf` model file):

```js
// smoke.js
const { LlamaModel } = require('./js/index.js');
const model = new LlamaModel(process.argv[2]);
console.log('contextLength:', model.contextLength);
model.dispose();
```

```bash
node smoke.js /path/to/model.gguf
# Expected: contextLength: 0
# (context is created lazily on first generate() call)
```

---

## 2. Publishing & Consuming the Package

### 2.1 Publishing to npm

The `files` field in `package.json` already limits what gets packed:

```json
"files": ["js/", "src/", "CMakeLists.txt", "package.json"]
```

The compiled `.node` binary is **not** included — consumers compile from source on `npm install`.

```bash
# Dry-run: inspect what will be published
npm pack --dry-run

# Bump version then publish
npm version patch   # or minor / major
npm publish
```

> If publishing a scoped package (e.g. `@myorg/llama-node`), add `"access": "public"` to the `publish` config or pass `--access public`.

Because the addon compiles on install via the `"install": "cmake-js compile"` script, consumers need the same build prerequisites (CMake, C++ compiler) on their machines. For a pre-built binary distribution strategy see node-pre-gyp or `@mapbox/node-pre-gyp`.

### 2.2 Installing in a Node.js project

```bash
npm install llama-node
# cmake-js compile runs automatically via the package's install script
```

After installation the addon is at `node_modules/llama-node/build/Release/llama_node.node` and the JS entry point is `node_modules/llama-node/js/index.js`.

```js
const { LlamaModel } = require('llama-node');
```

TypeScript users get types automatically via the `"types"` field:

```ts
import { LlamaModel, LlamaModelOptions, GenerateOptions } from 'llama-node';
```

### 2.3 Installing in an Electron project

The addon is pre-configured for **Electron 35**. If your project uses a different Electron version, either:

**Option A — override at install time using cmake-js flags:**

```bash
npm install llama-node --cmake-js-runtime=electron --cmake-js-runtime-version=<your-version>
```

**Option B — patch `cmake-js` config in your own `package.json`** (takes precedence during `npm install`):

```json
"cmake-js": {
  "runtime": "electron",
  "runtimeVersion": "<your-version>"
}
```

Then rebuild:

```bash
./node_modules/.bin/cmake-js rebuild --runtime electron --runtime-version <your-version>
```

In the Electron **main process** (not the renderer), require the addon as normal:

```js
// main.js (Electron main process)
const { LlamaModel } = require('llama-node');
```

Native addons cannot be loaded in the renderer process or web workers. Use IPC (`ipcMain` / `ipcRenderer`) to proxy calls from the renderer to the main process.

### 2.4 Consuming from a local path (monorepo / dev loop)

```bash
# From the consumer project
npm install /absolute/path/to/llama-node
# or
npm install ../relative/path/to/llama-node
```

npm creates a symlink in `node_modules/llama-node`. The `install` script runs once and the build output is shared from the source tree.

Alternatively, use `npm link`:

```bash
# In llama-node directory
npm link

# In the consumer project
npm link llama-node
```

---

## 3. JavaScript API

All public API is exported from `js/index.js` (CommonJS). TypeScript types are in `js/index.d.ts`.

```js
const { LlamaModel } = require('llama-node');
```

---

### 3.1 `new LlamaModel(modelPath, opts?)`

Loads a GGUF model file from disk into memory and prepares it for inference.

```ts
constructor(modelPath: string, opts?: LlamaModelOptions)
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `modelPath` | `string` | required | Absolute or resolvable path to a `.gguf` model file. |
| `opts.nGpuLayers` | `number` | `99` | Number of transformer layers to offload to the GPU. `0` = CPU only. `99` (or any large number) = offload all layers. |
| `opts.nCtx` | `number` | `2048` | Default context window size in tokens. Can be overridden per-call in `generate()`. |

**Behaviour**

- The constructor is **synchronous** and blocks the event loop while the model weights are mapped into memory. For large models (7B+) this can take several seconds. Consider calling it before the application starts serving requests.
- A `llama_context` is **not** created at construction time — it is created lazily on the first `generate()` call and reused afterwards unless `nCtx` changes.
- Throws a JS `Error` if the file is not found, is not a valid GGUF file, or the hardware cannot satisfy the requested GPU offload.

```js
const model = new LlamaModel('/models/mistral-7b-q4_k_m.gguf');

const model = new LlamaModel('/models/llama-3-8b.gguf', {
    nGpuLayers: 32,   // partial GPU offload
    nCtx: 8192,
});
```

---

### 3.2 `model.generate(prompt, opts?)`

Runs autoregressive inference and streams token text pieces as an async generator.

```ts
generate(prompt: string, opts?: GenerateOptions): AsyncGenerator<string, void, undefined>
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `string` | required | Raw prompt string, already formatted with the model's chat template. No template is applied by the addon. |
| `opts.nPredict` | `number` | `256` | Maximum number of new tokens to generate. `0` or negative = unlimited (runs until EOS or context full). |
| `opts.temperature` | `number` | `0.8` | Sampling temperature. `0` = greedy (deterministic), higher = more creative. |
| `opts.topP` | `number` | `0.95` | Nucleus sampling: only tokens whose cumulative probability ≤ `topP` are considered. `1.0` disables top-p filtering. |
| `opts.topK` | `number` | `40` | Only the top-k most probable tokens are kept before sampling. `0` disables top-k filtering. |
| `opts.nCtx` | `number` | model default | Override the context window size for this call. If different from the current context size, the context is recreated (KV cache is lost). |
| `opts.resetContext` | `boolean` | `false` | When `true`, the KV cache is cleared before generation begins, starting a fresh conversation. When `false`, previous context (prior turns) is retained and the new prompt is appended. |
| `opts.minP` | `number` | `0` | Min-p sampling: removes tokens whose probability is below `minP * p(top)`. Typical values 0.05–0.1. `0` disables. |
| `opts.repeatPenalty` | `number` | `1.0` | Penalty applied to recently generated tokens. `> 1` discourages repetition (e.g. `1.1`). `1.0` disables. |
| `opts.repeatLastN` | `number` | `64` | How many most-recent tokens the repeat penalty considers. `0` disables the penalty entirely. |
| `opts.grammar` | `string` | — | GBNF grammar constraining output. Takes precedence over `grammarFile`. |
| `opts.grammarFile` | `string` | — | Path to a `.gbnf` file; read synchronously and used as `grammar`. |
| `opts.stop` | `string[]` | `[]` | Stop sequences. Generation halts as soon as any of these strings appears; the stop string itself is not yielded. |
| `opts.signal` | `AbortSignal` | — | Cancels this specific generation when fired. Throws `AbortError` from the generator. Unlike `model.abort()`, other concurrent generations are unaffected. |
| `opts.grammarTriggerPatterns` | `string[]` | — | Lazy-grammar regex triggers. When set (and `grammar` is also set), the grammar stays inactive until one of these patterns appears in the output, then activates for the remainder. Used for tool calls where the model writes prose first. |
| `opts.grammarTriggerTokens` | `number[]` | — | Lazy-grammar vocab-token triggers. Same semantics as `grammarTriggerPatterns` but matches specific token IDs. |
| `opts.preservedTokens` | `string[]` | — | Tokens that must be preserved as atomic units during sampling (e.g. `<tool_call>`, `<|channel|>`). Reserved for future sampler integration; currently passed through. |

**Return value**

An `AsyncGenerator<string>` that yields token text pieces (one or more characters per token, as produced by the tokenizer's piece-to-text conversion).

**Constraints**

- Concurrent `generate()` calls on the same model instance are supported — each call returns its own independent async generator. Calls queue internally and serialise on the context mutex; only one runs `llama_decode` at a time, but the JS side sees them as fully parallel.
- `generate()` runs `llama_decode` on a libuv worker thread; the JS event loop is not blocked between `yield` points.
- The prompt is passed directly to the tokenizer. It must already include all special tokens and chat markers required by the model. Use `applyChatTemplate()` or `applyChatTemplateJinja()` (§3.9) to render from chat messages.

```js
for await (const token of model.generate(prompt, { nPredict: 512, temperature: 0.7 })) {
    process.stdout.write(token);
}
```

#### Context accumulation

By default (`resetContext: false`) the KV cache grows with each call. This allows multi-turn conversations where previous exchanges are retained in context:

```js
// Turn 1
for await (const t of model.generate('[INST] What is 2+2? [/INST]')) { ... }

// Turn 2 — previous exchange is still in the KV cache
for await (const t of model.generate('[INST] And multiplied by 3? [/INST]')) { ... }
```

When the accumulated context approaches `nCtx` tokens, generation will fail with `context size exceeded`. Track token usage manually or call `generate` with `resetContext: true` to start fresh.

---

### 3.3 Cancellation

There are two cancellation mechanisms, covering different scopes.

#### `model.abort()`

```ts
abort(): void
```

Signals **every** currently running and queued generation on this model to stop at the next token boundary. Returns immediately; each generator's `for await` loop ends after the in-flight token (if any) is yielded. Safe to call even with no generations in flight (no-op).

```js
const gen = model.generate(longPrompt, { nPredict: -1 });
setTimeout(() => model.abort(), 3000);

for await (const token of gen) process.stdout.write(token);
// loop exits after ~3 seconds
```

#### Per-request `AbortSignal`

Pass `opts.signal` to `generate()` to cancel *just that call* — other concurrent generations on the same model continue running. When the signal fires the generator throws `AbortError` from its next iteration.

```js
const ac = new AbortController();
setTimeout(() => ac.abort(), 3000);

try {
    for await (const t of model.generate(prompt, { signal: ac.signal })) {
        process.stdout.write(t);
    }
} catch (err) {
    if (err.name === 'AbortError') console.log('\n[aborted]');
    else throw err;
}
```

Both mechanisms are lock-free — aborts take effect inside `llama_decode` via an atomic flag read, not by acquiring any mutex.

---

### 3.4 `model.dispose()`

```ts
dispose(): void
```

Releases all native resources: the KV cache (`llama_context`) and model weights (`llama_model`). Waits for any in-progress generation to finish before freeing (holds the context mutex).

- Calling any method on a disposed instance throws (or returns silently for `dispose()` itself).
- After `dispose()`, the JS object can be garbage-collected normally.
- Prefer `dispose()` over relying on the garbage collector, which may hold native memory longer than expected.

```js
const model = new LlamaModel('/models/model.gguf');
try {
    // ... use model ...
} finally {
    model.dispose();
}
```

---

### 3.5 Model introspection

#### `model.contextLength`

```ts
readonly contextLength: number
```

Number of token slots in the current `llama_context`. Returns `0` before the first `generate()` call (context is created lazily). Reflects the configured `nCtx` value, not the tokens currently consumed.

```js
console.log(model.contextLength); // 0 before first generate()
for await (const t of model.generate('Hello', { nCtx: 4096 })) {}
console.log(model.contextLength); // 4096
```

#### `model.chatTemplate`

```ts
readonly chatTemplate: string | null
```

The Jinja2 chat template embedded in the GGUF metadata, or `null` if the file has no template. Note that some older GGUFs store a legacy template *alias* (e.g. the literal string `mistral-v7-tekken`) here rather than real Jinja source — see §3.8 for how `applyChatTemplateJinja` handles that case.

```js
if (model.chatTemplate) {
    console.log('Template length:', model.chatTemplate.length);
}
```

#### `model.getModelInfo()`

```ts
getModelInfo(): ModelInfo
```

Returns a snapshot of model metadata: description, parameter count, disk size, trained context length, layer count, vocabulary size, and the special-token IDs (BOS/EOS/EOT).

```ts
interface ModelInfo {
    description: string;         // e.g. "llama 8B Q4_0"
    nParams: number;
    modelSize: number;           // bytes on disk
    trainContextLength: number;
    embeddingSize: number;
    nLayer: number;
    vocabSize: number;
    specialTokens: { bos: number; eos: number; eot: number };
}
```

```js
const info = model.getModelInfo();
console.log(`${info.description}, ${(info.nParams / 1e9).toFixed(1)}B params`);
```

---

### 3.6 Tokenization

Cheap helpers that route directly to llama.cpp's tokenizer — useful for counting prompt tokens before generation, or for inspecting how the model segments text.

#### `model.tokenize(text, opts?)`

```ts
tokenize(text: string, opts?: TokenizeOptions): number[]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `addSpecial` | `boolean` | `true` | Prepend BOS / append EOS when the tokenizer requests them. |
| `parseSpecial` | `boolean` | `false` | Parse special-token syntax (e.g. `<\|im_start\|>`) instead of treating it as literal text. |

```js
const ids = model.tokenize('Hello, world!');
console.log(ids.length, 'tokens');
```

#### `model.detokenize(tokens, opts?)`

```ts
detokenize(tokens: number[], opts?: DetokenizeOptions): string
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `removeSpecial` | `boolean` | `false` | Skip special tokens in the output. |
| `unparseSpecial` | `boolean` | `false` | Render special tokens as their textual form instead of the bytes they represent. |

```js
const text = model.detokenize(ids);
```

---

### 3.7 `model.applyChatTemplate(messages, opts?)` (legacy)

Wraps llama.cpp's built-in template renderer (`llama_chat_apply_template`). This path does **not** use a Jinja parser — it matches a hardcoded list of known template shapes by name/header and only supports plain `{ role, content }` messages. Use it when:

- You need maximum compatibility with older GGUFs that store template aliases.
- You don't need tools / JSON schema / reasoning / template kwargs. For any of those, use `applyChatTemplateJinja` (§3.8).

```ts
applyChatTemplate(messages: ChatMessage[], opts?: { addAssistant?: boolean }): string
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `addAssistant` | `boolean` | `true` | Append the assistant-turn prefix so the model can continue from there. |

Throws when the embedded template can't be resolved by the legacy path.

```js
const prompt = model.applyChatTemplate([
    { role: 'system', content: 'You are helpful.' },
    { role: 'user',   content: 'Hello!' }
]);

for await (const t of model.generate(prompt)) process.stdout.write(t);
```

---

### 3.8 `model.applyChatTemplateJinja(messages, opts?)`

Renders chat messages through libcommon's Jinja template engine, returning the prompt *plus* any auto-generated grammar, lazy-grammar triggers, parser blob, and stop sequences the template family requires. Use this instead of `applyChatTemplate` when you want tool-calling, `json_schema` constraints, or reasoning toggles.

```ts
applyChatTemplateJinja(
    messages: Array<ChatMessage | object>,
    opts?: ApplyChatTemplateJinjaOptions
): ChatTemplateJinjaResult
```

**Options**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `tools` | `ChatTemplateTool[]` | — | OpenAI-format tool definitions (`{ type: 'function', function: { name, description, parameters } }`; the flat form `{ name, description, parameters }` is also accepted). |
| `toolChoice` | `'auto' \| 'required' \| 'none'` | `'auto'` | How strongly the template should push the model toward a tool call. |
| `parallelToolCalls` | `boolean` | `false` | Allow multiple tool calls in one turn. |
| `addGenerationPrompt` | `boolean` | `true` | Append the assistant-turn prefix. |
| `enableThinking` | `boolean` | `true` | Enable `<think>`/reasoning blocks when supported. |
| `grammar` | `string` | — | Raw GBNF grammar used when `tools` and `jsonSchema` are absent. Ignored when either is present (libcommon replaces it with the auto-generated tool/schema grammar). |
| `jsonSchema` | `string \| object` | — | JSON schema constraining free-form output. |
| `chatTemplateKwargs` | `Record<string, string>` | — | Arbitrary Jinja variables as JSON strings (e.g. `{ enable_thinking: 'true' }`). |
| `chatTemplateOverride` | `string` | — | Full Jinja template source used in place of the model's embedded template. Useful when the GGUF stores only a legacy alias (`mistral-v7-tekken`, etc.) — paste the template from the model's HuggingFace `tokenizer_config.json`. |

**Return value**

```ts
interface ChatTemplateJinjaResult {
    prompt: string;                    // rendered prompt, feed to generate()
    format: string;                    // 'Content-only' | 'peg-simple' | 'peg-native' | 'peg-gemma4' | 'legacy'
    parser?: string;                   // opaque PEG arena blob; pass to parseChatResponse()
    generationPrompt?: string;
    grammar?: string;                  // auto-generated; undefined when the format uses PEG-only parsing
    grammarLazy?: boolean;
    grammarTriggerPatterns?: string[];
    grammarTriggerTokens?: number[];
    preservedTokens?: string[];
    additionalStops?: string[];        // merge into generate()'s `stop` array
}
```

**Auto-fallback for alias templates.** Some GGUFs store a legacy template *name* (e.g. the literal string `mistral-v7-tekken`) instead of Jinja source. `applyChatTemplateJinja` detects this and falls back to `llama_chat_apply_template()` (the legacy C API that resolves aliases), returning `{ prompt, format: 'legacy' }`. If `tools` or `jsonSchema` were supplied, it throws — the legacy path can't honour them. Use `chatTemplateOverride` to supply full Jinja source when you need tools with such a model.

```js
// Content-only usage
const { prompt } = model.applyChatTemplateJinja([
    { role: 'system', content: 'You are a terse assistant.' },
    { role: 'user',   content: 'Name three primary colors.' }
]);
for await (const t of model.generate(prompt, { nPredict: 64 })) process.stdout.write(t);

// With tools — see §3.15 for the full end-to-end flow
const rendered = model.applyChatTemplateJinja(messages, {
    tools: [weatherTool], toolChoice: 'auto'
});
```

---

### 3.9 `model.parseChatResponse(text, opts)`

Parses a raw model response back into a structured message. Uses the `format` (and `parser`) tags from `applyChatTemplateJinja` to dispatch to the right per-format parser.

```ts
parseChatResponse(text: string, opts: ParseChatResponseOptions): ParsedChatMessage
```

**Options**

| Option | Type | Description |
|--------|------|-------------|
| `format` | `string` | Required. One of the values emitted by `applyChatTemplateJinja.format`. |
| `parser` | `string` | Opaque PEG blob from `applyChatTemplateJinja.parser`. Pass through unchanged. |
| `generationPrompt` | `string` | Pass through `applyChatTemplateJinja.generationPrompt` when present. |
| `parseToolCalls` | `boolean` | Extract tool calls. Default `true`. |
| `isPartial` | `boolean` | Set when `text` is a streaming partial; enables best-effort recovery. |

**Return value**

```ts
interface ParsedChatMessage {
    content: string;
    reasoningContent: string;
    toolCalls: Array<{ name: string; arguments: string; id: string }>;
    toolName?: string;       // only on tool-result messages
    toolCallId?: string;
}
```

`arguments` is a JSON string — parse with `JSON.parse` when you need the actual argument object. For `format === 'legacy'` the method returns the text verbatim as `content` (no parser available).

---

### 3.10 `model.chat(opts)`

One-shot helper that bundles `applyChatTemplateJinja` → `generate` → `parseChatResponse`. Buffers the full output internally and returns a structured result — it does **not** stream. For streaming, use `generate()` directly (and call `parseChatResponse` on the collected text). See §3.15 for both flows side-by-side.

```ts
chat(opts: ChatOptions): Promise<ChatResult>
```

**Options**

Takes everything from `GenerateOptions` (except `grammar` / `grammarTriggerPatterns` / `grammarTriggerTokens` / `preservedTokens` — those are derived from the template) plus:

| Option | Type | Description |
|--------|------|-------------|
| `messages` | `Array<ChatMessage>` | Required. Chat history. |
| `tools` | `ChatTemplateTool[]` | OpenAI-format tool definitions. |
| `toolChoice`, `parallelToolCalls`, `enableThinking`, `jsonSchema`, `chatTemplateKwargs`, `chatTemplateOverride` | — | Same as `applyChatTemplateJinja`. |
| `stop` | `string[]` | Merged with the template's `additionalStops`. |
| `signal` | `AbortSignal` | Forwarded to `generate()`. |

**Return value**

```ts
interface ChatResult {
    content: string;
    reasoningContent: string;
    toolCalls: Array<{ name: string; arguments: string; id: string }>;
    format: string;
    raw: string;              // full generated text, before parsing
}
```

```js
const result = await model.chat({
    messages: [{ role: 'user', content: "What's the weather in Paris?" }],
    tools: [weatherTool],
    nPredict: 200,
});

if (result.toolCalls.length) {
    for (const call of result.toolCalls) {
        const args = JSON.parse(call.arguments);
        const output = await runTool(call.name, args);
        // feed `output` back into a follow-up chat() call as a tool-role message
    }
} else {
    console.log(result.content);
}
```

---

### 3.11 `LlamaModelPool`

Registry that lazy-loads named models. Useful when you want to declare a set of available models up-front but only pay the load cost on first use. Not a worker pool — a single pool does **not** run concurrent models in parallel native threads; concurrency still happens within each `LlamaModel` instance (see §3.2).

```ts
class LlamaModelPool {
    register(name: string, modelPath: string, opts?: LlamaModelOptions): void;
    load(name: string): LlamaModel;
    generate(name: string, prompt: string, opts?: GenerateOptions): AsyncGenerator<string>;
    unload(name: string): void;
    dispose(): void;
    readonly chatTemplate: string | null;    // from the first loaded model, or null
    [Symbol.dispose](): void;                // same as dispose()
}
```

- `register(name, path, opts)` — record a model under `name`. Throws if `name` is already registered. Does **not** load the file yet.
- `load(name)` — return the live `LlamaModel` for `name`, constructing it on first call.
- `generate(name, prompt, opts)` — shorthand for `pool.load(name).generate(prompt, opts)`.
- `unload(name)` — dispose the underlying `LlamaModel` and **remove** its registration from the pool. Unlike some pool implementations, the name is not retained for auto-reload; re-register it explicitly if you want it back.
- `dispose()` — dispose every loaded model and clear the registry.

```js
const { LlamaModelPool } = require('llama-node');

const pool = new LlamaModelPool();
pool.register('chat',  '/models/llama-3.1-8b.gguf',  { nCtx: 4096 });
pool.register('coder', '/models/qwen2.5-coder.gguf', { nCtx: 8192 });

for await (const t of pool.generate('coder', 'function fizzBuzz() {')) {
    process.stdout.write(t);
}

pool.dispose();
```

---

### 3.12 `quantize` / `quantizeFtypes`

Module-level function that converts a GGUF file from one quantization to another, wrapping `llama_model_quantize`. **GGUF → GGUF only** — converting from Safetensors / HF format requires llama.cpp's `convert_hf_to_gguf.py` (a Python tool with ~1.5 GB of deps), which this package deliberately does **not** bundle.

```ts
function quantize(
    inputPath: string,
    outputPath: string,
    opts: QuantizeOptions
): Promise<void>;

function quantizeFtypes(): QuantizeFtype[];
```

**QuantizeOptions**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `ftype` | `QuantizeFtype \| number` | required | Target format. Accepts a name (`'Q4_K_M'`, `'Q5_0'`, …) or the raw enum value. Use `quantizeFtypes()` for the full name list. |
| `nthread` | `number` | `hw concurrency` | Number of threads used for the conversion. |
| `allowRequantize` | `boolean` | `false` | Permit re-quantizing tensors that are not f32/f16 (e.g. re-quantizing an already-quantized file). |
| `quantizeOutputTensor` | `boolean` | `true` | Also quantize `output.weight`. |
| `onlyCopy` | `boolean` | `false` | Skip quantization — copy tensors verbatim. Useful for shard repacking. |
| `pure` | `boolean` | `false` | Use the default ftype for every tensor (no per-tensor overrides). |
| `keepSplit` | `boolean` | `false` | Preserve the input's shard count. |
| `dryRun` | `boolean` | `false` | Compute + report final size without writing the output. |

Runs on a libuv worker thread; the returned Promise resolves once the output has been written.

```js
const { quantize, quantizeFtypes } = require('llama-node');

console.log('Supported ftypes:', quantizeFtypes().join(', '));

await quantize(
    '/models/llama-3-8b-f16.gguf',
    '/models/llama-3-8b-q4_k_m.gguf',
    { ftype: 'Q4_K_M', nthread: 8 }
);
```

---

### 3.13 Explicit resource management (`using`)

Both `LlamaModel` and `LlamaModelPool` implement the TC39 [Explicit Resource Management](https://github.com/tc39/proposal-explicit-resource-management) protocol via `Symbol.dispose`. In environments that support the `using` keyword (TypeScript 5.2+, Node.js 22+ with `--experimental-vm-modules` or transpilation):

```ts
{
    using model = new LlamaModel('/models/model.gguf');
    for await (const token of model.generate(prompt)) {
        process.stdout.write(token);
    }
} // model.dispose() runs automatically here
```

```ts
{
    using pool = new LlamaModelPool();
    pool.register('chat', '/models/model.gguf');
    for await (const t of pool.generate('chat', prompt)) process.stdout.write(t);
} // pool.dispose() runs automatically here
```

---

### 3.14 Error handling

All errors are thrown as standard JS `Error` objects.

| Scenario | Error name / message |
|----------|---------------------|
| Model file not found or invalid | `"Failed to load model: <path>"` |
| Method called after `dispose()` | `"LlamaModel has been disposed"` |
| `generate()` cancelled via `opts.signal` | `AbortError: Aborted` |
| Prompt too long for context | `"context size exceeded"` |
| Context creation failed | `"Failed to create llama_context"` |
| Internal decode error | `"llama_decode failed, ret=<N>"` |
| `applyChatTemplate` on unsupported template | `"Failed to apply chat template — template may be unsupported"` |
| `applyChatTemplateJinja` with tools on alias template | `"applyChatTemplateJinja: model's embedded template is a legacy alias; tools / jsonSchema cannot be honoured on this path. ..."` |
| Invalid `format` passed to `parseChatResponse` | `"Unknown chat format name: <name>"` |
| Pool: unknown name | `"LlamaModelPool: unknown model '<name>'"` |
| Pool: duplicate registration | `"LlamaModelPool: '<name>' is already registered"` |
| `quantize` missing or invalid `ftype` | `TypeError: "quantize: opts.ftype is required (string name or enum value)"` |

Errors from the worker thread (tokenization, decode, quantization) surface as a rejection of the async generator's next call (for `generate`) or as a Promise rejection (for `quantize`). Wrap `for await` in try/catch to intercept them.

```js
try {
    for await (const token of model.generate(prompt)) {
        process.stdout.write(token);
    }
} catch (err) {
    if (err.name === 'AbortError') console.log('[aborted]');
    else console.error('Generation failed:', err.message);
}
```

---

### 3.15 Full usage examples

#### Simple completion

```js
const { LlamaModel } = require('llama-node');

const model = new LlamaModel('/models/llama-3-8b-q4_k_m.gguf', {
    nGpuLayers: 99,
    nCtx: 4096,
});

const prompt = model.applyChatTemplateJinja([
    { role: 'user', content: 'Explain async generators in JavaScript.' }
]).prompt;

process.stdout.write('Response: ');
for await (const token of model.generate(prompt, { nPredict: 512, temperature: 0.7 })) {
    process.stdout.write(token);
}
process.stdout.write('\n');

model.dispose();
```

#### Collecting output as a string

```js
async function complete(model, prompt, opts = {}) {
    const parts = [];
    for await (const token of model.generate(prompt, opts)) {
        parts.push(token);
    }
    return parts.join('');
}

const reply = await complete(model, prompt, { nPredict: 256 });
```

#### Streaming to an HTTP response (Node.js)

```js
const http = require('http');
const { LlamaModel } = require('llama-node');

const model = new LlamaModel('/models/model.gguf');

http.createServer(async (req, res) => {
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('Transfer-Encoding', 'chunked');

    const prompt = decodeURIComponent(new URL(req.url, 'http://x').searchParams.get('q') ?? '');
    try {
        for await (const token of model.generate(prompt, { nPredict: 256 })) {
            res.write(token);
        }
    } catch (err) {
        res.write(`\n[Error: ${err.message}]`);
    }
    res.end();
}).listen(3000);
```

#### Multi-turn conversation

```js
const messages = [
    { role: 'system', content: 'You are a concise assistant.' }
];

async function turn(userMessage) {
    messages.push({ role: 'user', content: userMessage });
    const result = await model.chat({ messages, nPredict: 512 });
    messages.push({ role: 'assistant', content: result.content });
    return result.content;
}

console.log(await turn('Hello!'));
console.log(await turn('What did I just say?'));
```

#### Abort with a timeout (per-request)

```js
async function generateWithTimeout(model, prompt, ms) {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), ms);
    const parts = [];
    try {
        for await (const token of model.generate(prompt, { nPredict: -1, signal: ac.signal })) {
            parts.push(token);
        }
    } catch (err) {
        if (err.name !== 'AbortError') throw err;
    } finally {
        clearTimeout(timer);
    }
    return parts.join('');
}
```

#### Tool calling: end-to-end

Full round-trip render → generate → parse → execute → follow-up. Works with any tool-trained model whose GGUF has a Jinja tools template (Llama 3.1+, Qwen 2.5/3, Hermes, Functionary, Mistral Nemo Instruct, etc.).

```js
const { LlamaModel } = require('llama-node');

const model = new LlamaModel('/models/llama-3.1-8b-instruct-q4_k_m.gguf', {
    nGpuLayers: 99,
    nCtx: 4096,
});

const weatherTool = {
    type: 'function',
    function: {
        name: 'get_weather',
        description: 'Get the current weather for a city.',
        parameters: {
            type: 'object',
            properties: {
                city: { type: 'string' },
                units: { type: 'string', enum: ['c', 'f'], default: 'c' }
            },
            required: ['city'],
        }
    }
};

async function runTool(name, args) {
    if (name === 'get_weather') {
        // ...your real implementation here...
        return { temperature: 14, conditions: 'cloudy', units: args.units ?? 'c' };
    }
    throw new Error(`Unknown tool: ${name}`);
}

async function converse(messages) {
    // 1. First turn — the model may respond with content or a tool call.
    const first = await model.chat({
        messages,
        tools: [weatherTool],
        nPredict: 512,
        temperature: 0.3,
    });

    // Plain content response — we're done.
    if (first.toolCalls.length === 0) {
        return first.content;
    }

    // 2. Execute each tool call and append results to the conversation.
    const followUp = [...messages];
    followUp.push({
        role: 'assistant',
        content: first.content,
        tool_calls: first.toolCalls.map(c => ({
            id: c.id || `call_${Math.random().toString(36).slice(2, 10)}`,
            type: 'function',
            function: { name: c.name, arguments: c.arguments },
        })),
    });
    for (const call of first.toolCalls) {
        const args = JSON.parse(call.arguments);
        const output = await runTool(call.name, args);
        followUp.push({
            role: 'tool',
            tool_call_id: call.id,
            content: JSON.stringify(output),
        });
    }

    // 3. Second turn — model synthesises a natural-language reply from the tool results.
    const final = await model.chat({
        messages: followUp,
        tools: [weatherTool],
        nPredict: 512,
        temperature: 0.3,
    });
    return final.content;
}

const answer = await converse([
    { role: 'user', content: "What's the weather like in Paris today?" }
]);
console.log(answer);
model.dispose();
```

**Streaming variant.** `chat()` buffers the full output. If you need to stream tokens to a UI while the model is thinking, drop to the raw primitives:

```js
const rendered = model.applyChatTemplateJinja(messages, { tools });

let raw = '';
for await (const tok of model.generate(rendered.prompt, {
    grammar:                rendered.grammar,
    grammarTriggerPatterns: rendered.grammarTriggerPatterns,
    grammarTriggerTokens:   rendered.grammarTriggerTokens,
    preservedTokens:        rendered.preservedTokens,
    stop:                   rendered.additionalStops,
    nPredict:               512,
})) {
    process.stdout.write(tok);     // stream to UI
    raw += tok;
}

const parsed = model.parseChatResponse(raw, {
    format:           rendered.format,
    parser:           rendered.parser,
    generationPrompt: rendered.generationPrompt,
});
// parsed.toolCalls, parsed.content, parsed.reasoningContent
```

**Which models emit an auto-grammar.** libcommon tags each format with one of: `Content-only`, `peg-simple`, `peg-native`, `peg-gemma4`, or `legacy` (our fallback). Hermes/Qwen-style formats return a full `grammar` + `grammarTriggerPatterns` that constrains tool-call output at sample time. Llama 3.x and Gemma-style formats use the PEG parser post-hoc — `grammar` is undefined but `parseChatResponse` still recovers the tool calls from free-form output. Either way, the flow above works; you just don't need to forward grammar fields that are absent.

---
