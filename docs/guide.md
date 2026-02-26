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
   - 3.3 [`model.abort()`](#33-modelabort)
   - 3.4 [`model.dispose()`](#34-modeldispose)
   - 3.5 [`model.contextLength`](#35-modelcontextlength)
   - 3.6 [Explicit resource management (`using`)](#36-explicit-resource-management-using)
   - 3.7 [Error handling](#37-error-handling)
   - 3.8 [Full usage examples](#38-full-usage-examples)

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

**Return value**

An `AsyncGenerator<string>` that yields token text pieces (one or more characters per token, as produced by the tokenizer's piece-to-text conversion).

**Constraints**

- Only **one** generation can run at a time per model instance. Calling `generate()` while another generator is active throws `Error: Already generating — call abort() first`.
- `generate()` runs `llama_decode` on a libuv worker thread; the JS event loop is not blocked between `yield` points.
- The prompt is passed directly to the tokenizer. It must already include all special tokens and chat markers required by the model.

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

### 3.3 `model.abort()`

```ts
abort(): void
```

Signals the currently running generation to stop at the next token boundary. Returns immediately; the generator's `for await` loop will end after the in-flight token (if any) is yielded.

- Safe to call even if no generation is in progress (no-op).
- After calling `abort()`, the same model instance can be used for a new `generate()` call immediately — the `generating` flag is cleared when the generator finishes iterating.

```js
const gen = model.generate(longPrompt, { nPredict: -1 });
setTimeout(() => model.abort(), 3000);

for await (const token of gen) {
    process.stdout.write(token);
}
// Loop exits after ~3 seconds
```

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

### 3.5 `model.contextLength`

```ts
readonly contextLength: number
```

Returns the number of token slots available in the current `llama_context`. Returns `0` if no context has been created yet (before the first `generate()` call).

This reflects the configured `nCtx` value, not the number of tokens currently consumed.

```js
console.log(model.contextLength); // 0 before first generate()

for await (const t of model.generate('Hello', { nCtx: 4096 })) {}

console.log(model.contextLength); // 4096
```

---

### 3.6 Explicit resource management (`using`)

`LlamaModel` implements the TC39 [Explicit Resource Management](https://github.com/tc39/proposal-explicit-resource-management) protocol via `Symbol.dispose`. In environments that support the `using` keyword (TypeScript 5.2+, Node.js 22+ with `--experimental-vm-modules` or transpilation):

```ts
{
    using model = new LlamaModel('/models/model.gguf');
    for await (const token of model.generate(prompt)) {
        process.stdout.write(token);
    }
} // model.dispose() is called automatically here
```

---

### 3.7 Error handling

All errors are thrown as standard JS `Error` objects.

| Scenario | Error message |
|----------|---------------|
| Model file not found or invalid | `"Failed to load model: <path>"` |
| `generate()` called while generating | `"Already generating — call abort() first or await the previous generator"` |
| `generate()` called after `dispose()` | `"LlamaModel has been disposed"` |
| Prompt too long for context | `"context size exceeded"` |
| Context creation failed | `"Failed to create llama_context"` |
| Internal decode error | `"llama_decode failed, ret=<N>"` |

Errors from the worker thread (tokenization, decode) are surfaced as a rejection of the async generator's final `next()` call — i.e. they are thrown inside the `for await` loop.

```js
try {
    for await (const token of model.generate(prompt)) {
        process.stdout.write(token);
    }
} catch (err) {
    console.error('Generation failed:', err.message);
}
```

---

### 3.8 Full usage examples

#### Simple completion

```js
const { LlamaModel } = require('llama-node');

const model = new LlamaModel('/models/llama-3-8b-q4_k_m.gguf', {
    nGpuLayers: 99,
    nCtx: 4096,
});

const prompt = `<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Explain async generators in JavaScript.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
`;

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
let history = '';

async function chat(model, userMessage) {
    // Build prompt from accumulated history
    history += `[INST] ${userMessage} [/INST]`;

    const parts = [];
    for await (const token of model.generate(history, {
        nPredict: 512,
        resetContext: false,  // keep KV cache from previous turns
    })) {
        parts.push(token);
    }

    const reply = parts.join('');
    history += reply;   // append model reply for next turn
    return reply;
}

console.log(await chat(model, 'Hello!'));
console.log(await chat(model, 'What did I just say?'));
```

#### Abort with a timeout

```js
async function generateWithTimeout(model, prompt, ms) {
    const timer = setTimeout(() => model.abort(), ms);
    const parts = [];
    try {
        for await (const token of model.generate(prompt, { nPredict: -1 })) {
            parts.push(token);
        }
    } finally {
        clearTimeout(timer);
    }
    return parts.join('');
}
```
