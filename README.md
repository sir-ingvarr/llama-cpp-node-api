# llama-cpp-node-api

<p align="center">
  <img src="assets/logo.png" alt="llama-cpp-node-api logo" width="200" />
</p>

> **Warning: this is a vibe-coded library.** It was built with AI assistance, has no test suite, and has seen limited real-world use. The API may change. Use in production at your own risk.

Node.js native addon that runs [llama.cpp](https://github.com/ggml-org/llama.cpp) inference directly in the Node.js/Electron process — no child processes, no HTTP, no IPC overhead. Model weights stay loaded in memory across calls.

## Requirements

- Node.js 18+ or Electron 35+
- CMake 3.15+
- C++17 compiler (Xcode CLT on macOS, `build-essential` on Linux)
- A `.gguf` model file

## Install

```bash
git clone --recurse-submodules <repo-url>
npm install
```

The `vendor/llama.cpp` submodule is built automatically. On macOS, Metal acceleration is enabled by default. No extra flags needed.

## Usage

```js
const { LlamaModel } = require('llama-cpp-node-api');

const model = new LlamaModel('/path/to/model.gguf', { nGpuLayers: 99 });

for await (const token of model.generate(prompt, { nPredict: 256 })) {
    process.stdout.write(token);
}

model.dispose();
```

### Chat templates

Models embed a Jinja2-style chat template in their metadata. Use `applyChatTemplate()` to format messages instead of hand-crafting prompts:

```js
const formatted = model.applyChatTemplate([
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Hello!' },
]);

for await (const token of model.generate(formatted, { nPredict: 256 })) {
    process.stdout.write(token);
}
```

You can also inspect the raw template string via `model.chatTemplate`.

## API

### `new LlamaModel(modelPath, opts?)`

| Option | Default | Description |
|--------|---------|-------------|
| `nGpuLayers` | `99` | Layers to offload to GPU. `0` = CPU only. |
| `nCtx` | `2048` | Context window size in tokens. |

### `model.generate(prompt, opts?)`

Returns an `AsyncGenerator<string>` that yields token text pieces.

| Option | Default | Description |
|--------|---------|-------------|
| `nPredict` | `256` | Max tokens. `0` or negative = unlimited. |
| `temperature` | `0.8` | Sampling temperature. `0` = greedy. |
| `topP` | `0.95` | Nucleus sampling cutoff. `1` to disable. |
| `topK` | `40` | Top-k sampling. `0` to disable. |
| `minP` | `0` | Min-p sampling threshold. Typical: `0.05`. |
| `repeatPenalty` | `1.0` | Repetition penalty. `1.0` = disabled. |
| `repeatLastN` | `64` | Token window for repeat penalty. |
| `grammar` | — | GBNF grammar string to constrain output. |
| `grammarFile` | — | Path to a `.gbnf` file. |
| `stop` | `[]` | Stop sequences. Generation halts on first match; the sequence is not included in output. |
| `nCtx` | — | Override context size for this call. |
| `resetContext` | `false` | Clear KV cache before generating (start fresh). |

### `model.chatTemplate`

The Jinja2 chat template string embedded in the model's metadata, or `null` if the model doesn't include one.

### `model.applyChatTemplate(messages, opts?)`

Formats an array of `{ role, content }` messages using the model's built-in template. Returns the formatted prompt string.

| Option | Default | Description |
|--------|---------|-------------|
| `addAssistant` | `true` | Append the assistant turn prefix so the model continues naturally. |

### `model.abort()`

Signals the running generation to stop at the next token boundary.

### `model.dispose()`

Frees model weights and KV cache. Must be called when done.

### `model.contextLength`

Number of token slots in the current context window. `0` before the first `generate()` call.

### `LlamaModelPool`

Manages multiple named models, loading each lazily on first use.

```js
const { LlamaModelPool } = require('llama-cpp-node-api');

const pool = new LlamaModelPool();
pool.register('fast', '/models/phi-3-mini.gguf', { nGpuLayers: 99 });
pool.register('smart', '/models/llama-3-8b.gguf', { nGpuLayers: 99 });

for await (const token of pool.generate('fast', prompt, { nPredict: 128 })) {
    process.stdout.write(token);
}

pool.dispose();
```

## Build variants

```bash
npm run build          # default (Metal on macOS)
npm run build:cpu      # disable Metal
npm run build:cuda     # enable CUDA
npm run build:debug    # debug symbols
```

## License

MIT. The bundled `vendor/llama.cpp` is MIT licensed by its authors.
