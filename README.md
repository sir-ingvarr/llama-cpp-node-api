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

Concurrent calls on the same model are allowed — they queue internally and execute one at a time (llama.cpp's context is not thread-safe). The JS event loop is never blocked while a generation is running, so timers, I/O, and abort signals keep firing.

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
| `signal` | — | `AbortSignal` that cancels **this** call. The generator throws an `AbortError` on the next iteration. Other concurrent calls are unaffected. |

#### Cancelling generations

```js
// Cancel a single call:
const ac = new AbortController();
setTimeout(() => ac.abort(), 5_000);
try {
    for await (const t of model.generate(prompt, { signal: ac.signal })) {
        process.stdout.write(t);
    }
} catch (e) {
    if (e.name !== 'AbortError') throw e;
}

// Cancel every in-flight and queued call on this model:
model.abort();
```

### `model.chatTemplate`

The Jinja2 chat template string embedded in the model's metadata, or `null` if the model doesn't include one.

### `model.applyChatTemplate(messages, opts?)`

Formats an array of `{ role, content }` messages using the model's built-in template. Returns the formatted prompt string.

| Option | Default | Description |
|--------|---------|-------------|
| `addAssistant` | `true` | Append the assistant turn prefix so the model continues naturally. |

### `model.abort()`

Cancels every currently running and queued generation on this model. Each call stops at its next token boundary. For cancelling a single call only, use `opts.signal` on that call (see above).

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

### `quantize(inputPath, outputPath, opts)`

Requantizes a GGUF file to a different ftype. Runs on a libuv worker thread and returns a `Promise<void>` that resolves once the output has been written. Progress is printed to stderr by llama.cpp.

```js
const { quantize, quantizeFtypes } = require('llama-cpp-node-api');

console.log(quantizeFtypes());            // → ['F16', 'Q4_K_M', 'Q8_0', ...]

await quantize('/models/model-f16.gguf',
               '/models/model-q4_k_m.gguf',
               { ftype: 'Q4_K_M' });
```

| Option | Default | Description |
|--------|---------|-------------|
| `ftype` | **required** | Target quantization. Accepts a string name (e.g. `'Q4_K_M'`) or the raw `llama_ftype` enum value. Use `quantizeFtypes()` to list all accepted names. |
| `nthread` | hardware concurrency | Threads to use. |
| `allowRequantize` | `false` | Allow re-quantizing tensors that are not f32/f16. |
| `quantizeOutputTensor` | `true` | Quantize `output.weight` too. |
| `onlyCopy` | `false` | Copy tensors as-is (useful for shard repacking). |
| `pure` | `false` | Quantize every tensor to the default type (no per-tensor overrides). |
| `keepSplit` | `false` | Preserve the input's shard count. |
| `dryRun` | `false` | Compute and report final size without writing the output. |

Note: this wraps `llama_model_quantize` (C API). Converting from safetensors / Hugging Face formats to GGUF is not included — that path lives in llama.cpp's `convert_hf_to_gguf.py` and requires a Python environment with `torch`, `safetensors`, etc.

## Build variants

```bash
npm run build          # default (Metal on macOS)
npm run build:cpu      # disable Metal
npm run build:cuda     # enable CUDA
npm run build:debug    # debug symbols
```

## License

MIT. The bundled `vendor/llama.cpp` is MIT licensed by its authors.
