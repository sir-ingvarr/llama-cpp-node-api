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

Loads the model **synchronously** on the calling thread — fine for CLI tools and one-off scripts; for anything user-facing, prefer `LlamaModel.load()`.

| Option | Default | Description |
|--------|---------|-------------|
| `nGpuLayers` | `99` | Layers to offload to GPU. `0` = CPU only. |
| `nCtx` | `2048` | Context window size in tokens. |
| `embeddings` | `false` | Open in embedding mode — enables `model.embed()`. |
| `poolingType` | model default | `'mean'`, `'cls'`, `'last'`, `'rank'`, `'none'`, or `'unspecified'`. |
| `cacheTypeK` | `'f16'` | Quantization for the K (keys) tensor of the KV cache. See below. |
| `cacheTypeV` | `'f16'` | Quantization for the V (values) tensor of the KV cache. See below. |
| `flashAttention` | `'auto'` | `'auto'` (let llama.cpp decide), `'on'` (force-enable, required for quantized V cache on most backends), `'off'` (disable). |
| `nThreads` | llama.cpp default | Threads used during single-token decode. Tune for CPU-only setups; ignored when fully offloaded to GPU. |
| `nThreadsBatch` | llama.cpp default | Threads used during batched / prompt-processing decode. Often higher than `nThreads` on big-core CPUs. |
| `useMmap` | `true` | Memory-map the model file. Set `false` to avoid cold-cache page-fault stutter at the cost of a longer load and full RAM use. |
| `useMlock` | `false` | Pin model pages in physical memory so they cannot swap. May require elevated permissions (`ulimit -l` / `CAP_IPC_LOCK`). |

#### KV cache quantization

`cacheTypeK` / `cacheTypeV` cut KV-cache memory in exchange for a small quality cost — useful for fitting a bigger `nCtx` or running more concurrent sessions on the same GPU. Roughly: `q8_0` halves KV memory vs the `f16` default, `q4_0` quarters it.

Accepted values: `'f32' | 'f16' | 'bf16' | 'q8_0' | 'q4_0' | 'q4_1' | 'iq4_nl' | 'q5_0' | 'q5_1'` (the same allowlist `llama-cli` accepts via `--cache-type-k` / `--cache-type-v`).

```js
const model = await LlamaModel.load('/models/llama-3.1-8b.gguf', {
    nCtx: 32_768,
    cacheTypeK: 'q8_0',
    cacheTypeV: 'q8_0',  // see note below
});
```

> Quantized **V** cache (anything other than `f16`/`f32`/`bf16`) typically requires Flash Attention, which the underlying backend may or may not enable automatically. If context creation fails with a Flash-Attention-related error, fall back to `cacheTypeV: 'f16'` and quantize K only.

### `LlamaModel.load(modelPath, opts?)` → `Promise<LlamaModel>`

Async constructor — loads weights on a libuv worker thread so the JS event loop is **not** blocked while the model is mmap'd and uploaded to the GPU. Same options as the sync constructor; resolves to a ready-to-use model.

```js
// Use this in any code that runs on the Node.js main thread.
const model = await LlamaModel.load('/models/llama-3.1-8b.gguf', {
    nGpuLayers: 99,
    nCtx: 4096,
});
```

A 16 GB model that takes ~1.7s to load fires ~150 setInterval(10ms) ticks during the load — the event loop stays fully responsive.

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
| `signal` | — | `AbortSignal` that cancels **this** call. The generator throws — `signal.reason` if set (Web standard), otherwise an `AbortError`. Other concurrent calls are unaffected. |
| `logprobs` | `false` | When `true`, the generator yields `{ text, logprob, topLogprobs }` objects instead of strings. |
| `topLogprobs` | `0` | Include top-K alternative tokens with logprobs per step. Setting `> 0` implies `logprobs: true`. |
| `seed` | non-deterministic | Seed for the stochastic sampler. Pin to make generation reproducible for a given prompt + sampler config. |
| `logitBias` | `{}` | Per-token logit bias `{ tokenId: bias }`, applied additively before grammar. Use `model.tokenize()` to find token IDs. |

#### Cancelling generations

```js
// 1. AbortSignal — cancel a single call:
const ac = new AbortController();
setTimeout(() => ac.abort(), 5_000);
try {
    for await (const t of model.generate(prompt, { signal: ac.signal })) {
        process.stdout.write(t);
    }
} catch (e) {
    if (e.name !== 'AbortError') throw e;  // or check `e === ac.signal.reason`
}

// 2. break / return — exiting the for-await early also stops the worker:
for await (const t of model.generate(prompt)) {
    if (gotEnough(t)) break;          // native generation is aborted
}

// 3. model.abort() — cancel every in-flight and queued call (graceful stop,
//    no AbortError thrown unless the call also had its own signal):
model.abort();
```

`for await ... break` always aborts the underlying worker — there's no risk of a leaked decoder running to completion in the background.

#### Logprobs

```js
for await (const { text, logprob, topLogprobs } of model.generate(prompt,
        { nPredict: 8, temperature: 0, logprobs: true, topLogprobs: 5 })) {
    console.log(text, '→', logprob.toFixed(3));
    for (const { token, logprob: lp } of topLogprobs) {
        console.log('   ', JSON.stringify(token), lp.toFixed(3));
    }
}
```

Logprobs are computed from the **raw model distribution** (pre-grammar / pre-penalty / pre-top-k), so they're stable regardless of which sampler stages you have enabled. Useful for classification (compare logprobs of `' yes'` vs `' no'` from the top-K), evaluation harnesses, and OpenAI-API-compatibility shims.

> Caveat: stop-sequence lookahead can flush a chunk whose text spans multiple sampled tokens. In that case the chunk's `logprob` reflects the most recent sampled token (not the entire emitted text). Avoid stop sequences when you need exact token-level alignment.

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

Frees model weights and KV cache. Idempotent — safe to call more than once. Any subsequent call to `generate()`, `tokenize()`, `applyChatTemplate*()`, etc. throws.

### `model.contextLength`

Number of token slots in the current context window. `0` before the first `generate()` call.

### `model.embed(text)` → `Promise<Float32Array>`

Computes a vector embedding for `text`. Requires the model to have been opened with `{ embeddings: true }`; throws otherwise.

```js
const m = await LlamaModel.load('/models/nomic-embed-text-v1.5.gguf', {
    embeddings: true,
    poolingType: 'mean',  // optional — defaults to whatever the model was trained with
});

const v = await m.embed('the quick brown fox');
v.length;  // n_embd, e.g. 768
v.constructor.name;  // 'Float32Array'
```

Pooling: leave `poolingType` unset to use the model's training-time pooling (typically `'mean'` for BERT-like, `'last'` for last-token-pooling encoders). Set `'none'` to get the raw last-token embedding without pooling.

A model opened with `embeddings: true` can still call `generate()` — every decode populates the embedding buffer as a side effect on generative models, so the same instance can do both. For embedding-only encoders (no LM head) only `embed()` makes sense.

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

### `inspect(path, opts?)`

Reads a GGUF file's header, KV metadata, and tensor descriptors **without loading any tensor weights**. About 6× faster than constructing a `LlamaModel` and uses a fraction of the memory — useful for cheap model identification, shape inspection, or pre-flight validation.

```js
const { inspect, clearInspectCache } = require('llama-cpp-node-api');

const r = await inspect('/models/llama-3.1-8b.gguf');
r.version;                                  // 3
r.alignment;                                // 32
r.dataOffset;                               // bigint — byte offset of tensor data
r.metadata['general.architecture'];         // 'llama'
r.metadata['llama.context_length'];         // 131072
r.metadata['tokenizer.ggml.tokens'];        // string[] (full vocab)
r.tensors[0];                               // { name, type: 'F16', offset: bigint, size: bigint }
```

Returns `Promise<GgufInspectResult>`. Runs on a libuv worker thread.

**Memoization.** Results are cached by canonical path (LRU) and invalidated when the file's `mtime` or `size` changes. Repeat calls on the same file return the same object instance in <1ms.

| Option | Default | Description |
|--------|---------|-------------|
| `cache` | `true` | Set to `false` to bypass the cache for one call. |

`clearInspectCache()` flushes the cache manually. The cache size defaults to 8 entries; override via the `LLAMA_NODE_INSPECT_CACHE_MAX` env var (set to `0` to disable caching entirely).

**Type marshaling.** GGUF metadata values are mapped to JS primitives:
- 8/16/32-bit ints → `number`
- 64-bit ints → `bigint` (avoids 2^53 precision loss)
- floats → `number`, bool → `boolean`, strings → `string`
- arrays preserve element type
- tensor `offset` and `size`, plus `dataOffset`, are always `bigint`
- tensor `type` is the ggml type name string (`"F16"`, `"Q4_K"`, `"Q8_0"`, …)

## Build variants

```bash
npm run build          # default (Metal on macOS)
npm run build:cpu      # disable Metal
npm run build:cuda     # enable CUDA
npm run build:debug    # debug symbols
```

## License

MIT. The bundled `vendor/llama.cpp` is MIT licensed by its authors.
