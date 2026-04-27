'use strict';

const fs   = require('fs');
const path = require('path');

const PLATFORMS = {
    'darwin-arm64': '@llama-cpp-node-api/darwin-arm64',
    'darwin-x64':   '@llama-cpp-node-api/darwin-x64',
    'linux-x64':    '@llama-cpp-node-api/linux-x64',
    'win32-x64':    '@llama-cpp-node-api/win32-x64',
};

function loadAddon() {
    const errors = [];
    const key = `${process.platform}-${process.arch}`;
    const pkg = PLATFORMS[key];
    if (pkg) {
        try { return require(pkg); }
        catch (e) { errors.push(`  ${pkg}: ${e.message}`); }
    }
    try {
        return require(path.resolve(__dirname, '..', 'build', 'Release', 'llama_node.node'));
    } catch (e) {
        errors.push(`  build/Release/llama_node.node: ${e.message}`);
    }

    const tried = errors.length ? '\nTried:\n' + errors.join('\n') : '';
    throw new Error(
        `llama-cpp-node-api: no working binary for ${key}.${tried}\n` +
        'Supported platforms: darwin-arm64, darwin-x64, linux-x64, win32-x64.\n' +
        'To build from source, install from git instead:\n' +
        '  npm install git+https://github.com/sir-ingvarr/llama-cpp-node-api.git'
    );
}

const addon = loadAddon();

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Push/pull channel bridging the native callback producer to an async
 * iterator. The pull side returns a fresh Promise per next() so a producer
 * error gets delivered as a rejection on the pending next(), not swallowed.
 */
function createChannel() {
    const queue = [];
    let resolveNext = null;
    let rejectNext  = null;
    let closed   = false;
    let closeErr = null;

    function deliver() {
        if (!resolveNext) return;
        if (queue.length) {
            const r = resolveNext;
            resolveNext = null;
            rejectNext  = null;
            r({ value: queue.shift(), done: false });
        } else if (closed) {
            const r  = resolveNext;
            const rj = rejectNext;
            resolveNext = null;
            rejectNext  = null;
            if (closeErr) rj(closeErr);
            else          r({ value: undefined, done: true });
        }
    }

    return {
        push(value) { queue.push(value); deliver(); },
        close(err)  {
            if (closed) return;
            closed   = true;
            closeErr = err ?? null;
            deliver();
        },
        get isClosed() { return closed; },
        [Symbol.asyncIterator]() {
            return {
                next: () => new Promise((resolve, reject) => {
                    resolveNext = resolve;
                    rejectNext  = reject;
                    deliver();
                })
            };
        }
    };
}

/**
 * If opts.grammarFile is set, reads it asynchronously and injects its
 * contents as opts.grammar.
 */
async function resolveGrammar(opts) {
    if (!opts.grammarFile) return opts;
    const { grammarFile, ...rest } = opts;
    return { ...rest, grammar: await fs.promises.readFile(grammarFile, 'utf8') };
}

function makeAbortError(message = 'Aborted') {
    const err = new Error(message);
    err.name = 'AbortError';
    return err;
}

// Mirror llama.h's enum llama_pooling_type. JS surface accepts strings
// (`'mean'`, `'cls'`, …); the addon takes the integer enum value.
const POOLING_TYPES = {
    unspecified: -1,
    none:         0,
    mean:         1,
    cls:          2,
    last:         3,
    rank:         4,
};

// Mirror llama.h's enum llama_flash_attn_type.
const FLASH_ATTN_TYPES = {
    auto: -1,
    off:  0,
    on:   1,
};

// Mirror ggml.h's enum ggml_type values for the KV-cache-eligible subset
// (matches llama.cpp's `kv_cache_types` allowlist in common/arg.cpp).
const CACHE_TYPES = {
    f32:     0,
    f16:     1,
    q4_0:    2,
    q4_1:    3,
    q5_0:    6,
    q5_1:    7,
    q8_0:    8,
    iq4_nl: 20,
    bf16:   30,
};

function resolveCacheType(name, field) {
    if (typeof name !== 'string') return name;
    const code = CACHE_TYPES[name.toLowerCase()];
    if (code === undefined) {
        throw new TypeError(
            `LlamaModel: unknown ${field} '${name}'. ` +
            `Valid: ${Object.keys(CACHE_TYPES).join(', ')}`);
    }
    return code;
}

function normalizeModelOpts(opts) {
    if (!opts) return opts;
    let out = opts;
    if (typeof opts.poolingType === 'string') {
        const code = POOLING_TYPES[opts.poolingType.toLowerCase()];
        if (code === undefined) {
            throw new TypeError(
                `LlamaModel: unknown poolingType '${opts.poolingType}'. ` +
                `Valid: ${Object.keys(POOLING_TYPES).join(', ')}`);
        }
        out = { ...out, poolingType: code };
    }
    if (typeof opts.cacheTypeK === 'string') {
        out = { ...out, cacheTypeK: resolveCacheType(opts.cacheTypeK, 'cacheTypeK') };
    }
    if (typeof opts.cacheTypeV === 'string') {
        out = { ...out, cacheTypeV: resolveCacheType(opts.cacheTypeV, 'cacheTypeV') };
    }
    if (typeof opts.flashAttention === 'string') {
        const code = FLASH_ATTN_TYPES[opts.flashAttention.toLowerCase()];
        if (code === undefined) {
            throw new TypeError(
                `LlamaModel: unknown flashAttention '${opts.flashAttention}'. ` +
                `Valid: ${Object.keys(FLASH_ATTN_TYPES).join(', ')}`);
        }
        out = { ...out, flashAttention: code };
    }
    return out;
}

// ---------------------------------------------------------------------------
// LlamaModel
// ---------------------------------------------------------------------------

class LlamaModel {
    #native;
    #disposed = false;

    /**
     * @param {string} modelPath  Absolute path to the .gguf model file.
     * @param {import('./index').LlamaModelOptions} [opts]
     */
    constructor(modelPath, opts = {}, _handle = undefined) {
        if (typeof modelPath !== 'string' || !modelPath) {
            throw new TypeError('LlamaModel: modelPath must be a non-empty string');
        }
        const nopts = normalizeModelOpts(opts);
        // _handle is the External returned by addon.loadModel — used by
        // the static LlamaModel.load() factory to skip the sync load path.
        this.#native = _handle === undefined
            ? new addon.LlamaModel(modelPath, nopts)
            : new addon.LlamaModel(modelPath, nopts, _handle);
    }

    /**
     * Async constructor: loads the model on a libuv worker thread so the JS
     * event loop is not blocked while weights are mmap'd / GPU-uploaded.
     * Equivalent to `new LlamaModel(path, opts)` once it resolves.
     *
     * @param {string} modelPath
     * @param {import('./index').LlamaModelOptions} [opts]
     * @returns {Promise<LlamaModel>}
     */
    static load(modelPath, opts = {}) {
        if (typeof modelPath !== 'string' || !modelPath) {
            return Promise.reject(
                new TypeError('LlamaModel.load: modelPath must be a non-empty string'));
        }
        let nopts;
        try { nopts = normalizeModelOpts(opts); }
        catch (e) { return Promise.reject(e); }
        return new Promise((resolve, reject) => {
            addon.loadModel(modelPath, nopts, (err, handle) => {
                if (err) return reject(err);
                try {
                    resolve(new LlamaModel(modelPath, opts, handle));
                } catch (e) { reject(e); }
            });
        });
    }

    /**
     * Yields token strings as they are produced.
     *
     * Concurrent calls are allowed: each call is queued behind any running or
     * queued generations. Pass `opts.signal` (AbortSignal) to cancel an
     * individual request; calling `model.abort()` cancels *all* tracked
     * requests (current + queued).
     *
     * When `opts.signal` fires the generator throws — the original
     * `signal.reason` if set (Web standard), otherwise an `AbortError`.
     *
     * If the consumer breaks/throws out of `for await (...)` early, the
     * underlying native request is automatically cancelled so the worker
     * thread doesn't keep decoding into a dead channel.
     *
     * @param {string} prompt
     * @param {import('./index').GenerateOptions} [opts]
     * @yields {string}
     */
    async * generate(prompt, opts = {}) {
        if (this.#disposed) {
            throw new Error('LlamaModel: instance has been disposed');
        }
        const { signal, ...rest } = opts;
        if (signal?.aborted) {
            throw signal.reason ?? makeAbortError();
        }

        const ch = createChannel();
        const nativeOpts = await resolveGrammar(rest);

        // When logprobs are requested, the native callback fires with three
        // args (text, logprob, topLogprobs[]); JS bundles them into an object
        // so the consumer's iteration shape is uniform.
        const wantLogprobs = nativeOpts.logprobs === true ||
                             (typeof nativeOpts.topLogprobs === 'number' &&
                              nativeOpts.topLogprobs > 0);
        const onToken = wantLogprobs
            ? (text, logprob, topLogprobs) =>
                ch.push({ text, logprob, topLogprobs: topLogprobs ?? [] })
            : (text) => ch.push(text);

        const reqId = this.#native.generate(
            prompt,
            nativeOpts,
            onToken,
            err  => ch.close(err)
        );

        let aborted = false;
        let onAbort = null;
        if (signal) {
            onAbort = () => {
                aborted = true;
                try { this.#native.abortRequest(reqId); } catch (_) {}
            };
            signal.addEventListener('abort', onAbort, { once: true });
        }

        try {
            try {
                for await (const token of ch) yield token;
            } catch (err) {
                // Channel rejected (native error) OR consumer body threw.
                // Promote to AbortError only when we know the abort caused the
                // channel close (ch.isClosed) — otherwise let the consumer's
                // own exception propagate unchanged.
                if (aborted && ch.isClosed) {
                    throw signal.reason ?? makeAbortError();
                }
                throw err;
            }
            // Natural channel completion. Native exits cleanly on abort too
            // (ret==2 from llama_decode → close(null)), so we still need to
            // surface the abort intent here.
            if (aborted) throw signal.reason ?? makeAbortError();
        } finally {
            if (onAbort) signal.removeEventListener('abort', onAbort);
            // Consumer broke out early (or threw) before the channel closed:
            // stop the native worker instead of letting it decode into a void.
            if (!ch.isClosed) {
                try { this.#native.abortRequest(reqId); } catch (_) {}
            }
        }
    }

    /** Cancel every currently running and queued generation on this model. */
    abort() {
        if (this.#disposed) return;
        this.#native.abort();
    }

    /** Release model weights and KV cache. Idempotent. */
    dispose() {
        if (this.#disposed) return;
        this.#disposed = true;
        this.#native.dispose();
    }

    /** Number of token slots in the current context window. */
    get contextLength() {
        if (this.#disposed) return 0;
        return this.#native.contextLength;
    }

    /** Jinja2 chat template embedded in the model metadata, or null. */
    get chatTemplate() {
        if (this.#disposed) return null;
        return this.#native.chatTemplate;
    }

    #assertLive() {
        if (this.#disposed) {
            throw new Error('LlamaModel: instance has been disposed');
        }
    }

    /**
     * Format messages using the model's built-in chat template.
     *
     * @param {Array<{role: string, content: string}>} messages
     * @param {{ addAssistant?: boolean }} [opts]
     * @returns {string}
     */
    applyChatTemplate(messages, opts = {}) {
        this.#assertLive();
        return this.#native.applyChatTemplate(messages, opts);
    }

    /**
     * Jinja-based chat template renderer (libcommon). Supports tools,
     * json_schema, and reasoning toggles, and returns the auto-generated
     * GBNF grammar / stop sequences that constrain the model's output.
     *
     * Pass the returned `prompt` to `generate()`; if `grammar` is present,
     * forward `grammar`, `grammarTriggerPatterns`, `grammarTriggerTokens`,
     * `preservedTokens`, and merge `additionalStops` with your own stops.
     * Use `format` with `common_chat_parse` (not exposed yet) or your own
     * post-processor to recover tool calls from the generated text.
     *
     * @param {Array<object>} messages  OAI-format chat messages.
     * @param {import('./index').ApplyChatTemplateJinjaOptions} [opts]
     * @returns {import('./index').ChatTemplateJinjaResult}
     */
    applyChatTemplateJinja(messages, opts = {}) {
        this.#assertLive();
        const {
            tools,
            toolChoice,
            parallelToolCalls,
            addGenerationPrompt,
            enableThinking,
            grammar,
            jsonSchema,
            chatTemplateKwargs,
            chatTemplateOverride,
        } = opts;

        const nativeOpts = {};
        if (toolChoice           !== undefined) nativeOpts.toolChoice           = toolChoice;
        if (parallelToolCalls    !== undefined) nativeOpts.parallelToolCalls    = parallelToolCalls;
        if (addGenerationPrompt  !== undefined) nativeOpts.addGenerationPrompt  = addGenerationPrompt;
        if (enableThinking       !== undefined) nativeOpts.enableThinking       = enableThinking;
        if (grammar              !== undefined) nativeOpts.grammar              = grammar;
        if (chatTemplateKwargs   !== undefined) nativeOpts.chatTemplateKwargs   = chatTemplateKwargs;
        if (chatTemplateOverride !== undefined) nativeOpts.chatTemplateOverride = chatTemplateOverride;
        if (jsonSchema           !== undefined) {
            nativeOpts.jsonSchema =
                typeof jsonSchema === 'string' ? jsonSchema : JSON.stringify(jsonSchema);
        }

        return this.#native.applyChatTemplateJinja(
            JSON.stringify(messages),
            JSON.stringify(tools ?? []),
            nativeOpts
        );
    }

    /**
     * Parse a model response back into a structured message. Expects `format`
     * (and ideally `parser`) as returned by `applyChatTemplateJinja`.
     * Works for all libcommon formats; for `format === 'legacy'` the text
     * is returned as-is under `content`.
     *
     * @param {string} text
     * @param {import('./index').ParseChatResponseOptions} opts
     * @returns {import('./index').ParsedChatMessage}
     */
    parseChatResponse(text, opts) {
        this.#assertLive();
        return this.#native.parseChatResponse(text, opts);
    }

    /**
     * One-shot high-level chat: renders the template, runs generate with
     * any auto-emitted grammar/triggers/stops, parses the output, and
     * returns the structured message. Equivalent to
     * `applyChatTemplateJinja` → `generate` → `parseChatResponse`.
     *
     * @param {import('./index').ChatOptions} opts
     * @returns {Promise<import('./index').ChatResult>}
     */
    async chat(opts) {
        const { messages, tools, toolChoice, parallelToolCalls,
                enableThinking, jsonSchema, chatTemplateKwargs,
                chatTemplateOverride, signal, stop,
                ...generateOpts } = opts;

        const rendered = this.applyChatTemplateJinja(messages, {
            tools, toolChoice, parallelToolCalls, enableThinking,
            jsonSchema, chatTemplateKwargs, chatTemplateOverride,
            addGenerationPrompt: true,
        });

        const mergedStops = [
            ...(stop ?? []),
            ...(rendered.additionalStops ?? []),
        ];

        const genOpts = {
            ...generateOpts,
            signal,
            stop: mergedStops.length ? mergedStops : undefined,
        };
        if (rendered.grammar) {
            genOpts.grammar = rendered.grammar;
            if (rendered.grammarTriggerPatterns) genOpts.grammarTriggerPatterns = rendered.grammarTriggerPatterns;
            if (rendered.grammarTriggerTokens)   genOpts.grammarTriggerTokens   = rendered.grammarTriggerTokens;
            if (rendered.preservedTokens)        genOpts.preservedTokens        = rendered.preservedTokens;
        }

        let text = '';
        for await (const tok of this.generate(rendered.prompt, genOpts)) {
            text += tok;
        }

        const parsed = this.parseChatResponse(text, {
            format: rendered.format,
            parser: rendered.parser,
            generationPrompt: rendered.generationPrompt,
        });

        return {
            content:          parsed.content,
            reasoningContent: parsed.reasoningContent,
            toolCalls:        parsed.toolCalls,
            format:           rendered.format,
            raw:              text,
        };
    }

    /**
     * Convert text to token IDs.
     *
     * @param {string} text
     * @param {{ addSpecial?: boolean, parseSpecial?: boolean }} [opts]
     * @returns {number[]}
     */
    tokenize(text, opts = {}) {
        this.#assertLive();
        return this.#native.tokenize(text, opts);
    }

    /**
     * Convert token IDs back to text.
     *
     * @param {number[]} tokens
     * @param {{ removeSpecial?: boolean, unparseSpecial?: boolean }} [opts]
     * @returns {string}
     */
    detokenize(tokens, opts = {}) {
        this.#assertLive();
        return this.#native.detokenize(tokens, opts);
    }

    /**
     * Returns model metadata: description, parameter count, sizes, special tokens, etc.
     *
     * @returns {import('./index').ModelInfo}
     */
    getModelInfo() {
        this.#assertLive();
        return this.#native.getModelInfo();
    }

    /**
     * Compute an embedding for the given text. Requires the model to have
     * been constructed with `{ embeddings: true }`. Pooling is determined by
     * the model unless overridden at construction with `poolingType`.
     *
     * @param {string} text
     * @returns {Promise<Float32Array>}
     */
    embed(text) {
        this.#assertLive();
        if (typeof text !== 'string') {
            return Promise.reject(new TypeError('embed: text must be a string'));
        }
        return new Promise((resolve, reject) => {
            this.#native.embed(text, (err, vec) => {
                if (err) reject(err);
                else     resolve(vec);
            });
        });
    }

    [Symbol.dispose]() { this.dispose(); }
}

// ---------------------------------------------------------------------------
// LlamaModelPool
// ---------------------------------------------------------------------------

class LlamaModelPool {
    /** @type {Map<string, { modelPath: string, modelOpts: object, instance: LlamaModel|null }>} */
    #registry = new Map();
    #disposed = false;

    #assertLive() {
        if (this.#disposed) {
            throw new Error('LlamaModelPool: instance has been disposed');
        }
    }

    /**
     * Register a model under a name. Does not load it yet.
     * @param {string} name
     * @param {string} modelPath
     * @param {{ nGpuLayers?: number, nCtx?: number }} [opts]
     */
    register(name, modelPath, opts = {}) {
        this.#assertLive();
        if (this.#registry.has(name)) {
            throw new Error(`LlamaModelPool: '${name}' is already registered`);
        }
        this.#registry.set(name, { modelPath, modelOpts: opts, instance: null });
    }

    /**
     * Returns the live model for the given name, loading it if needed.
     * @param {string} name
     * @returns {LlamaModel}
     */
    load(name) {
        this.#assertLive();
        const entry = this.#registry.get(name);
        if (!entry) throw new Error(`LlamaModelPool: unknown model '${name}'`);
        entry.instance ??= new LlamaModel(entry.modelPath, entry.modelOpts);
        return entry.instance;
    }

    /**
     * Generate from a named model.
     * @param {string} name
     * @param {string} prompt
     * @param {import('./index').GenerateOptions} [opts]
     * @yields {string}
     */
    async * generate(name, prompt, opts = {}) {
        yield * this.load(name).generate(prompt, opts);
    }

    /**
     * Dispose a model and remove it from the pool. The `name` is released —
     * re-register it explicitly if you want to load it again.
     * @param {string} name
     */
    unload(name) {
        this.#assertLive();
        const entry = this.#registry.get(name);
        if (!entry) throw new Error(`LlamaModelPool: unknown model '${name}'`);
        entry.instance?.dispose();
        entry.instance = null;
        this.#registry.delete(name);
    }

    /** Dispose all loaded models and clear the registry. Idempotent. */
    dispose() {
        if (this.#disposed) return;
        this.#disposed = true;
        for (const entry of this.#registry.values()) {
            entry.instance?.dispose();
        }
        this.#registry.clear();
    }

    [Symbol.dispose]() { this.dispose(); }
}

// ---------------------------------------------------------------------------
// quantize — standalone GGUF requantization
// ---------------------------------------------------------------------------

/**
 * Convert a GGUF file from one quantization to another.
 * Runs on a libuv worker thread; returns a Promise that resolves when done.
 *
 * @param {string} inputPath    Source .gguf (typically F16 or F32).
 * @param {string} outputPath   Destination .gguf.
 * @param {import('./index').QuantizeOptions} opts
 * @returns {Promise<void>}
 */
function quantize(inputPath, outputPath, opts) {
    if (!opts || (typeof opts.ftype !== 'string' && typeof opts.ftype !== 'number')) {
        return Promise.reject(new TypeError('quantize: opts.ftype is required (string name or enum value)'));
    }
    return new Promise((resolve, reject) => {
        addon.quantize(inputPath, outputPath, opts, err => {
            if (err) reject(err);
            else     resolve();
        });
    });
}

/** List of ftype names accepted by `quantize()`. */
function quantizeFtypes() {
    return addon.quantizeFtypes();
}

// ---------------------------------------------------------------------------
// inspect — read GGUF metadata without loading tensor weights
// ---------------------------------------------------------------------------

// Default 8 entries. Override via env var LLAMA_NODE_INSPECT_CACHE_MAX
// (set to 0 to disable caching globally).
const INSPECT_CACHE_MAX = (() => {
    const raw = process.env.LLAMA_NODE_INSPECT_CACHE_MAX;
    if (raw && /^\d+$/.test(raw)) return parseInt(raw, 10);
    return 8;
})();
// LRU keyed by canonical realpath. Entries: { mtimeMs, size, promise }.
// The Map preserves insertion order; we delete + re-set on hit to bump.
const _inspectCache = new Map();

function _inspectNative(realPath) {
    return new Promise((resolve, reject) => {
        addon.inspect(realPath, (err, result) => {
            if (err) reject(err);
            else     resolve(result);
        });
    });
}

/**
 * Read GGUF header, KV metadata, and tensor descriptors without loading any
 * tensor data. Cheap: only the file header + metadata are read off disk.
 *
 * Results are memoized by canonical path, validated by file mtime + size.
 * Cache holds up to 8 entries (LRU); pass `{ cache: false }` to bypass.
 *
 * @param {string} path  Path to a .gguf file.
 * @param {{ cache?: boolean }} [opts]
 * @returns {Promise<import('./index').GgufInspectResult>}
 */
async function inspect(path, opts = {}) {
    const useCache = opts.cache !== false && INSPECT_CACHE_MAX > 0;
    const real     = await fs.promises.realpath(path);

    if (!useCache) return _inspectNative(real);

    const stat = await fs.promises.stat(real);
    const hit  = _inspectCache.get(real);
    if (hit && hit.mtimeMs === stat.mtimeMs && hit.size === stat.size) {
        // Bump to MRU position
        _inspectCache.delete(real);
        _inspectCache.set(real, hit);
        return hit.promise;
    }

    const promise = _inspectNative(real);
    // Drop on rejection so the next call can retry.
    promise.catch(() => {
        const entry = _inspectCache.get(real);
        if (entry && entry.promise === promise) _inspectCache.delete(real);
    });

    _inspectCache.set(real, { mtimeMs: stat.mtimeMs, size: stat.size, promise });
    while (_inspectCache.size > INSPECT_CACHE_MAX) {
        const oldest = _inspectCache.keys().next().value;
        _inspectCache.delete(oldest);
    }
    return promise;
}

/** Flush the inspect() memoization cache. */
function clearInspectCache() {
    _inspectCache.clear();
}

// ---------------------------------------------------------------------------

module.exports = { LlamaModel, LlamaModelPool, quantize, quantizeFtypes, inspect, clearInspectCache };
