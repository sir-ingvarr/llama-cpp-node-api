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
    // 1. Try platform-specific npm package
    const key = `${process.platform}-${process.arch}`;
    const pkg = PLATFORMS[key];
    if (pkg) {
        try { return require(pkg); } catch (_) {}
    }

    // 2. Fallback to local build (development / unsupported platform)
    try {
        return require(path.resolve(__dirname, '..', 'build', 'Release', 'llama_node.node'));
    } catch (_) {}

    throw new Error(
        `llama-cpp-node-api: no prebuilt binary for ${key}.\n` +
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
 * Push/pull channel that bridges the native callback producer to an async
 * iterator. Keeps a micro-queue so tokens produced before the consumer calls
 * next() are never dropped.
 */
function createChannel() {
    const queue  = [];
    let waiter   = null;
    let closed   = false;
    let closeErr = null;

    function wake(item) {
        if (waiter) { const w = waiter; waiter = null; w(item); }
        else         queue.push(item);
    }

    return {
        push(value)  { wake({ value, done: false }); },
        close(err)   { closeErr = err ?? null; closed = true; wake({ done: true }); },
        get error()  { return closeErr; },
        [Symbol.asyncIterator]() {
            return {
                next() {
                    if (queue.length) return Promise.resolve(queue.shift());
                    if (closed)       return Promise.resolve({ done: true });
                    return new Promise(r => { waiter = r; });
                }
            };
        }
    };
}

/**
 * If opts.grammarFile is set, reads it and injects its contents as opts.grammar.
 */
function resolveGrammar(opts) {
    if (!opts.grammarFile) return opts;
    const { grammarFile, ...rest } = opts;
    return { ...rest, grammar: fs.readFileSync(grammarFile, 'utf8') };
}

function makeAbortError(message = 'Aborted') {
    const err = new Error(message);
    err.name = 'AbortError';
    return err;
}

// ---------------------------------------------------------------------------
// LlamaModel
// ---------------------------------------------------------------------------

class LlamaModel {
    #native;

    /**
     * @param {string} modelPath  Absolute path to the .gguf model file.
     * @param {{ nGpuLayers?: number, nCtx?: number }} [opts]
     */
    constructor(modelPath, opts = {}) {
        this.#native = new addon.LlamaModel(modelPath, opts);
    }

    /**
     * Yields token strings as they are produced.
     *
     * Concurrent calls are allowed: each call is queued behind any running or
     * queued generations. Pass `opts.signal` (AbortSignal) to cancel an
     * individual request; calling `model.abort()` cancels *all* tracked
     * requests (current + queued).
     *
     * When `opts.signal` fires, the generator throws an `AbortError`.
     *
     * @param {string} prompt
     * @param {import('./index').GenerateOptions} [opts]
     * @yields {string}
     */
    async * generate(prompt, opts = {}) {
        const ch = createChannel();
        const { signal, ...nativeOpts } = opts;

        const reqId = this.#native.generate(
            prompt,
            resolveGrammar(nativeOpts),
            buf => ch.push(buf.toString('utf8')),
            err => ch.close(err)
        );

        let onAbort = null;
        if (signal) {
            if (signal.aborted) {
                this.#native.abortRequest(reqId);
            } else {
                onAbort = () => this.#native.abortRequest(reqId);
                signal.addEventListener('abort', onAbort, { once: true });
            }
        }

        try {
            for await (const token of ch) yield token;
        } finally {
            if (onAbort) signal.removeEventListener('abort', onAbort);
        }

        if (ch.error) throw ch.error;
        if (signal?.aborted) throw makeAbortError();
    }

    /** Cancel every currently running and queued generation on this model. */
    abort() { this.#native.abort(); }

    /** Release model weights and KV cache. */
    dispose() { this.#native.dispose(); }

    /** Number of token slots in the current context window. */
    get contextLength() { return this.#native.contextLength; }

    /** Jinja2 chat template embedded in the model metadata, or null. */
    get chatTemplate() { return this.#native.chatTemplate; }

    /**
     * Format messages using the model's built-in chat template.
     *
     * @param {Array<{role: string, content: string}>} messages
     * @param {{ addAssistant?: boolean }} [opts]
     * @returns {string}
     */
    applyChatTemplate(messages, opts = {}) {
        return this.#native.applyChatTemplate(messages, opts);
    }

    /**
     * Convert text to token IDs.
     *
     * @param {string} text
     * @param {{ addSpecial?: boolean, parseSpecial?: boolean }} [opts]
     * @returns {number[]}
     */
    tokenize(text, opts = {}) {
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
        return this.#native.detokenize(tokens, opts);
    }

    /**
     * Returns model metadata: description, parameter count, sizes, special tokens, etc.
     *
     * @returns {import('./index').ModelInfo}
     */
    getModelInfo() {
        return this.#native.getModelInfo();
    }

    [Symbol.dispose]() { this.dispose(); }
}

// ---------------------------------------------------------------------------
// LlamaModelPool
// ---------------------------------------------------------------------------

class LlamaModelPool {
    /** @type {Map<string, { modelPath: string, modelOpts: object, instance: LlamaModel|null }>} */
    #registry = new Map();

    /**
     * Register a model under a name. Does not load it yet.
     * @param {string} name
     * @param {string} modelPath
     * @param {{ nGpuLayers?: number, nCtx?: number }} [opts]
     */
    register(name, modelPath, opts = {}) {
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
     * Unload a single model, freeing its native resources.
     * The registration is kept; the model reloads automatically on next use.
     * @param {string} name
     */
    unload(name) {
        const entry = this.#registry.get(name);
        if (!entry) throw new Error(`LlamaModelPool: unknown model '${name}'`);
        entry.instance?.dispose();
        entry.instance = null;
        this.#registry.delete(name);
    }

    /** Dispose all loaded models and clear the registry. */
    dispose() {
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

module.exports = { LlamaModel, LlamaModelPool, quantize, quantizeFtypes };
