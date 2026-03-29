'use strict';

const fs   = require('fs');
const path = require('path');

const addon = require(path.resolve(__dirname, '..', 'build', 'Release', 'llama_node.node'));

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

// ---------------------------------------------------------------------------
// LlamaModel
// ---------------------------------------------------------------------------

class LlamaModel {
    #native;
    #generating = false;

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
     * @param {string} prompt
     * @param {import('./index').GenerateOptions} [opts]
     * @yields {string}
     */
    async * generate(prompt, opts = {}) {
        if (this.#generating) {
            throw new Error('Already generating — call abort() first or await the previous generator');
        }
        this.#generating = true;

        const ch = createChannel();
        this.#native.generate(
            prompt,
            resolveGrammar(opts),
            buf => ch.push(buf.toString('utf8')),
            err => ch.close(err)
        );

        try {
            for await (const token of ch) yield token;
        } finally {
            this.#generating = false;
        }

        if (ch.error) throw ch.error;
    }

    /** Signal the running generation to stop at the next token boundary. */
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

module.exports = { LlamaModel, LlamaModelPool };
