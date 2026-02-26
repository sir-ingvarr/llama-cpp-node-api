'use strict';

const fs   = require('fs');
const path = require('path');

// Resolve the native addon from the build output directory.
// cmake-js places it at build/Release/llama_node.node by default.
const addonPath = path.resolve(__dirname, '..', 'build', 'Release', 'llama_node.node');
const addon = require(addonPath);

/**
 * High-level wrapper around the native LlamaModel addon class.
 *
 * Turns the raw callback-based generate() into an async generator so callers
 * can use `for await (const token of model.generate(prompt))`.
 */
class LlamaModel {
    /** @type {object} */
    #native;
    #generating = false;

    /**
     * @param {string} modelPath   Absolute path to the .gguf model file.
     * @param {object} [opts]
     * @param {number} [opts.nGpuLayers=99]   GPU layers to offload.
     * @param {number} [opts.nCtx=2048]        Context size (tokens).
     */
    constructor(modelPath, opts = {}) {
        this.#native = new addon.LlamaModel(modelPath, opts);
    }

    /**
     * Generate tokens for a given prompt as an async iterable.
     *
     * @param {string} prompt
     * @param {object} [opts]
     * @param {number}   [opts.nPredict=256]
     * @param {number}   [opts.temperature=0.8]
     * @param {number}   [opts.topP=0.95]
     * @param {number}   [opts.topK=40]
     * @param {number}   [opts.minP=0]
     * @param {number}   [opts.repeatPenalty=1.0]
     * @param {number}   [opts.repeatLastN=64]
     * @param {string}   [opts.grammar]         GBNF grammar string.
     * @param {string}   [opts.grammarFile]      Path to a .gbnf file (read and passed as grammar).
     * @param {string[]} [opts.stop]             Stop sequences — generation halts when any is produced.
     * @param {number}   [opts.nCtx]
     * @param {boolean}  [opts.resetContext=false]
     * @yields {string} Token text pieces as they are produced.
     */
    async * generate(prompt, opts = {}) {
        if (this.#generating) {
            throw new Error('Already generating — call abort() first or await the previous generator');
        }
        this.#generating = true;

        // Resolve grammarFile → grammar string before entering the native layer.
        let nativeOpts = opts;
        if (opts.grammarFile) {
            const grammarStr = fs.readFileSync(opts.grammarFile, 'utf8');
            nativeOpts = { ...opts, grammar: grammarStr };
            delete nativeOpts.grammarFile;
        }

        // A small queue + promise bridge so tokens produced by the native
        // onToken callback are surfaced as async-generator yields.
        /** @type {Array<{value:string,done:false}|{done:true}>} */
        const queue = [];
        /** @type {((item:any)=>void)|null} */
        let resolve = null;
        let done = false;
        /** @type {Error|null} */
        let error = null;

        const enqueue = (item) => {
            if (resolve) {
                const r = resolve;
                resolve = null;
                r(item);
            } else {
                queue.push(item);
            }
        };

        this.#native.generate(
            prompt,
            nativeOpts,
            (token) => {
                enqueue({ value: token, done: false });
            },
            (err) => {
                done = true;
                error = err || null;
                enqueue({ done: true });
            }
        );

        try {
            while (true) {
                let item;
                if (queue.length > 0) {
                    item = queue.shift();
                } else if (done) {
                    break;
                } else {
                    item = await new Promise((r) => { resolve = r; });
                }

                if (item.done) {
                    break;
                }
                yield item.value;
            }
        } finally {
            this.#generating = false;
        }

        if (error) {
            throw error;
        }
    }

    /** Signal the running generation to stop at the next token boundary. */
    abort() {
        this.#native.abort();
    }

    /**
     * Release native resources (model weights + KV cache).
     * No methods should be called on this instance after dispose().
     */
    dispose() {
        this.#native.dispose();
    }

    /** Number of context tokens available in the current context window. */
    get contextLength() {
        return this.#native.contextLength;
    }

    /** Support `using model = new LlamaModel(...)` (TC39 explicit resource management). */
    [Symbol.dispose]() {
        this.dispose();
    }
}

/**
 * Manages a named set of LlamaModel instances, loading each lazily on first use.
 *
 * Solves the multi-model routing problem: instead of manually tracking multiple
 * LlamaModel instances and disposing/reloading them, register all models up
 * front and call generate() by name.
 *
 * @example
 * const pool = new LlamaModelPool();
 * pool.register('fast', '/models/phi-3-mini.gguf', { nGpuLayers: 99 });
 * pool.register('smart', '/models/llama-3-70b.gguf', { nGpuLayers: 99 });
 *
 * for await (const t of pool.generate('fast', prompt)) process.stdout.write(t);
 * pool.dispose();
 */
class LlamaModelPool {
    /** @type {Map<string, { modelPath: string, modelOpts: object, instance: LlamaModel|null }>} */
    #registry = new Map();

    /**
     * Register a model under a name. Does not load it yet.
     * @param {string} name       Unique name for this model.
     * @param {string} modelPath  Absolute path to the .gguf file.
     * @param {object} [opts]     LlamaModelOptions (nGpuLayers, nCtx).
     */
    register(name, modelPath, opts = {}) {
        if (this.#registry.has(name)) {
            throw new Error(`LlamaModelPool: model '${name}' is already registered`);
        }
        this.#registry.set(name, { modelPath, modelOpts: opts, instance: null });
    }

    /**
     * Returns the live LlamaModel for the given name, loading it if not yet loaded.
     * @param {string} name
     * @returns {LlamaModel}
     */
    load(name) {
        const entry = this.#registry.get(name);
        if (!entry) throw new Error(`LlamaModelPool: unknown model '${name}'`);
        if (!entry.instance) {
            entry.instance = new LlamaModel(entry.modelPath, entry.modelOpts);
        }
        return entry.instance;
    }

    /**
     * Generate from a named model.
     * @param {string} name    Registered model name.
     * @param {string} prompt
     * @param {object} [opts]  GenerateOptions.
     * @yields {string}
     */
    async * generate(name, prompt, opts = {}) {
        yield * this.load(name).generate(prompt, opts);
    }

    /**
     * Unload a single model and free its native resources.
     * The registration is kept so the model can be loaded again on next use.
     * @param {string} name
     */
    unload(name) {
        const entry = this.#registry.get(name);
        if (!entry) throw new Error(`LlamaModelPool: unknown model '${name}'`);
        if (entry.instance) {
            entry.instance.dispose();
            entry.instance = null;
        }
    }

    /** Dispose all loaded models and clear the registry. */
    dispose() {
        for (const [, entry] of this.#registry) {
            if (entry.instance) {
                entry.instance.dispose();
                entry.instance = null;
            }
        }
        this.#registry.clear();
    }

    /** Explicit resource management support. */
    [Symbol.dispose]() {
        this.dispose();
    }
}

module.exports = { LlamaModel, LlamaModelPool };
