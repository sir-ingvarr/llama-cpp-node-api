export interface LlamaModelOptions {
    /** Number of layers to offload to GPU. Default: 99 (all). */
    nGpuLayers?: number;
    /** Context size in tokens. Default: 2048. */
    nCtx?: number;
}

export interface GenerateOptions {
    /** Maximum number of tokens to generate. Default: 256. 0 or negative = unlimited. */
    nPredict?: number;
    /** Sampling temperature. Default: 0.8. */
    temperature?: number;
    /** Top-p (nucleus) sampling cutoff. Default: 0.95. Set to 1 to disable. */
    topP?: number;
    /** Top-k sampling. Default: 40. Set to 0 to disable. */
    topK?: number;
    /**
     * Min-p sampling: removes tokens whose probability is below `minP * (prob of top token)`.
     * Default: 0 (disabled). Typical value: 0.05–0.1.
     */
    minP?: number;
    /**
     * Repetition penalty applied to recently generated tokens.
     * Default: 1.0 (disabled). Values > 1 discourage repetition (e.g. 1.1).
     */
    repeatPenalty?: number;
    /**
     * Number of most-recent tokens considered for repeat penalty.
     * Default: 64. Set to 0 to disable penalty entirely.
     */
    repeatLastN?: number;
    /**
     * GBNF grammar string to constrain output structure (e.g. enforce JSON).
     * Takes precedence over `grammarFile` if both are provided.
     */
    grammar?: string;
    /**
     * Path to a .gbnf grammar file. The file is read synchronously before
     * generation starts and its contents are passed as `grammar`.
     */
    grammarFile?: string;
    /**
     * Stop sequences: generation halts as soon as any of these strings is
     * produced. The stop string itself is not included in the output.
     */
    stop?: string[];
    /** Override context size for this generation call. */
    nCtx?: number;
    /**
     * When true, the KV cache is cleared before generation begins.
     * Use this to start a fresh conversation. Default: false (accumulate context).
     */
    resetContext?: boolean;
}

export declare class LlamaModel {
    constructor(modelPath: string, opts?: LlamaModelOptions);
    generate(prompt: string, opts?: GenerateOptions): AsyncGenerator<string, void, undefined>;
    abort(): void;
    dispose(): void;
    readonly contextLength: number;
    [Symbol.dispose](): void;
}

export declare class LlamaModelPool {
    /**
     * Register a model under a name. Does not load it until first use.
     * @param name       Unique identifier for this model.
     * @param modelPath  Absolute path to the `.gguf` file.
     * @param opts       Optional model parameters.
     */
    register(name: string, modelPath: string, opts?: LlamaModelOptions): void;

    /**
     * Return the live LlamaModel for `name`, loading it if not yet loaded.
     */
    load(name: string): LlamaModel;

    /**
     * Generate from a named model. Equivalent to `pool.load(name).generate(prompt, opts)`.
     */
    generate(name: string, prompt: string, opts?: GenerateOptions): AsyncGenerator<string, void, undefined>;

    /**
     * Dispose a single model and free its native resources.
     * The registration is kept — the model reloads on next `load()` or `generate()`.
     */
    unload(name: string): void;

    /** Dispose all loaded models and clear the registry. */
    dispose(): void;

    [Symbol.dispose](): void;
}
