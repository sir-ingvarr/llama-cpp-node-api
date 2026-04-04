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

export interface ChatMessage {
    role: string;
    content: string;
}

export interface ApplyChatTemplateOptions {
    /** Whether to append the assistant turn prefix. Default: true. */
    addAssistant?: boolean;
}

export interface TokenizeOptions {
    /** Add BOS/EOS special tokens. Default: true. */
    addSpecial?: boolean;
    /** Parse special token syntax in the text (e.g. <|im_start|>). Default: false. */
    parseSpecial?: boolean;
}

export interface DetokenizeOptions {
    /** Remove special tokens from output. Default: false. */
    removeSpecial?: boolean;
    /** Render special tokens as their text representation. Default: false. */
    unparseSpecial?: boolean;
}

export interface SpecialTokens {
    /** Beginning-of-sequence token ID. */
    bos: number;
    /** End-of-sequence token ID. */
    eos: number;
    /** End-of-turn token ID. */
    eot: number;
}

export interface ModelInfo {
    /** Human-readable model description (e.g. "llama 8B Q4_0"). */
    description: string;
    /** Total number of parameters. */
    nParams: number;
    /** Model size on disk in bytes. */
    modelSize: number;
    /** Context length the model was trained with. */
    trainContextLength: number;
    /** Embedding vector dimension. */
    embeddingSize: number;
    /** Number of transformer layers. */
    nLayer: number;
    /** Vocabulary size (number of tokens). */
    vocabSize: number;
    /** Special token IDs. */
    specialTokens: SpecialTokens;
}

export declare class LlamaModel {
    constructor(modelPath: string, opts?: LlamaModelOptions);
    generate(prompt: string, opts?: GenerateOptions): AsyncGenerator<string, void, undefined>;
    /**
     * Format messages using the model's built-in chat template.
     * Uses llama.cpp's built-in template renderer (not a full Jinja parser).
     */
    applyChatTemplate(messages: ChatMessage[], opts?: ApplyChatTemplateOptions): string;
    /** Convert text to token IDs. */
    tokenize(text: string, opts?: TokenizeOptions): number[];
    /** Convert token IDs back to text. */
    detokenize(tokens: number[], opts?: DetokenizeOptions): string;
    /** Returns model metadata: description, parameter count, sizes, special tokens. */
    getModelInfo(): ModelInfo;
    abort(): void;
    dispose(): void;
    readonly contextLength: number;
    /** Jinja2 chat template embedded in the model metadata, or null if not present. */
    readonly chatTemplate: string | null;
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

    get chatTemplate(): string | null;

    [Symbol.dispose](): void;
}
