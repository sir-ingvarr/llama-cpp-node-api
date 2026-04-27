export interface LlamaModelOptions {
    /** Number of layers to offload to GPU. Default: 99 (all). */
    nGpuLayers?: number;
    /** Context size in tokens. Default: 2048. */
    nCtx?: number;
    /**
     * Open the model in embedding mode. `model.embed()` becomes available
     * (and is required for embedding-only models like nomic-embed, bge, etc.).
     * `model.generate()` still works on the same context — every decode
     * populates the embedding buffer as a side effect — so a generative
     * model can do both. For embedding-only encoders, only embed() makes sense.
     */
    embeddings?: boolean;
    /**
     * Pooling strategy applied to per-token embeddings. Defaults to whatever
     * the model was trained with — typically `'mean'` for BERT-like models,
     * `'last'` for last-token-pooling encoders. `'none'` returns the
     * last-token raw embedding without pooling.
     */
    poolingType?: 'unspecified' | 'none' | 'mean' | 'cls' | 'last' | 'rank';
    /**
     * Quantization type for the K (keys) tensor of the KV cache. Default
     * `'f16'`. Lower-precision types cut KV memory roughly in half (`'q8_0'`)
     * or more (`'q4_0'`), at a small quality cost — letting you fit a bigger
     * `nCtx` or run more concurrent contexts. Allowed: `'f32' | 'f16' | 'bf16'
     * | 'q8_0' | 'q4_0' | 'q4_1' | 'iq4_nl' | 'q5_0' | 'q5_1'`.
     */
    cacheTypeK?: CacheType;
    /**
     * Quantization type for the V (values) tensor of the KV cache. Same
     * options and trade-offs as `cacheTypeK`. Note: quantized V cache
     * generally requires Flash Attention to be enabled in the build.
     */
    cacheTypeV?: CacheType;
    /**
     * Flash Attention toggle. `'auto'` (default) lets llama.cpp decide based
     * on the backend; `'on'` forces it (required for quantized V cache on most
     * backends); `'off'` disables it. Falling back to `'off'` is sometimes
     * needed when a backend doesn't support FA for the model's head shape.
     */
    flashAttention?: 'auto' | 'on' | 'off';
    /**
     * Number of threads used for token generation (single-token decode).
     * Defaults to whatever llama.cpp picks (typically `min(n_logical, 8)` on
     * CPU, ignored when fully offloaded to GPU). Tune for CPU-only setups.
     */
    nThreads?: number;
    /**
     * Number of threads used for batch processing (prompt ingest, batched
     * decode). Same default rules as `nThreads`. Often higher than `nThreads`
     * on big-core machines because prompt processing parallelises well.
     */
    nThreadsBatch?: number;
    /**
     * Memory-map the model file instead of reading it into RAM. Default `true`.
     * Set `false` to avoid mmap stutter (cold-cache page faults during the
     * first decode) at the cost of a longer load and full RAM usage.
     */
    useMmap?: boolean;
    /**
     * Lock the model's pages in physical memory so they cannot be swapped
     * out. Default `false`. Useful for low-latency serving setups; may
     * require elevated permissions (`ulimit -l` / `CAP_IPC_LOCK`).
     */
    useMlock?: boolean;
}

export type CacheType =
    | 'f32' | 'f16' | 'bf16'
    | 'q8_0'
    | 'q4_0' | 'q4_1'
    | 'iq4_nl'
    | 'q5_0' | 'q5_1';

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
    /**
     * AbortSignal that cancels this specific generation when fired. The async
     * generator will throw an `AbortError` on the next iteration. Unlike
     * `model.abort()`, other concurrent generations are unaffected.
     */
    signal?: AbortSignal;
    /**
     * Lazy-grammar trigger patterns. When set (and `grammar` is also set),
     * the grammar stays inactive until one of these regex patterns appears
     * in the output, then activates for the remainder. Typically obtained
     * from `applyChatTemplateJinja` when tools are present.
     */
    grammarTriggerPatterns?: string[];
    /**
     * Lazy-grammar trigger token IDs. Like `grammarTriggerPatterns`, but
     * matches on specific vocab tokens instead of string patterns.
     */
    grammarTriggerTokens?: number[];
    /**
     * Special tokens that must be preserved as atomic units during sampling
     * (e.g. `<tool_call>`, `<|channel|>`). Forwarded by `chat()` from the
     * libcommon Jinja renderer when tools / json_schema are in use; reserved
     * for direct callers and not yet wired into the native sampler.
     */
    preservedTokens?: string[];
    /**
     * Include the chosen token's logprob with each yielded item. When set,
     * `generate()` yields `TokenWithLogprobs` objects instead of strings.
     */
    logprobs?: boolean;
    /**
     * Number of top alternative tokens to include per step (with their
     * logprobs). Setting `> 0` implies `logprobs: true`.
     */
    topLogprobs?: number;
    /**
     * Seed for the stochastic sampler (`dist`). When set, generation is
     * deterministic for a given prompt + sampler config. Defaults to a
     * non-deterministic seed (`LLAMA_DEFAULT_SEED`).
     */
    seed?: number;
    /**
     * Per-token logit bias, applied additively before grammar / penalty /
     * top-k. Keys are vocab token IDs (use `model.tokenize()` to find them);
     * values are added to the raw logits. Common idioms:
     *   - `{ 220: -100 }` to ban a specific token (e.g. a leading space).
     *   - `{ 13: 5 }` to nudge generation toward a token without forcing it.
     * Bias is applied before grammar, so grammar still has the final say on
     * which tokens are legal.
     */
    logitBias?: Record<number, number>;
}

export interface TokenLogprob {
    token: string;
    /** Natural-log probability under the raw model distribution. */
    logprob: number;
}

export interface TokenWithLogprobs {
    text: string;
    /** Logprob of the chosen token under the raw model distribution. */
    logprob: number;
    /** Top-K alternatives, present only when `topLogprobs` was set. */
    topLogprobs: TokenLogprob[];
}

export interface ChatTemplateTool {
    type?: 'function';
    function?: { name: string; description?: string; parameters?: unknown };
    /** Flat form also accepted: { name, description, parameters }. */
    name?: string;
    description?: string;
    parameters?: unknown;
}

export interface ApplyChatTemplateJinjaOptions {
    /** OpenAI-format tool definitions. */
    tools?: ChatTemplateTool[];
    /** 'auto' (default), 'required', or 'none'. */
    toolChoice?: 'auto' | 'required' | 'none';
    /** Allow the model to emit multiple tool calls in one turn. Default: false. */
    parallelToolCalls?: boolean;
    /** Append the assistant-turn prefix. Default: true. */
    addGenerationPrompt?: boolean;
    /** Enable `<think>`-style reasoning blocks when the template supports it. Default: true. */
    enableThinking?: boolean;
    /**
     * Raw GBNF grammar, used when `tools` and `jsonSchema` are both absent.
     * When tools or jsonSchema are present, this field is ignored by
     * libcommon (it replaces it with the auto-generated tool/schema grammar).
     */
    grammar?: string;
    /** JSON schema (object or JSON string) to constrain free-form output. */
    jsonSchema?: string | object;
    /**
     * Arbitrary Jinja template variables. Values are JSON strings as
     * consumed by the template (e.g. `{ enable_thinking: 'true' }`).
     */
    chatTemplateKwargs?: Record<string, string>;
    /**
     * Full Jinja template source used in place of the model's embedded
     * template. Useful when the GGUF stores only a legacy alias name
     * (e.g. `mistral-v7-tekken`) — paste the template from the model's
     * HuggingFace `tokenizer_config.json` here.
     */
    chatTemplateOverride?: string;
}

export interface ParsedToolCall {
    name: string;
    /** Arguments as a JSON string (parse via `JSON.parse`). */
    arguments: string;
    /** Call id, empty string if the format didn't provide one. */
    id: string;
}

export interface ParsedChatMessage {
    content: string;
    reasoningContent: string;
    toolCalls: ParsedToolCall[];
    /** Present only on tool-result messages. */
    toolName?: string;
    toolCallId?: string;
}

export interface ParseChatResponseOptions {
    /**
     * Format tag from `ChatTemplateJinjaResult.format`. One of:
     * 'Content-only', 'peg-simple', 'peg-native', 'peg-gemma4', or 'legacy'.
     */
    format: string;
    /** Opaque PEG parser blob from `ChatTemplateJinjaResult.parser`. */
    parser?: string;
    /** Pass through `ChatTemplateJinjaResult.generationPrompt` when available. */
    generationPrompt?: string;
    /** Enable tool-call extraction. Default: true. */
    parseToolCalls?: boolean;
    /**
     * Set true when `text` is a streaming partial; the parser will attempt
     * best-effort recovery rather than failing on incomplete output.
     */
    isPartial?: boolean;
}

export interface ChatOptions extends Omit<GenerateOptions, 'grammar' | 'grammarTriggerPatterns' | 'grammarTriggerTokens' | 'preservedTokens'> {
    /**
     * OAI-style chat messages. `unknown[]` is permitted because tool/result
     * messages and multipart content carry per-format fields not modelled
     * by `ChatMessage` — let the renderer validate.
     */
    messages: ReadonlyArray<ChatMessage | Record<string, unknown>>;
    tools?: ChatTemplateTool[];
    toolChoice?: 'auto' | 'required' | 'none';
    parallelToolCalls?: boolean;
    enableThinking?: boolean;
    jsonSchema?: string | object;
    chatTemplateKwargs?: Record<string, string>;
    chatTemplateOverride?: string;
}

export interface ChatResult {
    content: string;
    reasoningContent: string;
    toolCalls: ParsedToolCall[];
    format: string;
    /** Full generated text, before parsing. Useful for debugging. */
    raw: string;
}

export interface ChatTemplateJinjaResult {
    /** Rendered prompt, ready to pass to `generate()`. */
    prompt: string;
    /**
     * Parser-format tag. One of 'Content-only', 'peg-simple', 'peg-native',
     * 'peg-gemma4', or 'legacy' (the last is emitted when the embedded
     * template was an alias string and the legacy renderer was used).
     * Pass this to `parseChatResponse()` alongside `parser`.
     */
    format: string;
    /**
     * Opaque serialised PEG parser arena (from `common_peg_arena::save()`).
     * Hand it back to `parseChatResponse()` unchanged. May be absent for
     * formats that don't need a parser (content-only, legacy).
     */
    parser?: string;
    /**
     * Generation-prompt prefix used by some formats during parsing. Pass
     * along with `parser` to `parseChatResponse()` when present.
     */
    generationPrompt?: string;
    /**
     * Auto-generated GBNF grammar that constrains tool-call / json-schema
     * output. Present when tools or jsonSchema were provided, or when the
     * caller-supplied `grammar` was passed through unchanged.
     */
    grammar?: string;
    /**
     * When true, `grammar` is applied lazily — only after one of the
     * triggers fires. Feed this along with the trigger arrays into
     * `generate()`'s matching options.
     */
    grammarLazy?: boolean;
    /** Regex patterns that activate the lazy grammar. */
    grammarTriggerPatterns?: string[];
    /** Vocab token IDs that activate the lazy grammar. */
    grammarTriggerTokens?: number[];
    /** Tokens that must be preserved as atomic units during sampling. */
    preservedTokens?: string[];
    /**
     * Additional stop sequences required by the template (e.g. turn-end
     * markers). Merge with your own `stop` before calling `generate()`.
     */
    additionalStops?: string[];
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

/**
 * Names accepted by `QuantizeOptions.ftype`. Mirrors llama.cpp's CLI
 * ftype names (the `LLAMA_FTYPE_MOSTLY_` prefix stripped).
 */
export type QuantizeFtype =
    | 'F32' | 'F16' | 'BF16'
    | 'Q4_0' | 'Q4_1' | 'Q5_0' | 'Q5_1' | 'Q8_0'
    | 'Q2_K' | 'Q2_K_S'
    | 'Q3_K_S' | 'Q3_K_M' | 'Q3_K_L'
    | 'Q4_K_S' | 'Q4_K_M'
    | 'Q5_K_S' | 'Q5_K_M'
    | 'Q6_K'
    | 'IQ2_XXS' | 'IQ2_XS' | 'IQ2_S' | 'IQ2_M'
    | 'IQ3_XXS' | 'IQ3_XS' | 'IQ3_S' | 'IQ3_M'
    | 'IQ1_S' | 'IQ1_M'
    | 'IQ4_NL' | 'IQ4_XS'
    | 'TQ1_0' | 'TQ2_0'
    | 'MXFP4_MOE' | 'NVFP4' | 'Q1_0';

export interface QuantizeOptions {
    /** Target quantization ftype. Required. Accepts a string name or the raw enum value. */
    ftype: QuantizeFtype | number;
    /** Number of threads. Default: hardware concurrency. */
    nthread?: number;
    /** Allow re-quantizing tensors that are not f32/f16. Default: false. */
    allowRequantize?: boolean;
    /** Quantize `output.weight` too. Default: true. */
    quantizeOutputTensor?: boolean;
    /** Skip quantization — copy tensors as-is. Useful for shard repacking. */
    onlyCopy?: boolean;
    /** Quantize every tensor to the default type (no per-tensor overrides). */
    pure?: boolean;
    /** Preserve the input's shard count. */
    keepSplit?: boolean;
    /** Compute and report final size without actually writing the output. */
    dryRun?: boolean;
}

export declare class LlamaModel {
    constructor(modelPath: string, opts?: LlamaModelOptions);
    /**
     * Async constructor: loads weights on a libuv worker thread so the JS
     * event loop is not blocked. Equivalent to `new LlamaModel(path, opts)`
     * once it resolves. Use this in any code path that runs on the Node.js
     * main thread (Electron renderer, HTTP request handler, etc.).
     */
    static load(modelPath: string, opts?: LlamaModelOptions): Promise<LlamaModel>;
    /**
     * Yields tokens for a single generation. Concurrent calls are allowed —
     * they queue inside the model; only one generation runs at a time.
     * Use `opts.signal` to cancel an individual call; `abort()` cancels all.
     *
     * Yields `string` by default; yields `TokenWithLogprobs` when
     * `opts.logprobs` (or `opts.topLogprobs > 0`) is set.
     */
    generate(prompt: string, opts: GenerateOptions & { logprobs: true }): AsyncGenerator<TokenWithLogprobs, void, undefined>;
    generate(prompt: string, opts: GenerateOptions & { topLogprobs: number }): AsyncGenerator<TokenWithLogprobs, void, undefined>;
    generate(prompt: string, opts?: GenerateOptions): AsyncGenerator<string, void, undefined>;
    /**
     * Format messages using the model's built-in chat template.
     * Uses llama.cpp's built-in template renderer (not a full Jinja parser).
     */
    applyChatTemplate(messages: ChatMessage[], opts?: ApplyChatTemplateOptions): string;
    /**
     * Jinja-based renderer (libcommon). Supports tools, json_schema, and
     * reasoning toggles; returns the auto-generated grammar + triggers +
     * stops alongside the rendered prompt so the caller can feed them back
     * into `generate()` (or merge them with their own grammar).
     */
    applyChatTemplateJinja(
        messages: ReadonlyArray<ChatMessage | Record<string, unknown>>,
        opts?: ApplyChatTemplateJinjaOptions
    ): ChatTemplateJinjaResult;
    /**
     * Parse a model response into `{ content, reasoningContent, toolCalls }`.
     * Use the `format` (and `parser`) fields from `applyChatTemplateJinja`
     * so libcommon knows which per-model-family parser to run.
     */
    parseChatResponse(text: string, opts: ParseChatResponseOptions): ParsedChatMessage;
    /**
     * One-shot chat helper: render → generate → parse. Handles all the
     * template-returned grammar/triggers/stops internally. Equivalent to
     * calling `applyChatTemplateJinja` + `generate` + `parseChatResponse`
     * manually, but bundled for the common case.
     */
    chat(opts: ChatOptions): Promise<ChatResult>;
    /** Convert text to token IDs. */
    tokenize(text: string, opts?: TokenizeOptions): number[];
    /** Convert token IDs back to text. */
    detokenize(tokens: number[], opts?: DetokenizeOptions): string;
    /** Returns model metadata: description, parameter count, sizes, special tokens. */
    getModelInfo(): ModelInfo;
    /**
     * Compute an embedding for `text`. Requires the model to have been
     * constructed with `{ embeddings: true }`; throws otherwise. The returned
     * Float32Array has length `n_embd` (typically 384–4096 depending on the
     * model). Pooling is whatever the model was trained with unless
     * overridden via `poolingType` at construction.
     */
    embed(text: string): Promise<Float32Array>;
    /** Cancel every currently running and queued generation on this model. */
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
     * Dispose a model and remove it from the pool. The `name` is released —
     * re-register it explicitly to load again.
     */
    unload(name: string): void;

    /** Dispose all loaded models and clear the registry. Idempotent. */
    dispose(): void;

    [Symbol.dispose](): void;
}

/**
 * Convert a GGUF file from one quantization to another. Runs on a worker
 * thread; the returned Promise resolves once the output has been written.
 */
export declare function quantize(
    inputPath: string,
    outputPath: string,
    opts: QuantizeOptions
): Promise<void>;

/** List of ftype names accepted by `quantize()`. */
export declare function quantizeFtypes(): QuantizeFtype[];

export interface GgufTensorInfo {
    name: string;
    /** ggml type name, e.g. "F32", "F16", "Q4_K", "Q8_0". */
    type: string;
    /** Byte offset of the tensor data, relative to `dataOffset`. */
    offset: bigint;
    /** Byte size of the tensor data on disk. */
    size: bigint;
}

/**
 * GGUF metadata value. Scalar ints under 32 bits become `number`; 64-bit
 * ints become `bigint` to avoid precision loss. Arrays preserve element type.
 */
export type GgufMetaValue =
    | number
    | bigint
    | boolean
    | string
    | number[]
    | bigint[]
    | boolean[]
    | string[];

export interface GgufInspectResult {
    /** GGUF format version (typically 3). */
    version: number;
    /** Tensor data alignment in bytes (typically 32). */
    alignment: number;
    /** Absolute byte offset where tensor data begins in the file. */
    dataOffset: bigint;
    /** Full GGUF KV pair map. Keys are like "general.architecture", "llama.embedding_length", etc. */
    metadata: Record<string, GgufMetaValue>;
    /** Tensor descriptors (no data loaded). */
    tensors: GgufTensorInfo[];
}

export interface InspectOptions {
    /**
     * Use the in-process LRU cache (default `true`). Cached entries are keyed
     * by canonical path and invalidated when the file's mtime or size changes.
     */
    cache?: boolean;
}

/**
 * Read GGUF header, KV metadata, and tensor descriptors without loading any
 * tensor weights. Useful for cheap model identification, shape inspection,
 * or pre-flight validation before a full `LlamaModel` load.
 *
 * Results are memoized by canonical path (mtime + size validated). Pass
 * `{ cache: false }` to bypass the cache, or call `clearInspectCache()` to
 * flush it.
 */
export declare function inspect(path: string, opts?: InspectOptions): Promise<GgufInspectResult>;

/** Flush the in-process `inspect()` memoization cache. */
export declare function clearInspectCache(): void;
