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
     * (e.g. `<tool_call>`, `<|channel|>`). Reserved for future wiring into
     * the sampler's preserved-tokens list; currently passed through for
     * forward-compatibility.
     */
    preservedTokens?: string[];
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
    messages: Array<ChatMessage | object>;
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
     * Yields tokens for a single generation. Concurrent calls are allowed —
     * they queue inside the model; only one generation runs at a time.
     * Use `opts.signal` to cancel an individual call; `abort()` cancels all.
     */
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
        messages: ChatMessage[] | object[],
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

    /** Dispose all loaded models and clear the registry. */
    dispose(): void;

    get chatTemplate(): string | null;

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
