'use strict';

// Smoke test for applyChatTemplateJinja + lazy grammar.
//
// Usage:
//   node smoke_chat_jinja.js /path/to/model.gguf
//
// Works best with a tool-trained model (Qwen2.5/3, Hermes, Functionary, etc.).
// For content-only models the template still renders but no grammar is emitted.

const { LlamaModel } = require('./js/index.js');

const MODEL_PATH = process.argv[2];
if (!MODEL_PATH) {
    console.error('usage: node smoke_chat_jinja.js /path/to/model.gguf');
    process.exit(1);
}

const WEATHER_TOOL = {
    type: 'function',
    function: {
        name: 'get_weather',
        description: 'Get the current weather for a city.',
        parameters: {
            type: 'object',
            properties: {
                city: { type: 'string', description: 'City name, e.g. Paris' },
                units: { type: 'string', enum: ['c', 'f'], default: 'c' }
            },
            required: ['city']
        }
    }
};

(async () => {
    const model = new LlamaModel(MODEL_PATH, { nGpuLayers: 99, nCtx: 2048 });

    // ---- Case 1: content-only (no tools) ----
    const plain = model.applyChatTemplateJinja(
        [
            { role: 'system',  content: 'You are a terse assistant.' },
            { role: 'user',    content: 'Name three primary colors.' }
        ],
        { addGenerationPrompt: true }
    );
    console.log('=== content-only ===');
    console.log('format:        ', plain.format);
    console.log('has grammar?   ', !!plain.grammar);
    console.log('prompt preview:', JSON.stringify(plain.prompt.slice(0, 120)) + '…');

    // ---- Case 2: with tools ----
    const withTools = model.applyChatTemplateJinja(
        [
            { role: 'system', content: 'You are a helpful assistant. Use tools when useful.' },
            { role: 'user',   content: "What's the weather in Paris right now?" }
        ],
        {
            tools: [WEATHER_TOOL],
            toolChoice: 'auto',
            parallelToolCalls: false,
            addGenerationPrompt: true,
        }
    );
    console.log('\n=== with tools ===');
    console.log('format:                  ', withTools.format);
    console.log('has grammar?             ', !!withTools.grammar);
    console.log('grammarLazy:             ', withTools.grammarLazy);
    console.log('triggerPatterns:         ', withTools.grammarTriggerPatterns);
    console.log('triggerTokens:           ', withTools.grammarTriggerTokens);
    console.log('preservedTokens:         ', withTools.preservedTokens);
    console.log('additionalStops:         ', withTools.additionalStops);
    console.log('grammar preview:         ', (withTools.grammar || '').slice(0, 200) + '…');
    console.log('prompt length (chars):   ', withTools.prompt.length);

    // ---- Case 3: content-only generate (regression: template → generate still works) ----
    console.log('\n=== content-only generate ===');
    {
        let out = '';
        for await (const token of model.generate(plain.prompt, {
            nPredict: 40,
            temperature: 0.0,
            resetContext: true,
        })) {
            process.stdout.write(token);
            out += token;
        }
        console.log('\n--- end ---  (', out.length, 'chars )');
    }

    // ---- Case 4: generate with tool-call grammar (if the template emits one) ----
    if (withTools.grammar) {
        console.log('\n=== generate with tool-call grammar ===');
        let out = '';
        for await (const token of model.generate(withTools.prompt, {
            nPredict: 200,
            temperature: 0.3,
            grammar: withTools.grammar,
            grammarTriggerPatterns: withTools.grammarTriggerPatterns,
            grammarTriggerTokens:   withTools.grammarTriggerTokens,
            preservedTokens:        withTools.preservedTokens,
            stop: withTools.additionalStops ?? [],
            resetContext: true,
        })) {
            process.stdout.write(token);
            out += token;
        }
        console.log('\n--- end ---  (', out.length, 'chars )');
    } else {
        console.log('\n=== tool-call grammar: skipped (template did not emit one) ===');
    }

    // ---- Case 4b: parseChatResponse on the tools output ----
    if (withTools.grammar || withTools.format !== 'legacy') {
        console.log('\n=== parseChatResponse ===');
        let text = '';
        const genOpts = {
            nPredict: 200,
            temperature: 0.3,
            stop: withTools.additionalStops ?? [],
            resetContext: true,
        };
        if (withTools.grammar) {
            genOpts.grammar = withTools.grammar;
            genOpts.grammarTriggerPatterns = withTools.grammarTriggerPatterns;
            genOpts.grammarTriggerTokens   = withTools.grammarTriggerTokens;
            genOpts.preservedTokens        = withTools.preservedTokens;
        }
        for await (const tok of model.generate(withTools.prompt, genOpts)) {
            text += tok;
        }
        const parsed = model.parseChatResponse(text, {
            format:           withTools.format,
            parser:           withTools.parser,
            generationPrompt: withTools.generationPrompt,
        });
        console.log('raw text:       ', JSON.stringify(text).slice(0, 200));
        console.log('content:        ', JSON.stringify(parsed.content).slice(0, 120));
        console.log('reasoning:      ', JSON.stringify(parsed.reasoningContent).slice(0, 120));
        console.log('tool calls:     ', parsed.toolCalls);
    }

    // ---- Case 4c: chat() high-level helper ----
    console.log('\n=== chat() ===');
    {
        const result = await model.chat({
            messages: [
                { role: 'system', content: 'Use tools when helpful.' },
                { role: 'user',   content: "What's the weather in Paris?" }
            ],
            tools: [WEATHER_TOOL],
            toolChoice: 'auto',
            nPredict: 200,
            temperature: 0.3,
            resetContext: true,
        });
        console.log('format:    ', result.format);
        console.log('content:   ', JSON.stringify(result.content).slice(0, 120));
        console.log('reasoning: ', JSON.stringify(result.reasoningContent).slice(0, 120));
        console.log('tool calls:', result.toolCalls);
    }

    // ---- Case 5: lazy grammar (manual trigger — proves the wiring regardless of template) ----
    //
    // Trivial grammar that constrains tokens after the first '{' to a flat
    // JSON object. With a trigger pattern on '{', everything before it is
    // unconstrained — the model can write free-form prose first.
    console.log('\n=== lazy grammar (manual trigger) ===');
    {
        const prompt = model.applyChatTemplateJinja(
            [{ role: 'user', content: 'Reply with exactly: READY { "x": 1 }' }],
            { addGenerationPrompt: true }
        ).prompt;

        let out = '';
        for await (const token of model.generate(prompt, {
            nPredict: 60,
            temperature: 0.0,
            grammar: 'root ::= "{" [^}]* "}"',
            grammarTriggerPatterns: ['\\{'],
            stop: ['}'],
            resetContext: true,
        })) {
            process.stdout.write(token);
            out += token;
        }
        console.log('\n--- end ---');
        console.log('contains free-form prefix "READY"?', out.includes('READY'));
        console.log('contains "{"?                     ', out.includes('{'));
    }

    model.dispose();
})().catch(err => {
    console.error(err);
    process.exit(1);
});
