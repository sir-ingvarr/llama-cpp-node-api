// Functional tests — run against a real GGUF model.
//
//   LLAMA_NODE_MODEL_PATH=/path/to/model.gguf npm test
//
// The whole suite self-skips when LLAMA_NODE_MODEL_PATH is not set, so unit
// tests stay green in environments without a model. Any chat-capable
// instruct model works; a small one (e.g. Qwen3-0.6B Q8_0) keeps CI fast.
//
// Optional:
//   LLAMA_NODE_LORA_PATH — a LoRA adapter matching the model; enables the
//                          positive LoRA test (the error path runs always).
'use strict';

const { describe, it, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const { LlamaModel, LlamaModelPool, inspect } = require('../js');

const MODEL = process.env.LLAMA_NODE_MODEL_PATH;
const LORA  = process.env.LLAMA_NODE_LORA_PATH;

const describeModel = MODEL ? describe : describe.skip;

if (!MODEL) {
    it('functional suite (set LLAMA_NODE_MODEL_PATH to enable)', (t) => t.skip());
}

async function collect(gen) {
    let text = '';
    for await (const tok of gen) text += typeof tok === 'string' ? tok : tok.text;
    return text;
}

// Sanity-check that generated text looks like language rather than a stuck
// sampler: enough unique characters, and word breaks once there is room
// for several words. Grammar-constrained outputs are exempt (they may
// legitimately be a single word).
function assertReadableText(s, label) {
    const t = s.trim();
    assert.ok(t.length >= 3, `${label}: output too short: ${JSON.stringify(s)}`);
    const unique = new Set(t).size;
    assert.ok(unique >= 4,
        `${label}: only ${unique} unique characters in ${JSON.stringify(t)}`);
    assert.ok(!/^(.)\1+$/.test(t),
        `${label}: output is one repeated character: ${JSON.stringify(t)}`);
    if (t.length >= 24) {
        const spaces = (t.match(/\s/g) || []).length;
        assert.ok(spaces >= 2,
            `${label}: expected word breaks in ${JSON.stringify(t)}`);
    }
}

describeModel('LlamaModel functional', () => {
    /** @type {import('../js').LlamaModel} */
    let model;

    before(async () => {
        model = await LlamaModel.load(MODEL, { nCtx: 512 });
    });

    after(() => {
        model?.dispose();
    });

    // Fresh-KV generation with quiet defaults; tests override as needed.
    const gen = (prompt, opts = {}) =>
        model.generate(prompt, { nPredict: 16, resetContext: true, ...opts });

    describe('model metadata', () => {
        it('getModelInfo() has the documented shape', () => {
            const info = model.getModelInfo();
            assert.equal(typeof info.description, 'string');
            assert.ok(info.nParams > 0);
            assert.ok(info.modelSize > 0);
            assert.ok(info.trainContextLength > 0);
            assert.ok(info.embeddingSize > 0);
            assert.ok(info.nLayer > 0);
            assert.ok(info.vocabSize > 0);
            assert.equal(typeof info.specialTokens.bos, 'number');
            assert.equal(typeof info.specialTokens.eos, 'number');
            assert.ok(Array.isArray(info.suppressTokens));
        });

        it('memoryUsage() reports weights, and context state after a generate', async () => {
            const before = model.memoryUsage();
            assert.ok(before.modelBytes > 0);
            assert.ok(before.nParams > 0);
            assert.equal(before.modelBytes, model.getModelInfo().modelSize);
            assert.equal(before.nParams, model.getModelInfo().nParams);
            // Context may or may not exist yet depending on test order.
            assert.ok(before.stateBytes === null || before.stateBytes > 0);

            await collect(gen('Hi', { nPredict: 2 }));
            // No decode in flight now, so the context is observable.
            const after = model.memoryUsage();
            assert.ok(after.stateBytes > 0, `expected stateBytes > 0, got ${after.stateBytes}`);
        });

        it('chatTemplate is a string or null', () => {
            const t = model.chatTemplate;
            assert.ok(t === null || typeof t === 'string');
        });

        it('tokenize/detokenize round-trips plain text', () => {
            const text = 'The quick brown fox jumps over the lazy dog.';
            const tokens = model.tokenize(text, { addSpecial: false });
            assert.ok(Array.isArray(tokens) && tokens.length > 0);
            assert.equal(model.detokenize(tokens), text);
        });
    });

    describe('generate()', () => {
        it('yields strings and respects nPredict', async () => {
            const chunks = [];
            for await (const tok of gen('The sky is', { nPredict: 8 })) {
                assert.equal(typeof tok, 'string');
                chunks.push(tok);
            }
            assert.ok(chunks.length > 0, 'no tokens produced');
            // One chunk per token (no stop sequences configured) — the cap
            // bounds the chunk count.
            assert.ok(chunks.length <= 8, `expected <= 8 chunks, got ${chunks.length}`);
        });

        it('free-form output reads like language', async () => {
            const out = await collect(gen('Write one sentence about the sea.', {
                nPredict: 32, temperature: 0,
            }));
            assertReadableText(out, 'free-form generation');
        });

        it('temperature 0 is deterministic', async () => {
            const a = await collect(gen('Count: one, two,', { temperature: 0 }));
            const b = await collect(gen('Count: one, two,', { temperature: 0 }));
            assert.equal(a, b);
        });

        it('a fixed seed is deterministic at temperature > 0', async () => {
            const opts = { temperature: 0.9, seed: 1234 };
            const a = await collect(gen('Write a word:', opts));
            const b = await collect(gen('Write a word:', opts));
            assert.equal(a, b);
        });

        it('stop sequences truncate the output and are not included', async () => {
            const out = await collect(gen('List:\n- apple\n- banana\n-', {
                nPredict: 64, stop: ['\n'],
            }));
            assert.ok(!out.includes('\n'), `output contains a stop sequence: ${JSON.stringify(out)}`);
        });

        it('logprobs mode yields objects with logprob and topLogprobs', async () => {
            let items = 0;
            for await (const item of gen('Hello', { nPredict: 4, topLogprobs: 3 })) {
                items++;
                assert.equal(typeof item.text, 'string');
                assert.equal(typeof item.logprob, 'number');
                assert.ok(item.logprob <= 0, 'logprob must be <= 0');
                assert.ok(Array.isArray(item.topLogprobs));
                assert.ok(item.topLogprobs.length <= 3);
                for (const alt of item.topLogprobs) {
                    assert.equal(typeof alt.token, 'string');
                    assert.equal(typeof alt.logprob, 'number');
                }
            }
            assert.ok(items > 0);
        });

        it('accepts logitBias without error', async () => {
            const eos = model.getModelInfo().specialTokens.eos;
            const out = await collect(gen('Hi', { nPredict: 4, logitBias: { [eos]: -100 } }));
            assert.equal(typeof out, 'string');
        });
    });

    describe('samplers', () => {
        const configs = {
            'typical-p':   { typicalP: 0.9 },
            'xtc':         { xtcProbability: 0.5, xtcThreshold: 0.1 },
            'top-n-sigma': { topNSigma: 1.5 },
            'dry':         { dryMultiplier: 0.8 },
            'mirostat v1': { mirostat: 1 },
            'mirostat v2': { mirostat: 2 },
            'dynatemp':    { dynatempRange: 0.5 },
            'adaptive-p':  { adaptivePTarget: 0.3, minP: 0.05, topK: 0, topP: 1 },
        };
        for (const [name, cfg] of Object.entries(configs)) {
            it(`${name} produces readable output`, async () => {
                const out = await collect(gen('Describe the sea briefly.', {
                    seed: 42, nPredict: 24, ...cfg,
                }));
                assertReadableText(out, name);
            });
        }

        it('rejects an invalid mirostat version', async () => {
            await assert.rejects(collect(gen('x', { mirostat: 3 })), RangeError);
        });
    });

    describe('grammar', () => {
        const YES_NO = 'root ::= "yes" | "no"';

        it('eager grammar constrains output exactly', async () => {
            const out = await collect(gen('Is water wet? Answer:', {
                nPredict: 8, grammar: YES_NO,
            }));
            assert.ok(['yes', 'no'].includes(out),
                `grammar violated: ${JSON.stringify(out)}`);
        });

        it('a JSON grammar yields parseable JSON', async () => {
            const out = await collect(gen('Give me a JSON object with a name field.', {
                nPredict: 48,
                grammar: [
                    'root ::= "{" ws "\\"name\\"" ws ":" ws string ws "}"',
                    'string ::= "\\"" [a-zA-Z0-9 ]* "\\""',
                    'ws ::= [ \\t\\n]*',
                ].join('\n'),
            }));
            const parsed = JSON.parse(out);
            assert.equal(typeof parsed.name, 'string');
        });

        it('grammarFile is read and applied', async () => {
            const file = path.join(os.tmpdir(), `llama-node-test-${process.pid}.gbnf`);
            fs.writeFileSync(file, YES_NO);
            try {
                const out = await collect(gen('Is fire cold? Answer:', {
                    nPredict: 8, grammarFile: file,
                }));
                assert.ok(['yes', 'no'].includes(out),
                    `grammarFile violated: ${JSON.stringify(out)}`);
            } finally {
                fs.rmSync(file, { force: true });
            }
        });

        it('a lazy grammar stays inactive until its trigger fires', async () => {
            // The trigger pattern can never match, so the grammar must not
            // constrain the output at all.
            const out = await collect(gen('Write one sentence about the sea.', {
                nPredict: 24, temperature: 0,
                grammar: YES_NO,
                grammarTriggerPatterns: ['NEVER_MATCHES_\\d{9}'],
            }));
            assertReadableText(out, 'lazy grammar (untriggered)');
            assert.notEqual(out.trim(), 'yes');
            assert.notEqual(out.trim(), 'no');
        });
    });

    describe('chat templating', () => {
        it('parseChatResponse handles the legacy format standalone', () => {
            const msg = model.parseChatResponse('plain text answer', { format: 'legacy' });
            assert.equal(msg.content, 'plain text answer');
            assert.equal(msg.reasoningContent, '');
            assert.deepEqual(msg.toolCalls, []);
        });

        it('chat() returns readable content end to end', async (t) => {
            if (!model.chatTemplate) return t.skip('model has no chat template');
            const result = await model.chat({
                messages: [{ role: 'user', content: 'In one short sentence, what is the moon?' }],
                nPredict: 48, temperature: 0, resetContext: true,
            });
            assert.equal(typeof result.format, 'string');
            assertReadableText(result.content || result.reasoningContent, 'chat() content');
        });

        it('jsonSchema flows through the template into a working grammar', async (t) => {
            if (!model.chatTemplate) return t.skip('model has no chat template');
            let rendered;
            try {
                rendered = model.applyChatTemplateJinja(
                    [{ role: 'user', content: 'Give me a color as JSON.' }],
                    { jsonSchema: {
                        type: 'object',
                        properties: { color: { type: 'string' } },
                        required: ['color'],
                    } });
            } catch (e) {
                return t.skip(`template cannot render schemas: ${e.message}`);
            }
            assert.ok(rendered.grammar, 'no grammar emitted for jsonSchema');
            const out = await collect(gen(rendered.prompt, {
                nPredict: 64, grammar: rendered.grammar,
                stop: rendered.additionalStops,
            }));
            const parsed = JSON.parse(out);
            assert.equal(typeof parsed.color, 'string');
        });
    });

    describe('context options', () => {
        it('quantized KV cache + forced flash attention generate readable text', async () => {
            const scratch = await LlamaModel.load(MODEL, {
                nCtx: 256, cacheTypeK: 'q8_0', cacheTypeV: 'q8_0', flashAttention: 'on',
            });
            try {
                const out = await collect(scratch.generate('The sun is', {
                    nPredict: 16, resetContext: true, temperature: 0,
                }));
                assertReadableText(out, 'quantized KV generation');
            } finally {
                scratch.dispose();
            }
        });
    });

    describe('reasoning budget', () => {
        it('force-closes the thinking block when the budget runs out', async () => {
            // Pre-arm via the prefill so counting starts at token 0 even on
            // non-thinking models; the forced end tag must then appear early.
            const out = await collect(gen('Puzzle: 2+2=? Work it out.', {
                nPredict: 64,
                reasoningBudget: 8,
                reasoningBudgetStartTag: '<think>',
                reasoningBudgetEndTags: ['</think>'],
                reasoningBudgetPrefill: '<think>',
            }));
            assert.ok(out.includes('</think>'),
                `forced </think> not found in: ${JSON.stringify(out)}`);
        });
    });

    describe('session state', () => {
        it('saveState/loadState round-trips the token record', async () => {
            const statePath = path.join(os.tmpdir(), `llama-node-test-state-${process.pid}.bin`);
            try {
                const prompt = 'The capital of France is';
                await collect(gen(prompt, { nPredict: 4, temperature: 0 }));
                const tokens = model.tokenize(prompt);
                await model.saveState(statePath, { tokens });
                assert.ok(fs.statSync(statePath).size > 0);

                // Clobber the KV, then restore.
                await collect(gen('Count to five:', { nPredict: 4 }));
                const restored = await model.loadState(statePath);
                assert.deepEqual(restored, tokens);
            } finally {
                fs.rmSync(statePath, { force: true });
            }
        });

        it('loadState rejects a nonexistent file', async () => {
            await assert.rejects(model.loadState('/nonexistent/state.bin'),
                /llama_state_load_file failed/);
        });
    });

    describe('cancellation', () => {
        it('opts.signal aborts with AbortError and the model stays usable', async () => {
            const ac = new AbortController();
            const run = (async () => {
                for await (const _ of model.generate('Write a very long story.', {
                    nPredict: 512, resetContext: true, signal: ac.signal,
                })) {
                    ac.abort(); // cancel after the first token
                }
            })();
            await assert.rejects(run, (err) => err.name === 'AbortError');

            const out = await collect(gen('Hi', { nPredict: 4 }));
            assert.equal(typeof out, 'string');
        });

        it('signal.reason propagates per the AbortSignal contract', async () => {
            const ac = new AbortController();
            const myErr = new Error('custom cancel reason');
            const run = (async () => {
                for await (const _ of model.generate('Another long story.', {
                    nPredict: 512, resetContext: true, signal: ac.signal,
                })) {
                    ac.abort(myErr);
                }
            })();
            await assert.rejects(run, (err) => err === myErr);
        });

        it('model.abort() ends all in-flight generations without an error', async () => {
            // Documented contract: only opts.signal throws; global abort()
            // ends the streams silently (possibly truncated).
            const run = async () => {
                let n = 0;
                for await (const _ of model.generate('Tell me everything about cats.', {
                    nPredict: 512, resetContext: true,
                })) {
                    if (++n === 1) model.abort();
                }
                return n;
            };
            const produced = await run();
            assert.ok(produced >= 1);
        });

        it('concurrent generate() calls queue and both complete', async () => {
            const [a, b] = await Promise.all([
                collect(gen('Say alpha:', { nPredict: 4 })),
                collect(gen('Say bravo:', { nPredict: 4 })),
            ]);
            assert.equal(typeof a, 'string');
            assert.equal(typeof b, 'string');
        });
    });

    describe('embeddings', () => {
        it('embed() rejects when the model was not opened with embeddings: true', async () => {
            await assert.rejects(model.embed('hello'), /embeddings: true/);
        });

        it('embed() returns a Float32Array of n_embd length', async () => {
            const embedModel = await LlamaModel.load(MODEL, { nCtx: 256, embeddings: true });
            try {
                const vec = await embedModel.embed('hello world');
                assert.ok(vec instanceof Float32Array);
                assert.equal(vec.length, embedModel.getModelInfo().embeddingSize);
                assert.ok(vec.some((x) => x !== 0), 'embedding is all zeros');
            } finally {
                embedModel.dispose();
            }
        });
    });

    describe('LoRA', () => {
        it('rejects a nonexistent adapter path', async () => {
            await assert.rejects(
                LlamaModel.load(MODEL, { nCtx: 256, loras: ['/nonexistent/adapter.gguf'] }),
                /Failed to load LoRA adapter/);
        });

        const itLora = LORA ? it : it.skip;
        itLora('loads and generates with a real adapter (LLAMA_NODE_LORA_PATH)', async () => {
            const loraModel = await LlamaModel.load(MODEL, {
                nCtx: 256, loras: [{ path: LORA, scale: 1.0 }],
            });
            try {
                const out = await collect(loraModel.generate('Hello', {
                    nPredict: 4, resetContext: true,
                }));
                assert.equal(typeof out, 'string');
            } finally {
                loraModel.dispose();
            }
        });
    });

    describe('inspect()', () => {
        it('reads GGUF metadata without loading weights', async () => {
            const result = await inspect(MODEL);
            assert.ok(result.version >= 2);
            assert.equal(typeof result.metadata['general.architecture'], 'string');
            assert.ok(Array.isArray(result.tensors) && result.tensors.length > 0);
            assert.equal(typeof result.tensors[0].type, 'string');
        });

    });

    describe('lifecycle', () => {
        it('pool loads, generates, and unload disposes', async () => {
            const pool = new LlamaModelPool();
            pool.register('m', MODEL, { nCtx: 256 });
            const out = await collect(pool.generate('m', 'Hello', {
                nPredict: 4, resetContext: true,
            }));
            assert.equal(typeof out, 'string');
            const instance = pool.load('m');
            pool.unload('m');
            assert.throws(() => instance.tokenize('x'), /has been disposed/);
            pool.dispose();
        });

        it('use-after-dispose throws a clear error', async () => {
            const scratch = await LlamaModel.load(MODEL, { nCtx: 256 });
            scratch.dispose();
            scratch.dispose(); // idempotent
            assert.throws(() => scratch.tokenize('x'), /has been disposed/);
            assert.throws(() => scratch.getModelInfo(), /has been disposed/);
            assert.throws(() => scratch.memoryUsage(), /has been disposed/);
            await assert.rejects(collect(scratch.generate('x')), /has been disposed/);
        });
    });
});
