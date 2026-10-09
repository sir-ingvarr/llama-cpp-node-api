// Smoke test for the 0.15.0 feature batch: new samplers (DRY, XTC, typical-p,
// top-n-sigma, mirostat, dynatemp, adaptive-p), session state save/load,
// reasoning budget (synthetic tags on a non-thinking model), llamaVersion(),
// loadMode, suppressTokens, and LoRA error plumbing.
//
// Usage: node smoke_features.js /path/to/model.gguf
'use strict';

const os   = require('os');
const path = require('path');
const fs   = require('fs');
const { LlamaModel, llamaVersion } = require('./js');

const modelPath = process.argv[2];
if (!modelPath) {
    console.error('usage: node smoke_features.js /path/to/model.gguf');
    process.exit(1);
}

async function collect(gen) {
    let text = '';
    for await (const tok of gen) text += tok;
    return text;
}

(async () => {
    console.log(`llama.cpp version: ${llamaVersion()}`);

    console.log('\n=== load with loadMode: "mmap" ===');
    const model = await LlamaModel.load(modelPath, { nCtx: 512, loadMode: 'mmap' });

    const info = model.getModelInfo();
    if (!Array.isArray(info.suppressTokens)) {
        throw new Error('getModelInfo().suppressTokens missing');
    }
    console.log(`  ok (suppressTokens: ${info.suppressTokens.length} entries)`);

    const prompt = 'Write one short sentence about the sea.';
    const samplerConfigs = {
        'typicalP=0.9':             { typicalP: 0.9 },
        'xtc p=0.5 t=0.1':          { xtcProbability: 0.5, xtcThreshold: 0.1 },
        'topNSigma=1.5':            { topNSigma: 1.5 },
        'DRY multiplier=0.8':       { dryMultiplier: 0.8 },
        'mirostat v2':              { mirostat: 2 },
        'dynatemp range=0.5':       { dynatempRange: 0.5 },
        'adaptive-p target=0.3':    { adaptivePTarget: 0.3 },
    };
    for (const [name, cfg] of Object.entries(samplerConfigs)) {
        const text = await collect(model.generate(prompt, {
            nPredict: 24, resetContext: true, seed: 42, ...cfg,
        }));
        if (!text.trim()) throw new Error(`${name}: empty output`);
        console.log(`\n=== ${name} ===\n  ${text.trim().split('\n')[0]}`);
    }

    console.log('\n=== saveState / loadState round-trip ===');
    const statePath = path.join(os.tmpdir(), `llama-node-state-${process.pid}.bin`);
    const seedPrompt = 'The capital of France is';
    const first = await collect(model.generate(seedPrompt, {
        nPredict: 4, resetContext: true, temperature: 0,
    }));
    const promptTokens = model.tokenize(seedPrompt);
    await model.saveState(statePath, { tokens: promptTokens });
    // Clobber the KV with an unrelated generation, then restore.
    await collect(model.generate('Count to five:', { nPredict: 8, resetContext: true }));
    const restoredTokens = await model.loadState(statePath);
    if (restoredTokens.length !== promptTokens.length) {
        throw new Error(`loadState returned ${restoredTokens.length} tokens, expected ${promptTokens.length}`);
    }
    console.log(`  ok (state file ${fs.statSync(statePath).size} bytes, ${restoredTokens.length} tokens round-tripped)`);
    fs.unlinkSync(statePath);

    console.log('\n=== reasoning budget (synthetic <think> tags) ===');
    // Non-thinking model: pre-arm the state machine via the prefill so it
    // counts from token 0, then check the end tag was force-emitted early.
    const budgetText = await collect(model.generate('Puzzle: 2+2=? Work it out.', {
        nPredict: 64, resetContext: true,
        reasoningBudget: 8,
        reasoningBudgetStartTag: '<think>',
        reasoningBudgetEndTags: ['</think>'],
        reasoningBudgetPrefill: '<think>',
    }));
    const endIdx = budgetText.indexOf('</think>');
    if (endIdx === -1) {
        throw new Error(`forced </think> not found in: ${JSON.stringify(budgetText)}`);
    }
    console.log(`  ok (</think> forced at char ${endIdx}: ${JSON.stringify(budgetText.slice(0, endIdx + 9))})`);

    console.log('\n=== LoRA error plumbing (bad adapter path rejects) ===');
    let loraErr = null;
    try {
        await LlamaModel.load(modelPath, {
            nCtx: 256,
            loras: [{ path: '/nonexistent/adapter.gguf', scale: 0.5 }],
        });
    } catch (e) { loraErr = e; }
    if (!loraErr || !/LoRA/.test(loraErr.message)) {
        throw new Error(`expected LoRA load error, got: ${loraErr}`);
    }
    console.log(`  ok (${loraErr.message})`);

    model.dispose();
    console.log('\nAll feature checks passed.');
})().catch(err => {
    console.error(err);
    process.exit(1);
});
