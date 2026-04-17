'use strict';

// Exercises the request queue + abort semantics introduced in task 3.
//
//   1. Two overlapping generate() calls on the same model complete without
//      the old "Already generating" throw.
//   2. opts.signal cancels only that call — a second generate() still finishes.
//   3. model.abort() cancels everything in flight.

const { LlamaModel } = require('./js/index.js');

const MODEL_PATH = process.argv[2];
if (!MODEL_PATH) {
    console.error('usage: node smoke_concurrent.js /path/to/model.gguf');
    process.exit(1);
}

const userPrompt = (text) =>
    '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n' +
    text + '<|eot_id|>\n' +
    '<|start_header_id|>assistant<|end_header_id|>\n';

async function collect(iter) {
    let s = '';
    let tokens = 0;
    try {
        for await (const t of iter) { s += t; tokens++; }
    } catch (e) {
        return { text: s, tokens, error: e };
    }
    return { text: s, tokens, error: null };
}

let failures = 0;
const fail = (msg) => { console.error('  FAIL —', msg); failures++; };

(async () => {
    const model = new LlamaModel(MODEL_PATH, { nGpuLayers: 99, nCtx: 4096 });

    // --- 1. Two overlapping generate() calls -------------------------------
    console.log('=== concurrent: two overlapping generate() calls ===');
    const a = collect(model.generate(userPrompt('Say "alpha".'),
        { nPredict: 16, temperature: 0.0, stop: ['<|eot_id|>'] }));
    const b = collect(model.generate(userPrompt('Say "bravo".'),
        { nPredict: 16, temperature: 0.0, stop: ['<|eot_id|>'] }));
    const [ra, rb] = await Promise.all([a, b]);
    console.log('  a:', JSON.stringify(ra.text.trim()), `(${ra.tokens} tok)`);
    console.log('  b:', JSON.stringify(rb.text.trim()), `(${rb.tokens} tok)`);
    if (ra.error || rb.error) fail(`unexpected error: ${ra.error || rb.error}`);

    // --- 2. opts.signal cancels only that call -----------------------------
    console.log('\n=== AbortSignal: cancel one, let the other complete ===');
    const ac = new AbortController();
    // Long nPredict + unbounded prompt so the slow call is guaranteed to
    // still be running when the timer fires.
    const slow = collect(model.generate(userPrompt(
        'Write a detailed essay about the history of the printing press. ' +
        'Include at least 10 paragraphs.'),
        { nPredict: 2000, temperature: 0.0, signal: ac.signal, resetContext: true }));
    const fast = collect(model.generate(userPrompt('Say "charlie".'),
        { nPredict: 16, temperature: 0.0, stop: ['<|eot_id|>'] }));

    setTimeout(() => ac.abort(), 400);

    const [rs, rf] = await Promise.all([slow, fast]);
    console.log('  slow:', rs.error ? `threw ${rs.error.name}` : 'completed',
                `(${rs.tokens} tok)`);
    console.log('  fast:', JSON.stringify(rf.text.trim()), `(${rf.tokens} tok)`);
    if (!rs.error || rs.error.name !== 'AbortError') {
        fail('expected AbortError on the cancelled call');
    }
    if (rs.tokens >= 2000) fail('slow call was not cancelled — ran to completion');
    if (rf.error) fail(`fast call should not have errored: ${rf.error}`);

    // --- 3. model.abort() stops everything ---------------------------------
    console.log('\n=== model.abort(): cancel every in-flight request ===');
    const x = collect(model.generate(userPrompt('Write a long essay about cats.'),
        { nPredict: 2000, temperature: 0.0, resetContext: true }));
    const y = collect(model.generate(userPrompt('Write a long essay about dogs.'),
        { nPredict: 2000, temperature: 0.0 }));
    setTimeout(() => model.abort(), 300);
    const [rx, ry] = await Promise.all([x, y]);
    console.log('  x:', `${rx.tokens} tok, error=${rx.error?.name ?? 'none'}`);
    console.log('  y:', `${ry.tokens} tok, error=${ry.error?.name ?? 'none'}`);
    // No AbortSignal was attached, so these should complete silently
    // (graceful stop, not throw).
    if (rx.error || ry.error) {
        fail('model.abort() should not surface an error without a signal');
    }
    if (rx.tokens >= 2000 && ry.tokens >= 2000) {
        fail('model.abort() did not stop anything');
    }

    model.dispose();
    if (failures) {
        console.error(`\n${failures} check(s) failed.`);
        process.exitCode = 1;
    } else {
        console.log('\nAll concurrency checks passed.');
    }
})();
