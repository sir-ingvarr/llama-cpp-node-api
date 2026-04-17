'use strict';

// Measures event-loop responsiveness during streaming generation.
// Schedules a 100 ms interval; records the wall-clock drift of each tick.
// If the loop is stalled, ticks fire late and drift is large.

const { LlamaModel } = require('./js/index.js');

const MODEL_PATH = process.argv[2];
if (!MODEL_PATH) {
    console.error('usage: node smoke_event_loop.js /path/to/model.gguf');
    process.exit(1);
}

const userPrompt = (text) =>
    '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n' +
    text + '<|eot_id|>\n' +
    '<|start_header_id|>assistant<|end_header_id|>\n';

(async () => {
    const model = new LlamaModel(MODEL_PATH, { nGpuLayers: 99, nCtx: 2048 });

    const INTERVAL_MS = 100;
    const drifts = [];
    let last = Date.now();

    const timer = setInterval(() => {
        const now = Date.now();
        const drift = now - last - INTERVAL_MS;
        drifts.push(drift);
        last = now;
    }, INTERVAL_MS);

    const start = Date.now();
    let tokens = 0;
    for await (const _ of model.generate(
        userPrompt('Write a 200-word essay about the moon.'),
        { nPredict: 300, temperature: 0.0, resetContext: true }
    )) {
        tokens++;
    }
    const elapsed = Date.now() - start;
    clearInterval(timer);

    const max    = Math.max(...drifts);
    const over50 = drifts.filter(d => d > 50).length;
    const over200 = drifts.filter(d => d > 200).length;
    const avg    = drifts.reduce((a, b) => a + b, 0) / drifts.length;

    console.log(`generation: ${tokens} tokens in ${elapsed} ms`);
    console.log(`interval ticks: ${drifts.length}`);
    console.log(`drift avg=${avg.toFixed(1)} ms, max=${max} ms`);
    console.log(`ticks with drift > 50 ms:  ${over50}`);
    console.log(`ticks with drift > 200 ms: ${over200}`);

    if (max > 500) {
        console.error('FAIL — event loop stalled > 500 ms during streaming');
        process.exitCode = 1;
    } else {
        console.log('OK — event loop remained responsive during streaming');
    }

    model.dispose();
})();
