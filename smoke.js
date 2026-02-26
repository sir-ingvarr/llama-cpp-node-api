'use strict';

const { LlamaModel, LlamaModelPool } = require('./js/index.js');

const MODEL_PATH = process.argv[2] || '/Users/igorberezin/AI/models/llm/llama-3.1-8B-uncensored.gguf';

// Llama 3 instruct template
const userPrompt = (text) =>
    '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n' +
    text + '<|eot_id|>\n' +
    '<|start_header_id|>assistant<|end_header_id|>\n';

async function run(label, model, prompt, opts) {
    process.stdout.write(`\n=== ${label} ===\n`);
    for await (const token of model.generate(prompt, opts)) {
        process.stdout.write(token);
    }
    process.stdout.write('\n');
}

(async () => {
    const model = new LlamaModel(MODEL_PATH, { nGpuLayers: 99, nCtx: 2048 });

    // 1. Baseline
    await run('baseline (temperature=0)', model,
        userPrompt('What is 2 + 2?'),
        { nPredict: 64, temperature: 0.0, resetContext: true });

    // 2. minP — filters out tokens below minP * p(top token)
    await run('minP=0.05 (replaces top-p for long-tail filtering)', model,
        userPrompt('Name three primary colours.'),
        { nPredict: 64, temperature: 0.7, topP: 1.0, topK: 0, minP: 0.05, resetContext: true });

    // 3. repeatPenalty — discourages repeating recent tokens
    await run('repeatPenalty=1.3, repeatLastN=64', model,
        userPrompt('Write a short sentence without repeating any word.'),
        { nPredict: 80, temperature: 0.8, repeatPenalty: 1.3, repeatLastN: 64, resetContext: true });

    // 4. Stop sequences — halts at the first match, not included in output
    await run('stop=["\n"] — single line answer only', model,
        userPrompt('What is the capital of France? Reply in one line.'),
        { nPredict: 128, temperature: 0.0, stop: ['\n', '<|eot_id|>'], resetContext: true });

    // 5. LlamaModelPool — multi-model routing
    const pool = new LlamaModelPool();
    pool.register('main', MODEL_PATH, { nGpuLayers: 99, nCtx: 1024 });

    process.stdout.write('\n=== LlamaModelPool.generate ===\n');
    for await (const token of pool.generate('main',
        userPrompt('Say hello in exactly three words.'),
        { nPredict: 32, temperature: 0.0, stop: ['\n', '<|eot_id|>'] })) {
        process.stdout.write(token);
    }
    process.stdout.write('\n');
    pool.dispose();

    model.dispose();
})();
