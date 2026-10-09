#!/usr/bin/env node
// Test entry point: resolves the functional-test model path, then spawns
// `node --test`.
//
// Model path resolution order:
//   1. LLAMA_NODE_MODEL_PATH environment variable
//   2. test/.model-path cache file (written after a successful prompt)
//   3. interactive prompt (works from a git hook too — reads /dev/tty)
//
// An empty answer at the prompt (or no terminal, e.g. CI) runs the suite
// with the functional tests self-skipped.
//
// Usage: node test/run.js [unit|functional]
'use strict';

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const ENV_VAR = 'LLAMA_NODE_MODEL_PATH';
const CACHE_FILE = path.join(__dirname, '.model-path');

function readCache() {
    try {
        const p = fs.readFileSync(CACHE_FILE, 'utf8').trim();
        if (p && fs.existsSync(p)) return p;
    } catch (_) {}
    return null;
}

function writeCache(p) {
    try { fs.writeFileSync(CACHE_FILE, p + '\n'); } catch (_) {}
}

// Prompt on the controlling terminal. In a git hook stdin is not a TTY,
// but /dev/tty still reaches the user's terminal.
function prompt(question) {
    return new Promise((resolve) => {
        let input = process.stdin;
        let output = process.stderr;
        let ttyFd = null;
        if (!process.stdin.isTTY) {
            try {
                ttyFd = fs.openSync(process.platform === 'win32' ? '\\\\.\\CONIN$' : '/dev/tty', 'r+');
                const tty = require('tty');
                input = new tty.ReadStream(ttyFd);
                output = new tty.WriteStream(ttyFd);
            } catch (_) {
                resolve(null); // no terminal at all (CI) — don't block
                return;
            }
        }
        const rl = require('readline').createInterface({ input, output });
        rl.question(question, (answer) => {
            rl.close();
            if (ttyFd !== null) { try { fs.closeSync(ttyFd); } catch (_) {} }
            resolve(answer.trim());
        });
    });
}

async function resolveModelPath() {
    const fromEnv = process.env[ENV_VAR];
    if (fromEnv) {
        if (!fs.existsSync(fromEnv)) {
            console.error(`${ENV_VAR} points to a missing file: ${fromEnv}`);
            process.exit(1);
        }
        return fromEnv;
    }

    const cached = readCache();
    if (cached) {
        console.error(`Using cached model path: ${cached}  (from test/.model-path)`);
        return cached;
    }

    const answer = await prompt(
        `${ENV_VAR} is not set.\n` +
        'Absolute path to a GGUF model for the functional tests\n' +
        '(Enter to skip the functional suite): ');
    if (!answer) {
        console.error('No model path — functional tests will be skipped.');
        return null;
    }
    if (!path.isAbsolute(answer) || !fs.existsSync(answer)) {
        console.error(`Not an existing absolute path: ${answer}`);
        process.exit(1);
    }
    writeCache(answer);
    return answer;
}

async function main() {
    const suite = process.argv[2];
    const UNIT = path.join(__dirname, 'unit.test.js');
    const FUNCTIONAL = path.join(__dirname, 'functional.test.js');
    // Explicit file list — never pass the directory: node --test treats
    // every .js file inside a dir named test/ as a test file, which would
    // execute this runner recursively.
    const targets = {
        unit:       [UNIT],
        functional: [FUNCTIONAL],
    }[suite] ?? [UNIT, FUNCTIONAL];

    const env = { ...process.env };
    if (suite !== 'unit') {
        const modelPath = await resolveModelPath();
        if (modelPath) env[ENV_VAR] = modelPath;
    }

    const result = spawnSync(process.execPath, ['--test', ...targets], {
        stdio: 'inherit',
        env,
    });
    process.exit(result.status ?? 1);
}

main();
