// Unit tests — no model file required. Everything here must run in CI
// without downloading anything: API surface, option validation, and error
// paths that fail before (or without) touching model weights.
//
// Run: npm test            (runs unit + functional; functional self-skips
//                           when LLAMA_NODE_MODEL_PATH is not set)
'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const os = require('os');
const path = require('path');

const fs = require('fs');

const api = require('../js');
const { LlamaModel, LlamaModelPool, quantize, quantizeFtypes,
        inspect, clearInspectCache, llamaVersion } = api;

const MISSING_GGUF = path.join(os.tmpdir(), 'llama-node-test-no-such-model.gguf');

// Minimal valid GGUF v3 file: magic + version + 0 tensors + 0 KV pairs.
// Lets the whole inspect pipeline (native header parser, snapshot, JS
// marshaling, memoization) run in CI without downloading a model.
function writeGgufFixture(file) {
    const buf = Buffer.alloc(4 + 4 + 8 + 8);
    buf.write('GGUF', 0, 'ascii');
    buf.writeUInt32LE(3, 4);          // version
    buf.writeBigUInt64LE(0n, 8);      // tensor count
    buf.writeBigUInt64LE(0n, 16);     // metadata KV count
    fs.writeFileSync(file, buf);
    return file;
}

describe('module surface', () => {
    it('exports the documented API', () => {
        assert.equal(typeof LlamaModel, 'function');
        assert.equal(typeof LlamaModel.load, 'function');
        assert.equal(typeof LlamaModelPool, 'function');
        assert.equal(typeof quantize, 'function');
        assert.equal(typeof quantizeFtypes, 'function');
        assert.equal(typeof inspect, 'function');
        assert.equal(typeof clearInspectCache, 'function');
        assert.equal(typeof llamaVersion, 'function');
    });

    it('llamaVersion() returns a non-empty version string', () => {
        const v = llamaVersion();
        assert.equal(typeof v, 'string');
        assert.ok(v.length > 0);
        assert.match(v, /\d/);
    });

    it('quantizeFtypes() lists the CLI-style ftype names', () => {
        const names = quantizeFtypes();
        assert.ok(Array.isArray(names));
        for (const expected of ['F16', 'Q4_K_M', 'Q8_0', 'IQ4_NL', 'Q2_0']) {
            assert.ok(names.includes(expected), `missing ftype ${expected}`);
        }
    });
});

describe('constructor validation (fails before any native load)', () => {
    it('rejects a missing / empty model path', () => {
        assert.throws(() => new LlamaModel(), TypeError);
        assert.throws(() => new LlamaModel(''), TypeError);
        assert.throws(() => new LlamaModel(42), TypeError);
    });

    it('rejects an unknown poolingType', () => {
        assert.throws(() => new LlamaModel('x.gguf', { poolingType: 'bogus' }),
            /unknown poolingType/);
    });

    it('rejects unknown KV cache types', () => {
        assert.throws(() => new LlamaModel('x.gguf', { cacheTypeK: 'q9_9' }),
            /unknown cacheTypeK/);
        assert.throws(() => new LlamaModel('x.gguf', { cacheTypeV: 'nope' }),
            /unknown cacheTypeV/);
    });

    it('rejects an unknown flashAttention value', () => {
        assert.throws(() => new LlamaModel('x.gguf', { flashAttention: 'maybe' }),
            /unknown flashAttention/);
    });

    it('rejects an unknown loadMode', () => {
        assert.throws(() => new LlamaModel('x.gguf', { loadMode: 'turbo' }),
            /unknown loadMode/);
    });

    it('rejects malformed loras entries', () => {
        assert.throws(() => new LlamaModel('x.gguf', { loras: 'adapter.gguf' }),
            /loras must be an array/);
        assert.throws(() => new LlamaModel('x.gguf', { loras: [{ scale: 0.5 }] }),
            /path string or \{ path, scale \}/);
        assert.throws(() => new LlamaModel('x.gguf', { loras: [1] }),
            /path string or \{ path, scale \}/);
    });

    it('LlamaModel.load rejects the same validation errors asynchronously', async () => {
        await assert.rejects(LlamaModel.load(''), TypeError);
        await assert.rejects(LlamaModel.load('x.gguf', { poolingType: 'bogus' }),
            /unknown poolingType/);
    });

    it('LlamaModel.load rejects a nonexistent model file', async () => {
        await assert.rejects(LlamaModel.load(MISSING_GGUF), /Failed to load model/);
    });
});

describe('inspect()', () => {
    const fixture = path.join(os.tmpdir(), `llama-node-test-fixture-${process.pid}.gguf`);

    it('validates its path argument', async () => {
        await assert.rejects(inspect(), TypeError);
        await assert.rejects(inspect(''), TypeError);
    });

    it('rejects a nonexistent file', async () => {
        await assert.rejects(inspect(MISSING_GGUF));
    });

    it('parses a minimal GGUF header end to end', async () => {
        writeGgufFixture(fixture);
        try {
            const result = await inspect(fixture, { cache: false });
            assert.equal(result.version, 3);
            assert.equal(typeof result.alignment, 'number');
            assert.equal(typeof result.dataOffset, 'bigint');
            assert.deepEqual(result.metadata, {});
            assert.deepEqual(result.tensors, []);
        } finally {
            fs.rmSync(fixture, { force: true });
        }
    });

    it('rejects a corrupt file (bad magic)', async () => {
        fs.writeFileSync(fixture, Buffer.from('NOTAGGUFFILE....'));
        try {
            await assert.rejects(inspect(fixture, { cache: false }));
        } finally {
            fs.rmSync(fixture, { force: true });
        }
    });

    it('memoizes by path and invalidates on content change', async () => {
        writeGgufFixture(fixture);
        try {
            clearInspectCache();
            const a = await inspect(fixture);
            const b = await inspect(fixture);
            assert.equal(a, b, 'cache hit should return the identical object');
            const fresh = await inspect(fixture, { cache: false });
            assert.notEqual(a, fresh, 'cache: false should bypass the cache');

            // Change size + mtime → the cached entry must be dropped.
            fs.appendFileSync(fixture, Buffer.alloc(64));
            const after = await inspect(fixture);
            assert.notEqual(a, after, 'cache must invalidate when the file changes');
        } finally {
            fs.rmSync(fixture, { force: true });
            clearInspectCache();
        }
    });
});

describe('quantize()', () => {
    it('rejects missing paths and missing ftype', async () => {
        await assert.rejects(quantize(), TypeError);
        await assert.rejects(quantize('in.gguf'), TypeError);
        await assert.rejects(quantize('in.gguf', 'out.gguf'), /ftype is required/);
        await assert.rejects(quantize('in.gguf', 'out.gguf', {}), /ftype is required/);
    });

    it('rejects an unknown ftype name before touching any file', async () => {
        await assert.rejects(quantize('in.gguf', 'out.gguf', { ftype: 'Q99_X' }),
            /Unknown ftype/);
    });

    it('surfaces a native failure for a nonexistent input', async () => {
        const out = path.join(os.tmpdir(), `llama-node-test-quant-${process.pid}.gguf`);
        await assert.rejects(quantize(MISSING_GGUF, out, { ftype: 'Q8_0' }),
            /llama_model_quantize failed/);
    });
});

describe('LlamaModelPool registry', () => {
    it('register / duplicate / unknown-name errors', () => {
        const pool = new LlamaModelPool();
        pool.register('m', '/nonexistent.gguf');
        assert.throws(() => pool.register('m', '/other.gguf'), /already registered/);
        assert.throws(() => pool.load('missing'), /unknown model 'missing'/);
        assert.throws(() => pool.unload('missing'), /unknown model 'missing'/);
        pool.dispose();
    });

    it('generate() on an unknown name rejects asynchronously', async () => {
        const pool = new LlamaModelPool();
        await assert.rejects(
            (async () => { for await (const _ of pool.generate('nope', 'x')) {} })(),
            /unknown model 'nope'/);
        pool.dispose();
    });

    it('dispose is idempotent and guards later use', () => {
        const pool = new LlamaModelPool();
        pool.dispose();
        pool.dispose();
        assert.throws(() => pool.register('m', 'x.gguf'), /has been disposed/);
        assert.throws(() => pool.load('m'), /has been disposed/);
    });
});
