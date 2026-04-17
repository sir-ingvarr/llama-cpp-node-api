#!/usr/bin/env node
'use strict';

// Propagates the root package.json version to every platform subpackage
// under npm/<platform>-<arch>/ and to the matching entries in the root's
// optionalDependencies block.
//
// Usage:
//   node scripts/sync-versions.js               # sync from root package.json
//   node scripts/sync-versions.js 0.10.0        # set root + all subpackages to 0.10.0
//
// The script is idempotent. Files that already match are left untouched
// (preserving mtime); only files with a differing version are rewritten.

const fs   = require('fs');
const path = require('path');

const ROOT     = path.resolve(__dirname, '..');
const ROOT_PKG = path.join(ROOT, 'package.json');
const NPM_DIR  = path.join(ROOT, 'npm');
const SCOPE    = '@llama-cpp-node-api';

function readJson(file) {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function writeJson(file, obj) {
    // Preserve 4-space indent (matches the existing style in this repo)
    // and keep a trailing newline so git diffs stay clean.
    fs.writeFileSync(file, JSON.stringify(obj, null, 4) + '\n');
}

function main() {
    const root = readJson(ROOT_PKG);
    const target = process.argv[2] || root.version;

    if (!/^\d+\.\d+\.\d+(-[\w.]+)?$/.test(target)) {
        console.error(`sync-versions: invalid version "${target}"`);
        process.exit(1);
    }

    let changes = 0;

    // 1. Root package: version + optionalDependencies entries for each scope subpackage.
    if (root.version !== target) {
        console.log(`root: ${root.version} → ${target}`);
        root.version = target;
        changes++;
    }

    const optDeps = root.optionalDependencies || {};
    for (const key of Object.keys(optDeps)) {
        if (!key.startsWith(SCOPE + '/')) continue;
        if (optDeps[key] !== target) {
            console.log(`root optionalDependencies["${key}"]: ${optDeps[key]} → ${target}`);
            optDeps[key] = target;
            changes++;
        }
    }

    if (changes > 0) {
        root.optionalDependencies = optDeps;
        writeJson(ROOT_PKG, root);
    }

    // 2. Each platform subpackage.
    if (!fs.existsSync(NPM_DIR)) {
        console.error(`sync-versions: missing ${NPM_DIR}`);
        process.exit(1);
    }

    const platforms = fs.readdirSync(NPM_DIR, { withFileTypes: true })
        .filter(e => e.isDirectory())
        .map(e => e.name);

    for (const platform of platforms) {
        const file = path.join(NPM_DIR, platform, 'package.json');
        if (!fs.existsSync(file)) continue;

        const pkg = readJson(file);
        if (pkg.version === target) continue;

        console.log(`npm/${platform}: ${pkg.version} → ${target}`);
        pkg.version = target;
        writeJson(file, pkg);
        changes++;
    }

    if (changes === 0) {
        console.log(`sync-versions: everything already at ${target}, nothing to do.`);
    } else {
        console.log(`sync-versions: updated ${changes} file(s) to ${target}.`);
    }
}

main();
