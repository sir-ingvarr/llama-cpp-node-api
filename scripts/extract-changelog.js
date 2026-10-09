#!/usr/bin/env node
'use strict';

// Extracts one version's section from CHANGELOG.md (Keep a Changelog format)
// so the release pipeline can use it as the GitHub release notes.
//
// Usage:
//   node scripts/extract-changelog.js 0.15.0 > release_notes.md
//
// Prefers the exact "## [<version>]" section. When it is missing (e.g. a
// CI-only patch release with no dedicated entry), falls back to the latest
// version section and says so in the output. Exits non-zero only when the
// changelog has no version sections at all.

const fs = require('fs');
const path = require('path');

const version = process.argv[2];
if (!version) {
    console.error('usage: extract-changelog.js <version>');
    process.exit(1);
}

const changelog = fs.readFileSync(path.join(__dirname, '..', 'CHANGELOG.md'), 'utf8');
const lines = changelog.split('\n');

// All version headings, in file order (newest first by convention).
const headingRe = /^## \[(\d+\.\d+\.\d+(?:-[\w.]+)?)\]/;
const sections = [];
for (let i = 0; i < lines.length; i++) {
    const m = headingRe.exec(lines[i]);
    if (m) sections.push({ version: m[1], line: i });
}
if (sections.length === 0) {
    console.error('CHANGELOG.md has no version sections at all.');
    process.exit(1);
}

let picked = sections.find((s) => s.version === version);
let fallback = false;
if (!picked) {
    picked = sections[0]; // latest available
    fallback = true;
    console.error(
        `CHANGELOG.md has no "## [${version}]" section — falling back to ` +
        `the latest entry, [${picked.version}].`);
}

const start = picked.line;
let end = lines.length;
for (let i = start + 1; i < lines.length; i++) {
    if (/^## \[/.test(lines[i])) { end = i; break; }
}

let body = lines.slice(start + 1, end).join('\n').trim();
if (!body) {
    console.error(`CHANGELOG.md section "## [${picked.version}]" is empty.`);
    process.exit(1);
}

if (fallback) {
    body = `_No dedicated changelog entry for ${version}; ` +
           `showing the notes for [${picked.version}]. ` +
           `See CHANGELOG.md for full history._\n\n` + body;
}

process.stdout.write(body + '\n');
