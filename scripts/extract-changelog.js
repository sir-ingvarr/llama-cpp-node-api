#!/usr/bin/env node
'use strict';

// Extracts one version's section from CHANGELOG.md (Keep a Changelog format)
// so the release pipeline can use it as the GitHub release notes.
//
// Usage:
//   node scripts/extract-changelog.js 0.15.0 > release_notes.md
//
// Exits non-zero (failing the release before anything is published) when the
// section is missing or empty — which is exactly what happens if a version
// was bumped without renaming the [Unreleased] section.

const fs = require('fs');
const path = require('path');

const version = process.argv[2];
if (!version) {
    console.error('usage: extract-changelog.js <version>');
    process.exit(1);
}

const changelog = fs.readFileSync(path.join(__dirname, '..', 'CHANGELOG.md'), 'utf8');
const lines = changelog.split('\n');

const escaped = version.replace(/\./g, '\\.');
const heading = new RegExp(`^## \\[${escaped}\\]`);
const start = lines.findIndex((l) => heading.test(l));
if (start === -1) {
    console.error(
        `CHANGELOG.md has no "## [${version}]" section — rename [Unreleased] ` +
        'to the new version before tagging a release.');
    process.exit(1);
}

let end = lines.length;
for (let i = start + 1; i < lines.length; i++) {
    if (/^## \[/.test(lines[i])) { end = i; break; }
}

const body = lines.slice(start + 1, end).join('\n').trim();
if (!body) {
    console.error(`CHANGELOG.md section "## [${version}]" is empty.`);
    process.exit(1);
}

process.stdout.write(body + '\n');
