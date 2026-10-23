"""Reversible housekeeping; default is preview. Does not run experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'archive/2026-10-17_cleanup'
ITEMS = {
    'preview_evidence_validation_direct.py': 'Interim preview; final summarize_evidence_validation.py supersedes it.',
    'primary_evidence_validation_direct.py': 'Early statistics; final summary and verification recompute the same comparisons.',
    'migrate_fairsteer_improve_v1.py': 'Completed one-time migration, not a recurring experiment stage.',
    'probe_local_replication_models.py': 'Completed initial model feasibility survey; retain for provenance.',
    'smoke_identity_replication.py': 'Completed standalone smoke run; retained main experiment has its own smoke stage.',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    manifest = json.loads((ARCHIVE / 'manifest.json').read_text(encoding='utf-8'))
    for item in manifest['archived']:
        assert not (ROOT / item['original']).exists(), item['original']
        assert digest(ROOT / item['archived']) == item['sha256'], item['archived']
    for name, expected in manifest['retained_source_hashes'].items():
        assert digest(ROOT / name) == expected, name
    assert digest(ARCHIVE / 'README.before_cleanup.md') == manifest['previous_readme_sha256']
    print(json.dumps({'archive_verified': len(manifest['archived']),
                      'unchanged_source_files': len(manifest['retained_source_hashes']),
                      'previous_readme_preserved': True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        verify()
        return
    if not args.apply:
        print(json.dumps(ITEMS, indent=2))
        return
    if (ARCHIVE / 'manifest.json').exists():
        verify()
        return
    originals = {f'scripts/{name}' for name in ITEMS}
    retained = {}
    for folder, pattern in [('mechanism', '*.py'), ('scripts', '*.py'), ('tests', '*.py'), ('configs', '*.toml')]:
        for path in (ROOT / folder).rglob(pattern):
            relative = path.relative_to(ROOT).as_posix()
            if relative not in originals:
                retained[relative] = digest(path)
    # All targets are explicit relative files. No recursive deletion or movement.
    planned = []
    for name, reason in ITEMS.items():
        source = ROOT / 'scripts' / name
        target = ARCHIVE / 'scripts' / (name + '.disabled')
        assert source.is_file() and not target.exists(), name
        assert target.resolve().is_relative_to(ARCHIVE.resolve())
        planned.append({'original': source.relative_to(ROOT).as_posix(),
                        'archived': target.relative_to(ROOT).as_posix(),
                        'sha256': digest(source), 'reason': reason})
    (ARCHIVE / 'scripts').mkdir(parents=True, exist_ok=True)
    previous = ROOT / 'README.md'
    shutil.copy2(previous, ARCHIVE / 'README.before_cleanup.md')
    manifest = {'date': '2026-10-17', 'archived': planned,
                'previous_readme_sha256': digest(previous),
                'retained_source_hashes': retained,
                'scope': 'No experiment source, config, data, model, prediction, fit or result modification.'}
    # Save recovery information before moving any file.
    (ARCHIVE / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for item in planned:
        shutil.move(str(ROOT / item['original']), str(ROOT / item['archived']))
    verify()


if __name__ == '__main__':
    main()
