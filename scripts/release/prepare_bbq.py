"""Restore the exact BBQ inputs from pinned upstream files and UID manifests.

Uses the Python standard library only. Existing files with different bytes are
never overwritten. Dataset text is downloaded from upstream, not stored in Git.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'reproducibility/bbq/manifest.json'


def checksum(blob):
    return hashlib.sha256(blob).hexdigest()


def jsonl(values, newline):
    return ''.join(json.dumps(r, ensure_ascii=False) + newline for r in values).encode('utf-8')


def save_verified(path, blob, expected):
    if checksum(blob) != expected:
        raise ValueError(f'Reconstructed bytes differ from the frozen manifest: {path.name}')
    if path.exists():
        if path.read_bytes() != blob:
            raise FileExistsError(f'Refusing to replace existing different data: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.download')
    temporary.write_bytes(blob)
    temporary.replace(path)


def fetch(spec, raw_root, offline=False):
    path = raw_root / spec['path']
    if path.exists():
        blob = path.read_bytes()
        if checksum(blob) != spec['sha256']:
            raise ValueError(f'Upstream cache checksum mismatch: {path}')
        return blob
    if offline:
        raise FileNotFoundError(f'Offline source missing: {path}')
    errors = []
    for url in spec['urls']:
        for attempt in range(2):
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'EGUS-reproduction/1.0'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    blob = response.read()
                save_verified(path, blob, spec['sha256'])
                return blob
            except (OSError, ValueError) as error:
                errors.append(f'{type(error).__name__}: {error}')
                if attempt == 0:
                    time.sleep(1)
    raise RuntimeError(f'Unable to retrieve {spec["path"]}: {errors[-1]}')


def prepare(raw_root, data_root, source_root, *, offline=False):
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    inputs = {name: fetch(spec, raw_root, offline) for name, spec in manifest['upstream'].items()}
    metadata = {}
    for row in csv.DictReader(io.StringIO(inputs['metadata'].decode('utf-8-sig'), newline='')):
        metadata.setdefault((row['category'], int(row['example_id'])), row)
    source = []
    for category in manifest['categories']:
        for line in inputs[category].decode('utf-8').splitlines():
            row = json.loads(line)
            key = (row['category'], int(row['example_id']))
            target = metadata[key]['target_loc']
            row['target_loc'] = None if target in ('NA', '') else int(target)
            row['uid'] = f'{key[0]}:{key[1]}'
            row['unknown_index'] = next(i for i in range(3)
                                       if row['answer_info'][f'ans{i}'][1].lower() == 'unknown')
            source.append(row)
    if len(source) != manifest['source_rows']:
        raise ValueError('Unexpected source corpus size')
    save_verified(source_root / 'bbq.jsonl', jsonl(source, manifest['jsonl_newline']), manifest['source_bbq_sha256'])
    lookup = {r['uid']: r for r in source}
    if len(lookup) != len(source):
        raise ValueError('Duplicate source UIDs')
    mapping = json.loads((ROOT / 'configs/bbq_template_families.json').read_text(encoding='utf-8'))['mapping']
    mmlu = [json.loads(line) for line in inputs['mmlu'].decode('utf-8').splitlines() if line.strip()]
    for filename, spec in manifest['files'].items():
        values = []
        if spec['kind'] == 'bbq':
            for uid in spec['uids']:
                row = dict(lookup[uid])
                family = f'{row["category"]}:{row["question_index"]}'
                row['family'] = mapping.get(family, family)
                values.append(row)
        else:
            for uid in spec['uids']:
                row = dict(mmlu[int(uid.split(':')[1])])
                row.update(utility=True, uid=uid, label=row['answer'], family=uid)
                values.append(row)
        save_verified(data_root / filename, jsonl(values, manifest['jsonl_newline']), spec['sha256'])
    training = (MANIFEST.parent / 'training.json').read_bytes()
    save_verified(data_root / 'training.json', training, manifest['training_sha256'])
    print(json.dumps({'verified': True, 'source_rows': len(source),
                      'prepared_files': len(manifest['files']) + 1,
                      'data_dir': str(data_root), 'source_dir': str(source_root)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'data/raw/egus_release')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/prepared/egus_bbq_v1')
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'data/prepared/egus_bbq_source')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    prepare(args.raw_dir, args.data_dir, args.source_dir, offline=args.offline)


if __name__ == '__main__':
    main()
