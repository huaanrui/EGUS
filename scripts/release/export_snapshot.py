"""Maintainer export from the original local research archive (no GPU work)."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))


def main():
    destination = ROOT / 'reproducibility/bbq'
    data = ROOT / 'data/prepared/evidence_validation_20261017_v1'
    raw = ROOT / 'data/raw/bbq/bea11bd97d79217245b5871acd247b9d6eb24598'
    revision = raw.name
    categories = sorted(p.stem for p in (raw / 'data').glob('*.jsonl'))
    assert len(categories) == 11
    upstream = {}
    for category in categories:
        relative = f'data/{category}.jsonl'
        upstream[category] = {'path': f'bbq/{relative}', 'sha256': sha(raw / relative), 'urls': [
            f'https://raw.githubusercontent.com/nyu-mll/BBQ/{revision}/{relative}',
            f'https://cdn.jsdelivr.net/gh/nyu-mll/BBQ@{revision}/{relative}']}
    relative = 'supplemental/additional_metadata.csv'
    upstream['metadata'] = {'path': f'bbq/{relative}', 'sha256': sha(raw / relative), 'urls': [
        f'https://raw.githubusercontent.com/nyu-mll/BBQ/{revision}/{relative}',
        f'https://cdn.jsdelivr.net/gh/nyu-mll/BBQ@{revision}/{relative}']}
    mmlu = ROOT / 'research/sources/20261012/FairSteer/generate_data/mmlu.json'
    author_revision = '0545cdbaa71b3301030accf1b98130820e08026f'
    upstream['mmlu'] = {'path': 'fairsteer/mmlu.json', 'sha256': sha(mmlu), 'urls': [
        f'https://raw.githubusercontent.com/LiYichen99/FairSteer/{author_revision}/generate_data/mmlu.json',
        f'https://cdn.jsdelivr.net/gh/LiYichen99/FairSteer@{author_revision}/generate_data/mmlu.json']}
    filenames = ['dev.jsonl', 'test.jsonl', 'dev_utility.jsonl', 'test_utility.jsonl',
                 *[f'fit_{seed}.jsonl' for seed in range(20261017, 20261022)]]
    files = {}
    for filename in filenames:
        values = [json.loads(line) for line in (data / filename).read_text(encoding='utf-8').splitlines()]
        files[filename] = {'kind': 'mmlu' if 'utility' in filename else 'bbq',
                           'sha256': sha(data / filename), 'uids': [r['uid'] for r in values]}
    write(destination / 'manifest.json', {'schema_version': 1, 'jsonl_newline': '\r\n', 'bbq_revision': revision,
          'fairsteer_revision': author_revision, 'upstream': upstream, 'categories': categories,
          'source_rows': 58492, 'source_bbq_sha256': sha(ROOT / 'data/prepared/fairsteer_author_qwen3_v1/bbq.jsonl'),
          'training_sha256': sha(data / 'training.json'), 'files': files})
    shutil.copyfile(data / 'training.json', destination / 'training.json')
    evaluation = [json.loads(line) for line in (data / 'test.jsonl').read_text(encoding='utf-8').splitlines()]
    write(destination / 'evaluation_labels.json', {'uids': [r['uid'] for r in evaluation],
          'labels': [r['label'] for r in evaluation],
          'ambiguous': [r['context_condition'] == 'ambig' for r in evaluation],
          'families': [r['family'] for r in evaluation],
          'attribution': 'BBQ (Parrish et al., 2022), CC BY 4.0; labels selected in manifest.json order.'})
    shutil.copyfile(ROOT / 'artifacts/cloud_setup_20261023/models_manifest.json',
                    ROOT / 'reproducibility/cloud_models.json')
    for campaign, filename in [('evidence_validation_20261017_v1', 'small_models_bbq.json'),
                               ('cloud_bbq_20261023_v1', 'cloud_models_bbq.json')]:
        source = ROOT / 'results' / campaign / 'results.json'
        full = json.loads(source.read_text(encoding='utf-8'))
        exported = {'source_file': f'results/{campaign}/results.json', 'source_sha256': sha(source),
                    'scope': 'BBQ only; historical evaluation set, five training resamples',
                    'models': {name: value['test'] for name, value in full['models'].items()},
                    'head_budget_curves': full['head_budget_curves']}
        if 'planned_contrasts' in full:
            exported['planned_contrasts'] = full['planned_contrasts']
        else:
            exported['bbq_primary_contrasts'] = {k: v for k, v in full['primary'].items() if '/test/' in k}
            exported['statistical_note'] = 'Historical adjusted p values retain the original 12-comparison family.'
        write(ROOT / 'benchmarks' / filename, exported)
        for model in full['models']:
            for seed in range(20261017, 20261022):
                relative = Path(model) / f'seed_{seed}' / 'predictions_test.npz'
                target = ROOT / 'benchmarks/predictions' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source.parent / relative, target)
    lock = ROOT / 'environments/cloud.lock.txt'
    lock.parent.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / 'artifacts/cloud_setup_20261023/environment.lock.txt', lock)
    inv = json.loads((ROOT / 'artifacts/github_remote_inventory_20261023.json').read_text())
    # Documentation can change during integration; frozen scientific sources cannot.
    assert all(sha(ROOT / p) == v['sha256'] for p, v in inv['files'].items()
               if Path(p).suffix in ('.py', '.toml'))
    write(ROOT / 'reproducibility/local_cloud_source_audit.json', {
        'date': '2026-10-23', 'compared_files': len(inv['files']), 'identical': True,
        'different': [], 'remote_only': [], 'files': inv['files']})
    print({'manifests_exported': True, 'models': 6, 'prediction_files': 30})


if __name__ == '__main__':
    main()
