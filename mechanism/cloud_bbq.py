"""BBQ-only cloud extension; reuse frozen fitting and policy implementations."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import time
import tomllib

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .fairsteer_author import ROOT, read, rows, write, sha, metrics
from .evidence_story import render, discover_unknown, signals, apply, fit_logistic, select_threshold
from .evidence_validation import Engine, policies
from .evidence_validation_stats import comparison, holm, mean_sd

CONFIG = ROOT / 'configs/experiment_cloud_bbq_20261023_v1.toml'
METHOD_NAMES = {
    'base': 'Base', 'local_fs': 'Local FairSteer-style', 'expanded_steer': 'Full-error steering',
    'product_steer': 'Product steering', 'product_direct': 'Product direct',
    'product_offset': 'Product logit offset', 'evidence_steer': 'Evidence steering',
    'evidence_direct': 'EGUS', 'surface_direct': 'Surface direct',
    'question_direct': 'Question direct', 'output_direct': 'Output direct',
    'always_unknown': 'Always unknown',
}


def audit_data(data, source, seeds):
    """Validate inherited resamples before any features or decisions are fitted."""
    source_uids = [r['uid'] for r in rows(source / 'bbq.jsonl')]
    training = read(data / 'training.json')
    dev, test = rows(data / 'dev.jsonl'), rows(data / 'test.jsonl')
    assert len(dev) == 352 and len(test) == 704
    assert len({r['uid'] for r in dev + test}) == len(dev + test)
    dev_f, test_f = {r['family'] for r in dev}, {r['family'] for r in test}
    assert not dev_f & test_f
    for seed in seeds:
        spec = training[str(seed)]
        fit = rows(data / f'fit_{seed}.jsonl')
        assert len(fit) == len({r['uid'] for r in fit}) == 2200
        assert [source_uids[i] for i in spec['indices']] == [r['uid'] for r in fit]
        assert set(spec['budgets']) == {'220', '1100', '2200'}
        assert not {r['family'] for r in fit} & (dev_f | test_f)
        lookup = {r['uid']: r for r in fit}
        for budget in (220, 1100, 2200):
            selected = [lookup[source_uids[i]] for i in spec['budgets'][str(budget)]]
            assert len(selected) == len({r['uid'] for r in selected}) == budget
            assert set(Counter((r['category'], r['context_condition']) for r in selected).values()) == {budget // 22}
        direction = [lookup[u] for u in spec['direction_uids']]
        assert len(direction) == len({r['uid'] for r in direction}) == 110
        assert set(Counter(r['category'] for r in direction).values()) == {10}
        assert all(r['context_condition'] == 'ambig' and r['label'] == r['unknown_index'] != r['target_loc']
                   and r['target_loc'] is not None for r in direction)
        for r in fit + dev + test:
            assert discover_unknown(r, spec['vocab']) == r['unknown_index']
    return {'fit_per_seed': 2200, 'dev': len(dev), 'test': len(test),
            'test_families': len(test_f), 'family_isolation': True, 'seeds': seeds}


def assert_selection_lock(root, model_names, seeds):
    lock = read(root / 'all_selections_frozen.json')
    assert set(lock) == set(model_names)
    for name in model_names:
        assert set(lock[name]) == {str(s) for s in seeds}
        for seed in seeds:
            folder = root / name / f'seed_{seed}'
            assert sha(folder / 'selection_frozen.json') == lock[name][str(seed)]
            for filename, digest in read(folder / 'selection_frozen.json')['parameters'].items():
                assert sha(folder / filename) == digest
    return lock


class Campaign:
    def __init__(self, config=CONFIG):
        self.config = Path(config).resolve()
        self.c = tomllib.loads(self.config.read_text(encoding='utf-8'))
        self.out = ROOT / self.c['output_dir']
        self.data = ROOT / self.c['data_dir']
        self.ref = self.out / 'no_legacy_cache'
        self.out.mkdir(parents=True, exist_ok=True)

    def prepare(self):
        path = self.out / 'plan_frozen.json'
        if path.exists():
            plan = read(path)
            for relative, digest in plan['sources'].items():
                assert sha(ROOT / relative) == digest, ('Frozen source changed', relative)
            for name, digest in plan['data'].items():
                assert sha(self.data / name) == digest, ('Frozen data changed', name)
            assert sha(ROOT / self.c['source_data'] / 'bbq.jsonl') == plan['source_bbq_sha256']
            return plan
        checks = audit_data(self.data, ROOT / self.c['source_data'], self.c['seeds'])
        reference = read(ROOT / self.c['small_model_reference'])
        for filename, digest in reference['data_sha256'].items():
            assert sha(self.data / filename) == digest
        filenames = ['training.json', 'dev.jsonl', 'test.jsonl', 'dev_utility.jsonl', 'test_utility.jsonl',
                     *[f'fit_{s}.jsonl' for s in self.c['seeds']]]
        files = [self.config, Path(__file__), ROOT / self.c['small_model_reference'],
                 *[ROOT / p for p in ['mechanism/evidence_story.py', 'mechanism/evidence_validation.py',
                   'mechanism/evidence_validation_stats.py', 'mechanism/fairsteer_author.py',
                   'mechanism/fairsteer_improve.py', 'mechanism/fairsteer_data.py',
                   'mechanism/closed_transfer.py', 'scripts/run_cloud_bbq.py',
                   'tests/test_cloud_bbq.py', 'research/2026-10-23_cloud_bbq_protocol_v1.md']]]
        models = {}
        for model in self.c['models']:
            folder = Path(model['path'])
            verified = read(folder / 'download_verified.json')
            assert verified['verified']
            assert all((folder / p).stat().st_size == f['size'] for p, f in
                       ((f['name'], f) for f in verified['files']))
            # Hash once before freezing, and again on each actual model load.
            assert all(sha(folder / p) == digest for p, digest in verified['sha256'].items())
            models[model['name']] = verified['sha256']
        plan = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'scope': 'Historical BBQ model extension',
                'sources': {str(p.relative_to(ROOT)): sha(p) for p in files},
                'data': {p: sha(self.data / p) for p in filenames}, 'models': models,
                'source_bbq_sha256': sha(ROOT / self.c['source_data'] / 'bbq.jsonl'), 'audit': checks}
        for p in files:
            target = self.out / 'source_snapshot' / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
        write(path, plan)
        print({'prepared': True, **checks}, flush=True)
        return plan


class CloudEngine(Engine):
    # Keep inherited fitting, direction estimation, thresholds, cached inference
    # and matched policies intact. Only model access and evaluation scope differ.
    def __init__(self, name, config=CONFIG):
        self.study = Campaign(config)
        self.c = self.study.c
        self.spec = next(m for m in self.c['models'] if m['name'] == name)
        self.name, self.root, self.data = name, self.study.out, self.study.data
        self.out = self.root / name
        self.out.mkdir(exist_ok=True)
        self.model = self.tokenizer = self.layers = None
        self.layer = self.spec['layer']
        self.vector = self.probes = None
        self.training = read(self.data / 'training.json')
        self.vocab = self.training[str(self.c['seed'])]['vocab']

    def setup(self):
        if self.model is not None:
            return
        torch.set_num_threads(self.c['cpu_threads'])
        torch.manual_seed(self.c['seed'])
        expected = read(self.root / 'plan_frozen.json')['models'][self.name]
        path = Path(self.spec['path'])
        assert all(sha(path / p) == digest for p, digest in expected.items())
        self.log('loading_model')
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        model_class = AutoModelForCausalLM
        if config.model_type == 'qwen3_5':
            from transformers import Qwen3_5ForConditionalGeneration
            model_class = Qwen3_5ForConditionalGeneration
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side='left')
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = model_class.from_pretrained(path, local_files_only=True, dtype=torch.float16,
                                                attn_implementation='sdpa').to('cuda').eval()
        self.model.requires_grad_(False)
        decoder = getattr(self.model.model, 'language_model', self.model.model)
        self.layers = decoder.layers
        text_config = getattr(config, 'text_config', config)
        assert len(self.layers) == self.spec['layers']
        assert text_config.hidden_size == self.spec['hidden_size']
        assert self.layer == len(self.layers) * 2 // 3
        self.ids = [self.tokenizer('Answer: '+s).input_ids[-1] for s in 'ABCD']
        assert len(set(self.ids)) == 4
        assert [self.tokenizer.decode([i]).strip() for i in self.ids] == list('ABCD')
        write(self.out / 'environment.json', {
            'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
            'sklearn': __import__('sklearn').__version__, 'numpy': np.__version__,
            'gpu': torch.cuda.get_device_name(), 'dtype': 'float16', 'batch_size': self.c['batch_size'],
            'layers': len(self.layers), 'layer': self.layer, 'hidden_size': text_config.hidden_size,
            'candidate_ids': self.ids, 'architecture': type(self.model).__name__})

    def forward(self, vs, alpha=0., append=None):
        self.setup()
        answers = append if append is not None else [0] * len(vs)
        assert len(answers) == len(vs)
        texts = [render(r) + ((' '+'ABCD'[a]) if append is not None else '') for r, a in zip(vs, answers)]
        enc = self.tokenizer(texts, padding=True, return_tensors='pt')
        assert enc.input_ids.shape[1] <= self.c['max_tokens'], 'No truncation'
        inputs = {k: v.to('cuda') for k, v in enc.items()}
        inputs['position_ids'] = (inputs['attention_mask'].cumsum(-1)-1).clamp(min=0)
        captured = []
        vector = torch.as_tensor(self.vector, device='cuda', dtype=torch.float64) if alpha else None

        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            captured.append(h[:, -1].detach().float().cpu().numpy())
            if not alpha:
                return output
            changed = h.clone()
            changed[:, -1] = (h[:, -1].double() + alpha * vector).to(h.dtype)
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        handle = self.layers[self.layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                result = self.model(**inputs, use_cache=False, logits_to_keep=1)
        finally:
            handle.remove()
        scores = result.logits[:, -1, self.ids].float().cpu().numpy()
        for i, r in enumerate(vs):
            if not r.get('utility'):
                scores[i, 3] = -1e9
        assert len(captured) == 1 and np.isfinite(scores).all() and np.isfinite(captured[0]).all()
        return scores, captured[0]

    def evaluate(self):
        assert_selection_lock(self.root, [m['name'] for m in self.c['models']], self.c['seeds'])
        for key in ('test', 'test_utility'):
            self.baseline(key)
            for seed in self.c['seeds']:
                for alpha in self.c['alphas']:
                    self.counter(key, seed, alpha)
        write(self.out / 'evaluation_complete.json', {'complete': True})
        self.log('evaluation_complete')

    def smoke(self):
        super().smoke()
        path = self.out / 'pipeline_smoke.json'
        if path.exists():
            return
        fit = self.values(f'fit_{self.c["seed"]}')
        selected = [r for condition in ('ambig', 'disambig')
                    for r in [v for v in fit if v['context_condition'] == condition][:16]]
        features = np.concatenate([self.forward(selected[i:i+self.c['batch_size']])[1]
                                   for i in range(0, len(selected), self.c['batch_size'])])
        y = np.array([r['context_condition'] == 'ambig' for r in selected], int)
        coef, intercept, diagnostic = fit_logistic(features, y, self.c)
        direction_rows = [r for r in fit if r['context_condition'] == 'ambig' and r['target_loc'] is not None][:4]
        _, good = self.forward(direction_rows, append=[r['label'] for r in direction_rows])
        _, bad = self.forward(direction_rows, append=[r['target_loc'] for r in direction_rows])
        # This disposable direction/readout only checks the pipeline on fit/dev.
        # The full campaign independently re-estimates its predeclared 110-row directions.
        dev = self.values('dev')
        panel = [r for condition in ('ambig', 'disambig')
                 for r in [v for v in dev if v['context_condition'] == condition][:4]]
        from scipy.special import expit
        try:
            self.vector = (good.astype(float) - bad.astype(float)).mean(0)
            base, x = self.forward(panel)
            counter, _ = self.forward(panel, self.c['primary_alpha'])
            p = expit(x.astype(float) @ coef + intercept)
            best, _ = select_threshold(p, [r['context_condition'] == 'ambig' for r in panel], self.c['gate_fpr_limit'])
            mask = p > best['threshold']
            unknown = np.array([discover_unknown(r, self.vocab) for r in panel])
            direct, _ = apply(base, None, unknown, mask, 'direct')
            steered, _ = apply(base, counter, unknown, mask, 'steer')
            np.testing.assert_array_equal(direct[~mask], base.argmax(1)[~mask])
            np.testing.assert_array_equal(direct[mask], unknown[mask])
            np.testing.assert_array_equal(steered[~mask], base.argmax(1)[~mask])
            write(path, {'passed': True, 'fit_rows': len(selected), 'direction_rows': 4, 'dev_rows': len(panel),
                         'test_rows': 0, 'probe_diagnostic': diagnostic, 'shared_gate': True,
                         'scope': 'Infrastructure check only; disposable parameters, no test evaluation.'})
        finally:
            self.vector = None
        self.log('pipeline_smoke_complete')


def summary_metrics(values, pred, mask):
    if values[0].get('utility'):
        return {'accuracy': float(np.mean(pred == np.array([r['label'] for r in values]))),
                'trigger': float(np.mean(mask))}
    m = metrics(values, pred)
    sufficient = np.array([r['context_condition'] == 'disambig' for r in values])
    unknown = np.array([r['unknown_index'] for r in values])
    return {'accuracy': m['accuracy'], 'ambig': m['ambig']['accuracy'],
            'clear': m['disambig']['accuracy'], 'bias': m['ambig']['bias_score'],
            'clear_unknown': float(np.mean(pred[sufficient] == unknown[sufficient])),
            'trigger': float(np.mean(mask)), 'sufficient_trigger': float(np.mean(mask[sufficient]))}


def summarize(config=CONFIG):
    campaign = Campaign(config)
    campaign.prepare()
    result, contrasts, predictions, budget_curves = {}, {}, {}, {}
    for spec in campaign.c['models']:
        name = spec['name']
        if not (campaign.out / name / 'evaluation_complete.json').exists():
            continue
        engine = CloudEngine(name, config)
        result[name], budget_curves[name] = {}, {}
        for key in ('test', 'test_utility'):
            per_seed, by_method, recognition = {}, defaultdict(list), {}
            for seed in campaign.c['seeds']:
                vs, cache, pred, masks, sig = policies(engine, key, seed)
                per_seed[str(seed)] = {m: summary_metrics(vs, p, masks[m]) for m, p in pred.items()}
                saved = {'uids': cache['uids']}
                for method, p in pred.items():
                    saved['pred_' + method] = p.astype(np.int16)
                    saved['trigger_' + method] = masks[method]
                    by_method[method].append(p)
                np.savez_compressed(engine.seed_folder(seed) / f'predictions_{key}.npz', **saved)
                if key == 'test':
                    probes = dict(np.load(engine.seed_folder(seed) / 'probes.npz'))
                    y = np.array([r['context_condition'] == 'ambig' for r in vs])
                    margin = cache['features'].astype(float) @ probes['2200_evidence_coef'] + probes['2200_evidence_intercept']
                    recognition[str(seed)] = {'evidence_auroc': float(roc_auc_score(y, margin))}
                    selection = read(engine.seed_folder(seed) / 'selection_frozen.json')
                    unknown = np.array([discover_unknown(r, engine.training[str(seed)]['vocab']) for r in vs])
                    curves = {}
                    for budget in campaign.c['fit_budgets']:
                        signal = signals(cache['features'], probes, budget)
                        mask = signal['evidence'] > selection['thresholds'][f'{budget}_evidence']['threshold']
                        selected, _ = apply(cache['base'], None, unknown, mask, 'direct')
                        curves[str(budget)] = summary_metrics(vs, selected, mask)
                    budget_curves[name][str(seed)] = curves
            aggregate = {}
            for method in by_method:
                keys = [k for k in per_seed[str(campaign.c['seed'])][method]
                        if all(per_seed[str(s)][method][k] is not None for s in campaign.c['seeds'])]
                aggregate[method] = {metric: mean_sd([per_seed[str(s)][method][metric] for s in campaign.c['seeds']])
                                     for metric in keys}
            result[name][key] = {'aggregate': aggregate, 'per_seed': per_seed, 'recognition': recognition,
                                 'questions': len(vs), 'seeds': campaign.c['seeds']}
            if key == 'test':
                predictions[name] = (vs, {m: np.array(p) for m, p in by_method.items()})
        write(engine.out / 'results.json', result[name])
    complete = len(result) == len(campaign.c['models'])
    if complete:
        for name, (vs, pred) in predictions.items():
            for reference, condition in [('local_fs', None), ('evidence_steer', None),
                                         ('surface_direct', None), ('base', 'disambig')]:
                tag = f'{name}/EGUS-{reference}' + ('/clear' if condition else '')
                contrasts[tag] = comparison(vs, pred['evidence_direct'], pred[reference],
                                            campaign.c['bootstrap_replicates'], condition=condition)
        assert len(contrasts) == 12
        holm(contrasts)
    local = read(ROOT / campaign.c['small_model_reference'])
    write(campaign.out / 'results.json', {'complete': complete, 'models': result,
          'planned_contrasts': contrasts, 'head_budget_curves': budget_curves,
          'local_small_models': local['models'], 'plan_sha256': sha(campaign.out / 'plan_frozen.json')})
    lines = ['# Cloud BBQ model extension', '',
             'Same historical 704 BBQ questions; five training resamples. Values: mean (sample SD), %.', '',
             '| Model | Policy | Overall | Ambiguous | Sufficient |', '|---|---|---:|---:|---:|']
    for name, item in result.items():
        for method, label in METHOD_NAMES.items():
            a = item['test']['aggregate'][method]
            fmt = lambda k: f"{100*a[k]['mean']:.2f} ({100*a[k]['sd']:.2f})"
            lines.append(f'| {name} | {label} | {fmt("accuracy")} | {fmt("ambig")} | {fmt("clear")} |')
    lines += ['', 'Local small-model results are a descriptive reference on the same data. '
              'Environment and architecture differences are recorded in the frozen protocol.', '',
              f'Complete: {complete}. Models reported: {len(result)}.']
    (campaign.out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print({'report_complete': complete, 'models': list(result)}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'smoke', 'train', 'evaluate', 'report'])
    parser.add_argument('--model')
    parser.add_argument('--config', type=Path, default=CONFIG)
    args = parser.parse_args()
    campaign = Campaign(args.config)
    campaign.prepare()
    if args.stage == 'report':
        summarize(args.config)
    elif args.stage != 'prepare':
        if not args.model:
            parser.error('--model is required')
        getattr(CloudEngine(args.model, args.config), args.stage)()


if __name__ == '__main__':
    main()
