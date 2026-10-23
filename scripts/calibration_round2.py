"""Frozen output-only abstention control; no hidden activation interventions.

The unknown option is observable task content, not a ground-truth decision.
Select its logit offset on development only, before the new test is opened.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mechanism.data import PERMUTATIONS, neutral_controls, read_rows, sha256, write_json
from mechanism.evaluate import paired_cluster_ci
from mechanism.round2 import eligibility, extended_metrics, macro_cluster_ci


def calibrated(rows, predictions, strength):
    by_uid = {r['uid']: r for r in rows}
    changed = copy.deepcopy(predictions)
    for item in changed:
        row = by_uid[item['uid']]
        u = row['unknown_index']
        for rot, v in enumerate(item['rotations']):
            if strength == 0:
                continue
            scores = list(v['semantic_logprobs'])
            old_unknown_conditional = math.exp(scores[u])
            scores[u] += strength
            maximum = max(scores)
            normalizer = maximum + math.log(sum(math.exp(x-maximum) for x in scores))
            v['semantic_logprobs'] = [x-normalizer for x in scores]
            # Original label-order tie breaking is retained.
            perm = PERMUTATIONS[rot]
            v['prediction'] = max(perm, key=lambda i: scores[i])
            increment = v['candidate_mass'] * old_unknown_conditional * math.expm1(strength)
            v['candidate_mass'] = (v['candidate_mass'] + increment)/(1+increment)
    return changed


def wait_file(path):
    deadline = time.monotonic() + 7200
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f'Waiting for primary experiment: {path}')
        time.sleep(1)
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preregister', 'all'])
    args = parser.parse_args()
    c = tomllib.loads((ROOT / 'configs/experiment_round2.toml').read_text(encoding='utf-8'))
    out, data = ROOT / c['output_dir'], ROOT / c['dataset_dir']
    control = out / 'calibration'
    control.mkdir(exist_ok=True)
    prereg = dict(unknown_logit_offsets=[0., .25, .5, 1., 2., 4.], objective='Macro stereotype-rate reduction under the same full-selection guards; minimum 1 percentage point for acceptance. Lowest offset breaks ties.', test_feedback_allowed=False, meaning='Task-visible unknown-option score calibration, not an internal mechanism or general refusal feature.', fixed_before_new_test=True)
    prepath = control / 'preregistered.json'
    if prepath.exists():
        assert json.loads(prepath.read_text(encoding='utf-8')) == prereg
    else:
        if (out / 'test_lock.json').exists():
            raise ValueError('Cannot introduce an unfrozen control after test is opened')
        write_json(prepath, prereg)
    if args.stage == 'preregister':
        print('Abstention calibration control preregistered.', flush=True)
        return
    selection = read_rows(data / 'selection.jsonl')
    neutral = neutral_controls()
    planpath = control / 'plan_frozen.json'
    if not planpath.exists():
        b = wait_file(out / 'predictions/selection_baseline.json')
        n = wait_file(out / 'predictions/selection_neutral_baseline.json')
        if (out / 'test_lock.json').exists():
            raise ValueError('Calibration selection must precede test lock')
        results = []
        for offset in prereg['unknown_logit_offsets']:
            m = extended_metrics(selection, calibrated(selection, b['predictions'], offset))
            mn = extended_metrics(neutral['selection'], calibrated(neutral['selection'], n['predictions'], offset))
            ok, reasons = eligibility(m, mn, b['metrics'], n['metrics'], c)
            gain = b['metrics']['macro_stereotype']-m['macro_stereotype']
            results.append(dict(offset=offset, bbq=m, neutral=mn, feasible=ok, violations=reasons, macro_stereotype_reduction=gain, accepted=bool(ok and gain >= c['minimum_selection_stereotype_reduction']-1e-12)))
        chosen = sorted(results, key=lambda r: (not r['accepted'], not r['feasible'], -r['macro_stereotype_reduction'], r['offset']))[0]
        write_json(planpath, dict(selected=chosen, grid_results=results, preregistered_sha256=sha256(prepath), selection_baseline_sha256=sha256(out / 'predictions/selection_baseline.json'), selection_neutral_baseline_sha256=sha256(out / 'predictions/selection_neutral_baseline.json'), frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'), frozen_before_main_test_lock=True))
        print(f'Abstention control frozen: offset={chosen["offset"]}, accepted={chosen["accepted"]}', flush=True)
    plan = json.loads(planpath.read_text(encoding='utf-8'))
    b = wait_file(out / 'predictions/test_baseline.json')
    n = wait_file(out / 'predictions/test_neutral_baseline.json')
    test = read_rows(data / 'test.jsonl')
    changed = calibrated(test, b['predictions'], plan['selected']['offset'])
    cn = calibrated(neutral['validation'], n['predictions'], plan['selected']['offset'])
    m, mn = extended_metrics(test, changed), extended_metrics(neutral['validation'], cn)
    ok, reasons = eligibility(m, mn, b['metrics'], n['metrics'], c)
    ci = paired_cluster_ci(test, b['predictions'], changed, c['bootstrap_replicates'], c['seed'])
    ci.update(macro_cluster_ci(test, b['predictions'], changed, c['bootstrap_replicates'], c['seed']))
    write_json(control / 'test_results.json', dict(plan_sha256=sha256(planpath), offset=plan['selected']['offset'], accepted_selection=plan['selected']['accepted'], test_feasible=ok, violations=reasons, bbq=m, neutral=mn, ci=ci, predictions=changed, meaning=prereg['meaning']))
    print('Abstention control evaluated on frozen new-category test.', flush=True)


if __name__ == '__main__':
    main()
