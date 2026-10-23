"""Frozen BBQ -> CEB open-generation transfer; no CEB fitting or external API calls.

CEB source files are data, never imported/executed. Decode behavior and both
intervention scopes are explicit adaptations because FairSteer's CEB runner
is not present in its pinned public repository.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import time
import tomllib

import numpy as np
from scipy.special import expit
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/experiment_ceb_transfer.toml'
CATEGORIES = ('age', 'gender', 'race', 'religion')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_source(config):
    source = ROOT / config['source_dir']; manifest = read(source / 'snapshot_manifest.json')
    assert manifest['commit'] == config['source_commit']
    for item in manifest['files']: assert sha(source / item['path']) == item['sha256']
    rows = []
    for category in CATEGORIES:
        data = read(source / f'data/CEB-Continuation-S/{category}.json')
        assert len(data) == 100
        for i, row in enumerate(data):
            assert isinstance(row['prompt'], str) and row['prompt']
            rows.append({'uid': f'{category}:{i:03d}', 'category': category, **row})
    assert len({r['prompt'] for r in rows}) == 400
    return rows


def selected_rows(rows, config, smoke):
    if not smoke: return rows
    selected = []
    for category in CATEGORIES:
        group = [r for r in rows if r['category'] == category]
        selected.extend(sorted(group, key=lambda r: digest([config['seed'], r['uid']]))[:config['smoke_per_category']])
    return selected


def specifications(config):
    values = [{'name': 'baseline', 'gate': 'none', 'scope': 'none', 'alpha': 0., 'threshold': .5}]
    for scope in config['scopes']:
        values.extend([
            {'name': f'fairsteer_{scope}', 'gate': 'original', 'scope': scope, 'alpha': 1., 'threshold': .5},
            {'name': f'evidence_{scope}', 'gate': 'evidence', 'scope': scope, 'alpha': 1.5, 'threshold': .25},
        ])
    return values


def gate_values(hidden, probes, spec):
    probs = {name: expit(hidden @ probes[name + '_coef'] + float(probes[name + '_intercept']))
             for name in ('original', 'expanded', 'evidence')}
    if spec['gate'] == 'none': score = np.zeros(len(hidden))
    elif spec['gate'] == 'original': score = 1 - probs['original']
    elif spec['gate'] == 'evidence': score = (1 - probs['expanded']) * probs['evidence']
    else: raise ValueError(spec['gate'])
    return probs, score, score > spec['threshold']


class SteeringHook:
    def __init__(self, vector, probes, spec):
        self.vector = vector; self.probes = probes; self.spec = spec; self.events = []; self.calls = 0
        self.device_vector = None

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        value = hidden[:, -1, :].detach().float().cpu().numpy()
        probs, score, fire = gate_values(value, self.probes, self.spec)
        eligible = self.spec['scope'] == 'every_step' or (self.spec['scope'] == 'prefill' and self.calls == 0)
        fire = fire & eligible
        self.events.append({'original': probs['original'].tolist(), 'expanded': probs['expanded'].tolist(),
                            'evidence': probs['evidence'].tolist(), 'gate_score': score.tolist(), 'trigger': fire.tolist()})
        self.calls += 1
        if not fire.any(): return output
        if self.device_vector is None:
            self.device_vector = torch.as_tensor(self.vector, device=hidden.device, dtype=torch.float64)
        indices = torch.as_tensor(np.flatnonzero(fire), device=hidden.device)
        changed = hidden.clone()
        changed[indices, -1, :] = (hidden[indices, -1, :].double() + self.spec['alpha'] * self.device_vector).to(hidden.dtype)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed


def clip_output(ids, eos_ids):
    ids = list(map(int, ids)); eos_ids = set(eos_ids)
    for i, token in enumerate(ids):
        if token in eos_ids: return ids[:i+1], 'eos'
    return ids, 'max_new_tokens'


def trim_copied_prefix(response, given_prefix):
    """Conservative auxiliary view, never silently alter the primary response."""
    stripped = response.lstrip()
    if stripped.startswith(given_prefix):
        return stripped[len(given_prefix):].lstrip(), True
    return response, False


def behavior_diagnostics(response, prefix):
    novel, stripped = trim_copied_prefix(response, prefix)
    words = re.findall(r"\b[\w'-]+\b", response.lower())
    grams = [tuple(words[i:i+4]) for i in range(max(0, len(words)-3))]
    # Flags support auditing only; none is a validated bias/refusal score.
    return {'empty': not response.strip(), 'copied_prefix_exact': stripped, 'novel_text': novel,
            'word_count': len(words), 'repeated_4gram_fraction': 1-len(set(grams))/len(grams) if grams else 0.,
            'refusal_phrase_flag': bool(re.search(r"\b(?:i cannot|i can't|i am unable|i'm unable|i won't|i will not)\b", response.lower())),
            'unknown_phrase_flag': bool(re.search(r'\b(?:cannot determine|not enough information|insufficient information)\b', response.lower()))}


def reuse_inactive_prefill(records, spec):
    """Exact same-batch reuse only when EVERY row's sole eligible edit is off."""
    if spec['scope'] != 'prefill': return None
    def score(event):
        if spec['gate'] == 'original': return np.float32(1)-np.float32(event['original'])
        if spec['gate'] == 'evidence': return (np.float32(1)-np.float32(event['expanded']))*np.float32(event['evidence'])
        raise ValueError(spec['gate'])
    if any(score(r['trace'][0]) > spec['threshold'] for r in records): return None
    reused=copy.deepcopy(records)
    for row in reused:
        row['method']=spec['name'];row['reused_from']='baseline_same_batch_gate_off'
        for event in row['trace']:
            event['gate_score']=float(score(event));event['trigger']=False
    return reused


class Experiment:
    def __init__(self, config_path=CONFIG, smoke=False):
        self.config_path = Path(config_path); self.c = tomllib.loads(self.config_path.read_text(encoding='utf-8'))
        self.smoke = smoke; self.out = ROOT / (self.c['output_dir'] + ('_smoke' if smoke else ''))
        self.rows = selected_rows(load_source(self.c), self.c, smoke)
        self.specs = specifications(self.c); self.fit = ROOT / self.c['fit_dir']
        plan = read(self.fit / 'test_plan_frozen.json')
        for name in ('probes.npz', 'direction_dim.npy'): assert sha(self.fit / name) == plan['fitted_sha256'][name]
        assert next(x for x in plan['configs'] if x['name'] == 'evidence') == {'name':'evidence','gate':'evidence','vector':'dim','alpha':1.5,'threshold':.25}
        self.probes = dict(np.load(self.fit / 'probes.npz')); self.vector = np.load(self.fit / 'direction_dim.npy')
        self.tokenizer = None; self.model = None
        self.freeze()

    def freeze(self):
        config_relative=self.config_path.resolve().relative_to(ROOT).as_posix()
        sources = ('mechanism/ceb_transfer.py', 'scripts/run_ceb_transfer.py', config_relative)
        model_dir = Path(self.c['model_path'])
        identity = {'config': self.c, 'smoke': self.smoke, 'data_hash': digest(self.rows), 'uids': [r['uid'] for r in self.rows],
                    'source_hashes': {p: sha(ROOT/p) for p in sources},
                    'artifact_hashes': {n: sha(self.fit/n) for n in ('probes.npz','direction_dim.npy','test_plan_frozen.json')},
                    'model_metadata_hashes': {p.name: sha(p) for p in model_dir.iterdir() if p.suffix in ('.json','.jinja')},
                    'model_weight_inventory': [{'name': p.name, 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                                               for p in sorted(model_dir.glob('*.safetensors'))],
                    'specifications': self.specs}
        path = self.out/'plan_frozen.json'
        if path.exists(): assert read(path)['identity'] == identity, 'Frozen code/config/data changed; use a new output version'
        else:
            write(path, {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'identity': identity,
                         'purpose': 'Open-generation transfer, no training or parameter selection on CEB',
                         'adaptations': ['Qwen3 chat template with thinking disabled', 'Fixed batches with left padding, FP16',
                                         'Both prefill-only and every-step gates; paper does not specify a reproducible CEB implementation',
                                         'Same family-disjoint BBQ fits as the improvement pilot, sklearn solver',
                                         'Independent bias judging is pending; behavior flags are not bias scores']})
            for relative in sources:
                dst=self.out/'source_snapshot'/relative; dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,dst)
            write(self.out/'input_rows.json', self.rows)

    def setup(self):
        if self.model is not None: return
        torch.manual_seed(self.c['seed']); torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(self.c['model_path'], local_files_only=True, padding_side='left')
        if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.c['model_path'], local_files_only=True,
                       dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
        self.model.requires_grad_(False)
        eos = self.model.generation_config.eos_token_id
        self.eos = eos if isinstance(eos, list) else [eos if eos is not None else self.tokenizer.eos_token_id]
        self.generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=self.c['max_new_tokens'],
                         repetition_penalty=1., eos_token_id=self.eos, pad_token_id=self.tokenizer.pad_token_id,
                         bos_token_id=self.tokenizer.bos_token_id, use_cache=True)
        write(self.out/'runtime.json', {'torch':torch.__version__, 'python':platform.python_version(),
              'gpu':torch.cuda.get_device_name(), 'generation':self.generation.to_dict(), 'dtype':'float16',
              'attention':'sdpa', 'batch_size':self.c['batch_size'], 'padding':'left'})

    def inputs(self, rows):
        if self.c['prompt_format'] != 'qwen_chat_no_think': raise ValueError('Unsupported prompt format')
        texts = [self.tokenizer.apply_chat_template([{'role':'user','content':r['prompt']}],
                  tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
        data = self.tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False)
        if data.input_ids.shape[1] > self.c['max_input_tokens']: raise ValueError('Input too long; no truncation')
        return {k:v.to('cuda') for k,v in data.items()}, texts

    def checks(self):
        self.setup(); inputs,_=self.inputs(self.rows[:2])
        gate_off = {**self.specs[0], 'gate':'evidence', 'scope':'every_step', 'threshold':2.}
        hook = SteeringHook(self.vector,self.probes,gate_off)
        with torch.inference_mode():
            baseline=self.model(**inputs,use_cache=False,logits_to_keep=1).logits.detach().clone()
            handle=self.model.model.layers[self.c['layer']].register_forward_hook(hook)
            try: off=self.model(**inputs,use_cache=False,logits_to_keep=1).logits
            finally: handle.remove()
            assert torch.equal(baseline,off), 'Gate-off logits changed'
            forced=SteeringHook(self.vector,self.probes,{'gate':'original','scope':'prefill','alpha':1.,'threshold':-1.})
            handle=self.model.model.layers[self.c['layer']].register_forward_hook(forced)
            try: on=self.model(**inputs,use_cache=False,logits_to_keep=1).logits
            finally: handle.remove()
            assert torch.isfinite(on).all() and not torch.equal(baseline,on)
            restored=self.model(**inputs,use_cache=False,logits_to_keep=1).logits
            assert torch.equal(restored,baseline)
            plain=self.model.generate(**inputs,generation_config=self.generation,max_new_tokens=8,logits_to_keep=1)
            hook=SteeringHook(self.vector,self.probes,gate_off);handle=self.model.model.layers[self.c['layer']].register_forward_hook(hook)
            try: controlled=self.model.generate(**inputs,generation_config=self.generation,max_new_tokens=8,logits_to_keep=1)
            finally: handle.remove()
            assert torch.equal(plain,controlled), 'Gate-off generation changed'
        write(self.out/'hook_checks.json',{'passed':True,'gate_off_logits_exact':True,'gate_off_generation_exact':True,
              'forced_gate_changes_logits':True,'hook_removal_restores_logits':True,'batch_rows':2,'decode_forward_calls':hook.calls})

    def generate(self):
        self.setup(); self.checks(); size=self.c['batch_size']
        for start in range(0,len(self.rows),size):
            rows=self.rows[start:start+size]; inputs,texts=self.inputs(rows)
            for spec in self.specs:
                path=self.out/'batches'/f'{start:04d}_{spec["name"]}.json'
                if path.exists():
                    cached=read(path);assert cached['plan_sha256']==sha(self.out/'plan_frozen.json')
                    assert [r['uid'] for r in cached['rows']]==[r['uid'] for r in rows];continue
                base_path=self.out/'batches'/f'{start:04d}_baseline.json'
                reused=reuse_inactive_prefill(read(base_path)['rows'],spec) if spec['scope']=='prefill' and base_path.exists() else None
                if reused is not None:
                    write(path,{'plan_sha256':sha(self.out/'plan_frozen.json'),'seconds':0.,'rows':reused,
                                'exact_reuse_from_sha256':sha(base_path)})
                    print(json.dumps({'event':'reused_gate_off','method':spec['name'],'completed':start+len(rows),'total':len(self.rows)}),flush=True)
                    continue
                hook=SteeringHook(self.vector,self.probes,spec)
                handle=self.model.model.layers[self.c['layer']].register_forward_hook(hook); began=time.perf_counter()
                try:
                    with torch.inference_mode():
                        outputs=self.model.generate(**inputs,generation_config=self.generation,logits_to_keep=1)
                    torch.cuda.synchronize()
                finally: handle.remove()
                generated=outputs[:,inputs['input_ids'].shape[1]:].cpu().tolist(); records=[]
                for j,(row,ids) in enumerate(zip(rows,generated)):
                    ids,stop=clip_output(ids,self.eos); events=[{key:value[j] for key,value in event.items()} for event in hook.events[:len(ids)]]
                    assert len(events)==len(ids)
                    response=self.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    records.append({'uid':row['uid'],'category':row['category'],'method':spec['name'],
                      'prompt':row['prompt'],'formatted_prompt':texts[j],'given_prefix':row['gpt_rewrite_text'],
                      'response':response,'token_ids':ids,'generated_tokens':len(ids),'stop_reason':stop,
                      'trace':events,'diagnostics':behavior_diagnostics(response,row['gpt_rewrite_text'])})
                write(path,{'plan_sha256':sha(self.out/'plan_frozen.json'),'seconds':time.perf_counter()-began,'rows':records})
                print(json.dumps({'event':'generated','method':spec['name'],'completed':start+len(rows),
                   'total':len(self.rows),'tokens':sum(r['generated_tokens'] for r in records),
                   'seconds':round(time.perf_counter()-began,2)},ensure_ascii=False),flush=True)
        self.export()

    def export(self):
        records=[]
        for path in sorted((self.out/'batches').glob('*.json')):
            batch=read(path);assert batch['plan_sha256']==sha(self.out/'plan_frozen.json');records.extend(batch['rows'])
        expected={(r['uid'],s['name']) for r in self.rows for s in self.specs}
        actual={(r['uid'],r['method']) for r in records}; assert len(actual)==len(records) and actual<=expected
        complete=actual==expected; by_method={}; anonymous=[]; mapping=[]; content_to_id={}
        for spec in self.specs:
            items=[r for r in records if r['method']==spec['name']]
            if not items: continue
            total_tokens=sum(r['generated_tokens'] for r in items)
            by_method[spec['name']]={'rows':len(items),'mean_generated_tokens':float(np.mean([r['generated_tokens'] for r in items])),
               'max_length_fraction':float(np.mean([r['stop_reason']=='max_new_tokens' for r in items])),
               'prompt_trigger_fraction':float(np.mean([r['trace'][0]['trigger'] for r in items])),
               'token_trigger_fraction':sum(sum(e['trigger'] for e in r['trace']) for r in items)/total_tokens,
               'empty_fraction':float(np.mean([r['diagnostics']['empty'] for r in items])),
               'refusal_phrase_fraction':float(np.mean([r['diagnostics']['refusal_phrase_flag'] for r in items]))}
            for r in items:
                content=r['response']; content_hash=digest(content)
                if content_hash not in content_to_id:
                    aid=digest([self.c['seed'],'blind',content_hash])[:20];content_to_id[content_hash]=aid
                    anonymous.append({'evaluation_id':aid,'response':content,'bias_score_0_99':None,
                                      'judge_model':None,'evaluation_status':'pending'})
                mapping.append({'uid':r['uid'],'category':r['category'],'method':r['method'],
                                'evaluation_id':content_to_id[content_hash]})
        anonymous.sort(key=lambda r:r['evaluation_id'])
        write(self.out/'generations.json',records);write(self.out/'judge_inputs_blinded.json',anonymous)
        write(self.out/'judge_mapping_private.json',mapping)
        write(self.out/'summary.json',{'complete':complete,'expected_generations':len(expected),'actual_generations':len(records),
              'distinct_responses_to_judge':len(anonymous),'by_method':by_method,'bias_evaluation':'pending',
              'scoring_unit':'Entire assistant response, excludes input prompt; identical responses judged once',
              'diagnostic_caution':'Phrase flags are heuristic; they are not validated refusal or bias scores.'})
        reference={r['uid']:r for r in records if r['method']=='baseline'}
        agreement={s['name']:sum(r['token_ids']==reference[r['uid']]['token_ids'] for r in records
                    if r['method']==s['name'] and r['uid'] in reference) for s in self.specs}
        write(self.out/'generation_audit.json',{'complete':complete,'unique_method_sample_pairs':True,
             'identical_to_baseline_counts':agreement,'no_ceb_fitting':True,'independent_bias_scores_available':False})
        title='CEB transfer '+('smoke' if self.smoke else 'full generation')
        lines=['# '+title,'','生成已完成。' if complete else '生成尚未全部完成。','',
            '偏见评分尚未进行；下表仅检查运行行为，不能据此判断哪种方法去偏更好。','',
            '|方法|输出数|平均新token|达到长度上限|提示末尾触发率|所有步触发率|与原模型完全相同输出|',
            '|---|---:|---:|---:|---:|---:|---:|']
        for name,r in by_method.items():
            lines.append(f"|{name}|{r['rows']}|{r['mean_generated_tokens']:.1f}|{r['max_length_fraction']:.1%}|{r['prompt_trigger_fraction']:.1%}|{r['token_trigger_fraction']:.1%}|{agreement[name]}|")
        lines+=['','使用官方CEB完整prompt、Qwen3非思考聊天模板、FP16、固定左填充批次、贪心解码、最多512个新token。',
                'prefill只编辑提示末尾一次；every_step在提示末尾及后续每步重算门控。两者均使用BBQ冻结参数。',
                'judge_inputs_blinded.json为随机匿名且按完整输出去重的待评分文本；judge_mapping_private.json为本地映射。',
                '完整输出包含模型自己复述的前缀；diagnostics.novel_text仅在精确匹配时去除一次给定前缀，用于辅助检查。',
                '没有调用外部API。模型、提示包装、求解器与原论文不同，且缺少作者CEB专用脚本，属于明确记录差异的迁移实验。']
        (self.out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print(json.dumps({'event':'exported','complete':complete,'rows':len(records),'judge_rows':len(anonymous)}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','smoke','generate','export']);p.add_argument('--config',type=Path,default=CONFIG)
    a=p.parse_args(); experiment=Experiment(a.config,smoke=a.stage=='smoke')
    if a.stage in ('generate','smoke'):experiment.generate()
    elif a.stage=='export':experiment.export()
    else:print(experiment.out/'plan_frozen.json')


if __name__=='__main__':main()
