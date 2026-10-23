"""Layer-7 source tracing, receiver mediation and controlled attention edges.

All donor patches are diagnostic. This module does not train a deployment gate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from .identity_combo import Experiment as Combo, task_row, SCOPES
from .identity_causal import variants, render, metrics, comparison
from .closed_transfer import digest, rank
from .fairsteer_author import ROOT, read, rows, write, sha
from .identity_answer_parser import parse_answer_line

CONFIG = ROOT/'configs/experiment_identity_path_v3.toml'
PROTOCOL = ROOT/'research/2026-10-15_identity_path_protocol_v3.md'
RECEIVERS = ('identity', 'question_identity', 'decision')
SOURCE_LAYER = 7
LAYERS = tuple(range(8, 28))
PARTS = ('v_source', 'k_source', 'kv_source', 'q_receiver',
         'all_k', 'all_v', 'weights', 'all_qkv', 'direct_v', 'direct_k', 'direct_kv')


def array_sha(a):
    a = np.asarray(a)
    return hashlib.sha256(str((a.shape, str(a.dtype))).encode()+a.tobytes()).hexdigest()


def semantic_margin(vs, scores):
    # Fixed lexical entity order, independent of the gold and AB permutation.
    signs = np.array([1 if r['choices'][0] == sorted(r['choices'])[0] else -1 for r in vs])
    return (np.asarray(scores)[:, 0].astype(float)-np.asarray(scores)[:, 1].astype(float))*signs


def transmission(vs, a, b, restore, transplant):
    sa, sb = semantic_margin(vs, a), semantic_margin(vs, b)
    t = sb-sa
    r = sb-semantic_margin(vs, restore)
    p = semantic_margin(vs, transplant)-sa
    energy = float(np.mean(t*t))
    defined = energy > 1e-8
    return {
        'fidelity': float(1-np.mean((r-t)**2+(p-t)**2)/(2*energy)) if defined else None,
        'restore_alignment': float(np.mean(r*t)/energy) if defined else None,
        'transplant_alignment': float(np.mean(p*t)/energy) if defined else None,
        'total_rms': float(np.sqrt(energy)),
        'restore_rms': float(np.sqrt(np.mean(r*r))),
        'transplant_rms': float(np.sqrt(np.mean(p*p))),
        'restore_mean': float(r.mean()), 'transplant_mean': float(p.mean()),
    }


def score_order(item):
    value = item['transmission']['fidelity']
    return value if value is not None else -float('inf')


def patch_heads(x, positions, sites, donor=None, factor=None):
    """Pre-o_proj head output edits; no hidden residual or other token is edited."""
    y = x.clone().reshape(*x.shape[:2], 16, 128)
    for s in sites:
        p = positions[:, SCOPES.index(s['position'])]
        valid = p >= 0
        ix = torch.arange(len(p), device=x.device)[valid]
        pp = p[valid]
        hs = list(range(16)) if s['head'] == -1 else [s['head']]
        for h in hs:
            if donor is not None:
                value = torch.as_tensor(donor[:, s['layer']-8, RECEIVERS.index(s['position']), h],
                                        device=x.device, dtype=x.dtype)[valid]
                y[ix, pp, h] = value
            else:
                y[ix, pp, h] *= factor
    return y.reshape_as(x)


def reverse_indices(vs, kind='gender'):
    if kind == 'gender':
        keys = {(r['pair_id'], r['permutation'], r['gender']): i for i, r in enumerate(vs)}
        return np.array([keys[(r['pair_id'], r['permutation'],
                              'female' if r['gender'] == 'male' else 'male')] for r in vs])
    keys = {(r['origin_uid'], r['record_index']): i for i, r in enumerate(vs)}
    return np.array([keys[(r['origin_uid'], 1-r['record_index'])] for r in vs])


def record_rows(vs):
    result = []
    for r in vs:
        for j, ent in enumerate(sorted(r['entities'])):
            result.append({**r, 'context': r['context']+f' Reference record: {ent}.',
                           'question': 'Which occupation is explicitly named in the reference record?',
                           'gold': ent, 'label': r['choices'].index(ent), 'task': 'record',
                           'record_index': j, 'origin_uid': r['uid'],
                           'uid': r['uid']+f':record{j}', 'pair_id': r['pair_id']+f':record{j}'})
    return result


def bootstrap_continuous(vs, a, b, restore, transplant, reps, seed):
    # Recompute ratio from aggregate moments on each family bootstrap, rather
    # than averaging unstable per-example ratios.
    sa, sb = semantic_margin(vs, a), semantic_margin(vs, b)
    t = sb-sa; r = sb-semantic_margin(vs, restore); p = semantic_margin(vs, transplant)-sa
    by = defaultdict(list)
    for i, row in enumerate(vs):
        by[row['family_id']].append(i)
    moments = np.array([[len(ix), np.sum(t[ix]**2),
                         np.sum((r[ix]-t[ix])**2+(p[ix]-t[ix])**2),
                         np.sum(r[ix]*t[ix]), np.sum(p[ix]*t[ix])]
                        for ix in by.values()])
    rng = np.random.default_rng(seed); estimates = []
    for _ in range(reps):
        z = moments[rng.integers(0, len(moments), len(moments))].sum(0)
        if z[1]/z[0] > 1e-8:
            estimates.append([1-z[2]/(2*z[1]), z[3]/z[1], z[4]/z[1]])
    return {'clusters': len(by), 'cluster': 'family_id',
            'ci95': dict(zip(('fidelity','restore_alignment','transplant_alignment'),
                            np.quantile(estimates, [.025, .975], axis=0).T.tolist())) if estimates else None}


class Experiment(Combo):
    def __init__(self, config=CONFIG):
        super().__init__(config)
        self.source_data = ROOT/self.c['source_data']

    def prepare(self):
        paths = [Path(__file__), self.config, PROTOCOL,
                 ROOT/'mechanism/identity_combo.py', ROOT/'mechanism/identity_causal.py',
                 ROOT/'mechanism/identity_answer_parser.py', ROOT/'tests/test_identity_path.py']
        identity = {str(p): sha(p) for p in paths}
        path = self.out/'plan_frozen.json'
        if path.exists():
            plan = read(path)
            assert identity == plan['implementation'], 'Changed code: use a new experiment version'
            for p, h in plan['source_inputs'].items():
                assert sha(p) == h, p
        else:
            sources = {str(self.source_data/f'{k}.jsonl'): sha(self.source_data/f'{k}.jsonl')
                       for k in ('rank','screen','dev','historical','new_type1','winogender')}
            write(path, {'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                         'implementation': identity, 'source_inputs': sources,
                         'status': 'exploratory source-fixed causal follow-up; all target sets historically inspected'})
            for p in paths:
                (self.out/('snapshot_'+p.name)).write_bytes(p.read_bytes())
        self.values = {k: rows(self.source_data/f'{k}.jsonl')
                       for k in ('rank','screen','dev','historical','new_type1','winogender')}
        # Explicitly materialize target subsets before any predictions.
        groups = defaultdict(list)
        for r in self.values['historical']:
            groups[r['pair_id']].append(r)
        ordered = sorted(groups, key=lambda pid: rank(self.c['seed'], pid))
        self.values['controls'] = sum([groups[p] for p in ordered[:self.c['control_pairs']]], [])
        self.values['generation'] = sum([groups[p] for p in ordered[:self.c['generation_pairs']]], [])
        for k in ('controls','generation'):
            p = self.data/f'{k}.jsonl'
            text = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in self.values[k])
            if p.exists(): assert p.read_text(encoding='utf-8') == text
            else: p.write_text(text, encoding='utf-8')
        self.log('prepared', pairs={k: len(v)//2 for k,v in self.values.items()})

    def setup(self):
        if self.model is not None: return
        torch.set_num_threads(self.c['cpu_threads']); torch.manual_seed(self.c['seed'])
        self.tok = AutoTokenizer.from_pretrained(self.c['model_path'], local_files_only=True, padding_side='left')
        if self.tok.pad_token_id is None: self.tok.pad_token = self.tok.eos_token
        self.ids = [self.tok('Answer: '+x, add_special_tokens=False).input_ids[-1] for x in 'AB']
        self.log('loading_model')
        self.model = AutoModelForCausalLM.from_pretrained(self.c['model_path'], local_files_only=True,
                    dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
        self.model.requires_grad_(False)
        assert self.model.config.num_attention_heads == 16 and self.model.config.num_key_value_heads == 8
        write(self.out/'environment.json', {'torch': torch.__version__,
              'transformers': __import__('transformers').__version__, 'gpu': torch.cuda.get_device_name(),
              'dtype': 'float16', 'attention': 'sdpa', 'candidate_ids': self.ids,
              'model_config_sha256': sha(Path(self.c['model_path'])/'config.json')})

    def spans(self, vs, texts, offsets, width):
        base = super().spans(vs, texts, offsets, width)
        extra = []
        for r, text, off in zip(vs, texts, offsets):
            if r['task'] != 'record': extra.append(-1); continue
            start = text.index(r['context'])+r['context'].rindex('Reference record: ')+len('Reference record: ')
            ent = sorted(r['entities'])[r['record_index']]
            ix = [j for j,(a,b) in enumerate(off.tolist()) if b>a and a<start+len(ent) and b>start]
            assert ix
            extra.append(ix[-1])
        return torch.cat([base, torch.tensor(extra, device=base.device)[:,None]], dim=1)

    def run_forward(self, vs, source=None, sites=(), donor=None, head_factor=None,
                    capture=False, full_layers=(), source_position=0, inputs_pos=None,
                    edge=None):
        self.setup()
        inputs, pos = self.encode(vs) if inputs_pos is None else inputs_pos
        caps, handles, raw, attn_args = {}, [], {}, {}
        n = len(vs); idx = torch.arange(n, device='cuda'); sp = pos[:, source_position]
        assert (sp >= 0).all()
        def source_hook(mod, args, output):
            x = output[idx,sp]
            if source is None: return output
            y = output.clone()
            if isinstance(source, str):
                assert source == 'half'; y[idx,sp] = x*.5
            else:
                y[idx,sp] = torch.as_tensor(source, device=x.device, dtype=x.dtype)
            return y
        handles.append(self.model.model.layers[7].mlp.register_forward_hook(source_hook))
        if capture:
            def source_capture(mod,args,output): caps['mlp'] = output[idx,sp].detach().cpu().numpy()
            handles.append(self.model.model.layers[7].mlp.register_forward_hook(source_capture))
        for l, block in enumerate(self.model.model.layers):
            ss = [s for s in sites if s['layer'] == l]
            if ss:
                def hp(mod,args,ss=ss):
                    return (patch_heads(args[0],pos,ss,donor,head_factor), *args[1:])
                handles.append(block.self_attn.o_proj.register_forward_pre_hook(hp))
            if capture and l in (*LAYERS, 7):
                def cap(mod,args,l=l):
                    x = args[0].reshape(n,-1,16,128)
                    pp = pos[:, [SCOPES.index(s) for s in RECEIVERS]]
                    z = x[idx[:,None],pp.clamp(min=0)]
                    z = z.masked_fill((pp<0)[:,:,None,None],0)
                    caps['h'+str(l)] = z.detach().cpu().numpy()
                handles.append(block.self_attn.o_proj.register_forward_pre_hook(cap))
            if l in full_layers or (edge is not None and l == edge['site']['layer']):
                def rawcap(mod,args,l=l): raw[l] = args[0].detach()
                handles.append(block.input_layernorm.register_forward_pre_hook(rawcap))
                def acap(mod,args,kwargs,l=l):
                    attn_args[l] = kwargs
                    if l in full_layers:
                        caps['norm'+str(l)] = kwargs['hidden_states'].detach().cpu().numpy()
                        caps['raw'+str(l)] = raw[l].cpu().numpy()
                handles.append(block.self_attn.register_forward_pre_hook(acap, with_kwargs=True))
            if edge is not None and l == edge['site']['layer']:
                def ep(mod,args,l=l,block=block):
                    return (self.edge_output(args[0],pos,raw[l],attn_args[l],block,edge,sp),*args[1:])
                handles.append(block.self_attn.o_proj.register_forward_pre_hook(ep))
        try:
            with torch.inference_mode():
                out = self.model(**inputs,use_cache=False,logits_to_keep=1)
            scores = out.logits[:,-1,self.ids].float().cpu().numpy()
        finally:
            for h in handles: h.remove()
        assert np.isfinite(scores).all()
        if capture:
            caps['heads'] = np.stack([caps.pop('h'+str(l)) for l in LAYERS],axis=1)
            caps['prior_heads'] = caps.pop('h7')
        return {'scores': scores, **caps}

    def qkv(self, attn, hidden, embeddings):
        shape = (*hidden.shape[:2], -1, 128)
        q = attn.q_norm(attn.q_proj(hidden).view(shape)).transpose(1,2)
        k = attn.k_norm(attn.k_proj(hidden).view(shape)).transpose(1,2)
        v = attn.v_proj(hidden).view(shape).transpose(1,2)
        q,k = apply_rotary_pos_emb(q,k,*embeddings)
        return q,k,v

    def edge_output(self, x, positions, raw, kwargs, block, edge, source_pos):
        """Recompute one query head with explicitly specified incoming edges."""
        s, recipe = edge['site'], edge['recipe']; h = s['head']; g = h//2
        n = len(x); ix = torch.arange(n,device=x.device)
        receiver = positions[:,SCOPES.index(s['position'])]; valid = receiver>=0
        q0,k0,v0 = self.qkv(block.self_attn,kwargs['hidden_states'],kwargs['position_embeddings'])
        direct = recipe.startswith('direct_')
        if direct:
            # Add source delta to baseline residual while fixing every other
            # intervening computed contribution. Actual nonlinear norm follows.
            hybrid = raw.clone()
            p = source_pos if edge.get('relay_pos') is None else positions[:,SCOPES.index(edge['relay_pos'])]
            vv = p>=0
            delta = torch.as_tensor(edge['delta'],device=x.device,dtype=raw.dtype)
            hybrid[ix[vv],p[vv]] += delta[vv]
            donor_hidden = block.input_layernorm(hybrid)
        else:
            p = source_pos
            donor_hidden = torch.as_tensor(edge['donor_norm'],device=x.device,dtype=raw.dtype)
        q1,k1,v1 = self.qkv(block.self_attn,donor_hidden,kwargs['position_embeddings'])
        q,k,v = q0[:,h:h+1].clone(),k0[:,g:g+1].clone(),v0[:,g:g+1].clone()
        if recipe in ('q_receiver','all_qkv','weights'):
            q[ix[valid],0,receiver[valid]] = q1[ix[valid],h,receiver[valid]]
        if recipe in ('all_k','all_qkv','weights'): k = k1[:,g:g+1]
        if recipe in ('all_v','all_qkv'): v = v1[:,g:g+1]
        vv = p>=0
        if recipe in ('k_source','kv_source','direct_k','direct_kv'):
            k[ix[vv],0,p[vv]] = k1[ix[vv],g,p[vv]]
        if recipe in ('v_source','kv_source','direct_v','direct_kv'):
            v[ix[vv],0,p[vv]] = v1[ix[vv],g,p[vv]]
        mask = kwargs.get('attention_mask')
        changed = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0,
                    is_causal=mask is None,scale=block.self_attn.scaling).transpose(1,2)
        y = x.clone().reshape(n,-1,16,128)
        y[ix[valid],receiver[valid],h] = changed[ix[valid],receiver[valid],0]
        return y.reshape_as(x)

    def infer(self, key, vs, source=None, sites=(), donor=None, head_factor=None,
              capture=False, source_position=0):
        folder = self.out/'shards'/key; folder.mkdir(parents=True,exist_ok=True)
        result = []
        for start in range(0,len(vs),self.c['batch_size']):
            batch = vs[start:start+self.c['batch_size']]; end = start+len(batch)
            src = source if source is None or isinstance(source,str) else source[start:end]
            d = donor[start:end] if donor is not None else None
            signature = digest([sha(self.out/'plan_frozen.json'),batch,sites,head_factor,capture,source_position,
                                src if isinstance(src,(str,type(None))) else array_sha(src),
                                array_sha(d) if d is not None else None])
            path = folder/f'{start:06d}.npz'
            if path.exists():
                with np.load(path) as f: cache = {k:f[k] for k in f.files}
                assert str(cache.pop('identity')) == signature
            else:
                cache = self.run_forward(batch,src,sites,d,head_factor,capture,source_position=source_position)
                np.savez(path, identity=signature, **cache)
            result.append(cache)
        return {k:np.concatenate([r[k] for r in result]) for k in result[0]}

    def ab(self, key, vs, capture=True):
        return (self.infer(key+'_A',vs,capture=capture),
                self.infer(key+'_B',vs,'half',capture=capture))

    def trial(self,key,vs,a,b,sites,source='half',source_position=0):
        r = self.infer(key+'_restore',vs,source,sites,a['heads'],source_position=source_position)['scores']
        p = self.infer(key+'_transplant',vs,None,sites,b['heads'],source_position=source_position)['scores']
        return {'sites': sites, 'transmission': transmission(vs,a['scores'],b['scores'],r,p),
                'restore': r, 'transplant': p}

    def smoke(self):
        path = self.out/'smoke.json'
        if path.exists(): return
        vs = [variants(r) for r in self.values['screen'][:8]]
        a = self.run_forward(vs,capture=True); b = self.run_forward(vs,'half',capture=True)
        assert np.array_equal(a['prior_heads'],b['prior_heads'])
        site = {'layer':8,'position':'decision','head':0}
        z = self.run_forward(vs,sites=[site],donor=a['heads'])['scores']
        assert np.array_equal(z,a['scores'])
        restored = self.run_forward(vs)['scores']; assert np.array_equal(restored,a['scores'])
        c = self.run_forward(vs,full_layers=[8])
        edge = {'site':site,'recipe':'all_qkv','donor_norm':c['norm8']}
        identical = self.run_forward(vs,edge=edge)['scores']
        err = float(abs(identical-a['scores']).max()); assert err<.04,err
        edge = {'site':site,'recipe':'direct_kv','delta':np.zeros_like(a['mlp'])}
        zerodirect = self.run_forward(vs,edge=edge)['scores']
        de = float(abs(zerodirect-a['scores']).max()); assert de<.04,de
        rr = record_rows(vs); rc = self.run_forward(rr,capture=True)
        ind = reverse_indices(rr,'record')
        # Facts occur after the pronoun: they cannot alter source m7(p).
        rd=rc['mlp'].astype(float)-rc['mlp'][ind].astype(float)
        record_rel=float(np.max(np.linalg.norm(rd,axis=1)/np.maximum(np.linalg.norm(rc['mlp'].astype(float),axis=1),1e-8)))
        # Verify the causal premise with identical prefix inputs, separately
        # from length-dependent FP16 rounding in the full-sequence comparison.
        ri,rpos=self.encode(rr);prefixes=[]
        for i in range(len(rr)):
            prefixes.append(ri['input_ids'][i,:rpos[i,0]+1][ri['attention_mask'][i,:rpos[i,0]+1].bool()].tolist())
        assert all(prefixes[i]==prefixes[j] for i,j in enumerate(ind))
        pi=self.tok.pad({'input_ids':prefixes},padding=True,return_tensors='pt').to('cuda')
        pi['position_ids']=(pi['attention_mask'].cumsum(-1)-1).clamp(min=0)
        pp=torch.zeros((len(rr),6),dtype=torch.long,device='cuda');pp[:,0]=pp[:,1]=pi.input_ids.shape[1]-1;pp[:,4:]=-1
        replay=self.run_forward(rr,capture=True,inputs_pos=(pi,pp))
        assert np.array_equal(replay['mlp'],replay['mlp'][ind])
        write(path,{'prior_attention_exact':True,'self_patch_exact':True,'hook_restore_exact':True,
                    'single_head_sdpa_recompute_max_logit_delta':err,'zero_direct_edge_max_delta':de,
                    'later_record_earlier_source_max_relative_delta':record_rel,
                    'identical_prefix_replay_exact':True,
                    'full_sequence_length_numerical_difference_is_descriptive':True})
        self.log('smoke_complete',**read(path))

    def scan(self):
        path = self.out/'head_shortlist.json'
        if path.exists(): return
        vs = [variants(r) for r in self.values['rank']]
        a,b = self.ab('rank',vs)
        layers = []
        for l in LAYERS:
            for pos in ('decision','question_identity'):
                site = {'layer':l,'position':pos,'head':-1}
                t = self.trial(f'layer_{l}_{pos}',vs,a,b,[site])
                layers.append({'site':site,'transmission':t['transmission']})
            self.log('layer_scan',layer=l,results=layers[-2:])
        layers.sort(key=score_order,reverse=True)
        write(self.out/'layer_scan.json',layers)
        heads=[]
        for candidate in layers[:self.c['layer_sites']]:
            for h in range(16):
                site={**candidate['site'],'head':h}
                t=self.trial(f"head_{site['layer']}_{site['position']}_{h}",vs,a,b,[site])
                heads.append({'site':site,'transmission':t['transmission']})
            self.log('head_scan',layer=candidate['site']['layer'],position=candidate['site']['position'])
        heads.sort(key=score_order,reverse=True)
        write(self.out/'head_scan.json',heads)
        write(path,heads[:self.c['head_shortlist']])

    def select(self):
        path=self.out/'selection_frozen.json'
        if path.exists(): return
        shortlist=read(self.out/'head_shortlist.json')
        vs=self.task_values('screen','coref');a,b=self.ab('screen',vs)
        screen=[]
        for i,item in enumerate(shortlist):
            t=self.trial(f'screen_head_{i}',vs,a,b,[item['site']])
            screen.append({'site':item['site'],'transmission':t['transmission']})
        screen.sort(key=score_order,reverse=True); frozen=[r['site'] for r in screen[:self.c['final_heads']]]
        write(self.out/'screen_heads.json',screen)
        # Approximate norm matching is fixed from rank, before dev/targets.
        rv=[variants(r) for r in self.values['rank']];ra,rb=self.ab('rank',rv)
        used={(s['layer'],s['position'],s['head']) for s in frozen}; random=[]; matches=[]
        for s in frozen:
            l=s['layer']-8;p=RECEIVERS.index(s['position']);h=s['head']
            target=float(np.linalg.norm(ra['heads'][:,l,p,h].astype(float),axis=1).mean())
            pool=[]
            for ll in LAYERS:
                for hh in range(16):
                    if (ll,s['position'],hh) in used:continue
                    value=float(np.linalg.norm(ra['heads'][:,ll-8,p,hh].astype(float),axis=1).mean())
                    pool.append((abs(np.log(max(value,1e-8)/max(target,1e-8))),ll,hh,value))
            _,ll,hh,value=min(pool);used.add((ll,s['position'],hh))
            random.append({'layer':ll,'position':s['position'],'head':hh})
            matches.append({'target_norm':target,'control_norm':value})
        vs=self.task_values('dev','coref');a,b=self.ab('dev',vs);candidates=[]
        for k in (1,2,4):
            t=self.trial(f'dev_set{k}',vs,a,b,frozen[:k])
            rt=self.trial(f'dev_random{k}',vs,a,b,random[:k])
            candidates.append({'k':k,'sites':frozen[:k],'transmission':t['transmission'],
                               'random_transmission':rt['transmission']})
        best=max(candidates,key=lambda r:(score_order(r),-r['k']))
        write(path,{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'heads':frozen,
                    'random_heads':random,'norm_matches':matches,'candidates':candidates,
                    'selected':best,'criterion':'two-way transmission fidelity only; not target accuracy',
                    'target_sets_previously_inspected':True})
        self.log('selection_frozen',selected=best,heads=frozen)

    def save_predictions(self,key,vs,method_scores):
        path=self.out/'predictions'/f'{key}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        output=[]
        for method,s in method_scores.items():
            for r,z in zip(vs,s):
                output.append({'uid':r['uid'],'pair_id':r['pair_id'],'family_id':r['family_id'],
                     'method':method,'scores':z.tolist(),'prediction':int(z.argmax()),'label':r['label']})
        path.write_text(''.join(json.dumps(r)+'\n' for r in output),encoding='utf-8')

    def evaluate(self):
        path=self.out/'evaluation.json'
        if path.exists():return
        selected=read(self.out/'selection_frozen.json'); ss=selected['selected']['sites']; random=selected['random_heads'][:len(ss)]
        result={}
        for split in ('historical','new_type1','winogender'):
            vs=self.task_values(split,'coref');a,b=self.ab(split,vs); methods={'baseline':a['scores'],'source_half':b['scores']}
            for name,sites in [('selected',ss),('matched',random)]:
                trial=self.trial(split+'_'+name,vs,a,b,sites)
                methods[name+'_restore']=trial['restore'];methods[name+'_transplant']=trial['transplant']
                methods[name+'_half']=self.infer(split+'_'+name+'_half',vs,sites=sites,head_factor=.5)['scores']
                result.setdefault(split,{})[name]={'transmission':trial['transmission'],
                     'bootstrap':bootstrap_continuous(vs,a['scores'],b['scores'],trial['restore'],trial['transplant'],self.c['bootstrap_replicates'],self.c['seed'])}
            result[split]['metrics']={k:metrics(vs,s) for k,s in methods.items()}
            result[split]['behavior_comparisons']={name:{f:comparison(vs,methods[name].argmax(1),methods['baseline'].argmax(1),f,self.c['bootstrap_replicates'],self.c['seed'])
                  for f in ('accuracy','both_correct','flip')} for name in ('source_half','selected_half','selected_transplant')}
            self.save_predictions(split,vs,methods)
            self.log('evaluation',split=split,selected=result[split]['selected'])
        write(path,result)

    def controls(self):
        path=self.out/'controls.json'
        if path.exists():return
        ss=read(self.out/'selection_frozen.json')['selected']['sites'];vs=self.task_values('controls','coref')
        a,b=self.ab('controls',vs);other=reverse_indices(vs)
        neutral=[{**r,'context':'Yesterday, '+r['context'][0].lower()+r['context'][1:]} for r in vs]
        nc=self.infer('neutral_prefix_donor',neutral,capture=True)
        dm=a['mlp'][other].astype(float)-a['mlp'].astype(float)
        dn=nc['mlp'].astype(float)-a['mlp'].astype(float)
        norms=np.linalg.norm(dm,axis=1);nn=np.linalg.norm(dn,axis=1)
        matched=dn*(norms/np.maximum(nn,1e-12))[:,None]
        rng=np.random.default_rng(self.c['seed']);sources={'identity_swap':a['mlp'][other],
            'neutral_prefix':nc['mlp'],'neutral_norm_matched':a['mlp'].astype(float)+matched}
        for k in range(3):
            z=rng.normal(size=dm.shape);z*= (norms/np.linalg.norm(z,axis=1))[:,None]
            sources[f'random_norm_{k}']=a['mlp'].astype(float)+z
        source_results={};preds={'baseline':a['scores'],'source_half':b['scores']}
        for name,source in sources.items():
            c=self.infer('control_'+name,vs,source,capture=True)
            t=self.trial('control_'+name,vs,a,c,ss,source)
            norm=np.linalg.norm(source.astype(float)-a['mlp'].astype(float),axis=1)
            head=[]
            for s in ss:
                z=c['heads'][:,s['layer']-8,RECEIVERS.index(s['position']),s['head']].astype(float)-a['heads'][:,s['layer']-8,RECEIVERS.index(s['position']),s['head']].astype(float)
                head.append(float(np.mean(np.linalg.norm(z,axis=1)/np.maximum(norm,1e-12))))
            source_results[name]={'transmission':t['transmission'],'source_mean_norm':float(norm.mean()),
                'receiver_response_per_source_norm':head,'margin_absolute_response_per_source_norm':
                float(np.mean(abs(semantic_margin(vs,c['scores'])-semantic_margin(vs,a['scores']))/np.maximum(norm,1e-12))),
                'metrics':metrics(vs,c['scores'])}
            preds[name]=c['scores'];preds[name+'_restore']=t['restore'];preds[name+'_transplant']=t['transplant']
            self.log('source_control',method=name,result=source_results[name])
        self.save_predictions('source_controls',vs,preds)
        # Record reading has its own genuine evidence source after the pronoun.
        rv=record_rows(vs); ra=self.infer('record_A',rv,capture=True,source_position=5)
        ri=reverse_indices(rv,'record'); donor=ra['mlp'][ri]
        rb=self.infer('record_swap_B',rv,donor,capture=True,source_position=5)
        rt=self.trial('record_swap',rv,ra,rb,ss,donor,source_position=5)
        rhalf=self.infer('record_heads_half',rv,sites=ss,head_factor=.5,source_position=5)['scores']
        self.save_predictions('record',rv,{'baseline':ra['scores'],'record_source_swap':rb['scores'],
                    'restore':rt['restore'],'transplant':rt['transplant'],'selected_half':rhalf})
        utilities={}
        for task in ('literal','reading','binding','role_swap'):
            tv=[task_row(r,task,p) for r in self.values['controls'] for p in (0,1)]
            scores={}
            for name,source,sites,factor in [('baseline',None,(),None),('source_half','half',(),None),('selected_half',None,ss,.5)]:
                scores[name]=self.infer('utility_'+task+'_'+name,tv,source,sites,head_factor=factor)['scores']
            utilities[task]={k:metrics(tv,z) for k,z in scores.items()};self.save_predictions('utility_'+task,tv,scores)
        from scripts.identity_combo_mechanism_panel import anonymize
        av=[variants(anonymize(r),permutation=p) for r in self.values['controls'] for p in (0,1)]
        am={name:self.infer('anonymous_'+name,av,source,sites,head_factor=factor)['scores']
            for name,source,sites,factor in [('baseline',None,(),None),('source_half','half',(),None),('selected_half',None,ss,.5)]}
        self.save_predictions('anonymous',av,am)
        write(path,{'source_controls':source_results,'zero_neutral_delta_rows':int((nn<1e-12).sum()),
              'record':{'transmission':rt['transmission'],'baseline':metrics(rv,ra['scores']),
                        'record_source_swap':metrics(rv,rb['scores']),'selected_half':metrics(rv,rhalf)},
              'utilities':utilities,'anonymous':{k:metrics(av,z) for k,z in am.items()},
              'limits':'Derived record task measures fact reading, not native disambiguation. Norm matching is at source only. All historical exploratory.'})

    def edges(self):
        path=self.out/'edges.json'
        if path.exists():return
        heads=read(self.out/'selection_frozen.json')['heads'];vs=self.task_values('controls','coref')
        full_layers=sorted({s['layer'] for s in heads});folder=self.out/'edge_shards';folder.mkdir(exist_ok=True)
        chunks=[]
        for start in range(0,len(vs),self.c['batch_size']):
            batch=vs[start:start+self.c['batch_size']];p=folder/f'{start:06d}.npz'
            if p.exists():
                with np.load(p) as f: chunks.append({k:f[k] for k in f.files})
                continue
            a=self.run_forward(batch,capture=True,full_layers=full_layers)
            b=self.run_forward(batch,'half',capture=True,full_layers=full_layers)
            delta=b['mlp'].astype(float)-a['mlp'].astype(float);r={'baseline':a['scores'],'source_half':b['scores']}
            for i,s in enumerate(heads):
                for recipe in PARTS:
                    edge={'site':s,'recipe':recipe,'donor_norm':b['norm'+str(s['layer'])],'delta':delta}
                    r[f'h{i}_{recipe}']=self.run_forward(batch,edge=edge)['scores']
            # Relay is a fixed eligible earlier question receiver -> later
            # decision receiver, selected without looking at this panel.
            eligible=[(i,j) for i,u in enumerate(heads) for j,v in enumerate(heads)
                      if u['position']=='question_identity' and v['position']=='decision' and u['layer']<v['layer']]
            if eligible:
                i,j=eligible[0];u,v=heads[i],heads[j];l=u['layer']-8;pp=RECEIVERS.index('question_identity')
                hd=b['heads'][:,l,pp,u['head']].astype(float)-a['heads'][:,l,pp,u['head']].astype(float)
                weight=self.model.model.layers[u['layer']].self_attn.o_proj.weight[:,u['head']*128:(u['head']+1)*128].detach().float().cpu().numpy()
                rd=hd@weight.T
                edge={'site':v,'recipe':'direct_kv','delta':rd,'relay_pos':'question_identity'}
                r['relay_kv']=self.run_forward(batch,edge=edge)['scores']
            np.savez(p,**r);chunks.append(r)
            if start%(self.c['batch_size']*4)==0:self.log('edge_progress',done=start+len(batch),total=len(vs))
        allscores={k:np.concatenate([z[k] for z in chunks]) for k in chunks[0]}
        a=allscores['baseline'];b=allscores['source_half'];t=semantic_margin(vs,b)-semantic_margin(vs,a);energy=np.mean(t*t)
        result={}
        for name,z in allscores.items():
            d=semantic_margin(vs,z)-semantic_margin(vs,a)
            result[name]={'alignment':float(np.mean(t*d)/energy) if energy>1e-8 else None,
                    'effect_rms':float(np.sqrt(np.mean(d*d))),
                    'reproduction_fidelity':float(1-np.mean((d-t)**2)/energy) if energy>1e-8 else None,
                    'metrics':metrics(vs,z)}
        self.save_predictions('edges',vs,allscores)
        write(path,{'heads':heads,'results':result,'relay_eligible':'relay_kv' in result,
             'limits':'Per-query GQA incoming-edge interventions; controlled additive edges are not full natural mediation. All recipes exploratory.'})

    def generate(self):
        path=self.out/'chat_results.json'
        if path.exists():return
        self.setup();ss=read(self.out/'selection_frozen.json')['selected']['sites']
        vs=[variants(r) for r in self.values['generation']];folder=self.out/'chat';folder.mkdir(exist_ok=True)
        output=[]
        for start in range(0,len(vs),self.c['batch_size']):
            batch=vs[start:start+self.c['batch_size']];p=folder/f'{start:06d}.json'
            if p.exists():output.extend(read(p));continue
            prompts=[self.tok.apply_chat_template([{'role':'user','content':r['context']+' '+r['question']+'\nAnswer with only the occupation name.'}],
                        tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in batch]
            enc=self.tok(prompts,padding=True,return_offsets_mapping=True,return_tensors='pt',add_special_tokens=False)
            offsets=enc.pop('offset_mapping');pos=self.spans(batch,prompts,offsets,enc.input_ids.shape[1]);inputs=enc.to('cuda')
            inputs['position_ids']=(inputs['attention_mask'].cumsum(-1)-1).clamp(min=0)
            baseline=self.run_forward(batch,capture=True,inputs_pos=(inputs,pos));part=[]
            for name in ('baseline','source_half','selected_half','source_half_restore'):
                handles=[];state={'source':False,'head':set()}
                if name in ('source_half','source_half_restore'):
                    def sh(mod,args,out):
                        if state['source']:return out
                        state['source']=True;y=out.clone();ii=torch.arange(len(batch),device='cuda');y[ii,pos[:,0]]*=.5;return y
                    handles.append(self.model.model.layers[7].mlp.register_forward_hook(sh))
                if name in ('selected_half','source_half_restore'):
                    for l in sorted({s['layer'] for s in ss}):
                        sites=[s for s in ss if s['layer']==l]
                        def hp(mod,args,l=l,sites=sites):
                            if l in state['head']:return args
                            state['head'].add(l)
                            y=patch_heads(args[0],pos,sites,baseline['heads'] if name=='source_half_restore' else None,.5)
                            return (y,*args[1:])
                        handles.append(self.model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hp))
                try:
                    with torch.inference_mode():
                        generated=self.model.generate(**inputs,max_new_tokens=32,do_sample=False,use_cache=True,
                                   pad_token_id=self.tok.pad_token_id,eos_token_id=self.tok.eos_token_id)
                    texts=self.tok.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
                finally:
                    for h in handles:h.remove()
                for r,text in zip(batch,texts):
                    clean=re.sub(r'^the\s+','',text.strip().strip(' \t\r\n.,:;!?\"\'').lower())
                    pred=next((i for i,x in enumerate(r['choices']) if clean==x.lower()),-1)
                    part.append({'method':name,'uid':r['uid'],'raw':text,'label':r['label'],
                            'strict_prediction':pred,'line_prediction':parse_answer_line(text,r['choices'])})
            write(p,part);output.extend(part)
            if start%(self.c['batch_size']*4)==0:self.log('chat_progress',done=start+len(batch),total=len(vs))
        lookup={(r['method'],r['uid']):r for r in output};result={}
        for name in ('baseline','source_half','selected_half','source_half_restore'):
            p=[lookup[(name,r['uid'])]['strict_prediction'] for r in vs]
            result[name]={'strict':metrics(vs,predictions=p),'unparsed':float(np.mean(np.array(p)<0)),
                'line':metrics(vs,predictions=[lookup[(name,r['uid'])]['line_prediction'] for r in vs])}
        write(path,{'metrics':result,'pairs':len(vs)//2,'status':'hash-selected historical native chat short answers; prefill-only edits'})
        (self.out/'chat_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in output),encoding='utf-8')

    def verify(self):
        plan=read(self.out/'plan_frozen.json')
        for p,h in {**plan['source_inputs'],**plan['implementation']}.items():assert sha(p)==h,p
        checked=0
        for p in (self.out/'predictions').glob('*.jsonl'):
            for r in rows(p):
                assert np.isfinite(r['scores']).all()
                assert r['prediction']==int(np.argmax(r['scores']))
                checked+=1
        write(self.out/'verification.json',{'records_recomputed':checked,'source_inputs_unchanged':True,
              'implementation_unchanged':True,'target_selection_forbidden':True,
              'historical_status':'all external behavior sets previously inspected; exploratory'})
        self.log('verification_complete',records=checked)

    def run(self,stage='all'):
        self.prepare()
        stages={'smoke':self.smoke,'scan':self.scan,'select':self.select,'evaluate':self.evaluate,
                'controls':self.controls,'edges':self.edges,'generate':self.generate,'verify':self.verify}
        if stage=='all':
            for name,fn in stages.items():
                self.log('stage_start',stage=name);fn()
            write(self.out/'complete.json',{'time':time.strftime('%Y-%m-%d %H:%M:%S')})
        elif stage!='prepare':stages[stage]()
        if self.model is not None:
            del self.model;self.model=None;gc.collect();torch.cuda.empty_cache()
        self.log('stage_complete',stage=stage)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',default='all',
          choices=['all','prepare','smoke','scan','select','evaluate','controls','edges','generate','verify'])
    parser.add_argument('--config',type=Path,default=CONFIG);args=parser.parse_args()
    Experiment(args.config).run(args.stage)


if __name__=='__main__':main()
