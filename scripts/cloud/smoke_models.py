"""Verify text forwards and last-prompt activation edits before running EGUS."""
from pathlib import Path
import argparse
import gc
import importlib
import json
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/hy-tmp/models')
    ap.add_argument('--models', nargs='+', default=['Qwen3-8B', 'Qwen3.5-4B', 'Qwen3.5-9B'])
    ap.add_argument('--dtype', choices=['float16', 'bfloat16'], default='float16')
    args = ap.parse_args()
    torch.set_num_threads(8)
    result_dir = Path('logs/smoke')
    result_dir.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        path = Path(args.root) / name
        assert (path / 'download_verified.json').exists(), f'Download unverified: {name}'
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        cls = AutoModelForCausalLM
        if config.model_type == 'qwen3_5':
            from transformers import Qwen3_5ForConditionalGeneration
            cls = Qwen3_5ForConditionalGeneration
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side='left')
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        started = time.time()
        torch.cuda.reset_peak_memory_stats()
        print(json.dumps({'event': 'loading', 'model': name}), flush=True)
        model = cls.from_pretrained(path, local_files_only=True, dtype=getattr(torch, args.dtype),
                                    attn_implementation='sdpa').to('cuda').eval()
        model.requires_grad_(False)
        text_model = getattr(model.model, 'language_model', model.model)
        layers = text_model.layers
        text_config = getattr(config, 'text_config', config)
        layer = len(layers) * 2 // 3
        ids = [tokenizer('Answer: '+s).input_ids[-1] for s in 'ABC']
        assert len(set(ids)) == 3
        assert [tokenizer.decode([i]).strip() for i in ids] == list('ABC')
        texts = ['A. Yes\nB. No\nC. Cannot be determined\nAnswer:',
                 'Context: A teacher and a doctor entered a room. No other information is available. '
                 'Who arrived first?\nA. The teacher\nB. The doctor\nC. Cannot be determined\nAnswer:']
        enc = tokenizer(texts, padding=True, return_tensors='pt').to('cuda')
        enc['position_ids'] = (enc['attention_mask'].cumsum(-1)-1).clamp(min=0)
        captured = []

        def hook(module, inputs, output, edit=None):
            h = output[0] if isinstance(output, tuple) else output
            captured.append(h[:, -1].detach().float().cpu())
            if edit is None:
                return output
            changed = h.clone()
            # Only the first example is edited; verify isolation in a mixed batch.
            changed[0, -1] = (changed[0, -1].float()+edit).to(changed.dtype)
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        def forward(edit=None):
            captured.clear()
            handle = layers[layer].register_forward_hook(lambda m, i, o: hook(m, i, o, edit))
            try:
                with torch.inference_mode():
                    scores = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, ids].float().cpu()
            finally:
                handle.remove()
            assert len(captured) == 1 and captured[0].shape == (2, text_config.hidden_size)
            assert torch.isfinite(scores).all() and torch.isfinite(captured[0]).all()
            return scores, captured[0].clone()

        scores, features = forward()
        noop, _ = forward(torch.zeros(text_config.hidden_size, device='cuda'))
        assert torch.equal(scores, noop), 'Zero edit changes outputs'
        torch.manual_seed(20261023)
        edit = torch.randn(text_config.hidden_size, device='cuda') * 0.1
        edited, _ = forward(edit)
        assert torch.equal(scores[1], edited[1]), 'Edit leaks across batch rows'
        delta = float((scores[0]-edited[0]).abs().max())
        assert delta > 1e-5, 'Edit has no observable downstream effect'
        with torch.inference_mode():
            single = tokenizer(texts[0], return_tensors='pt').to('cuda')
            single['position_ids'] = torch.arange(single.input_ids.shape[-1], device='cuda').unsqueeze(0)
            alone = model(**single, use_cache=False, logits_to_keep=1).logits[0, -1, ids].float().cpu()
        padding_difference = float((scores[0]-alone).abs().max())
        assert padding_difference < 0.25, f'Padding changes scores excessively: {padding_difference}'
        # Import the existing experiment stack without modifying the frozen implementation.
        importlib.import_module('mechanism.evidence_story')
        info = {'model': name, 'passed': True, 'dtype': args.dtype,
                'architecture': type(model).__name__, 'layers': len(layers), 'probe_layer': layer,
                'hidden_size': text_config.hidden_size, 'candidate_ids': ids,
                'scores': scores.tolist(), 'edited_scores': edited.tolist(),
                'edit_max_logit_change': delta, 'padding_max_logit_difference': padding_difference,
                'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
                'elapsed_seconds': time.time()-started}
        (result_dir / f'{name}.{args.dtype}.json').write_text(json.dumps(info, indent=2))
        print(json.dumps(info), flush=True)
        del model, text_model, layers, enc, edit
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
