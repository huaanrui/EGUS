"""Checkpoint loading and prompt-position interventions, shared across model sizes."""

from pathlib import Path
import gc
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .io import read, write, sha, digest
from .policies import render


class ModelEngine:
    def __init__(self, spec, config, output, plan_hash, expected):
        self.spec, self.c = spec, config
        self.name, self.out = spec["name"], Path(output) / spec["name"]
        self.out.mkdir(parents=True, exist_ok=True)
        self.plan_hash, self.expected = plan_hash, expected
        self.model = self.tokenizer = self.layers = self.decoder = None
        self.edit_callback = None

    def close(self):
        self.model = self.tokenizer = self.layers = self.decoder = None
        gc.collect()
        torch.cuda.empty_cache()

    def log(self, event, **details):
        import time
        import json

        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.name,
            "event": event,
            **details,
        }
        print(json.dumps(record), flush=True)
        with (self.out / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def load(self, dtype=torch.float16):
        if self.model is not None and self.model.dtype == dtype:
            return
        self.model = self.tokenizer = self.layers = self.decoder = None
        gc.collect()
        torch.cuda.empty_cache()
        path = Path(self.spec["path"])
        if any(sha(path / f) != h for f, h in self.expected.items()):
            raise ValueError(f"Checkpoint differs from its pinned revision: {self.name}")
        torch.set_num_threads(self.c["cpu_threads"])
        torch.manual_seed(self.c["seed"])
        cfg = AutoConfig.from_pretrained(path, local_files_only=True)
        cls = AutoModelForCausalLM
        if cfg.model_type == "qwen3_5":
            from transformers import Qwen3_5ForConditionalGeneration

            cls = Qwen3_5ForConditionalGeneration
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, padding_side="left"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = (
            cls.from_pretrained(
                path, local_files_only=True, dtype=dtype, attn_implementation="sdpa"
            )
            .to("cuda")
            .eval()
        )
        self.model.requires_grad_(False)
        self.decoder = getattr(self.model.model, "language_model", self.model.model)
        self.layers = self.decoder.layers
        assert len(self.layers) == self.spec["layers"]
        text_config = getattr(cfg, "text_config", cfg)
        assert text_config.hidden_size == self.spec["hidden_size"]
        self.ids = [self.tokenizer("Answer: " + a).input_ids[-1] for a in "ABCD"]
        assert [self.tokenizer.decode([v]).strip() for v in self.ids] == list("ABCD")
        if not hasattr(self.model, "hf_device_map"):
            self.model.hf_device_map = {"": 0}
        write(
            self.out / f"environment_{str(dtype).split('.')[-1]}.json",
            {
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "numpy": np.__version__,
                "sklearn": __import__("sklearn").__version__,
                "gpu": torch.cuda.get_device_name(),
                "dtype": str(dtype),
            },
        )

    def forward(self, values, layer=None, alpha=0.0, vector=None, append=None, capture=False):
        texts = [
            render(r) + ((" " + "ABCD"[a]) if append is not None else "")
            for r, a in zip(values, append if append is not None else [0] * len(values))
        ]
        if append is not None and len(append) != len(values):
            raise ValueError("Appended answers must match the input count")
        enc = self.tokenizer(texts, padding=True, return_tensors="pt")
        assert enc.input_ids.shape[1] <= self.c["max_tokens"], "No silent truncation"
        inputs = {k: v.to("cuda") for k, v in enc.items()}
        inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        features, handles = [None] * len(self.layers), []
        edit = torch.as_tensor(vector, device="cuda", dtype=torch.float64) if alpha else None

        def make_hook(index):
            def hook(module, args, output):
                state = output[0] if isinstance(output, tuple) else output
                if capture:
                    features[index] = state[:, -1].detach().clone()
                if index != layer or not alpha:
                    return output
                changed = state.clone()
                if self.edit_callback is None:
                    changed[:, -1] = (state[:, -1].double() + alpha * edit).to(state.dtype)
                else:
                    changed = self.edit_callback(changed, vector, alpha)
                return (changed, *output[1:]) if isinstance(output, tuple) else changed

            return hook

        for i in range(len(self.layers)) if capture else ([] if not alpha else [layer]):
            handles.append(self.layers[i].register_forward_hook(make_hook(i)))
        try:
            with torch.inference_mode():
                output = self.model(**inputs, use_cache=False, logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
        scores = output.logits[:, -1, self.ids].float().cpu().numpy()
        scores[:, 3] = -1e9
        assert np.isfinite(scores).all()
        states = torch.stack(features).cpu().numpy() if capture else None
        if capture:
            assert states.dtype == np.float16 and np.isfinite(states).all()
        return scores, states

    def capture(self, key, values, append=None):
        folder = self.out / "states" / key
        folder.mkdir(parents=True, exist_ok=True)
        identity = digest([self.plan_hash, self.name, key, [r["uid"] for r in values], append])
        path = folder / "progress.json"
        info = read(path) if path.exists() else {"identity": identity, "done": 0}
        assert info["identity"] == identity, "Activation cache belongs to different inputs"
        if info.get("complete"):
            assert sha(folder / "features.npy") == info["features_sha256"]
            assert sha(folder / "scores.npy") == info["scores_sha256"]
            return np.load(folder / "features.npy", mmap_mode="r"), np.load(
                folder / "scores.npy", mmap_mode="r"
            )
        shape = (self.spec["layers"], len(values), self.spec["hidden_size"])
        write(path, info)
        features = np.lib.format.open_memmap(
            folder / "features.npy",
            mode="r+" if (folder / "features.npy").exists() else "w+",
            dtype=np.float16,
            shape=shape,
        )
        scores = np.lib.format.open_memmap(
            folder / "scores.npy",
            mode="r+" if (folder / "scores.npy").exists() else "w+",
            dtype=np.float32,
            shape=(len(values), 4),
        )
        for start in range(info["done"], len(values), self.c["batch_size"]):
            end = min(start + self.c["batch_size"], len(values))
            s, h = self.forward(
                values[start:end],
                capture=True,
                append=None if append is None else append[start:end],
            )
            features[:, start:end], scores[start:end] = h, s
            if end == len(values) or (start // self.c["batch_size"] + 1) % 16 == 0:
                features.flush()
                scores.flush()
                write(path, {"identity": identity, "done": end})
                self.log("capture", dataset=key, done=end, total=len(values))
        features.flush()
        scores.flush()
        write(
            path,
            {
                "identity": identity,
                "done": len(values),
                "complete": True,
                "features_sha256": sha(folder / "features.npy"),
                "scores_sha256": sha(folder / "scores.npy"),
            },
        )
        return features, scores
