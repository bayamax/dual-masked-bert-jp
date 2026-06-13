"""trigger_gate.py — runtime recall gate (TRIGGER_V3 head, wiring stage).

Scores the current question in the production layout [BOS][SP(kept)][raw window]
[question] with the question-position pre-RoPE queries (layers 8/14/20) and a
4608-dim logistic head. fires() == True means "the answer needs something that
was evicted from this session's window" — only then does the app pay for an
archive retrieve + verbatim re-injection.

Threshold note: the gate npz carries a TPR-first threshold (-2.6: heldout TPR
1.000, FPR 0.043) — a false fire costs one redundant retrieve (today's per-turn
status quo), a miss costs the recall itself.
"""
import numpy as np
import torch

LAYERS = (8, 14, 20)


class TriggerGate:
    def __init__(self, llm, tok, pooler, head_path, thresh=None):
        z = np.load(head_path)
        self.coef = z["coef"].astype(np.float32).ravel()
        self.b = float(np.ravel(z["intercept"])[0])
        self.mean = z["mean"].astype(np.float32)
        self.scale = z["scale"].astype(np.float32)
        if thresh is not None:
            self.thresh = float(thresh)
        elif "thresh" in z.files:
            self.thresh = float(z["thresh"])
        else:
            self.thresh = 0.0
        self.llm, self.tok, self.pooler = llm, tok, pooler
        self.dev = next(llm.parameters()).device
        self.embT = llm.get_input_embeddings()
        self.last_score = 0.0

    @torch.no_grad()
    def score(self, kept_ids, window_ids, q_text):
        from transformers.cache_utils import DynamicCache
        pre = self.tok.encode(f"<｜end▁of▁sentence｜><｜User｜>{q_text}<｜Assistant｜>"
                              f"<think>\n\n</think>\n\n", add_special_tokens=False)
        cache = DynamicCache()
        bos = self.tok.bos_token_id
        self.llm.model(input_ids=torch.tensor([[bos]], device=self.dev),
                       past_key_values=cache, use_cache=True)
        parts = []
        if kept_ids:
            sp = self.pooler(self.embT(torch.tensor([kept_ids], device=self.dev))
                             .float()).to(self.embT.weight.dtype)
            parts.append(sp)
        if window_ids:
            parts.append(self.embT(torch.tensor([window_ids], device=self.dev)))
        if parts:
            self.llm.model(inputs_embeds=torch.cat(parts, 1), past_key_values=cache,
                           use_cache=True)
        outs, hooks = {}, []
        for li in LAYERS:
            attn = self.llm.model.layers[li].self_attn

            def mk(li):
                def fn(_m, _i, o):
                    outs[li] = o.detach()
                return fn
            hooks.append(attn.q_proj.register_forward_hook(mk(li)))
        try:
            self.llm.model(inputs_embeds=self.embT(torch.tensor([pre], device=self.dev)),
                           past_key_values=cache, use_cache=True)
        finally:
            for h in hooks:
                h.remove()
        f = np.concatenate([outs[li][0].mean(0).float().cpu().numpy().ravel()
                            for li in LAYERS])
        self.last_score = float(((f - self.mean) / self.scale) @ self.coef + self.b)
        return self.last_score

    def fires(self, kept_ids, window_ids, q_text):
        return self.score(kept_ids, window_ids, q_text) > self.thresh
