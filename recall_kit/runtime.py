"""recall_kit.runtime — production wiring of the per-position recall mechanism.

prepare(seq_ids) builds, from the PRODUCTION-available context [BOS][SP(evicted)][window]:
  - qfeat[win, L, Hq, D] : pre-RoPE query at each window position (the live hidden state),
  - keys[nB, L, Hkv, 2D] : pooled pre-RoPE block keys (for QKIndexer; harvested for free),
  - block_ids / block_texts : for the state-free BGERetriever (embed on demand).
decide(ctx, p) runs the GATE on position p and, only when it fires, retrieves the top
evicted block. This is the whole loop: detect rare recall moments, then pull one block.
"""
import numpy as np
import torch

LAYERS = (8, 14, 20)
WINDOW, BLOCK = 512, 128


class _Cap:
    def __init__(self, model, layers, q, k):
        self.model, self.layers, self.wq, self.wk = model, layers, q, k
        self.q, self.k, self.h = {}, {}, []
    def __enter__(self):
        for li in self.layers:
            a = self.model.model.layers[li].self_attn
            if self.wq:
                self.h.append(a.q_proj.register_forward_hook(
                    lambda m, i, o, li=li: self.q.__setitem__(li, o.detach())))
            if self.wk:
                self.h.append(a.k_proj.register_forward_hook(
                    lambda m, i, o, li=li: self.k.__setitem__(li, o.detach())))
        return self
    def __exit__(self, *a):
        for h in self.h:
            h.remove()


class Ctx:
    pass


class RecallRuntime:
    def __init__(self, model, tok, pooler, gate, qk=None, bge=None, device="cpu"):
        self.model, self.tok, self.pooler, self.gate = model, tok, pooler, gate
        self.qk, self.bge, self.device = qk, bge, device
        cfg = model.config
        self.Hq = cfg.num_attention_heads
        self.Hkv = cfg.num_key_value_heads
        self.D = cfg.hidden_size // self.Hq
        self.embT = model.get_input_embeddings()

    def _emb(self, ids):
        return self.embT(torch.tensor([ids], device=self.device))

    @torch.no_grad()
    def prepare(self, seq_ids):
        try:
            from transformers import DynamicCache
        except Exception:
            from transformers.cache_utils import DynamicCache
        n_evict = ((len(seq_ids) - WINDOW) // BLOCK) * BLOCK
        nB = n_evict // BLOCK
        if nB < 1:
            return None
        kept, window = seq_ids[:n_evict], seq_ids[n_evict:]
        block_ids = [seq_ids[i * BLOCK:(i + 1) * BLOCK] for i in range(nB)]

        # contextual pooled block keys (one full-KV prefill, capture pre-RoPE k_proj).
        # Match the training-time harvest exactly: full-model call, no cache.
        with _Cap(self.model, LAYERS, q=False, k=True) as cap:
            self.model(input_ids=torch.tensor([seq_ids], device=self.device),
                       use_cache=False)
        kf = []
        for li in LAYERS:
            k = cap.k[li][0][:n_evict].view(nB, BLOCK, self.Hkv, self.D).float()
            kf.append(torch.cat([k.mean(1), k.amax(1)], -1).cpu())
        keys = torch.stack(kf, 1).half().float().numpy()   # [nB,L,Hkv,2D] (fp16 like train)

        # SP-compressed context -> pre-RoPE q at window positions
        c = DynamicCache()
        self.model.model(input_ids=torch.tensor([[self.tok.bos_token_id]], device=self.device),
                         past_key_values=c, use_cache=True)
        sp = self.pooler(self._emb(kept).float()).to(self.embT.weight.dtype)
        self.model.model(inputs_embeds=sp, past_key_values=c, use_cache=True)
        with _Cap(self.model, LAYERS, q=True, k=False) as cap2:
            self.model.model(inputs_embeds=self._emb(window), past_key_values=c, use_cache=True)
        win_len = len(window)
        q_layers = [cap2.q[li][0].float().cpu().numpy().reshape(win_len, self.Hq, self.D)
                    for li in LAYERS]
        ctx = Ctx()
        ctx.qfeat = np.stack(q_layers, 1)                  # [win, L, Hq, D]
        ctx.keys = keys
        ctx.block_ids = block_ids
        ctx.block_texts = [self.tok.decode(b) for b in block_ids]
        ctx.nB = nB
        ctx.win_len = win_len
        return ctx

    def decide(self, ctx, p, backend="qk", topk=1):
        """Gate position p; if it fires, return the top-k evicted block indices."""
        qf = ctx.qfeat[p]
        if not self.gate.fires(qf.reshape(-1)):
            return {"fired": False, "blocks": []}
        if backend == "qk" and self.qk is not None:
            order = self.qk.rank(qf, ctx.keys)
        elif backend == "bge" and self.bge is not None:
            order = self.bge.rank(qf.reshape(-1), ctx.block_texts)
        else:
            raise ValueError(f"backend {backend} unavailable")
        return {"fired": True, "blocks": order[:topk]}
