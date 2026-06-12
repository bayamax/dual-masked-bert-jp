"""ensemble_archive.py — BlockArchive-compatible ensemble scorer (PHASEB e2e).

Scores evicted blocks with z(BGE cosine) + z(learned indexer on contextual keys).
Contextual keys are harvested with ONE hooked forward over the history (in streaming
production they arrive for free during normal rebuild prefills; trained on full-context
keys, so this harvest matches training). Implements the .blocks/.buf/.extend/.retrieve
interface AppSession expects, with retrieval precomputed per question text.
Held-out retrieval top-2: bge .894 / indexer .883 / ensemble .978 (n=180).
"""
import numpy as np
import torch

LAYERS, BLOCK = (8, 14, 20), 128


class EnsembleArchive:
    def __init__(self, llm, tok, bge, indexer_npz="phaseb_indexer.npz"):
        self.llm, self.tok, self.bge = llm, tok, bge
        ix = np.load(indexer_npz)
        self.Aq, self.Bk, w = ix["Aq"], ix["Bk"], ix["w"]
        L, Hq, Hkv, D, d = ix["meta"]
        self.Hq, self.Hkv, self.D, self.group = int(Hq), int(Hkv), int(D), int(Hq // Hkv)
        e = np.exp(w - w.max()); self.wsm = e / e.sum()
        self.dev = next(llm.parameters()).device
        self.blocks, self.buf, self._k = [], [], None

    def _hooked(self, ids, want_q):
        outs, hooks = {}, []
        for li in LAYERS:
            attn = self.llm.model.layers[li].self_attn
            tgt = attn.q_proj if want_q else attn.k_proj

            def mk(li):
                def fn(_m, _i, o):
                    outs[li] = o.detach()
                return fn
            hooks.append(tgt.register_forward_hook(mk(li)))
        try:
            with torch.no_grad():
                self.llm.model(input_ids=torch.tensor([ids], device=self.dev))
        finally:
            for h in hooks:
                h.remove()
        H = self.Hq if want_q else self.Hkv
        return {li: outs[li][0].view(len(ids), H, self.D).float().cpu() for li in LAYERS}

    def extend(self, ids):
        self.buf.extend(ids)
        while len(self.buf) >= BLOCK:
            self.blocks.append({"ids": self.buf[:BLOCK]})
            self.buf = self.buf[BLOCK:]
        self._k = None                                  # features stale

    def _featurize(self):
        ids = [t for b in self.blocks for t in b["ids"]]
        kc = self._hooked(ids, want_q=False)            # contextual keys, full history
        nB = len(self.blocks)
        kf = []
        for li in LAYERS:
            k = kc[li].view(nB, BLOCK, self.Hkv, self.D)
            kf.append(torch.cat([k.mean(1), k.amax(1)], -1).numpy())
        self._k = np.stack(kf, 1)                       # [nB, L, Hkv, 2D]
        texts = [self.tok.decode(b["ids"]) for b in self.blocks]
        self._emb = self.bge._encode(texts, is_query=False)

    def retrieve(self, query_ids, k=2):
        if not self.blocks:
            return []
        if self._k is None:
            self._featurize()
        q_text = self.tok.decode(query_ids)
        pre = self.tok.encode(f"<｜end▁of▁sentence｜><｜User｜>{q_text}<｜Assistant｜>"
                              f"<think>\n\n</think>\n\n", add_special_tokens=False)
        qc = self._hooked(pre, want_q=True)
        qf = np.stack([qc[li].mean(0).view(self.Hq, self.D).numpy() for li in LAYERS])
        qp = np.einsum("lhd,lhde->lhe", qf, self.Aq)
        kp = np.einsum("blkf,lkfe->blke", self._k.astype(np.float32), self.Bk)
        kp = np.repeat(kp, self.group, axis=2)
        sim = np.maximum(np.einsum("lhe,blhe->blh", qp, kp), 0)
        s_idx = np.einsum("blh,lh->b", sim, self.wsm)
        s_bge = self._emb @ self.bge._encode([q_text], is_query=True)[0]

        def z(s):
            return (s - s.mean()) / (s.std() + 1e-8)
        s = z(s_idx) + z(np.asarray(s_bge))
        top = sorted(np.argsort(-s)[:k].tolist())
        out = []
        for i in top:
            out.extend(self.blocks[i]["ids"])
        return out
