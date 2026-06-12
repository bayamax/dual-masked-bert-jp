"""block_recall.py — Phase A of the zero-base memory redesign (BLOCKRECALL_V1).

Training-free verbatim recall of tokens that left the exposure window, scored in the
MODEL'S OWN attention space (InfLLM / EM-LLM style), not in an external text-embedding
space. Motivation: EM-LLM (arXiv:2407.09450) shows query-vs-representative-key matching
in the LLM's Q/K space beats SOTA external-embedder RAG by +30.5% LongBench for recalling
a session's own context; gist/SP compression is measurably lossy exactly on high-entropy
values (arXiv:2412.17483: 93.9% -> 40.6% exact recall at 4x compression).

Mechanics:
  - Tokens absorbed into `kept` (i.e. no longer verbatim-visible, SP-gist only) are also
    appended here and sealed into fixed-size blocks.
  - Each sealed block gets PRE-RoPE key projections from a few mid layers (one cheap
    forward per block, lm_head skipped). Pre-RoPE keeps the match position-independent —
    the same trick InfLLM uses by collapsing distant positions.
  - At turn start the current message's pre-RoPE query projections score every block:
    score = sum over layers of mean_q( max_k( mean_heads( q.k / sqrt(D) ) ) ),
    with GQA query heads folded onto their KV head.
  - Top-k blocks are re-injected VERBATIM (token ids) between the SP and the raw window,
    so exact values survive even after SP compression.

The archive is append-only token ids + fp16 keys: ~0.2 MB per 128-token block on CPU
(keys for 3 layers); on device the same blocks live in flash (~8 MB / 1k tok at q4 KV).
"""
import torch


class BlockArchive:
    def __init__(self, llm, tok, layers=(8, 14, 20), block=128, mode="qk", bge=None):
        self.llm, self.tok = llm, tok
        self.layers, self.block, self.mode, self.bge = tuple(layers), block, mode, bge
        cfg = llm.config
        self.Hq = cfg.num_attention_heads
        self.Hkv = cfg.num_key_value_heads
        self.D = cfg.hidden_size // cfg.num_attention_heads
        self.group = self.Hq // self.Hkv
        self.dev = next(llm.parameters()).device     # keys are stored on CPU regardless
        self.buf = []                # pending ids, not yet a full block
        self.blocks = []             # {"ids": [...], "k": [L,T,Hkv,D] fp16, "emb": vec}

    # ---- capture ---------------------------------------------------------------------
    @torch.no_grad()
    def _capture(self, ids, want_q):
        """One lm_head-free forward over `ids`; returns {layer: [T, H, D]} pre-RoPE."""
        outs, hooks = {}, []
        for li in self.layers:
            attn = self.llm.model.layers[li].self_attn
            tgt = attn.q_proj if want_q else attn.k_proj

            def mk(li):
                def fn(_m, _i, o):
                    outs[li] = o.detach()
                return fn
            hooks.append(tgt.register_forward_hook(mk(li)))
        try:
            self.llm.model(input_ids=torch.tensor([ids], device=self.dev))
        finally:
            for h in hooks:
                h.remove()
        H = self.Hq if want_q else self.Hkv
        return {li: outs[li][0].view(len(ids), H, self.D).to(torch.float16).cpu()
                for li in self.layers}

    def _seal(self, ids):
        b = {"ids": list(ids)}
        if self.mode == "qk":
            kc = self._capture(ids, want_q=False)
            b["k"] = torch.stack([kc[li] for li in self.layers])      # [L,T,Hkv,D]
        else:                                                          # bge baseline arm
            b["emb"] = self.bge._encode([self.tok.decode(ids)], is_query=False)[0]
        self.blocks.append(b)

    def extend(self, ids):
        self.buf.extend(ids)
        while len(self.buf) >= self.block:
            self._seal(self.buf[:self.block])
            self.buf = self.buf[self.block:]

    # ---- retrieval -------------------------------------------------------------------
    @torch.no_grad()
    def retrieve(self, query_ids, k=2):
        """Top-k blocks for the current message, chronological order, as one id list.

        The pending buffer (tokens evicted from the window but not yet a full block) is
        scored too, as a temporary pseudo-block — otherwise the most recently evicted
        <=block-1 tokens are invisible to recall while ALSO being outside the window
        (the smoke-run boundary gap: needles parked there were unreachable)."""
        cands = list(self.blocks)
        if len(self.buf) >= 16:
            b = {"ids": list(self.buf)}
            if self.mode == "qk":
                kc = self._capture(b["ids"], want_q=False)
                b["k"] = torch.stack([kc[li] for li in self.layers])
            else:
                b["emb"] = self.bge._encode([self.tok.decode(b["ids"])], is_query=False)[0]
            cands.append(b)
        if not cands or not query_ids:
            return []
        if self.mode == "qk":
            qc = self._capture(query_ids, want_q=True)
            scores = []
            for b in cands:
                s = 0.0
                for lidx, li in enumerate(self.layers):
                    q = qc[li].float()                                 # [Tq,Hq,D]
                    kk = b["k"][lidx].float().repeat_interleave(self.group, 1)
                    sim = torch.einsum("qhd,khd->qhk", q, kk) / self.D ** 0.5
                    s += sim.mean(1).max(-1).values.mean().item()
                scores.append(s)
        else:
            qe = self.bge._encode([self.tok.decode(query_ids)], is_query=True)[0]
            scores = [float((qe * b["emb"]).sum()) for b in cands]
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        out = []
        for i in sorted(top):
            out.extend(cands[i]["ids"])
        self._last_scores = scores
        return out
