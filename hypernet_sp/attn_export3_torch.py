"""STEP1 (torch port, for CPU/Linux boxes without MLX): identical scenario set and npz
schema to attn_export3.py, but the SP vectors come from the ORIGINAL PyTorch AttnPoolSP
(train_sched_evict.py) loaded from fft_out/pooler.pt, with token embeddings taken from the
fft_hf student. RESULTS.md puts the MLX/PyTorch pooler diff at 2.4e-7, so the two exports
are interchangeable; this one exists so the probe can run end-to-end off-Mac.

Run next to fft_hf/ and fft_out/pooler.pt:  python3 attn_export3_torch.py
"""
import math, numpy as np, torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from attn_scenarios import SCEN                    # single source of truth for the scenarios


def sinusoidal(L, H, device):
    pos = torch.arange(L, device=device).float().unsqueeze(1)
    i = torch.arange(0, H, 2, device=device).float()
    div = torch.exp(-math.log(10000.0) * i / H)
    pe = torch.zeros(L, H, device=device)
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Block(nn.Module):
    def __init__(self, H, heads, ffn):
        super().__init__()
        self.lnq1 = nn.LayerNorm(H); self.lnk = nn.LayerNorm(H)
        self.cross = nn.MultiheadAttention(H, heads, batch_first=True)
        self.lnq2 = nn.LayerNorm(H)
        self.selfa = nn.MultiheadAttention(H, heads, batch_first=True)
        self.lnq3 = nn.LayerNorm(H)
        self.ffn = nn.Sequential(nn.Linear(H, ffn), nn.GELU(), nn.Linear(ffn, H))

    def forward(self, q, past):
        if past is not None and past.size(1) > 0:
            kn = self.lnk(past)
            a, _ = self.cross(self.lnq1(q), kn, kn, need_weights=False)
            q = q + a
        s, _ = self.selfa(self.lnq2(q), self.lnq2(q), self.lnq2(q), need_weights=False); q = q + s
        return q + self.ffn(self.lnq3(q))


class AttnPoolSP(nn.Module):
    def __init__(self, H, n_sp=128, heads=8, layers=3, ffn=2048, target_norm=1.0):
        super().__init__()
        self.H = H; self.n_sp = n_sp
        self.query = nn.Parameter(torch.randn(n_sp, H) * 0.02)
        self.blocks = nn.ModuleList([Block(H, heads, ffn) for _ in range(layers)])
        self.ln_out = nn.LayerNorm(H)
        self.out_scale = nn.Parameter(torch.tensor(float(target_norm)))
        self._pe = None

    def forward(self, past_emb):
        B, L = past_emb.size(0), past_emb.size(1)
        if self._pe is None or self._pe.size(0) < L:
            self._pe = sinusoidal(max(L, 1024), self.H, past_emb.device)
        past = past_emb + self._pe[:L].unsqueeze(0) if L > 0 else past_emb
        q = self.query.unsqueeze(0).expand(B, -1, -1).contiguous()
        for blk in self.blocks:
            q = blk(q, past)
        sp = self.ln_out(q)
        return sp / sp.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self.out_scale.abs()


def load_pooler(path="fft_out/pooler.pt"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k: v.float() for k, v in ck["pooler"].items()}
    a = ck.get("args") or ck.get("src_args") or {}
    n_sp, H = sd["query"].shape
    layers = sum(1 for k in sd if k.endswith(".lnq1.weight"))
    ffn = sd["blocks.0.ffn.0.weight"].shape[0]
    p = AttnPoolSP(H, n_sp=n_sp, heads=(a.get("heads", 8) or 8), layers=layers, ffn=ffn)
    p.load_state_dict(sd, strict=True)
    return p.eval()


def main():
    tok = AutoTokenizer.from_pretrained("fft_hf")
    llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32).eval()
    embT = llm.get_input_embeddings()
    pooler = load_pooler()
    out = {"n": np.array([len(SCEN)]),
           "bos": np.array([tok.bos_token_id if tok.bos_token_id is not None else tok.encode("")[0]]),
           "cat": np.array([c for c, *_ in SCEN])}
    with torch.no_grad():
        for i, (cat, t1, ref, ctrl) in enumerate(SCEN):
            ids = tok.encode(t1, add_special_tokens=False)
            sp = pooler(embT(torch.tensor([ids])).float())
            out[f"sp_{i}"] = sp[0].numpy()
            out[f"ref_{i}"] = np.array(tok.encode(ref, add_special_tokens=False))
            out[f"ctrl_{i}"] = np.array(tok.encode(ctrl, add_special_tokens=False))
    np.savez("attn_probe3.npz", **out)
    print(f"saved {len(SCEN)} scenarios, SP shape {out['sp_0'].shape}")
    print("ATTN_EXPORT3_TORCH_DONE")


if __name__ == "__main__":
    main()
