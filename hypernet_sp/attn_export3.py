"""STEP1 (MLX), scaled (STATUS 2026-06-10, next-step 1): 30 scenario triples instead of 5.
For each scenario, compress turn-1 into SP and tokenize a REFERENTIAL turn-2 (needs a turn-1
value, which therefore lives ONLY in the SP) and a matched CONTROL turn-2 (self-contained —
the value is in the question, so the SP isn't needed). Same SP for both. Categories cover
arithmetic chains, codes/IDs, names, schedule, preferences and measurements so the probe's
heads aren't an artifact of one task family. Save for attn_probe3.py.

Run next to sp_mlx.py in the HF repo:  python3 attn_export3.py
"""
import numpy as np, mlx.core as mx
import sp_mlx

M = sp_mlx.get()
tok, pooler, embT = M["tok"], M["pooler"], M["embT"]
emb = lambda ids: embT(mx.array([ids]))

from attn_scenarios import SCEN
out = {"n": np.array([len(SCEN)]),
       "bos": np.array([tok.bos_token_id if tok.bos_token_id is not None else tok.encode("")[0]]),
       "cat": np.array([c for c, *_ in SCEN])}
for i, (cat, t1, ref, ctrl) in enumerate(SCEN):
    sp = pooler.forward(emb(tok.encode(t1, add_special_tokens=False)).astype(mx.float32))
    out[f"sp_{i}"] = np.array(sp.astype(mx.float32))[0]
    out[f"ref_{i}"] = np.array(tok.encode(ref, add_special_tokens=False))
    out[f"ctrl_{i}"] = np.array(tok.encode(ctrl, add_special_tokens=False))
np.savez("attn_probe3.npz", **out)
print(f"saved {len(SCEN)} scenarios, SP shape {out['sp_0'].shape}")
print("ATTN_EXPORT3_DONE")
