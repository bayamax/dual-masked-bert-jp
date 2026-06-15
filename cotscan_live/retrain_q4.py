"""retrain_q4.py — recalibrate GATE + BGE bridge on 4-bit features, with FP32-attention labels.

On-device the 1.5B runs in 4-bit (nf4). The heads were trained on fp32 q, so we recalibrate on
4-bit q. Labels stay principled (attention mass, NOT hand-made): bitsandbytes 4-bit returns NaN
for output_attentions, so we take the recall LABELS (which positions/blocks) from the FP32 model
and the q FEATURES from the 4-BIT model on the SAME sequences/positions. Refit gate (logistic) +
bge_head (bridge). Report held-out BEFORE (fp32 head on 4-bit feats) vs AFTER (refit). Saves q4 heads.
"""
import os, json, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import DynamicCache
except Exception:
    from transformers.cache_utils import DynamicCache
from attn_export3_torch import load_pooler
import mix_scan as M
import gate_contrastive as GC

DEV = "cuda"
LAYERS, WINDOW, BLOCK = M.LAYERS, M.WINDOW, M.BLOCK
N_MIX_DOL = int(os.environ.get("N_MIX_DOL", "55"))
N_MIX_CHAT = int(os.environ.get("N_MIX_CHAT", "55"))
N_NEG_TR = int(os.environ.get("N_NEG_TR", "110"))
NEG_PER_SEQ = int(os.environ.get("NEG_PER_SEQ", "40"))
POS_FRAC = M.POS_FRAC
OUT = os.environ.get("OUT_DIR", "results_q4train"); ART = os.environ.get("ART_DIR", "artifacts_q4")
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    os.makedirs(OUT, exist_ok=True); open(os.path.join(OUT, "live.log"), "w").write("\n".join(LOG))


@torch.no_grad()
def sp_qfeat(model, embT, pooler, seq, dims):
    """SP-context pre-RoPE q at every window position, from `model` (the 4-bit one). [win,L,Hq,D]."""
    Hq, Hkv, D = dims
    if len(seq) < WINDOW + BLOCK * 4 + 4:
        return None
    n_evict = ((len(seq) - WINDOW) // BLOCK) * BLOCK
    kept, window = seq[:n_evict], seq[n_evict:]; win_len = len(seq) - n_evict
    c = DynamicCache()
    model.model(input_ids=torch.tensor([[seq[0]]], device=DEV), past_key_values=c, use_cache=True)
    sp = pooler(embT(torch.tensor([kept], device=DEV)).float()).to(embT.weight.dtype)
    model.model(inputs_embeds=sp, past_key_values=c, use_cache=True)
    with M.Cap(model, LAYERS, q=True, k=False) as cap:
        model.model(inputs_embeds=embT(torch.tensor([window], device=DEV)), past_key_values=c, use_cache=True)
    return np.stack([cap.q[li][0].float().cpu().numpy().reshape(win_len, Hq, D) for li in LAYERS], 1)


def collect_dual(seqs, fp32, q4, tok, pooler, embT32, embTq4, dims, bge, tag, t0):
    """labels (mass) from fp32 M.process; gate/bge q-features from the 4-bit model. Same positions."""
    Xg, yg, IX = [], [], []
    for n, (_s, seq) in enumerate(seqs):
        try:
            r = M.process(fp32, tok, pooler, embT32, seq, dims, bge=bge)     # fp32 labels + bge_k
            q4f = sp_qfeat(q4, embTq4, pooler, seq, dims)                    # 4-bit q features
        except Exception as e:
            log(f"  {tag} {n} fail: {e}"); continue
        if r is None or q4f is None or q4f.shape[0] != r["win_len"]:
            continue
        mm = r["maxmass"]; thr = np.quantile(mm, 1 - POS_FRAC)
        pos = np.where(mm >= max(thr, 0.05))[0]; neg_all = np.where(mm < max(thr, 0.05))[0]
        rng = np.random.RandomState(n)
        neg = rng.choice(neg_all, min(len(neg_all), max(1, len(pos) * 6)), replace=False) if len(neg_all) else np.array([], int)
        for p in list(pos) + list(neg):
            Xg.append(q4f[p].reshape(-1)); yg.append(int(mm[p] >= max(thr, 0.05)))
        for p in pos:
            row = {"q": q4f[p].astype(np.float32), "gold": int(r["goldblk"][p]),
                   "labels": r["blockmass"][p][None, :].astype(np.float32)}
            if "bge_k" in r:
                row["bge_k"] = r["bge_k"].astype(np.float32); row["bge_q"] = np.zeros(r["bge_k"].shape[1], np.float32)
            IX.append(row)
        if (n + 1) % 20 == 0:
            log(f"  {tag} {n+1}/{len(seqs)} pts={len(yg)} pos={int(np.sum(yg))} IX={len(IX)} t={time.time()-t0:.0f}s")
    return np.array(Xg, np.float32), np.array(yg, np.int32), IX


def harvest_neg_q4(q4, embTq4, pooler, dims, seqs, per):
    X = []
    for si, seq in enumerate(seqs):
        try:
            q4f = sp_qfeat(q4, embTq4, pooler, seq, dims)
        except Exception:
            continue
        if q4f is None:
            continue
        win = q4f.shape[0]; lo = max(0, win - per)
        for p in range(lo, win):
            X.append(q4f[p].reshape(-1))
    return np.array(X, np.float32)


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); os.makedirs(ART, exist_ok=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from sentence_transformers import SentenceTransformer
    from transformers import BitsAndBytesConfig
    from huggingface_hub import hf_hub_download
    R = "baya1116/hypernet-sp-distill"; htok = os.environ["HF_TOKEN"]
    tok = AutoTokenizer.from_pretrained("fft_hf")
    fp32 = AutoModelForCausalLM.from_pretrained("fft_hf", torch_dtype=torch.float32,
                                                attn_implementation="eager").eval().to(DEV)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    q4 = AutoModelForCausalLM.from_pretrained("fft_hf", quantization_config=bnb, device_map={"": 0}).eval()
    pooler = load_pooler().to(DEV).eval(); embT32 = fp32.get_input_embeddings(); embTq4 = q4.get_input_embeddings()
    bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device=DEV)
    cfg = fp32.config; Hq = cfg.num_attention_heads; Hkv = cfg.num_key_value_heads
    D = cfg.hidden_size // Hq; dims = (Hq, Hkv, D); bos = tok.bos_token_id
    log(f"[q4] fp32-labels + 4bit-features | mix={N_MIX_DOL}/{N_MIX_CHAT} neg={N_NEG_TR}")

    dol = M.dolphin_seqs(tok, N_MIX_DOL + 12, bos); uc = M.ultrachat_seqs(tok, N_MIX_CHAT + 12, bos)
    Xm, ym, IXm = collect_dual(dol[:N_MIX_DOL] + uc[:N_MIX_CHAT], fp32, q4, tok, pooler, embT32, embTq4, dims, bge, "MIX", t0)
    Xdt, ydt, IXdt = collect_dual(dol[N_MIX_DOL:N_MIX_DOL + 12], fp32, q4, tok, pooler, embT32, embTq4, dims, bge, "DOLTE", t0)
    Xch, ych, IXch = collect_dual(uc[N_MIX_CHAT:N_MIX_CHAT + 12], fp32, q4, tok, pooler, embT32, embTq4, dims, bge, "CHATTE", t0)
    pos = np.vstack([Xdt[ydt == 1], Xch[ych == 1]]) if (int(ydt.sum()) + int(ych.sum())) else np.zeros((0, Xm.shape[1]), np.float32)
    neg_seqs = [GC.casual_seq(tok, s, bos) for s in range(N_NEG_TR)]
    Xneg = harvest_neg_q4(q4, embTq4, pooler, dims, [s for s in neg_seqs if M._seq_ok(s)], NEG_PER_SEQ)
    log(f"[q4] mix pts={len(ym)} pos={int(ym.sum())} | held pos={len(pos)} | hard-neg={len(Xneg)} | bge IX tr={len(IXm)} te={len(IXdt)+len(IXch)}")

    zf = np.load(hf_hub_download(R, "trigger_experiment/recall_contrastive_v3/artifacts/gate.npz", token=htok))
    cf, bf, mf, sf = zf["coef"].ravel(), float(np.ravel(zf["intercept"])[0]), zf["mean"], zf["scale"]
    def fp32_gate(X): return ((X - mf) / sf) @ cf + bf
    sc = StandardScaler().fit(np.vstack([Xm, Xneg]))
    clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(
        sc.transform(np.vstack([Xm, Xneg])), np.concatenate([ym, np.zeros(len(Xneg), np.int32)]))
    def refit_gate(X): return clf.decision_function(sc.transform(X))
    def auc(fn, X, y):
        y = np.asarray(y)
        return round(float(roc_auc_score(y, fn(X))), 4) if 0 < int(y.sum()) < len(y) else None
    res = {"gate_auc": {"cot": {"fp32_head_on_q4": auc(fp32_gate, Xdt, ydt), "q4_refit": auc(refit_gate, Xdt, ydt)},
                        "chat": {"fp32_head_on_q4": auc(fp32_gate, Xch, ych), "q4_refit": auc(refit_gate, Xch, ych)}}}
    ps = refit_gate(pos); thr = float(np.quantile(ps, 0.1)) if len(ps) else 0.0
    held_neg = harvest_neg_q4(q4, embTq4, pooler, dims, [GC.casual_seq(tok, 80000 + s, bos) for s in range(40)], NEG_PER_SEQ)
    res["gate_thresh_q4"] = round(thr, 4)
    res["gate_false_fire_q4@recall90"] = round(float(np.mean(refit_gate(held_neg) >= thr)), 4) if len(held_neg) else None
    np.savez(os.path.join(ART, "gate.npz"), coef=clf.coef_.ravel().astype(np.float32),
             intercept=np.float32(clf.intercept_[0]), mean=sc.mean_.astype(np.float32),
             scale=sc.scale_.astype(np.float32), thresh=np.float32(thr))
    log(f"[GATE] AUC cot {res['gate_auc']['cot']} chat {res['gate_auc']['chat']}")

    IXte = IXdt + IXch
    def bge_top2(state, qd, kd, dd, rows):
        m = M.BGEHead(qd, kd, dd).to(DEV); m.load_state_dict(state); m.eval(); h = 0
        with torch.no_grad():
            for r in rows:
                q = torch.tensor(r["q"].reshape(-1), device=DEV); k = torch.tensor(r["bge_k"], dtype=torch.float32, device=DEV)
                h += int(r["gold"]) in m(q, k).topk(min(2, len(k))).indices.tolist()
        return round(h / max(len(rows), 1), 4)
    zb = np.load(hf_hub_download(R, "trigger_experiment/recall_mix_v1/artifacts/bge_head.npz", token=htok))
    qd, kd, dd = int(zb["qdim"]), int(zb["kdim"]), int(zb["d"])
    fp32_state = {k: torch.tensor(zb[k]) for k in zb.files if k not in ("qdim", "kdim", "d")}
    before = bge_top2(fp32_state, qd, kd, dd, IXte) if IXte else None
    best, bstate, (qd2, kd2, dd2) = M.train_bge_head([dict(r) for r in IXm], IXte) if (len(IXm) > 5 and len(IXte) > 2) else (None, None, (4608, 768, 128))
    res["bge_top2"] = {"fp32_head_on_q4": before, "q4_refit": round(best, 4) if best is not None else None}
    log(f"[BGE] top2 fp32head_on_q4={before} q4_refit={best}")
    if bstate is not None:
        sd = {k: v.numpy() for k, v in bstate.items()}
        np.savez(os.path.join(ART, "bge_head.npz"), qdim=np.int64(qd2), kdim=np.int64(kd2), d=np.int64(dd2), **sd)
    res["final"] = True
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    open(os.path.join(OUT, ".done"), "w").write("ok")
    log("\n=== 4-BIT RECALIBRATION (fp32 labels, 4-bit features) ===")
    log(f"  GATE AUC cot {res['gate_auc']['cot']['fp32_head_on_q4']}->{res['gate_auc']['cot']['q4_refit']} "
        f"chat {res['gate_auc']['chat']['fp32_head_on_q4']}->{res['gate_auc']['chat']['q4_refit']}")
    log(f"  BGE  top2 {res['bge_top2']['fp32_head_on_q4']}->{res['bge_top2']['q4_refit']}")
    log(f"  gate thresh_q4={thr:.3f} false-fire@recall90={res['gate_false_fire_q4@recall90']}")
    log(f"DONE t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
