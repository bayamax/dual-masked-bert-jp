"""gate_v3.py — close the train/generation gap with GENERATION-distribution hard negatives.

v2 cut held-out casual false-fire to 3% but generation casual-fire stayed ~62%: the gate scores
SELF-GENERATED casual reply positions higher than the teacher-forced ones it trained on. Fix:
harvest the gate's pre-RoPE q at the ACTUAL decision points reached while the model generates a
casual reply under SP compression (the real inference distribution), label them 0, and retrain.

Positives stay 100% mass-labeled (mix: Dolphin + UltraChat). Negatives = teacher-forced casual
tails (v2) + GENERATION casual q-snapshots (new). Reports held-out false-fire on BOTH a
teacher-forced held set and a generation held set, mix vs v3. Saves gate.npz.
"""
import os, json, time, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import DynamicCache
except Exception:
    from transformers.cache_utils import DynamicCache
from attn_export3_torch import load_pooler
from block_recall import BlockArchive
import mix_scan as M
from gate_contrastive import FACTS, CASUAL, approx_tok, to_ids, casual_seq, harvest_neg

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS, WINDOW, BLOCK = M.LAYERS, M.WINDOW, M.BLOCK
RW = int(os.environ.get("RW", "512"))
C = 64
N_GEN_TR = int(os.environ.get("N_GEN_TR", "140"))
N_GEN_TE = int(os.environ.get("N_GEN_TE", "40"))
N_NEG_TR = int(os.environ.get("N_NEG_TR", "120"))
NEG_PER_SEQ = int(os.environ.get("NEG_PER_SEQ", "40"))
N_MIX_DOL = int(os.environ.get("N_MIX_DOL", "70"))
N_MIX_CHAT = int(os.environ.get("N_MIX_CHAT", "70"))
GEN_CAP = int(os.environ.get("GEN_CAP", "150"))
OUT = os.environ.get("OUT_DIR", "results_gate_v3")
ART = os.environ.get("ART_DIR", "artifacts_v3")
EOS = "<｜end▁of▁sentence｜>"
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    os.makedirs(OUT, exist_ok=True); open(os.path.join(OUT, "live.log"), "w").write("\n".join(LOG))


def casual_feed(tok, seed, bos):
    """A casual scenario ending at <｜Assistant｜> so we GENERATE its reply (decision region)."""
    r = random.Random(seed)
    nf = r.choice([1, 2, 3])
    turns = []
    for k in r.sample(range(len(FACTS)), nf):
        kw, val = FACTS[k]
        turns.append((f"Important to note: the {kw} is {val}.", "Got it, noted."))
    fills = r.sample(CASUAL, len(CASUAL)); i = 0
    while approx_tok(turns) < WINDOW + r.choice([360, 500, 640]):
        turns.append(fills[i % len(fills)]); i += 1
    cu, _ = r.choice(CASUAL)                                 # final casual user turn, no reply yet
    ids = to_ids(tok, turns) + tok.encode(EOS + f"<｜User｜>{cu}<｜Assistant｜>", add_special_tokens=False)
    return [bos] + ids


def emb(embT, ids):
    if ids:
        return embT(torch.tensor([ids], device=DEV))
    return torch.zeros(1, 0, embT.weight.shape[1], dtype=embT.weight.dtype, device=DEV)


@torch.no_grad()
def gen_negatives(llm, tok, pooler, embT, dims, feeds, tag, t0):
    """Run the SP generation loop on each casual feed; snapshot the gate q at every rebuild
    decision point (once eviction exists) -> label-0 features from the TRUE inference path."""
    Hq, Hkv, D = dims
    X = []
    for si, feed in enumerate(feeds):
        gen, kept, absorbed = list(feed), [], 0
        gq = {}; hooks = []
        for li in LAYERS:
            def mk(li):
                def fn(_m, _i, o): gq[li] = o.detach()[0, -1].float().cpu().numpy().ravel()
                return fn
            hooks.append(llm.model.layers[li].self_attn.q_proj.register_forward_hook(mk(li)))
        cache = DynamicCache(); bos = tok.bos_token_id
        llm(input_ids=torch.tensor([[bos]], device=DEV), past_key_values=cache, use_cache=True)
        MQ = cache.get_seq_length(); newtok = 0; done = False
        fed = False
        while not done:
            c0 = len(gen); R = min(c0, RW); nd = c0 - R
            if nd > absorbed:
                kept.extend(gen[absorbed:nd]); absorbed = nd
            # snapshot the gate feature at this rebuild (eviction present, mid casual reply)
            if fed and kept and all(li in gq for li in LAYERS):
                X.append(np.concatenate([gq[li] for li in LAYERS]).astype(np.float32))
            parts = []
            if kept:
                parts.append(pooler(emb(embT, kept).float()).to(embT.weight.dtype))
            if R > 0:
                parts.append(emb(embT, gen[c0 - R:c0]))
            cache.crop(MQ)
            last = llm(inputs_embeds=torch.cat(parts, 1), past_key_values=cache,
                       use_cache=True).logits[:, -1, :].float()
            for _ in range(C):
                t = int(last[0].argmax())
                if t == tok.eos_token_id or newtok >= GEN_CAP:
                    done = True; break
                newtok += 1; gen.append(t); fed = True
                last = llm(inputs_embeds=emb(embT, [t]), past_key_values=cache,
                           use_cache=True).logits[:, -1, :].float()
            if done:
                break
        for h in hooks:
            h.remove()
        if (si + 1) % 20 == 0:
            log(f"  gen-neg {tag} {si+1}/{len(feeds)} rows={len(X)} t={time.time()-t0:.0f}s")
    return np.array(X, np.float32)


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); os.makedirs(ART, exist_ok=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    tok = AutoTokenizer.from_pretrained("fft_hf")
    try:
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", torch_dtype=torch.float32, attn_implementation="eager")
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32, attn_implementation="eager")
    llm = llm.eval().to(DEV)
    pooler = load_pooler().to(DEV).eval(); embT = llm.get_input_embeddings()
    cfg = llm.config; Hq = cfg.num_attention_heads; Hkv = cfg.num_key_value_heads
    D = cfg.hidden_size // Hq; dims = (Hq, Hkv, D); bos = tok.bos_token_id
    log(f"[v3] gen-neg tr/te={N_GEN_TR}/{N_GEN_TE} cap={GEN_CAP} | tf-neg={N_NEG_TR} mix={N_MIX_DOL}/{N_MIX_CHAT}")

    dol = M.dolphin_seqs(tok, N_MIX_DOL + 15, bos)
    uc = M.ultrachat_seqs(tok, N_MIX_CHAT + 15, bos)
    Xm, ym, _ = M.collect(dol[:N_MIX_DOL] + uc[:N_MIX_CHAT], llm, tok, pooler, embT, dims, "MIX_TR", t0)
    Xdt, ydt, _ = M.collect(dol[N_MIX_DOL:N_MIX_DOL + 15], llm, tok, pooler, embT, dims, "DOL_TE", t0)
    Xch, ych, _ = M.collect(uc[N_MIX_CHAT:N_MIX_CHAT + 15], llm, tok, pooler, embT, dims, "CHAT_TE", t0)
    pos = np.vstack([Xdt[ydt == 1], Xch[ych == 1]])
    log(f"[mix] tr pts={len(ym)} pos={int(ym.sum())} | held real pos={len(pos)}")

    # teacher-forced tail negatives (v2 style)
    tf_seqs = [casual_seq(tok, s, bos) for s in range(N_NEG_TR)]
    Xtf = harvest_neg(llm, tok, pooler, embT, dims, [s for s in tf_seqs if M._seq_ok(s)], NEG_PER_SEQ)
    # GENERATION-distribution negatives (new)
    gfeeds_tr = [casual_feed(tok, 1000 + s, bos) for s in range(N_GEN_TR)]
    gfeeds_te = [casual_feed(tok, 50000 + s, bos) for s in range(N_GEN_TE)]
    Xgn = gen_negatives(llm, tok, pooler, embT, dims, gfeeds_tr, "TR", t0)
    Xgn_te = gen_negatives(llm, tok, pooler, embT, dims, gfeeds_te, "TE", t0)
    log(f"[neg] tf={len(Xtf)} gen-tr={len(Xgn)} gen-te={len(Xgn_te)}")

    def fit(X, y):
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
        return sc, clf
    def sco(scl, X): return scl[1].decision_function(scl[0].transform(X))
    sc_mix, clf_mix = fit(Xm, ym)                                          # baseline (recall_mix style)
    Xall = np.vstack([Xm, Xtf, Xgn]); yall = np.concatenate([ym, np.zeros(len(Xtf) + len(Xgn), np.int32)])
    sc_v3, clf_v3 = fit(Xall, yall)

    def pr(scl, neg, tr):
        ps = sco(scl, pos); ns = sco(scl, neg)
        thr = float(np.quantile(ps, 1 - tr))
        return round(float(np.mean(ns >= thr)), 4)
    res = {"elapsed_s": round(time.time()-t0, 1),
           "data": {"mix_pos": int(ym.sum()), "tf_neg": int(len(Xtf)), "gen_neg": int(len(Xgn)),
                    "held_pos": int(len(pos)), "held_gen_neg": int(len(Xgn_te))},
           "auc": {"cot": {"mix": round(float(roc_auc_score(ydt, sco((sc_mix, clf_mix), Xdt))), 4),
                           "v3": round(float(roc_auc_score(ydt, sco((sc_v3, clf_v3), Xdt))), 4)},
                   "chat": {"mix": round(float(roc_auc_score(ych, sco((sc_mix, clf_mix), Xch))), 4),
                            "v3": round(float(roc_auc_score(ych, sco((sc_v3, clf_v3), Xch))), 4)}},
           "false_fire_GEN": {  # the metric that matters: false-fire on the GENERATION negatives
               "recall90": {"mix": pr((sc_mix, clf_mix), Xgn_te, 0.9), "v3": pr((sc_v3, clf_v3), Xgn_te, 0.9)},
               "recall80": {"mix": pr((sc_mix, clf_mix), Xgn_te, 0.8), "v3": pr((sc_v3, clf_v3), Xgn_te, 0.8)}}}
    log(f"[GEN false-fire] @recall90: mix={res['false_fire_GEN']['recall90']['mix']} -> v3={res['false_fire_GEN']['recall90']['v3']}")
    log(f"[GEN false-fire] @recall80: mix={res['false_fire_GEN']['recall80']['mix']} -> v3={res['false_fire_GEN']['recall80']['v3']}")
    log(f"[AUC] cot {res['auc']['cot']['mix']}->{res['auc']['cot']['v3']} chat {res['auc']['chat']['mix']}->{res['auc']['chat']['v3']}")

    ps = sco((sc_v3, clf_v3), pos); thr = float(np.quantile(ps, 0.1))     # 90% recall operating point
    res["saved_thresh"] = round(thr, 4)
    np.savez(os.path.join(ART, "gate.npz"),
             coef=clf_v3.coef_.ravel().astype(np.float32), intercept=np.float32(clf_v3.intercept_[0]),
             mean=sc_v3.mean_.astype(np.float32), scale=sc_v3.scale_.astype(np.float32), thresh=np.float32(thr))
    res["final"] = True
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    open(os.path.join(OUT, ".done"), "w").write("ok")
    log(f"\n=== GATE v3 (generation-distribution hard negatives) ===")
    log(f"  GEN false-fire @recall90: mix {res['false_fire_GEN']['recall90']['mix']} -> v3 {res['false_fire_GEN']['recall90']['v3']}")
    log(f"  AUC preserved: cot {res['auc']['cot']['v3']} chat {res['auc']['chat']['v3']}")
    log(f"  saved thresh={thr:.3f}")
    log(f"DONE t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
