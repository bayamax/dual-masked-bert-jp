"""gate_v3_q4.py — 4-bit recall GATE with GENERATION-distribution contrastive negatives.

The on-device target runs the 1.5B in 4-bit (nf4). The open problem is recall FALSE-FIRE: in
generation the gate over-fires on casual turns (~62% per-turn at fp32; ~67% at 4-bit thresh -3.5).
gate_v3 showed the fix on fp32 — harvest the gate q at the model's ACTUAL self-generated casual
decision points (under SP compression) as label-0 hard negatives, then retrain. This script does
exactly that ON THE 4-BIT MODEL, so the gate is calibrated to the on-device inference distribution.

Principled labels (never hand-made):
  * POSITIVES = mass-labeled recall (full-KV attention concentrates on a single evicted block),
    taken from the FP32 model (bitsandbytes 4-bit returns NaN for output_attentions), with the
    gate q FEATURES read from the 4-BIT model at the same window positions.
  * NEGATIVES = teacher-forced casual tails (4-bit feats) + GENERATION-distribution casual
    q-snapshots produced by running the SP loop on the 4-BIT model itself (the true on-device path).

Reports, both for the current fp32 gate scored on 4-bit features (baseline) and the new 4-bit
contrastive gate: cot/chat detection AUC and GENERATION false-fire @recall90/80. Saves gate.npz
(4-bit-calibrated, thresh = 90%-recall operating point). The threshold is then swept end-to-end
by recall_gen.py (Q4=1) for the recall/false-fire knee.
"""
import os, json, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from attn_export3_torch import load_pooler
import mix_scan as M
import gate_contrastive as GC
import gate_v3 as V3
import retrain_q4 as Q4

DEV = "cuda"
LAYERS, WINDOW, BLOCK = M.LAYERS, M.WINDOW, M.BLOCK
N_MIX_DOL = int(os.environ.get("N_MIX_DOL", "45"))
N_MIX_CHAT = int(os.environ.get("N_MIX_CHAT", "45"))
N_MIX_TE = int(os.environ.get("N_MIX_TE", "12"))
N_NEG_TR = int(os.environ.get("N_NEG_TR", "70"))          # teacher-forced casual tails
NEG_PER_SEQ = int(os.environ.get("NEG_PER_SEQ", "30"))
N_GEN_TR = int(os.environ.get("N_GEN_TR", "70"))          # generation-distribution negatives
N_GEN_TE = int(os.environ.get("N_GEN_TE", "30"))
GEN_CAP = int(os.environ.get("GEN_CAP", "140"))
OUT = os.environ.get("OUT_DIR", "results_gate_v3_q4")
ART = os.environ.get("ART_DIR", "artifacts_v3_q4")
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    os.makedirs(OUT, exist_ok=True); open(os.path.join(OUT, "live.log"), "w").write("\n".join(LOG))


def _finite(X):
    if len(X) == 0:
        return X
    return X[np.isfinite(X).all(1)]


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); os.makedirs(ART, exist_ok=True)
    # patch GEN_CAP into gate_v3 so its generation loop matches our budget
    V3.GEN_CAP = GEN_CAP
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from transformers import BitsAndBytesConfig
    from huggingface_hub import hf_hub_download
    R = "baya1116/hypernet-sp-distill"; htok = os.environ["HF_TOKEN"]

    tok = AutoTokenizer.from_pretrained("fft_hf")
    fp32 = AutoModelForCausalLM.from_pretrained("fft_hf", torch_dtype=torch.float32,
                                                attn_implementation="eager").eval().to(DEV)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    q4 = AutoModelForCausalLM.from_pretrained("fft_hf", quantization_config=bnb, device_map={"": 0}).eval()
    pooler = load_pooler().to(DEV).eval()
    embT32 = fp32.get_input_embeddings(); embTq4 = q4.get_input_embeddings()
    cfg = fp32.config; Hq = cfg.num_attention_heads; Hkv = cfg.num_key_value_heads
    D = cfg.hidden_size // Hq; dims = (Hq, Hkv, D); bos = tok.bos_token_id
    log(f"[q4v3] mix={N_MIX_DOL}/{N_MIX_CHAT} tf-neg={N_NEG_TR} gen-neg tr/te={N_GEN_TR}/{N_GEN_TE} cap={GEN_CAP}")

    # ---- POSITIVES (mass labels fp32, q-features 4-bit) ----
    dol = M.dolphin_seqs(tok, N_MIX_DOL + N_MIX_TE, bos)
    uc = M.ultrachat_seqs(tok, N_MIX_CHAT + N_MIX_TE, bos)
    Xm, ym, _ = Q4.collect_dual(dol[:N_MIX_DOL] + uc[:N_MIX_CHAT], fp32, q4, tok, pooler,
                                embT32, embTq4, dims, None, "MIX", t0)
    Xdt, ydt, _ = Q4.collect_dual(dol[N_MIX_DOL:N_MIX_DOL + N_MIX_TE], fp32, q4, tok, pooler,
                                  embT32, embTq4, dims, None, "DOLTE", t0)
    Xch, ych, _ = Q4.collect_dual(uc[N_MIX_CHAT:N_MIX_CHAT + N_MIX_TE], fp32, q4, tok, pooler,
                                  embT32, embTq4, dims, None, "CHATTE", t0)
    pos = np.vstack([Xdt[ydt == 1], Xch[ych == 1]]) if (int(ydt.sum()) + int(ych.sum())) else np.zeros((0, Xm.shape[1]), np.float32)
    log(f"[pos] mix pts={len(ym)} pos={int(ym.sum())} | held real pos={len(pos)}")

    # ---- NEGATIVES on the 4-bit model ----
    tf_seqs = [GC.casual_seq(tok, s, bos) for s in range(N_NEG_TR)]
    Xtf = _finite(Q4.harvest_neg_q4(q4, embTq4, pooler, dims, [s for s in tf_seqs if M._seq_ok(s)], NEG_PER_SEQ))
    gfeeds_tr = [V3.casual_feed(tok, 1000 + s, bos) for s in range(N_GEN_TR)]
    gfeeds_te = [V3.casual_feed(tok, 50000 + s, bos) for s in range(N_GEN_TE)]
    Xgn = _finite(V3.gen_negatives(q4, tok, pooler, embTq4, dims, gfeeds_tr, "TR", t0))
    Xgn_te = _finite(V3.gen_negatives(q4, tok, pooler, embTq4, dims, gfeeds_te, "TE", t0))
    log(f"[neg] tf={len(Xtf)} gen-tr={len(Xgn)} gen-te={len(Xgn_te)}")

    # ---- baseline = current fp32 ondevice gate, scored on 4-bit features ----
    zf = np.load(hf_hub_download(R, "trigger_experiment/ondevice_recall/gate.npz", token=htok))
    cf0, b0, m0, s0 = zf["coef"].ravel(), float(np.ravel(zf["intercept"])[0]), zf["mean"], zf["scale"]
    def base_sco(X): return ((X - m0) / s0) @ cf0 + b0

    # ---- new 4-bit contrastive gate ----
    Xall = np.vstack([Xm, Xtf, Xgn]); yall = np.concatenate([ym, np.zeros(len(Xtf) + len(Xgn), np.int32)])
    sc = StandardScaler().fit(Xall)
    clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(sc.transform(Xall), yall)
    def new_sco(X): return clf.decision_function(sc.transform(X))

    def auc(fn, X, y):
        y = np.asarray(y)
        return round(float(roc_auc_score(y, fn(X))), 4) if 0 < int(y.sum()) < len(y) else None

    def ff_at_recall(fn, tr):
        if not len(pos) or not len(Xgn_te):
            return None
        thr = float(np.quantile(fn(pos), 1 - tr))
        return round(float(np.mean(fn(Xgn_te) >= thr)), 4)

    res = {"elapsed_s": round(time.time() - t0, 1),
           "data": {"mix_pts": int(len(ym)), "mix_pos": int(ym.sum()), "tf_neg": int(len(Xtf)),
                    "gen_neg": int(len(Xgn)), "held_pos": int(len(pos)), "held_gen_neg": int(len(Xgn_te))},
           "gate_auc": {"cot": {"baseline_fp32_on_q4": auc(base_sco, Xdt, ydt), "q4_contrastive": auc(new_sco, Xdt, ydt)},
                        "chat": {"baseline_fp32_on_q4": auc(base_sco, Xch, ych), "q4_contrastive": auc(new_sco, Xch, ych)}},
           "false_fire_GEN": {
               "recall90": {"baseline_fp32_on_q4": ff_at_recall(base_sco, 0.9), "q4_contrastive": ff_at_recall(new_sco, 0.9)},
               "recall80": {"baseline_fp32_on_q4": ff_at_recall(base_sco, 0.8), "q4_contrastive": ff_at_recall(new_sco, 0.8)}}}
    log(f"[AUC] cot base={res['gate_auc']['cot']['baseline_fp32_on_q4']} -> q4con={res['gate_auc']['cot']['q4_contrastive']} | "
        f"chat base={res['gate_auc']['chat']['baseline_fp32_on_q4']} -> q4con={res['gate_auc']['chat']['q4_contrastive']}")
    log(f"[GEN false-fire] @recall90: base={res['false_fire_GEN']['recall90']['baseline_fp32_on_q4']} -> q4con={res['false_fire_GEN']['recall90']['q4_contrastive']}")
    log(f"[GEN false-fire] @recall80: base={res['false_fire_GEN']['recall80']['baseline_fp32_on_q4']} -> q4con={res['false_fire_GEN']['recall80']['q4_contrastive']}")

    thr = float(np.quantile(new_sco(pos), 0.1)) if len(pos) else 0.0   # 90%-recall operating point
    res["saved_thresh"] = round(thr, 4)
    np.savez(os.path.join(ART, "gate.npz"),
             coef=clf.coef_.ravel().astype(np.float32), intercept=np.float32(clf.intercept_[0]),
             mean=sc.mean_.astype(np.float32), scale=sc.scale_.astype(np.float32), thresh=np.float32(thr))
    res["final"] = True
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    open(os.path.join(OUT, ".done"), "w").write("ok")
    log("\n=== 4-BIT CONTRASTIVE GATE (generation-distribution hard negatives) ===")
    log(f"  GEN false-fire @recall90: baseline {res['false_fire_GEN']['recall90']['baseline_fp32_on_q4']} "
        f"-> q4-contrastive {res['false_fire_GEN']['recall90']['q4_contrastive']}")
    log(f"  AUC cot {res['gate_auc']['cot']['q4_contrastive']} chat {res['gate_auc']['chat']['q4_contrastive']}")
    log(f"  saved thresh={thr:.3f}  (sweep end-to-end with recall_gen.py Q4=1)")
    log(f"DONE t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
