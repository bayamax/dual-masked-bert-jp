"""composite_test.py — end-to-end integration test of the wired recall_kit package.

On unseen held Dolphin-R1 items: RecallRuntime builds production-context features, the GATE
fires per position, and (only on fires) a RETRIEVER pulls the top evicted block. We score
the WIRED system against the full-KV attention ground truth:
  * gate precision / recall / F1 (did it fire on the real recall positions?)
  * retrieval top-1/top-2 at true recall positions (QK backend and state-free BGE backend)
  * end-to-end: fired on a real recall AND pulled the right block.
"""
import os, json, time
import numpy as np
import torch

import dolphin_scan as DS
from recall_kit import RecallGate, QKIndexer, BGERetriever, RecallRuntime

ART = os.environ.get("ART_DIR", "recall_kit_artifacts")
OUT = os.environ.get("OUT_DIR", "recall_kit_artifacts")
POS_FRAC = float(os.environ.get("POS_FRAC", "0.08"))
DEV = DS.DEV
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    open(os.path.join(OUT, "composite.log"), "w").write("\n".join(LOG))


def seq_for(tok, usr, cot):
    pre = tok.encode(f"<｜User｜>{usr}<｜Assistant｜><think>\n", add_special_tokens=False)
    return [tok.bos_token_id] + pre + tok.encode(cot, add_special_tokens=False)


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    tok, llm, pooler, embT, bge = DS.build_model()
    cfg = llm.config; dims = (cfg.num_attention_heads, cfg.num_key_value_heads,
                              cfg.hidden_size // cfg.num_attention_heads)
    gate = RecallGate.load(os.path.join(ART, "gate.npz"))
    qk = QKIndexer.load(os.path.join(ART, "indexer.npz"), dims, device=DEV)
    bge_ret = BGERetriever.load(os.path.join(ART, "bge_head.npz"), bge, device=DEV)
    rt = RecallRuntime(llm, tok, pooler, gate, qk=qk, bge=bge_ret, device=DEV)

    ci = os.path.join(ART, "composite_items.json")
    if os.path.exists(ci):                        # same sorted pool as training (matched regime)
        items = [(0, u, c) for u, c in json.load(open(ci))]
    else:
        items = DS.pick_items(420)[360:400]
    log(f"[composite] {len(items)} unseen held items (from build pool: {os.path.exists(ci)})")

    TP = FP = FN = TN = 0
    qk_t1 = qk_t2 = bge_t1 = bge_t2 = nfire_pos = 0
    e2e_qk = e2e_bge = n_realpos = 0
    done = 0
    for n, (_, usr, cot) in enumerate(items):
        seq = seq_for(tok, usr, cot)
        try:
            r = DS.process(llm, tok, pooler, embT, usr, cot, dims)        # gold labels
            ctx = rt.prepare(seq)                                          # production features
        except Exception as e:
            log(f"  item {n} skipped: {e}"); continue
        if r is None or ctx is None or ctx.win_len != r["win_len"]:
            continue
        if done == 0:                                   # one-time wiring sanity check
            kd = float(np.abs(r["keys"].astype(np.float32) - ctx.keys.astype(np.float32)).max())
            qd = float(np.abs(r["qfeat"].astype(np.float32) - ctx.qfeat.astype(np.float32)).max())
            mmp = r["maxmass"]; thr0 = max(np.quantile(mmp, 1 - POS_FRAC), 0.05)
            pp = np.where(mmp >= thr0)[0][:80]
            hit_proc = np.mean([int(r["goldblk"][p]) in qk.rank(r["qfeat"][p], r["keys"])[:2] for p in pp])
            hit_ctx = np.mean([int(r["goldblk"][p]) in qk.rank(ctx.qfeat[p], ctx.keys)[:2] for p in pp])
            log(f"  [debug] key_maxdiff={kd:.4g} q_maxdiff={qd:.4g} | QK top2 process-keys={hit_proc:.3f} ctx-keys={hit_ctx:.3f}")
        mm = r["maxmass"]; goldblk = r["goldblk"]
        thr = max(np.quantile(mm, 1 - POS_FRAC), 0.05)
        realpos = mm >= thr
        n_realpos += int(realpos.sum())
        for p in range(ctx.win_len):
            d = rt.decide(ctx, p, backend="qk", topk=2)
            fired = d["fired"]; rp = bool(realpos[p])
            TP += fired and rp; FP += fired and not rp
            FN += (not fired) and rp; TN += (not fired) and not rp
            if rp:                                  # retrieval quality at TRUE recall positions
                oq = qk.rank(ctx.qfeat[p], ctx.keys)
                ob = bge_ret.rank(ctx.qfeat[p].reshape(-1), ctx.block_texts)
                g = int(goldblk[p])
                qk_t1 += g == oq[0]; qk_t2 += g in oq[:2]
                bge_t1 += g == ob[0]; bge_t2 += g in ob[:2]
                nfire_pos += 1
                if fired:
                    e2e_qk += g in oq[:2]; e2e_bge += g in ob[:2]
        done += 1
        if (n + 1) % 5 == 0:
            log(f"  {n+1}/{len(items)} TP={TP} FP={FP} FN={FN} t={time.time()-t0:.0f}s")

    prec = TP / max(TP + FP, 1); rec = TP / max(TP + FN, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    rep = {
        "held_items": done, "real_recall_positions": int(n_realpos),
        "gate": {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
                 "TP": TP, "FP": FP, "FN": FN, "TN": TN},
        "retrieval_at_true_recall": {
            "qk_top1": round(qk_t1 / max(nfire_pos, 1), 3),
            "qk_top2": round(qk_t2 / max(nfire_pos, 1), 3),
            "bge_top1": round(bge_t1 / max(nfire_pos, 1), 3),
            "bge_top2": round(bge_t2 / max(nfire_pos, 1), 3), "n": nfire_pos},
        "end_to_end_fired_and_correct": {
            "qk_top2": round(e2e_qk / max(TP, 1), 3),
            "bge_top2": round(e2e_bge / max(TP, 1), 3), "n_fired_true": TP},
        "elapsed_s": round(time.time() - t0, 1), "final": True}
    json.dump(rep, open(os.path.join(OUT, "composite_report.json"), "w"), indent=2)
    log("\n=== COMPOSITE ===")
    log(f"  GATE  precision {prec:.3f} recall {rec:.3f} f1 {f1:.3f}")
    log(f"  RETRIEVAL@true-recall  QK top2 {rep['retrieval_at_true_recall']['qk_top2']} | "
        f"BGE top2 {rep['retrieval_at_true_recall']['bge_top2']} (n={nfire_pos})")
    log(f"  END-TO-END (fired & right block, top2)  QK {rep['end_to_end_fired_and_correct']['qk_top2']} | "
        f"BGE {rep['end_to_end_fired_and_correct']['bge_top2']}")
    open(os.path.join(OUT, ".composite_done"), "w").write("ok")
    log(f"DONE t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
