"""gate_contrastive.py — sharpen the recall GATE with constructed HARD NEGATIVES.

POSITIVES are never hand-made: they come only from the mass-labeled mixed data (Dolphin +
UltraChat), where a window position is positive iff full-KV attention concentrates on a single
evicted block (dolphin_scan / mix_scan rule). The gate's weakness is false-firing on
post-eviction CASUAL turns, a distribution under-represented in training. So we ADD exactly that
distribution as label-0 hard negatives — "state a fact, then talk about something UNRELATED" —
which is legitimate hard-negative mining (we assert the SCENARIO needs no recall; we do NOT
fabricate any positive). Train logistic; compare false-fire @ matched recall, mix vs contrastive.
"""
import os, json, time, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from attn_export3_torch import load_pooler
import mix_scan as M

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS, WINDOW, BLOCK = M.LAYERS, M.WINDOW, M.BLOCK
N_NEG_TR = int(os.environ.get("N_NEG_TR", "120"))
N_NEG_TE = int(os.environ.get("N_NEG_TE", "40"))
NEG_PER_SEQ = int(os.environ.get("NEG_PER_SEQ", "40"))
N_MIX_DOL = int(os.environ.get("N_MIX_DOL", "70"))
N_MIX_CHAT = int(os.environ.get("N_MIX_CHAT", "70"))
OUT = os.environ.get("OUT_DIR", "results_gate_con")
ART = os.environ.get("ART_DIR", "artifacts_con")
EOS = "<｜end▁of▁sentence｜>"
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    os.makedirs(OUT, exist_ok=True); open(os.path.join(OUT, "live.log"), "w").write("\n".join(LOG))

FACTS = [
    ("project codename", "Bluefin"), ("vault code", "7382"), ("flight number", "JL412"),
    ("hotel confirmation number", "88231"), ("launch date", "October 9th"),
    ("budget cap", "47000 dollars"), ("locker number", "514"), ("meeting room", "C12"),
    ("reference code", "ZX9043"), ("account pin", "6207"), ("server name", "atlas03"),
    ("delivery window", "Tuesday at 3pm"),
]
CASUAL = [
    ("How's your day going so far?", "Pretty good, thanks for asking! It's been a steady kind of day, nothing too dramatic. I got through my morning tasks earlier than expected, so I'm feeling fairly relaxed about the afternoon. How has yours been treating you?"),
    ("I had great ramen for lunch today.", "Oh that sounds wonderful, ramen is such a comforting meal. Tonkotsu broth on a cool day is honestly hard to beat, especially with a soft egg and some extra scallions. Did you go to a regular spot or try somewhere new this time?"),
    ("The weather's been grey and rainy all week.", "Ugh, I hear you, long stretches of grey skies can really weigh on your mood after a while. It makes it so tempting to just stay curled up indoors with something warm. Hopefully there's a bit of sunshine coming to break it up soon."),
    ("I'm trying to read more before bed lately.", "That's a really nice habit to build, much kinder to your brain than scrolling on a bright screen late at night. Even a few pages can help you wind down and sleep better. What kind of books are you reaching for these days?"),
    ("My neighbor just got a golden retriever puppy.", "Aw, that's adorable, golden retriever puppies are pure sunshine, though they come with a serious amount of chaotic energy. Lots of zoomies and chewed shoes in their future, I'd guess. I bet your neighborhood walks just got a lot more entertaining."),
    ("I'm thinking of repainting my home office.", "Ooh, a fresh coat of paint can completely change how a room feels, that sounds like a fun little project. A calm, soft color can really help you focus and feel settled while you work. Do you already have a shade in mind or are you still browsing swatches?"),
    ("I started learning to play the guitar.", "That's awesome, picking up an instrument is such a rewarding thing to do. The first few weeks can be tough on the fingertips, but the chords start coming together faster than you'd expect once muscle memory kicks in. Are you following lessons or just figuring things out?"),
    ("Traffic this morning was an absolute nightmare.", "Oh no, there's really no worse way to start the day than being stuck bumper to bumper when you just want to get going. It drains your energy before anything has even happened. I hope the roads cleared up and the rest of your commute was smoother."),
    ("I tried a new coffee place downtown.", "Nice, discovering a cozy new cafe is one of life's small but genuine pleasures. There's something satisfying about finding a spot with good coffee and a comfortable corner to sit in. Did the pastries live up to the coffee, or was it drinks-only?"),
    ("I've been listening to a lot of jazz lately.", "Jazz is such a great choice, it has this wonderful way of filling a room without demanding all your attention. It makes for perfect background music whether you're working, cooking, or just relaxing in the evening. Any artists on repeat?"),
    ("I might take a short trip next month.", "That sounds refreshing, a little change of scenery can do wonders for resetting your head. Even a couple of days away can feel like a proper reset. Are you thinking of revisiting an old favorite or exploring somewhere new?"),
    ("The sunset yesterday evening was unreal.", "I love when the sky puts on a show like that, those deep oranges and pinks that make you stop whatever you're doing just to stare. They're such a nice reminder to slow down for a moment. Did you catch a photo?"),
]


def approx_tok(turns):
    return sum(len(u + a) for u, a in turns) // 4


def to_ids(tok, turns):
    ids = []
    for u, a in turns:
        ids += tok.encode((EOS if ids else "") + f"<｜User｜>{u}<｜Assistant｜>{a}",
                          add_special_tokens=False)
    return ids


def casual_seq(tok, seed, bos):
    """Match the GENERATION distribution: state 1-3 facts early, lots of casual filler so the
    facts are evicted, then a final unrelated casual turn. Every window position needs nothing
    evicted => HARD NEGATIVE. We harvest the TAIL (recent casual turn) = the decision region."""
    r = random.Random(seed)
    nfacts = r.choice([1, 2, 3])
    fi = r.sample(range(len(FACTS)), nfacts)
    turns = []
    for k in fi:
        kw, val = FACTS[k]
        turns.append((f"Important to note: the {kw} is {val}.", "Got it, noted."))
    fills = r.sample(CASUAL, len(CASUAL)); i = 0
    while approx_tok(turns) < WINDOW + r.choice([360, 500, 640]):
        turns.append(fills[i % len(fills)]); i += 1
    cu, ca = r.choice(CASUAL)
    turns.append((cu, ca))                                  # final casual turn (decision point)
    return [bos] + to_ids(tok, turns)


@torch.no_grad()
def harvest_neg(llm, tok, pooler, embT, dims, seqs, per):
    """SP-context q at the TAIL window positions (recent casual turn = the decision region the
    gate scores in generation) of casual scenarios -> label-0 hard negatives."""
    X = []
    for si, seq in enumerate(seqs):
        try:
            r = M.process(llm, tok, pooler, embT, seq, dims, bge=None)
        except Exception:
            continue
        if r is None:
            continue
        win = r["win_len"]; qf = r["qfeat"]
        lo = max(0, win - per)                               # tail = most recent casual turn
        for p in range(lo, win):
            X.append(qf[p].reshape(-1))
    return np.array(X, np.float32)


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); os.makedirs(ART, exist_ok=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    tok = AutoTokenizer.from_pretrained("fft_hf")
    try:
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", torch_dtype=torch.float32,
                                                   attn_implementation="eager")
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32,
                                                   attn_implementation="eager")
    llm = llm.eval().to(DEV)
    pooler = load_pooler().to(DEV).eval(); embT = llm.get_input_embeddings()
    cfg = llm.config; Hq = cfg.num_attention_heads; Hkv = cfg.num_key_value_heads
    D = cfg.hidden_size // Hq; dims = (Hq, Hkv, D); bos = tok.bos_token_id
    log(f"[con] dims Hq={Hq} D={D} | neg tr/te={N_NEG_TR}/{N_NEG_TE} per={NEG_PER_SEQ} mix dol/chat={N_MIX_DOL}/{N_MIX_CHAT}")

    # ---- mass-labeled mixed data (the source of ALL positives) ----
    dol = M.dolphin_seqs(tok, N_MIX_DOL + 18, bos)
    uc = M.ultrachat_seqs(tok, N_MIX_CHAT + 18, bos)
    Xm, ym, _ = M.collect(dol[:N_MIX_DOL] + uc[:N_MIX_CHAT], llm, tok, pooler, embT, dims, "MIX_TR", t0)
    Xdt, ydt, _ = M.collect(dol[N_MIX_DOL:N_MIX_DOL + 18], llm, tok, pooler, embT, dims, "DOL_TE", t0)
    Xch, ych, _ = M.collect(uc[N_MIX_CHAT:N_MIX_CHAT + 18], llm, tok, pooler, embT, dims, "CHAT_TE", t0)
    log(f"[mix] tr pts={len(ym)} pos={int(ym.sum())} | dol_te pos={int(ydt.sum())}/{len(ydt)} chat_te pos={int(ych.sum())}/{len(ych)}")

    # ---- constructed casual HARD NEGATIVES (label 0 by construction) ----
    neg_tr_seqs = [casual_seq(tok, s, bos) for s in range(N_NEG_TR)]
    neg_te_seqs = [casual_seq(tok, 70000 + s, bos) for s in range(N_NEG_TE)]
    Xneg = harvest_neg(llm, tok, pooler, embT, dims, [s for s in neg_tr_seqs if M._seq_ok(s)], NEG_PER_SEQ)
    Xneg_te = harvest_neg(llm, tok, pooler, embT, dims, [s for s in neg_te_seqs if M._seq_ok(s)], NEG_PER_SEQ)
    log(f"[con] hard-neg tr={len(Xneg)} te={len(Xneg_te)}")

    def fit(X, y):
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
        return sc, clf
    sc_mix, clf_mix = fit(Xm, ym)                                  # = recall_mix_v1 style
    Xn = np.vstack([Xm, Xneg]); yn = np.concatenate([ym, np.zeros(len(Xneg), np.int32)])
    sc_new, clf_new = fit(Xn, yn)                                  # + casual hard negatives
    def sco(scl, X): return scl[1].decision_function(scl[0].transform(X))

    # held POSITIVES = mass-labeled real recall (cot + chat); NEGATIVES = constructed casual
    pos = np.vstack([Xdt[ydt == 1], Xch[ych == 1]])
    log(f"[con-held] real positives={len(pos)} | casual negatives={len(Xneg_te)}")

    def prec_at_recall(scl, tr):
        ps = sco(scl, pos); ns = sco(scl, Xneg_te)
        thr = float(np.quantile(ps, 1 - tr))
        return thr, float(np.mean(ps >= thr)), float(np.mean(ns >= thr))

    def auc(scl, X, y):
        y = np.asarray(y)
        return round(float(roc_auc_score(y, sco(scl, X))), 4) if 0 < int(y.sum()) < len(y) else None

    res = {"elapsed_s": round(time.time()-t0, 1),
           "data": {"mix_tr": int(len(ym)), "mix_pos": int(ym.sum()), "hard_neg_tr": int(len(Xneg)),
                    "held_real_pos": int(len(pos)), "held_casual_neg": int(len(Xneg_te))},
           "auc": {"cot": {"mix": auc((sc_mix, clf_mix), Xdt, ydt), "contrastive": auc((sc_new, clf_new), Xdt, ydt)},
                   "chat": {"mix": auc((sc_mix, clf_mix), Xch, ych), "contrastive": auc((sc_new, clf_new), Xch, ych)}},
           "false_fire": {}}
    for tr in (0.9, 0.95):
        _, rm, ffm = prec_at_recall((sc_mix, clf_mix), tr)
        _, rn, ffn = prec_at_recall((sc_new, clf_new), tr)
        res["false_fire"][f"recall{int(tr*100)}"] = {"mix": round(ffm, 4), "contrastive": round(ffn, 4)}
        log(f"[con-eval] @recall{int(tr*100)}: false-fire(casual)  mix={ffm:.3f} -> contrastive={ffn:.3f}")
    log(f"[AUC] cot mix={res['auc']['cot']['mix']} con={res['auc']['cot']['contrastive']} | "
        f"chat mix={res['auc']['chat']['mix']} con={res['auc']['chat']['contrastive']}")

    thr, rec, ff = prec_at_recall((sc_new, clf_new), 0.9)
    res["saved_thresh"] = round(thr, 4); res["saved_recall"] = round(rec, 4); res["saved_false_fire"] = round(ff, 4)
    np.savez(os.path.join(ART, "gate.npz"),
             coef=clf_new.coef_.ravel().astype(np.float32),
             intercept=np.float32(clf_new.intercept_[0]),
             mean=sc_new.mean_.astype(np.float32), scale=sc_new.scale_.astype(np.float32),
             thresh=np.float32(thr))
    res["final"] = True
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    open(os.path.join(OUT, ".done"), "w").write("ok")
    log("\n=== CONTRASTIVE GATE (positives=mass-labeled; hard-neg=constructed casual) ===")
    log(f"  false-fire(casual) @recall90: mix {res['false_fire']['recall90']['mix']} "
        f"-> contrastive {res['false_fire']['recall90']['contrastive']}")
    log(f"  AUC cot {res['auc']['cot']['mix']}->{res['auc']['cot']['contrastive']} | "
        f"chat {res['auc']['chat']['mix']}->{res['auc']['chat']['contrastive']}")
    log(f"  saved thresh={thr:.3f} (real recall {rec:.2f}, casual false-fire {ff:.2f})")
    log(f"DONE t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
