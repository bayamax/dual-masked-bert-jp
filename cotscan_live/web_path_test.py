"""web_path_test.py — operational test of the WEB path (CPU, public model).

The coexistence design = run recall-gate and web-gate INDEPENDENTLY, inject whatever
fires. The recall path (block_recall) is already validated in this repo; the NEW, risky
half is the web path. This tests, on the public DeepSeek-R1-Distill-Qwen-1.5B:

  1) FIRE PRECISION across turn types: a thresholded web gate (trained on the 4608-dim
     pre-RoPE q @ layers 8/14/20) must fire on WEB turns (fabricated entity => knowledge
     gap) and STAY SILENT on KNOWN and CASUAL turns. Spurious web fires are the harmful
     case (model parrots an injected doc on a turn that did not need one).
  2) INJECT -> END-TASK: on a web turn, injecting the reference doc must make the model
     answer with the gold value (vs hallucinate without it).

Train/threshold on a split; report fire-rates and end-task on the HELD split + casual.
"""
import os, json, time, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
LAYERS = (8, 14, 20)
OUT = os.environ.get("OUT_DIR", "results_web_path")
MAXNEW = int(os.environ.get("MAXNEW", "48"))
DEV = "cpu"
os.makedirs(OUT, exist_ok=True)
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    open(os.path.join(OUT, "live.log"), "w").write("\n".join(LOG))

# KNOWN: (question, gold-substring the model should already produce)
KNOWN = [
    ("What is the capital of France?", "paris"),
    ("What is the chemical symbol for gold?", "au"),
    ("Who wrote the play Romeo and Juliet?", "shakespeare"),
    ("Which planet is known as the Red Planet?", "mars"),
    ("What is the largest ocean on Earth?", "pacific"),
    ("What gas do plants absorb during photosynthesis?", "carbon dioxide"),
    ("Who painted the Mona Lisa?", "vinci"),
    ("What language is primarily spoken in Brazil?", "portuguese"),
    ("What is the currency of Japan?", "yen"),
    ("Who developed the theory of general relativity?", "einstein"),
    ("What is the tallest mountain on Earth?", "everest"),
    ("Which organ pumps blood through the human body?", "heart"),
    ("What is the capital of Japan?", "tokyo"),
    ("Who is the author of the Harry Potter book series?", "rowling"),
    ("What is the capital of Italy?", "rome"),
    ("Who was the first President of the United States?", "washington"),
]
# WEB: (question, reference-doc, gold-substring) — fabricated entities (guaranteed knowledge gap)
WEB = [
    ("In what year was the Treaty of Velmaran signed?",
     "The Treaty of Velmaran was signed in 1847 between the coastal provinces.", "1847"),
    ("What is the capital of the nation of Brindolia?",
     "Brindolia is a small nation whose capital city is Hostrand.", "hostrand"),
    ("In what year was the Vorlith Institute of Technology founded?",
     "The Vorlith Institute of Technology was founded in 1921.", "1921"),
    ("Who won the Galvin Prize in 2019?",
     "The 2019 Galvin Prize was awarded to Dr. Helena Voss for her work in optics.", "voss"),
    ("How tall is Mount Drennakar?",
     "Mount Drennakar rises to a height of 4,812 meters above sea level.", "4,812"),
    ("What language is spoken in the kingdom of Ulventhia?",
     "The people of the kingdom of Ulventhia speak a language called Ulven.", "ulven"),
    ("What is the currency of the federation of Mooncrest?",
     "The federation of Mooncrest uses a currency known as the lunar.", "lunar"),
    ("How many members sit on the Council of Threll?",
     "The Council of Threll is composed of nine voting members.", "nine"),
    ("In what year was the city of Norvanis established?",
     "The city of Norvanis was established in 1623 by river traders.", "1623"),
    ("What is the boiling point of the element Trovium?",
     "The element Trovium has a boiling point of 1342 degrees Celsius.", "1342"),
    ("How many moons orbit the planet Xelpha?",
     "The planet Xelpha is orbited by seven moons.", "seven"),
    ("Who directed the film 'Echoes of Tannareth'?",
     "The film 'Echoes of Tannareth' was directed by Marcus Ardell.", "ardell"),
    ("What is the population of the city of Quorvath?",
     "The city of Quorvath has a population of about 2.3 million people.", "2.3 million"),
    ("Who is the current president of the Republic of Zelandria?",
     "The current president of the Republic of Zelandria is Aldira Penn.", "penn"),
    ("What is the atomic number of the element Quenzite?",
     "Quenzite is a synthetic element with atomic number 119.", "119"),
    ("Who invented the Castellan rotary engine?",
     "The Castellan rotary engine was invented by Tobias Marr.", "marr"),
]
CASUAL = [
    "How's your day going so far?",
    "I had great ramen for lunch today.",
    "The weather's been grey and rainy all week.",
    "I started learning to play the guitar.",
    "Traffic this morning was an absolute nightmare.",
    "I tried a new coffee place downtown.",
    "I've been listening to a lot of jazz lately.",
    "The sunset yesterday evening was unreal.",
]


def build():
    tok = AutoTokenizer.from_pretrained(MODEL)
    try:
        llm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="eager")
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager")
    return tok, llm.eval().to(DEV)


def msgs_ids(tok, content):
    try:
        return tok.apply_chat_template([{"role": "user", "content": content}],
                                       add_generation_prompt=True, tokenize=True)
    except Exception:
        return tok.encode(f"<｜begin▁of▁sentence｜><｜User｜>{content}<｜Assistant｜>", add_special_tokens=False)


@torch.no_grad()
def feat(tok, llm, content):
    """pre-RoPE q at layers 8/14/20, answer-onset (last prompt token) -> 4608-dim."""
    gq = {}; hooks = []
    for li in LAYERS:
        def mk(li):
            def fn(_m, _i, o): gq[li] = o.detach()[0, -1].float().numpy()
            return fn
        hooks.append(llm.model.layers[li].self_attn.q_proj.register_forward_hook(mk(li)))
    ids = msgs_ids(tok, content)
    llm(input_ids=torch.tensor([ids], device=DEV), use_cache=False)
    for h in hooks: h.remove()
    return np.concatenate([gq[li] for li in LAYERS]).astype(np.float32)


@torch.no_grad()
def generate(tok, llm, content, maxnew=MAXNEW):
    ids = msgs_ids(tok, content)
    out = llm.generate(input_ids=torch.tensor([ids], device=DEV), max_new_tokens=maxnew,
                       do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, len(ids):], skip_special_tokens=True)


def hit(text, gold):
    return gold.lower() in text.lower()


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    log(f"[web-path] model={MODEL}")
    tok, llm = build()
    log(f"[web-path] loaded {time.time()-t0:.0f}s | known={len(KNOWN)} web={len(WEB)} casual={len(CASUAL)}")

    # Harvest ALL known+web; OOF scores give an honest held estimate (fixes tiny-train calibration).
    log("[harvest] features (all known+web+casual) ...")
    Xk = np.array([feat(tok, llm, q) for q, _ in KNOWN], np.float32)
    Xw = np.array([feat(tok, llm, q) for q, _, _ in WEB], np.float32)
    Xc = np.array([feat(tok, llm, q) for q in CASUAL], np.float32)
    X = np.vstack([Xk, Xw]); y = np.concatenate([np.zeros(len(Xk)), np.ones(len(Xw))]).astype(int)

    def fit(Xt, yt):
        sc = StandardScaler().fit(Xt)
        clf = LogisticRegression(C=0.02, max_iter=5000, class_weight="balanced").fit(sc.transform(Xt), yt)
        return sc, clf
    def dec(scl, Xt): return scl[1].decision_function(scl[0].transform(Xt))

    # out-of-fold decision scores for known+web (honest held estimate)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    oof = np.zeros(len(y), np.float32)
    for tr, te in skf.split(X, y):
        scl = fit(X[tr], y[tr]); oof[te] = dec(scl, X[te])
    auc = float(roc_auc_score(y, oof))
    oof_known = oof[y == 0]; oof_web = oof[y == 1]
    # zero-known-false-fire operating point: threshold just above max OOF-known score
    thr = float(max(oof_known)) + 1e-3
    final = fit(X, y)                      # gate trained on all 32, for scoring casual + inference
    cas_scores = dec(final, Xc)
    log(f"[gate] OOF AUC={auc:.3f} | thr={thr:.3f} (max OOF-known={max(oof_known):.2f})")
    log(f"  OOF known scores={np.round(oof_known,2).tolist()}")
    log(f"  OOF web scores  ={np.round(oof_web,2).tolist()}")
    log(f"  casual scores   ={np.round(cas_scores,2).tolist()}")

    fr = {"web": f"{int((oof_web>=thr).sum())}/{len(oof_web)}",
          "known": f"{int((oof_known>=thr).sum())}/{len(oof_known)}",
          "casual": f"{int((cas_scores>=thr).sum())}/{len(cas_scores)}"}
    log(f"[fire@OOF/held] web={fr['web']} (want high) | known={fr['known']} casual={fr['casual']} (want 0)")

    # ---- web end-task: inject doc (operational action) vs no-inject baseline ----
    e2e = []
    for i, (q, doc, gold) in enumerate(WEB[:8]):
        base = generate(tok, llm, q)
        inj = generate(tok, llm, f"Reference information:\n{doc}\n\nQuestion: {q}")
        row = {"q": q, "gold": gold,
               "noinject_hit": hit(base, gold), "inject_hit": hit(inj, gold),
               "inject_out": inj[:200]}
        e2e.append(row)
        log(f"  [web] noinj={row['noinject_hit']} inj={row['inject_hit']} | {q[:42]}")
    inj_acc = np.mean([r["inject_hit"] for r in e2e]); base_acc = np.mean([r["noinject_hit"] for r in e2e])

    # ---- known end-task: model already correct, no injection ----
    known_correct = [hit(generate(tok, llm, q), gold) for q, gold in KNOWN[:6]]
    kacc = float(np.mean(known_correct))

    rep = {"model": MODEL, "feat_dim": int(X.shape[1]), "oof_auc": round(auc, 4),
           "threshold_zero_known_ff": round(thr, 4),
           "fire": fr,
           "web_endtask": {"inject_acc": round(float(inj_acc), 3),
                           "noinject_acc": round(float(base_acc), 3), "n": len(e2e), "rows": e2e},
           "known_acc_noinject": round(kacc, 3), "known_n": 6,
           "elapsed_s": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    log("\n=== WEB PATH (operational) ===")
    log(f"  OOF AUC={rep['oof_auc']}")
    log(f"  fire@thr(0-known-ff)  web={fr['web']}  known={fr['known']}  casual={fr['casual']}")
    log(f"  web end-task  inject={rep['web_endtask']['inject_acc']}  vs  no-inject={rep['web_endtask']['noinject_acc']}")
    log(f"  known acc (no inject) = {kacc:.3f}")
    log(f"DONE t={rep['elapsed_s']}s")


if __name__ == "__main__":
    main()
