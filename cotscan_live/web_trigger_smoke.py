"""web_trigger_smoke.py — does the WEB (knowledge-gap) trigger drive at all?

Minimal, CPU-runnable separability check on the PUBLIC base model
(deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B). No SP/pooler, no private fft weights:
the web trigger does NOT depend on eviction (that is recall's job) — it depends on
KNOWN vs UNKNOWN. So the correct first test is a clean single-turn answer-onset.

For each question we feed `<｜User｜>{q}<｜Assistant｜>` and harvest the pre-RoPE query
(q_proj output) at layers 8/14/20 at the answer-onset position (last prompt token) —
the SAME 4608-dim feature the shipped trigger head uses. KNOWN = real questions the
model can answer; WEB = fabricated entities the model provably cannot know.

We fit a logistic on the onset feature (and a mean-pooled variant) and report
5-fold stratified CV AUC + out-of-fold confusion. AUC ~0.5 => trigger is dead;
AUC >> 0.5 => the web trigger is driveable (the model's internal state separates
"I can answer" from "I don't know this entity").
"""
import os, json, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
LAYERS = (8, 14, 20)
OUT = os.environ.get("OUT_DIR", "results_web_smoke")
DEV = "cpu"
os.makedirs(OUT, exist_ok=True)
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    open(os.path.join(OUT, "live.log"), "w").write("\n".join(LOG))

# --- KNOWN: real, unambiguous, answerable from parametric knowledge (incl. real proper nouns) ---
KNOWN = [
    "What is the capital of France?",
    "What is the chemical symbol for gold?",
    "How many continents are there on Earth?",
    "Who wrote the play Romeo and Juliet?",
    "What is the boiling point of water in Celsius at sea level?",
    "Which planet is known as the Red Planet?",
    "What is the largest ocean on Earth?",
    "What gas do plants absorb from the atmosphere during photosynthesis?",
    "What is the square root of 144?",
    "Who painted the Mona Lisa?",
    "What language is primarily spoken in Brazil?",
    "How many sides does a hexagon have?",
    "What is the currency of Japan?",
    "Who developed the theory of general relativity?",
    "What is the tallest mountain on Earth?",
    "What is 15 multiplied by 6?",
    "What is the smallest prime number?",
    "Which organ pumps blood through the human body?",
    "What is the capital of Japan?",
    "How many days are in a leap year?",
    "Who is the author of the Harry Potter book series?",
    "What is the hardest known natural material?",
    "What is the capital of Italy?",
    "Who was the first President of the United States?",
]

# --- WEB: fabricated entities => guaranteed knowledge gap (model cannot know these) ---
UNKNOWN = [
    "In what year was the Treaty of Velmaran signed?",
    "What is the population of the city of Quorvath?",
    "Who is the current president of the Republic of Zelandria?",
    "What is the boiling point of the element Trovium?",
    "How many moons orbit the planet Xelpha?",
    "What is the capital of the nation of Brindolia?",
    "Who composed the symphony 'Aurelian Tides' by Festus Marleve?",
    "What is the GDP of the province of Karnathia?",
    "In what year was the Vorlith Institute of Technology founded?",
    "What is the wingspan of the Greater Plumed Yendari bird?",
    "Who won the Galvin Prize in 2019?",
    "What is the chemical formula for zorbium oxide?",
    "How tall is Mount Drennakar?",
    "What language is spoken in the kingdom of Ulventhia?",
    "Who directed the film 'Echoes of Tannareth'?",
    "What is the currency of the federation of Mooncrest?",
    "In what year did the Pellucid War between Vandara and Korreth end?",
    "What is the atomic number of the element Quenzite?",
    "Who is the CEO of the Brammelton Corporation?",
    "What is the length of the Yarvuth River in kilometers?",
    "How many members sit on the Council of Threll?",
    "In what year was the city of Norvanis established?",
    "Who invented the Castellan rotary engine?",
    "What is the diameter of the asteroid Pherelon?",
]


def build():
    tok = AutoTokenizer.from_pretrained(MODEL)
    try:
        llm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="eager")
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager")
    return tok, llm.eval().to(DEV)


def prompt_ids(tok, q):
    msgs = [{"role": "user", "content": q}]
    try:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    except Exception:
        ids = tok.encode(f"<｜begin▁of▁sentence｜><｜User｜>{q}<｜Assistant｜>", add_special_tokens=False)
    return ids


@torch.no_grad()
def harvest(tok, llm, qs):
    """Return (X_onset, X_mean): pre-RoPE q at layers 8/14/20.
    onset = last prompt token (answer-onset decision point). mean = mean over all prompt tokens."""
    gq = {}
    hooks = []
    for li in LAYERS:
        def mk(li):
            def fn(_m, _i, o): gq[li] = o.detach()[0].float()  # [T, Dq]
            return fn
        hooks.append(llm.model.layers[li].self_attn.q_proj.register_forward_hook(mk(li)))
    Xo, Xm = [], []
    t0 = time.time()
    for i, q in enumerate(qs):
        ids = prompt_ids(tok, q)
        gq.clear()
        llm(input_ids=torch.tensor([ids], device=DEV), use_cache=False)
        onset = np.concatenate([gq[li][-1].numpy() for li in LAYERS]).astype(np.float32)
        mean = np.concatenate([gq[li].mean(0).numpy() for li in LAYERS]).astype(np.float32)
        Xo.append(onset); Xm.append(mean)
        if (i + 1) % 8 == 0:
            log(f"  harvested {i+1}/{len(qs)} t={time.time()-t0:.0f}s dim={onset.shape[0]}")
    for h in hooks:
        h.remove()
    return np.array(Xo, np.float32), np.array(Xm, np.float32)


def evaluate(X, y, tag):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, confusion_matrix
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(len(y), np.float32)
    aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        s = clf.decision_function(sc.transform(X[te]))
        oof[te] = s
        if len(set(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], s))
    auc_oof = float(roc_auc_score(y, oof))
    thr = 0.0  # decision_function midpoint
    pred = (oof >= thr).astype(int)
    cm = confusion_matrix(y, pred).tolist()
    res = {"tag": tag, "auc_oof": round(auc_oof, 4),
           "auc_perfold_mean": round(float(np.mean(aucs)), 4),
           "confusion_[[TN,FP],[FN,TP]]": cm,
           "n_known": int((y == 0).sum()), "n_web": int((y == 1).sum())}
    log(f"[{tag}] AUC(oof)={res['auc_oof']} perfold={res['auc_perfold_mean']} "
        f"confusion(TN,FP/FN,TP)={cm}")
    return res


def main():
    t0 = time.time()
    log(f"[web-smoke] model={MODEL} layers={LAYERS}")
    tok, llm = build()
    log(f"[web-smoke] loaded in {time.time()-t0:.0f}s | known={len(KNOWN)} web={len(UNKNOWN)}")
    Xo_k, Xm_k = harvest(tok, llm, KNOWN)
    Xo_w, Xm_w = harvest(tok, llm, UNKNOWN)
    Xo = np.vstack([Xo_k, Xo_w]); Xm = np.vstack([Xm_k, Xm_w])
    y = np.concatenate([np.zeros(len(Xo_k), int), np.ones(len(Xo_w), int)])
    res = {"model": MODEL, "layers": list(LAYERS), "feat_dim": int(Xo.shape[1]),
           "onset": evaluate(Xo, y, "onset(last-token)"),
           "mean_pooled": evaluate(Xm, y, "mean-pooled")}
    res["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    log("\n=== WEB TRIGGER SMOKE ===")
    log(f"  feat_dim={res['feat_dim']} (3 layers x q_proj)")
    log(f"  onset      AUC={res['onset']['auc_oof']}")
    log(f"  mean-pooled AUC={res['mean_pooled']['auc_oof']}")
    log("  AUC ~0.5 => trigger dead; >>0.5 => web trigger is driveable")
    log(f"DONE t={res['elapsed_s']}s")


if __name__ == "__main__":
    main()
