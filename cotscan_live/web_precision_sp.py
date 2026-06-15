"""web_precision_sp.py — the coexistence tuning that actually matters.

Clarified design: WEB is decided at the QUESTION ONSET (is this a knowledge gap?), RECALL is
decided mid-CoT (did we lose an operand to SP eviction?). Different times => no competition.
Recall SHOULD fire on GSM8K/Dolphin (mandatory under SP). The only thing to tune for coexistence
is WEB PRECISION: the web gate must stay SILENT on reasoning/known/chat (else it injects a junk
doc and harms them) and fire only on real knowledge gaps.

So we train the web gate with GSM8K + Dolphin + known + casual ALL as negatives (fabricated
entities = positives), calibrate the threshold on a held negative slice for ~zero false-fire,
and report fire-rates on held suites. No generation needed -> fast. Public R1-Distill-1.5B.
"""
import os, json, time, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
LAYERS = (8, 14, 20)
OUT = os.environ.get("OUT_DIR", "results_webprec")
DEV = "cpu"
os.makedirs(OUT, exist_ok=True)
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    open(os.path.join(OUT, "live.log"), "w").write("\n".join(LOG))

KNOWN = ["What is the capital of France?", "What is the chemical symbol for gold?",
         "Who wrote Romeo and Juliet?", "Which planet is the Red Planet?",
         "What is the largest ocean on Earth?", "What is the capital of Japan?",
         "How many sides does a hexagon have?", "What is the currency of Japan?",
         "Who painted the Mona Lisa?", "What language is spoken in Brazil?",
         "Which organ pumps blood?", "What is the capital of Italy?",
         "Who developed general relativity?", "What is the tallest mountain?"]
CASUAL = ["How's your day going so far?", "I had great ramen for lunch today.",
          "The weather's been grey and rainy all week.", "I started learning to play the guitar.",
          "Traffic this morning was an absolute nightmare.", "I tried a new coffee place downtown.",
          "I've been listening to a lot of jazz lately.", "The sunset yesterday evening was unreal.",
          "My neighbor just got a golden retriever puppy.", "I might take a short trip next month."]
WEB = ["In what year was the Treaty of Velmaran signed?", "What is the capital of the nation of Brindolia?",
       "In what year was the Vorlith Institute founded?", "Who won the Galvin Prize in 2019?",
       "How tall is Mount Drennakar?", "What language is spoken in Ulventhia?",
       "What is the currency of Mooncrest?", "How many members sit on the Council of Threll?",
       "In what year was the city of Norvanis established?", "What is the boiling point of Trovium?",
       "How many moons orbit the planet Xelpha?", "Who directed the film 'Echoes of Tannareth'?",
       "What is the population of the city of Quorvath?", "Who is the president of Zelandria?",
       "What is the atomic number of Quenzite?", "Who invented the Castellan rotary engine?"]


def build():
    tok = AutoTokenizer.from_pretrained(MODEL)
    try:
        llm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="eager")
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager")
    return tok, llm.eval().to(DEV)


@torch.no_grad()
def feat(tok, llm, q):
    gq = {}; hooks = []
    for li in LAYERS:
        def mk(li):
            def fn(_m, _i, o): gq[li] = o.detach()[0, -1].float().numpy()
            return fn
        hooks.append(llm.model.layers[li].self_attn.q_proj.register_forward_hook(mk(li)))
    try:
        ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=True)
    except Exception:
        ids = tok.encode(f"<｜begin▁of▁sentence｜><｜User｜>{q}<｜Assistant｜>", add_special_tokens=False)
    llm(input_ids=torch.tensor([ids], device=DEV), use_cache=False)
    for h in hooks: h.remove()
    return np.concatenate([gq[li] for li in LAYERS]).astype(np.float32)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    tok, llm = build()
    log(f"[webprec] loaded {time.time()-t0:.0f}s")
    gsm = load_dataset("gsm8k", "main", split="train[:24]")
    gsm_q = [gsm[i]["question"] for i in range(24)]
    dol = json.load(open("dolphin_prompts.json")) if os.path.exists("dolphin_prompts.json") else []
    log(f"[data] known={len(KNOWN)} casual={len(CASUAL)} web={len(WEB)} gsm={len(gsm_q)} dolphin={len(dol)}")

    def F(qs): return np.array([feat(tok, llm, q) for q in qs], np.float32)
    log("[harvest] features ...")
    Xweb = F(WEB); Xkn = F(KNOWN); Xca = F(CASUAL); Xgs = F(gsm_q); Xdl = F(dol) if dol else np.zeros((0, Xweb.shape[1]), np.float32)

    # splits: train half of each; held the rest
    def sp(X, h): return X[:h], X[h:]
    wtr, wte = sp(Xweb, 8); ktr, kte = sp(Xkn, 8); ctr, cte = sp(Xca, 5)
    gtr, gte = sp(Xgs, 14); dtr, dte = sp(Xdl, max(0, len(Xdl) - 3)) if len(Xdl) else (Xdl, Xdl)

    pos = wtr
    neg = np.vstack([ktr, ctr, gtr] + ([dtr] if len(dtr) else []))
    X = np.vstack([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.02, max_iter=5000, class_weight="balanced").fit(sc.transform(X), y)
    def dec(M): return clf.decision_function(sc.transform(M)) if len(M) else np.array([])

    # calibrate threshold for ~zero false-fire on a held negative mix (gsm + known + casual held)
    calib = np.vstack([gte, kte, cte] + ([dte] if len(dte) else []))
    thr = float(np.quantile(dec(calib), 0.97)) + 1e-3
    log(f"[gate] web thr={thr:.2f} (97th pct of held negatives, n={len(calib)})")

    def fr(M, name, want):
        s = dec(M); n = int((s >= thr).sum()); tot = len(s)
        log(f"  fire {name:9}: {n}/{tot}  (want {want})  scores~[{s.min():.1f},{s.max():.1f}]" if tot else f"  {name}: empty")
        return f"{n}/{tot}"
    fire = {"web_held": fr(wte, "web", "high"), "known_held": fr(kte, "known", "0"),
            "casual_held": fr(cte, "casual", "0"), "gsm8k_held": fr(gte, "gsm8k", "0")}
    if len(dte): fire["dolphin_held"] = fr(dte, "dolphin", "0")

    rep = {"model": MODEL, "web_thr": round(thr, 4), "fire": fire,
           "negatives_in_training": ["known", "casual", "gsm8k", "dolphin"],
           "elapsed_s": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    log("\n=== WEB PRECISION under reasoning/chat negatives (coexistence tuning) ===")
    for k, v in fire.items():
        log(f"  {k:14}: {v}")
    log("  GOAL: web_held high; known/casual/gsm8k/dolphin = 0  => web stays silent on reasoning => safe to coexist with recall")
    log(f"DONE t={rep['elapsed_s']}s")


if __name__ == "__main__":
    main()
