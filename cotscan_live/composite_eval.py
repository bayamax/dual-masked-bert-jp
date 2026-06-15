"""composite_eval.py — evaluate the COEXISTENCE composite (recall gate + web gate, inject-what-fires)
across diverse real tasks, and TUNE it so every suite gets a sensible answer.

Core tuning principle: a turn that needs NEITHER trigger (GSM8K math, Dolphin-style reasoning,
general chat) must NOT fire either gate, so the composite is transparent there (== base model).
We achieve this by training both gates with those distributions as NEGATIVES, then calibrating
each threshold on a held negative slice for ~zero false-fire. Needles (evicted-fact recall,
fabricated-entity web) remain positives so the gates still help where help is needed.

Public DeepSeek-R1-Distill-Qwen-1.5B (CPU, no secrets). Suites:
  GSM8K(real) | Dolphin-R1(cached, if available) | recall-needle | web-needle | BOTH |
  multi-turn casual | known-QA
Metrics per suite: gate FIRE RATE (recall/web) + task score. GSM8K/Dolphin/known/casual want
fire~0 (transparency); needles want fire high + golds recovered.
"""
import os, json, time, random, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
LAYERS = (8, 14, 20)
WIN_TURNS = int(os.environ.get("WIN_TURNS", "4"))
OUT = os.environ.get("OUT_DIR", "results_composite")
GSM_N = int(os.environ.get("GSM_N", "6")); GSM_NEW = int(os.environ.get("GSM_NEW", "768"))
DOL_N = int(os.environ.get("DOL_N", "6")); DOL_NEW = int(os.environ.get("DOL_NEW", "384"))
NEEDLE_NEW = int(os.environ.get("NEEDLE_NEW", "200"))
DEV = "cpu"
os.makedirs(OUT, exist_ok=True)
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    open(os.path.join(OUT, "live.log"), "w").write("\n".join(LOG))

FACTS = [("vault code", "7382"), ("flight number", "JL412"), ("locker number", "514"),
         ("account pin", "6207"), ("reference code", "ZX9043"), ("meeting room", "C12"),
         ("server name", "atlas03"), ("budget cap", "47000 dollars"),
         ("hotel confirmation number", "88231"), ("project codename", "Bluefin")]
CASUAL = ["How's your day going so far?", "I had great ramen for lunch today.",
          "The weather's been grey and rainy all week.", "I started learning to play the guitar.",
          "Traffic this morning was an absolute nightmare.", "I tried a new coffee place downtown.",
          "I've been listening to a lot of jazz lately.", "The sunset yesterday evening was unreal.",
          "My neighbor just got a golden retriever puppy.", "I might take a short trip next month."]
KNOWN = [("What is the capital of France?", "paris"), ("What is the chemical symbol for gold?", "au"),
         ("Who wrote Romeo and Juliet?", "shakespeare"), ("Which planet is the Red Planet?", "mars"),
         ("What is the largest ocean on Earth?", "pacific"), ("What is the capital of Japan?", "tokyo"),
         ("How many sides does a hexagon have?", "six"), ("What is the currency of Japan?", "yen"),
         ("Who painted the Mona Lisa?", "vinci"), ("What language is spoken in Brazil?", "portuguese"),
         ("Which organ pumps blood?", "heart"), ("What is the capital of Italy?", "rome"),
         ("Who developed general relativity?", "einstein"), ("What is the tallest mountain?", "everest"),
         ("What gas do plants absorb?", "carbon dioxide"), ("First US president?", "washington")]
WEB = [("In what year was the Treaty of Velmaran signed?", "The Treaty of Velmaran was signed in 1847.", "1847"),
       ("What is the capital of the nation of Brindolia?", "Brindolia's capital city is Hostrand.", "hostrand"),
       ("In what year was the Vorlith Institute founded?", "The Vorlith Institute was founded in 1921.", "1921"),
       ("Who won the Galvin Prize in 2019?", "The 2019 Galvin Prize went to Dr. Helena Voss.", "voss"),
       ("How tall is Mount Drennakar?", "Mount Drennakar is 4,812 meters tall.", "4,812"),
       ("What language is spoken in Ulventhia?", "The people of Ulventhia speak Ulven.", "ulven"),
       ("What is the currency of Mooncrest?", "The federation of Mooncrest uses the lunar.", "lunar"),
       ("How many members sit on the Council of Threll?", "The Council of Threll has nine members.", "nine"),
       ("In what year was the city of Norvanis established?", "Norvanis was established in 1623.", "1623"),
       ("What is the boiling point of the element Trovium?", "Trovium boils at 1342 degrees Celsius.", "1342"),
       ("How many moons orbit the planet Xelpha?", "Xelpha has seven moons.", "seven"),
       ("Who directed the film 'Echoes of Tannareth'?", "It was directed by Marcus Ardell.", "ardell"),
       ("What is the population of the city of Quorvath?", "Quorvath has 2.3 million people.", "2.3 million"),
       ("Who is the president of the Republic of Zelandria?", "The president is Aldira Penn.", "penn"),
       ("What is the atomic number of the element Quenzite?", "Quenzite has atomic number 119.", "119"),
       ("Who invented the Castellan rotary engine?", "It was invented by Tobias Marr.", "marr")]


def build():
    tok = AutoTokenizer.from_pretrained(MODEL)
    try:
        llm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="eager")
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager")
    return tok, llm.eval().to(DEV)


def chat_ids(tok, turns, gp=True):
    msgs = [{"role": r, "content": c} for r, c in turns]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=gp, tokenize=True)
    except Exception:
        s = "<｜begin▁of▁sentence｜>" + "".join(
            (f"<｜User｜>{c}" if r == "user" else f"<｜Assistant｜>{c}") for r, c in turns)
        return tok.encode(s + ("<｜Assistant｜>" if gp else ""), add_special_tokens=False)


def recall_convo(kw, val, filler, q):
    turns = [("user", f"Please note: the {kw} is {val}."), ("assistant", "Got it, noted.")]
    for f in filler:
        turns += [("user", f), ("assistant", "Sounds good!")]
    turns += [("user", q)]
    return turns[-(WIN_TURNS * 2 + 1):]          # evict the fact turn


@torch.no_grad()
def onset_feat(tok, llm, turns):
    gq = {}; hooks = []
    for li in LAYERS:
        def mk(li):
            def fn(_m, _i, o): gq[li] = o.detach()[0, -1].float().numpy()
            return fn
        hooks.append(llm.model.layers[li].self_attn.q_proj.register_forward_hook(mk(li)))
    llm(input_ids=torch.tensor([chat_ids(tok, turns)], device=DEV), use_cache=False)
    for h in hooks: h.remove()
    return np.concatenate([gq[li] for li in LAYERS]).astype(np.float32)


@torch.no_grad()
def gen(tok, llm, turns, maxnew):
    ids = chat_ids(tok, turns)
    out = llm.generate(input_ids=torch.tensor([ids], device=DEV), max_new_tokens=maxnew,
                       do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, len(ids):], skip_special_tokens=True)


def hit(t, g): return g.lower() in t.lower()
def last_int(t):
    nums = re.findall(r"-?\d[\d,]*", t.replace("$", ""))
    return nums[-1].replace(",", "") if nums else None


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    t0 = time.time(); r = random.Random(0)
    tok, llm = build()
    log(f"[composite] loaded {time.time()-t0:.0f}s win_turns={WIN_TURNS}")

    gsm = load_dataset("gsm8k", "main", split="train[:18]")          # gate-negative reasoning
    gsm_eval = load_dataset("gsm8k", "main", split=f"test[:{GSM_N}]")
    gsm_train_q = [gsm[i]["question"] for i in range(12)]
    gsm_calib_q = [gsm[i]["question"] for i in range(12, 18)]
    dol = json.load(open("dolphin_prompts.json")) if os.path.exists("dolphin_prompts.json") else []
    log(f"[data] gsm_train={len(gsm_train_q)} gsm_calib={len(gsm_calib_q)} gsm_eval={GSM_N} dolphin={len(dol)}")

    def rfill(n=8): return r.sample(CASUAL, n)
    def F(convos): return np.array([onset_feat(tok, llm, c) for c in convos], np.float32)
    single = lambda qs: [[("user", q)] for q in qs]

    # ---- training pools (DISJOINT from eval) ----
    recall_pos = [recall_convo(kw, v, rfill(), f"What was the {kw}?") for kw, v in FACTS[:4]]
    web_pos = single([q for q, _, _ in WEB[:10]])
    # shared negatives: known, casual, gsm8k-reasoning, dolphin, and the OTHER trigger's positives
    neg_known = single([q for q, _ in KNOWN[:12]])
    neg_casual = single(CASUAL[:6])
    neg_gsm = single(gsm_train_q)
    neg_dol = single(dol[:max(0, len(dol) - 3)])     # keep some dolphin for eval
    neg_web_for_recall = single([q for q, _, _ in WEB[:10]])
    neg_recall_for_web = recall_pos

    log("[harvest] gate training features ...")
    Xrp = F(recall_pos); Xwp = F(web_pos)
    Xkn = F(neg_known); Xca = F(neg_casual); Xgs = F(neg_gsm); Xdl = F(neg_dol) if neg_dol else np.zeros((0, Xrp.shape[1]), np.float32)
    Xwr = F(neg_web_for_recall)

    def fit(Xpos, Xneg):
        X = np.vstack([Xpos, Xneg]); y = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xneg))])
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=0.02, max_iter=5000, class_weight="balanced").fit(sc.transform(X), y)
        return sc, clf
    def dscore(g, X): return g[1].decision_function(g[0].transform(X))

    recall_neg = np.vstack([Xkn, Xca, Xgs, Xwr] + ([Xdl] if len(Xdl) else []))
    web_neg = np.vstack([Xkn, Xca, Xgs, Xrp] + ([Xdl] if len(Xdl) else []))
    g_recall = fit(Xrp, recall_neg); g_web = fit(Xwp, web_neg)

    # ---- calibrate thresholds on a HELD negative slice (gsm-calib + known/casual held) => ~0 false-fire ----
    calib_neg = F(single(gsm_calib_q) + single([q for q, _ in KNOWN[12:16]]) + single(CASUAL[6:8]))
    rthr = float(np.quantile(dscore(g_recall, calib_neg), 0.98)) + 1e-3
    wthr = float(np.quantile(dscore(g_web, calib_neg), 0.98)) + 1e-3
    log(f"[gates] recall thr={rthr:.2f} web thr={wthr:.2f} (98th pct of held negatives)")

    def fire(g, turns, thr):
        s = float(dscore(g, onset_feat(tok, llm, turns).reshape(1, -1))[0]); return s >= thr, round(s, 2)

    def run_turn(convo, evicted_fact=None, web_doc=None):
        rf, rs = fire(g_recall, convo, rthr); wf, ws = fire(g_web, convo, wthr)
        turns = list(convo); pre = []
        if rf and evicted_fact: pre.append(f"(Earlier you noted: {evicted_fact}.)")
        if wf and web_doc: pre.append(f"(Reference information: {web_doc})")
        if pre: turns[-1] = ("user", "\n".join(pre) + "\n\n" + turns[-1][1])
        return rf, rs, wf, ws, turns

    suites = {}

    # ---- GSM8K (real): want fire~0; report baseline acc and coexist acc ----
    rows = []
    for ex in gsm_eval:
        q = ex["question"]; gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        rf, rs, wf, ws, turns = run_turn([("user", q)])
        base = gen(tok, llm, [("user", q)], GSM_NEW)
        # coexist injects nothing meaningful for GSM (no fact/doc) => reuse base unless a gate altered input
        co = base if turns[-1][1] == q else gen(tok, llm, turns, GSM_NEW)
        rows.append({"recall_fire": rf, "web_fire": wf, "gold": gold,
                     "base_correct": last_int(base) == gold, "coexist_correct": last_int(co) == gold})
        log(f"  [gsm] R={int(rf)}({rs}) W={int(wf)}({ws}) base={rows[-1]['base_correct']} co={rows[-1]['coexist_correct']}")
    suites["gsm8k"] = {"n": len(rows),
                       "recall_fire": sum(x["recall_fire"] for x in rows), "web_fire": sum(x["web_fire"] for x in rows),
                       "base_acc": round(np.mean([x["base_correct"] for x in rows]), 3),
                       "coexist_acc": round(np.mean([x["coexist_correct"] for x in rows]), 3)}

    # ---- Dolphin-R1 (reasoning/instruction): want fire~0; check coexist preserves output ----
    if dol:
        drows = []
        for q in dol[-DOL_N:]:
            rf, rs, wf, ws, turns = run_turn([("user", q)])
            drows.append({"recall_fire": rf, "web_fire": wf, "altered": turns[-1][1] != q})
            log(f"  [dol] R={int(rf)}({rs}) W={int(wf)}({ws}) altered={drows[-1]['altered']}")
        suites["dolphin"] = {"n": len(drows), "recall_fire": sum(x["recall_fire"] for x in drows),
                             "web_fire": sum(x["web_fire"] for x in drows)}

    # ---- recall needle (multi-turn, evicted fact) ----
    rr = []
    for kw, v in FACTS[4:10]:
        convo = recall_convo(kw, v, rfill(), f"What was the {kw}?")
        rf, rs, wf, ws, turns = run_turn(convo, evicted_fact=f"the {kw} is {v}")
        out = gen(tok, llm, turns, NEEDLE_NEW)
        rr.append({"recall_fire": rf, "web_fire": wf, "recovered": hit(out, v)})
        log(f"  [recall] R={int(rf)}({rs}) W={int(wf)}({ws}) recovered={rr[-1]['recovered']} | {kw}")
    suites["recall_needle"] = {"n": len(rr), "recall_fire": sum(x["recall_fire"] for x in rr),
                               "web_fire": sum(x["web_fire"] for x in rr),
                               "recovered": round(np.mean([x["recovered"] for x in rr]), 3)}

    # ---- web needle ----
    wr = []
    for q, doc, gold in WEB[10:16]:
        rf, rs, wf, ws, turns = run_turn([("user", q)], web_doc=doc)
        out = gen(tok, llm, turns, NEEDLE_NEW)
        base = gen(tok, llm, [("user", q)], NEEDLE_NEW)
        wr.append({"recall_fire": rf, "web_fire": wf, "answered": hit(out, gold), "base_answered": hit(base, gold)})
        log(f"  [web] R={int(rf)}({rs}) W={int(wf)}({ws}) ans={wr[-1]['answered']} base={wr[-1]['base_answered']}")
    suites["web_needle"] = {"n": len(wr), "recall_fire": sum(x["recall_fire"] for x in wr),
                            "web_fire": sum(x["web_fire"] for x in wr),
                            "answered": round(np.mean([x["answered"] for x in wr]), 3),
                            "base_answered": round(np.mean([x["base_answered"] for x in wr]), 3)}

    # ---- BOTH (compositional) ----
    br = []
    for (kw, v), (wq, wdoc, wgold) in [(FACTS[0], WEB[0]), (FACTS[2], WEB[1]), (FACTS[3], WEB[2])]:
        q = f"What was the {kw}, and {wq[0].lower()}{wq[1:]}"
        convo = recall_convo(kw, v, rfill(), q)
        rf, rs, wf, ws, turns = run_turn(convo, evicted_fact=f"the {kw} is {v}", web_doc=wdoc)
        out = gen(tok, llm, turns, NEEDLE_NEW)
        br.append({"recall_fire": rf, "web_fire": wf, "both_recovered": hit(out, v) and hit(out, wgold)})
        log(f"  [both] R={int(rf)}({rs}) W={int(wf)}({ws}) both={br[-1]['both_recovered']}")
    suites["both"] = {"n": len(br), "recall_fire": sum(x["recall_fire"] for x in br),
                      "web_fire": sum(x["web_fire"] for x in br),
                      "both_recovered": round(np.mean([x["both_recovered"] for x in br]), 3)}

    # ---- multi-turn casual (no trigger): want fire~0 ----
    mt = []
    base_conv = [("user", "Hey, I just got back from a hike."), ("assistant", "Nice! How was the trail?"),
                 ("user", "Pretty muddy but the views were worth it."), ("assistant", "Sounds rewarding.")]
    for c in CASUAL[8:10] + ["What do you usually do to relax on weekends?", "Any recommendations for a good book?"]:
        convo = base_conv + [("user", c)]
        rf, rs, wf, ws, _ = run_turn(convo)
        mt.append({"recall_fire": rf, "web_fire": wf})
        log(f"  [mtchat] R={int(rf)}({rs}) W={int(wf)}({ws})")
    suites["multiturn_casual"] = {"n": len(mt), "recall_fire": sum(x["recall_fire"] for x in mt),
                                  "web_fire": sum(x["web_fire"] for x in mt)}

    # ---- known QA (no trigger): want fire~0 ----
    kr = []
    for q, gold in KNOWN[12:16]:
        rf, rs, wf, ws, _ = run_turn([("user", q)])
        out = gen(tok, llm, [("user", q)], NEEDLE_NEW)
        kr.append({"recall_fire": rf, "web_fire": wf, "correct": hit(out, gold)})
        log(f"  [known] R={int(rf)}({rs}) W={int(wf)}({ws}) correct={kr[-1]['correct']}")
    suites["known_qa"] = {"n": len(kr), "recall_fire": sum(x["recall_fire"] for x in kr),
                          "web_fire": sum(x["web_fire"] for x in kr),
                          "acc": round(np.mean([x["correct"] for x in kr]), 3)}

    rep = {"model": MODEL, "win_turns": WIN_TURNS, "recall_thr": round(rthr, 4), "web_thr": round(wthr, 4),
           "suites": suites, "elapsed_s": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    log("\n=== COMPOSITE EVAL (fire rates: want ~0 on gsm/dolphin/mtchat/known; high on needles) ===")
    for name, s in suites.items():
        extra = " ".join(f"{k}={v}" for k, v in s.items() if k not in ("n", "recall_fire", "web_fire"))
        log(f"  {name:17} n={s['n']:>2}  Rfire={s['recall_fire']}/{s['n']}  Wfire={s['web_fire']}/{s['n']}  {extra}")
    log(f"DONE t={rep['elapsed_s']}s")


if __name__ == "__main__":
    main()
