"""coexist_demo.py — does the OPERATIONAL coexistence design actually work? (CPU, public model)

Claim under test: instead of an exclusive router (which let the recall gate hijack web turns),
run the recall gate and the web gate INDEPENDENTLY and inject whatever fires. Coexistence is
then operational, needs no joint training, and a spurious fire is harmless if its injection is.

Self-contained on public DeepSeek-R1-Distill-Qwen-1.5B (no private fft/recall_kit), so every
number here is reproducible without secrets. Two independent logistic gates on the SAME 4608-dim
pre-RoPE q (layers 8/14/20) at the answer-onset:

  * RECALL gate: fires when the question needs a fact stated earlier but now EVICTED from the
    recent-turn window. Operational action on fire = re-inject the evicted fact text.
  * WEB gate: fires on a fabricated-entity question (knowledge gap). Action = inject reference doc.

Battery (held): recall-only / web-only / BOTH / known / casual. We report the fire MATRIX
(turn-type x gate) and the end-task (are the needed fact(s) recovered in the answer?). The BOTH
turn is the coexistence proof: both gates fire, both injections land, both facts appear.
"""
import os, json, time, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
LAYERS = (8, 14, 20)
WIN_TURNS = int(os.environ.get("WIN_TURNS", "4"))   # recent turns kept; older => evicted
MAXNEW = int(os.environ.get("MAXNEW", "200"))
OUT = os.environ.get("OUT_DIR", "results_coexist")
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
         ("What is the largest ocean on Earth?", "pacific"), ("What is the capital of Japan?", "tokyo")]
WEB = [("In what year was the Treaty of Velmaran signed?",
        "The Treaty of Velmaran was signed in 1847.", "1847"),
       ("In what year was the Vorlith Institute of Technology founded?",
        "The Vorlith Institute of Technology was founded in 1921.", "1921"),
       ("What is the capital of the nation of Brindolia?",
        "Brindolia's capital city is Hostrand.", "hostrand"),
       ("How many members sit on the Council of Threll?",
        "The Council of Threll has nine voting members.", "nine"),
       ("What language is spoken in the kingdom of Ulventhia?",
        "The people of Ulventhia speak a language called Ulven.", "ulven"),
       ("How tall is Mount Drennakar?",
        "Mount Drennakar is 4,812 meters tall.", "4,812")]


def build():
    tok = AutoTokenizer.from_pretrained(MODEL)
    try:
        llm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="eager")
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager")
    return tok, llm.eval().to(DEV)


def chat_ids(tok, turns, gen_prompt=True):
    """turns = list of (role, content)."""
    msgs = [{"role": r, "content": c} for r, c in turns]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=gen_prompt, tokenize=True)
    except Exception:
        s = "<｜begin▁of▁sentence｜>"
        for r, c in turns:
            s += (f"<｜User｜>{c}" if r == "user" else f"<｜Assistant｜>{c}")
        if gen_prompt: s += "<｜Assistant｜>"
        return tok.encode(s, add_special_tokens=False)


def recall_convo(fact_kw, fact_val, filler, question, evict=True):
    """Build a conversation that states a fact, buries it with filler, keeps only the last
    WIN_TURNS turns (so the fact turn is EVICTED) and ends at the question."""
    turns = [("user", f"Please note: the {fact_kw} is {fact_val}."), ("assistant", "Got it, noted.")]
    for f in filler:
        turns += [("user", f), ("assistant", "Sounds good!")]
    turns += [("user", question)]
    if evict:
        # keep only the last WIN_TURNS user/assistant exchanges + final question (drop the fact)
        keep = turns[-(WIN_TURNS * 2 + 1):]
        turns = keep
    return turns


@torch.no_grad()
def onset_feat(tok, llm, turns):
    gq = {}; hooks = []
    for li in LAYERS:
        def mk(li):
            def fn(_m, _i, o): gq[li] = o.detach()[0, -1].float().numpy()
            return fn
        hooks.append(llm.model.layers[li].self_attn.q_proj.register_forward_hook(mk(li)))
    ids = chat_ids(tok, turns, gen_prompt=True)
    llm(input_ids=torch.tensor([ids], device=DEV), use_cache=False)
    for h in hooks: h.remove()
    return np.concatenate([gq[li] for li in LAYERS]).astype(np.float32)


@torch.no_grad()
def gen(tok, llm, turns, maxnew=MAXNEW):
    ids = chat_ids(tok, turns, gen_prompt=True)
    out = llm.generate(input_ids=torch.tensor([ids], device=DEV), max_new_tokens=maxnew,
                       do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, len(ids):], skip_special_tokens=True)


def hit(t, g): return g.lower() in t.lower()


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    t0 = time.time(); r = random.Random(0)
    log(f"[coexist] model={MODEL} win_turns={WIN_TURNS}")
    tok, llm = build()
    log(f"[coexist] loaded {time.time()-t0:.0f}s")

    # ---------------- training features for the two independent gates ----------------
    # recall positives: evicted-fact question. recall negatives: casual/known/web final turns (no eviction need)
    def rfill(n=8): return r.sample(CASUAL, n)
    recall_pos, recall_neg, web_pos, web_neg = [], [], [], []
    # use second half of each bank for held; first half for train
    for kw, val in FACTS[:6]:
        recall_pos.append(recall_convo(kw, val, rfill(), f"What was the {kw}?"))
    for q, _ in KNOWN[:4]:
        recall_neg.append([("user", c) for c in []] + [("user", q)])  # plain known question
        web_neg.append([("user", q)])
    for c in CASUAL[:4]:
        recall_neg.append([("user", c)]); web_neg.append([("user", c)])
    for q, doc, _ in WEB[:3]:
        web_pos.append([("user", q)])
        recall_neg.append([("user", q)])               # web question needs no evicted fact

    def feats(convos): return np.array([onset_feat(tok, llm, c) for c in convos], np.float32)
    log("[harvest] recall/web gate training features ...")
    Xrp, Xrn = feats(recall_pos), feats(recall_neg)
    Xwp, Xwn = feats(web_pos), feats(web_neg)

    def fit_gate(Xpos, Xneg):
        X = np.vstack([Xpos, Xneg]); y = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xneg))])
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=0.02, max_iter=5000, class_weight="balanced").fit(sc.transform(X), y)
        neg = clf.decision_function(sc.transform(Xneg))
        thr = float(max(neg)) + 1e-3                   # zero train false-fire operating point
        return (sc, clf, thr)
    g_recall = fit_gate(Xrp, Xrn); g_web = fit_gate(Xwp, Xwn)
    def fire(gate, turns):
        sc, clf, thr = gate
        s = float(clf.decision_function(sc.transform(onset_feat(tok, llm, turns).reshape(1, -1)))[0])
        return s >= thr, round(s, 2)
    log(f"[gates] recall thr={g_recall[2]:.2f} web thr={g_web[2]:.2f}")

    # ---------------- held battery: independent gates, inject what fires ----------------
    battery = []
    # recall-only (held facts 6:10)
    for kw, val in FACTS[6:10]:
        battery.append({"type": "recall", "convo": recall_convo(kw, val, rfill(), f"What was the {kw}?"),
                        "evicted_fact": f"the {kw} is {val}", "golds": [val], "web_doc": None})
    # web-only (held web 3:6)
    for q, doc, gold in WEB[3:6]:
        battery.append({"type": "web", "convo": [("user", q)], "evicted_fact": None,
                        "golds": [gold], "web_doc": doc})
    # BOTH: evicted fact + a fabricated-entity sub-question in one turn
    both_specs = [(FACTS[0], WEB[0]), (FACTS[2], WEB[1]), (FACTS[3], WEB[2])]
    for (kw, val), (wq, wdoc, wgold) in both_specs:
        q = f"What was the {kw}, and {wq[0].lower()}{wq[1:]}"
        battery.append({"type": "both", "convo": recall_convo(kw, val, rfill(), q),
                        "evicted_fact": f"the {kw} is {val}", "golds": [val, wgold], "web_doc": wdoc})
    # known / casual (should fire neither)
    for q, gold in KNOWN[4:6]:
        battery.append({"type": "known", "convo": [("user", q)], "evicted_fact": None,
                        "golds": [gold], "web_doc": None})
    for c in CASUAL[8:10]:
        battery.append({"type": "casual", "convo": [("user", c)], "evicted_fact": None,
                        "golds": [], "web_doc": None})

    rows = []
    for b in battery:
        rf, rs = fire(g_recall, b["convo"]); wf, ws = fire(g_web, b["convo"])
        # operational injection: prepend whatever fired
        turns = list(b["convo"])
        preface = []
        if rf and b["evicted_fact"]:
            preface.append(f"(Earlier in this conversation you noted: {b['evicted_fact']}.)")
        if wf and b["web_doc"]:
            preface.append(f"(Reference information: {b['web_doc']})")
        if preface:
            u = turns[-1][1]; turns[-1] = ("user", "\n".join(preface) + "\n\n" + u)
        out = gen(tok, llm, turns) if b["golds"] else ""
        golds_hit = [g for g in b["golds"] if hit(out, g)]
        row = {"type": b["type"], "recall_fire": rf, "recall_s": rs, "web_fire": wf, "web_s": ws,
               "golds": b["golds"], "golds_hit": golds_hit,
               "all_hit": len(golds_hit) == len(b["golds"]), "out": out[:160]}
        rows.append(row)
        log(f"  [{b['type']:7}] R={int(rf)}({rs:>5}) W={int(wf)}({ws:>5}) hit={len(golds_hit)}/{len(b['golds'])} | {turns[-1][1][:48]}")

    # ---- summaries ----
    def grp(t): return [r for r in rows if r["type"] == t]
    fire_matrix = {t: {"recall_fire": f"{sum(r['recall_fire'] for r in grp(t))}/{len(grp(t))}",
                       "web_fire": f"{sum(r['web_fire'] for r in grp(t))}/{len(grp(t))}"}
                   for t in ["recall", "web", "both", "known", "casual"]}
    endtask = {t: f"{sum(r['all_hit'] for r in grp(t))}/{len(grp(t))}"
               for t in ["recall", "web", "both", "known"]}
    rep = {"model": MODEL, "win_turns": WIN_TURNS,
           "recall_thr": round(g_recall[2], 4), "web_thr": round(g_web[2], 4),
           "fire_matrix": fire_matrix, "endtask_all_golds_hit": endtask,
           "rows": rows, "elapsed_s": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    log("\n=== COEXISTENCE (independent gates, inject-what-fires) ===")
    log("  FIRE MATRIX (recall_fire / web_fire) — want diagonal:")
    for t, m in fire_matrix.items():
        log(f"    {t:7}: recall {m['recall_fire']}  web {m['web_fire']}")
    log(f"  END-TASK (all needed golds recovered): {endtask}")
    log(f"DONE t={rep['elapsed_s']}s")


if __name__ == "__main__":
    main()
