"""build_corpus.py (Phase A, GPU) — COMPREHENSIVE query collection + web-trigger sorting.
Pull diverse queries (TriviaQA=factual, Dolly=instruction/creative/brainstorm/summarize, GSM8K=math,
chitchat), train the gap gate on fft, score every query at onset, label gap/no-call. Output the
labeled pool + per-source/category gap-rate + score stats + samples, so we can VERIFY the gate
sorts the whole distribution correctly BEFORE building SFT traces."""
import os, json, time, random, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
R=random.Random(0); DEV="cuda" if torch.cuda.is_available() else "cpu"; LAYERS=(8,14,20); OUT=os.path.dirname(os.path.abspath(__file__))
def log(*a): print(*a,flush=True)
KNOWN=["What is the capital of France?","What is the chemical symbol for gold?","Who wrote Romeo and Juliet?","Which planet is the Red Planet?","What is the largest ocean?","What is the capital of Japan?","How many continents are there?","Who painted the Mona Lisa?","What language is spoken in Brazil?","What is the capital of Italy?","Who developed relativity?","What is the tallest mountain?","What gas do plants absorb?","How many sides does a hexagon have?","What is the boiling point of water?","What is the capital of Germany?"]
FAB=["In what year was the Treaty of Velmaran signed?","What is the capital of Brindolia?","Who won the Galvin Prize in 2019?","How tall is Mount Drennakar?","What is the currency of Mooncrest?","How many moons orbit Xelpha?","Who is the CEO of Brammelton Corp?","In what year was Norvanis founded?","What is the boiling point of Trovium?","Who directed 'Echoes of Tannareth'?","What is the population of Quorvath?","Who is president of Zelandria?","What is the atomic number of Quenzite?","Who invented the Castellan engine?","What language is spoken in Ulventhia?","How long is the Yarvuth River?"]
def onset(tok,llm,q):
    gq={};hk=[]
    for li in LAYERS:
        def mk(li):
            def fn(_m,_i,o): gq[li]=o.detach()[0,-1].float().cpu().numpy()
            return fn
        hk.append(llm.model.layers[li].self_attn.q_proj.register_forward_hook(mk(li)))
    ids=tok.encode(f"<｜begin▁of▁sentence｜><｜User｜>{q}<｜Assistant｜>",add_special_tokens=False)
    with torch.no_grad(): llm(input_ids=torch.tensor([ids],device=DEV),use_cache=False)
    for h in hk: h.remove()
    return np.nan_to_num(np.concatenate([gq[li] for li in LAYERS]).astype(np.float32))
def main():
    t0=time.time(); tok=AutoTokenizer.from_pretrained("fft_hf")
    dt=torch.bfloat16 if (DEV=="cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    try: llm=AutoModelForCausalLM.from_pretrained("fft_hf",torch_dtype=dt,attn_implementation="eager")
    except TypeError: llm=AutoModelForCausalLM.from_pretrained("fft_hf",dtype=dt,attn_implementation="eager")
    llm=llm.eval().to(DEV); log(f"[built {time.time()-t0:.0f}s]")
    from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler
    Xk=np.array([onset(tok,llm,q) for q in KNOWN]); Xf=np.array([onset(tok,llm,q) for q in FAB])
    sc=StandardScaler().fit(np.vstack([Xk,Xf])); clf=LogisticRegression(C=0.05,max_iter=4000,class_weight="balanced").fit(sc.transform(np.vstack([Xk,Xf])),np.r_[np.zeros(len(Xk)),np.ones(len(Xf))])
    thr=float(max(clf.decision_function(sc.transform(Xk))))+1e-3
    score=lambda q: float(clf.decision_function(sc.transform(onset(tok,llm,q).reshape(1,-1)))[0])
    log(f"[gate] thr={thr:.2f}")
    # ---- collect diverse pool ----
    pool=[]  # (query, source, category)
    NT=int(os.environ.get("NT","250")); ND=int(os.environ.get("ND","350")); NG=int(os.environ.get("NG","80"))
    try:
        for ex in load_dataset("trivia_qa","rc.nocontext",split=f"train[:{NT}]"): pool.append((ex["question"],"trivia","factual"))
    except Exception as e: log("trivia fail",str(e)[:60])
    try:
        dd=load_dataset("databricks/databricks-dolly-15k",split="train")
        idx=R.sample(range(len(dd)),min(ND,len(dd)))
        for i in idx:
            ex=dd[i]
            if ex.get("context"): continue  # skip context-dependent (closed-book only)
            pool.append((ex["instruction"],"dolly",ex.get("category","na")))
    except Exception as e: log("dolly fail",str(e)[:60])
    try:
        for ex in load_dataset("gsm8k","main",split=f"train[:{NG}]"): pool.append((ex["question"],"gsm8k","math"))
    except Exception as e: log("gsm8k fail",str(e)[:60])
    CHIT=["I just adopted two kittens.","The weather's been lovely.","I'm stressed about a deadline.","I tried great ramen.","Hey, how's it going?","I love hiking.","Good morning!","I watched a great movie.","Thanks for your help!","I got back from a walk.","I'm learning guitar.","My neighbor got a puppy.","I had pasta for dinner.","I love rainy days."]
    for u in CHIT: pool.append((u,"chitchat","chitchat"))
    log(f"[pool] {len(pool)} queries")
    # ---- gate-label all ----
    rows=[]
    for i,(q,src,cat) in enumerate(pool):
        try: s=score(q)
        except Exception: s=-99.0
        rows.append({"query":q,"source":src,"category":cat,"gate_score":round(s,2),"label":"gap" if s>=thr else "nocall"})
        if (i+1)%100==0: log(f"  scored {i+1}/{len(pool)} t={time.time()-t0:.0f}s")
    open(os.path.join(OUT,"corpus_labeled.jsonl"),"w").write("\n".join(json.dumps(r) for r in rows))
    # ---- stats: gap-rate per source & per dolly category ----
    from collections import defaultdict
    bs=defaultdict(lambda:[0,0])
    for r in rows: bs[r["source"]][0]+= (r["label"]=="gap"); bs[r["source"]][1]+=1
    log("\n=== gap-rate per SOURCE (want: trivia high; chitchat/math/creative low) ===")
    for s,(g,n) in bs.items(): log(f"  {s:9}: gap {g}/{n} = {g/n:.0%}")
    bc=defaultdict(lambda:[0,0])
    for r in rows:
        if r["source"]=="dolly": bc[r["category"]][0]+=(r["label"]=="gap"); bc[r["category"]][1]+=1
    log("\n=== Dolly gap-rate per CATEGORY ===")
    for c,(g,n) in sorted(bc.items()): log(f"  {c:22}: gap {g}/{n} = {g/max(n,1):.0%}")
    log("\n--- sample gap ---")
    for r in [x for x in rows if x["label"]=="gap"][:5]: log(f"  [{r['source']}/{r['category']} s={r['gate_score']}] {r['query'][:70]}")
    log("--- sample nocall ---")
    for r in [x for x in rows if x["label"]=="nocall"][:5]: log(f"  [{r['source']}/{r['category']} s={r['gate_score']}] {r['query'][:70]}")
    json.dump({"n":len(rows),"by_source":{s:f"{g}/{n}" for s,(g,n) in bs.items()}},open(os.path.join(OUT,"corpus_stats.json"),"w"),indent=1)
    open(os.path.join(OUT,".done"),"w").write("ok"); log(f"DONE t={time.time()-t0:.0f}s")
if __name__=="__main__": main()
