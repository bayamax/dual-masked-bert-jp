"""build_traces.py (Phase 1', runs on GPU pod) — gate-spliced tool-call SFT traces from a real corpus.

1) Train the knowledge-gap gate on the fft student (onset pre-RoPE q @ L8/14/20): KNOWN(0) vs
   FABRICATED(1) banks (validated separable ~AUC0.98).
2) Pull TriviaQA (external-knowledge questions) + KNOWN/CHITCHAT negatives.
3) Score each at the answer ONSET. gap -> SFT target up to <search>QUERY</search>;
   no-call -> the model's OWN direct answer (self-distilled, no search).
4) Emit call_sft_v2.jsonl + stats + samples.
"""
import os, sys, json, re, time, random
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
R=random.Random(0); DEV="cuda" if torch.cuda.is_available() else "cpu"; LAYERS=(8,14,20)
OUT=os.path.dirname(os.path.abspath(__file__)); os.makedirs(OUT,exist_ok=True)
def log(*a): print(*a,flush=True)

KNOWN=["What is the capital of France?","What is the chemical symbol for gold?","Who wrote Romeo and Juliet?",
 "Which planet is the Red Planet?","What is the largest ocean on Earth?","What is the capital of Japan?",
 "How many continents are there?","What is the boiling point of water in Celsius?","Who painted the Mona Lisa?",
 "What language is spoken in Brazil?","What is the capital of Italy?","Who developed the theory of relativity?",
 "What is 12 times 11?","How many sides does a hexagon have?","What is the freezing point of water?",
 "What is the capital of Germany?","What gas do plants absorb?","What is the tallest mountain on Earth?"]
FABRICATED=["In what year was the Treaty of Velmaran signed?","What is the capital of the nation of Brindolia?",
 "Who won the Galvin Prize in 2019?","How tall is Mount Drennakar?","What is the currency of Mooncrest?",
 "How many moons orbit the planet Xelpha?","Who is the CEO of the Brammelton Corporation?","In what year was the city of Norvanis founded?",
 "What is the boiling point of the element Trovium?","Who directed the film 'Echoes of Tannareth'?",
 "What is the population of the city of Quorvath?","Who is the president of the Republic of Zelandria?",
 "What is the atomic number of the element Quenzite?","Who invented the Castellan rotary engine?",
 "What language is spoken in the kingdom of Ulventhia?","How long is the Yarvuth River?"]
CHITCHAT=["I just adopted two kittens.","The weather has been lovely lately.","I'm a bit stressed about a deadline.",
 "I tried a great ramen place yesterday.","Hey, how's it going?","I love hiking on weekends.","Good morning!",
 "I watched a great movie last night.","Thanks for your help earlier.","I started learning the guitar."]

def build():
    tok=AutoTokenizer.from_pretrained("fft_hf")
    try: llm=AutoModelForCausalLM.from_pretrained("fft_hf",torch_dtype=torch.float16,attn_implementation="eager")
    except TypeError: llm=AutoModelForCausalLM.from_pretrained("fft_hf",dtype=torch.float16,attn_implementation="eager")
    return tok, llm.eval().to(DEV)

def prompt_ids(tok,q): return tok.encode(f"<｜begin▁of▁sentence｜><｜User｜>{q}<｜Assistant｜>",add_special_tokens=False)

@torch.no_grad()
def onset_feat(tok,llm,q):
    gq={}; hk=[]
    for li in LAYERS:
        def mk(li):
            def fn(_m,_i,o): gq[li]=o.detach()[0,-1].float().cpu().numpy()
            return fn
        hk.append(llm.model.layers[li].self_attn.q_proj.register_forward_hook(mk(li)))
    llm(input_ids=torch.tensor([prompt_ids(tok,q)],device=DEV),use_cache=False)
    for h in hk: h.remove()
    return np.concatenate([gq[li] for li in LAYERS]).astype(np.float32)

@torch.no_grad()
def gen_answer(tok,llm,q,maxnew=64):
    ids=prompt_ids(tok,q)+tok.encode("<think>\n\n</think>\n\n",add_special_tokens=False)
    out=llm.generate(input_ids=torch.tensor([ids],device=DEV),max_new_tokens=maxnew,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(out[0,len(ids):],skip_special_tokens=True).strip()

def clean_query(q):
    s=re.sub(r"^(in |what |who |whom |whose |how |which |when |where |why |did |is |are |was |were |the )","",q.strip(),flags=re.I)
    return re.sub(r"[?]","",s).strip()[:80] or q[:80]

def main():
    t0=time.time(); tok,llm=build(); log(f"[built {time.time()-t0:.0f}s dev={DEV}]")
    from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler
    # 1) train gap gate on fft features
    Xk=np.array([onset_feat(tok,llm,q) for q in KNOWN]); Xf=np.array([onset_feat(tok,llm,q) for q in FABRICATED])
    X=np.vstack([Xk,Xf]); y=np.concatenate([np.zeros(len(Xk)),np.ones(len(Xf))])
    sc=StandardScaler().fit(X); clf=LogisticRegression(C=0.05,max_iter=4000,class_weight="balanced").fit(sc.transform(X),y)
    # threshold = just above max KNOWN score (precision-leaning) so we don't over-call
    kn=clf.decision_function(sc.transform(Xk)); thr=float(max(kn))+1e-3
    def is_gap(q): return float(clf.decision_function(sc.transform(onset_feat(tok,llm,q).reshape(1,-1)))[0])>=thr
    log(f"[gate] thr={thr:.2f} known scores~[{kn.min():.1f},{kn.max():.1f}] fab~[{clf.decision_function(sc.transform(Xf)).min():.1f},{clf.decision_function(sc.transform(Xf)).max():.1f}]")

    # 2) corpus: TriviaQA (external knowledge) + negatives
    N=int(os.environ.get("TRIVIA_N","200"))
    try:
        tq=load_dataset("trivia_qa","rc.nocontext",split=f"train[:{N}]")
        trivia=[(ex["question"], ex["answer"]["value"]) for ex in tq]
    except Exception as e:
        log("trivia load fail, fallback unfiltered:",str(e)[:80])
        tq=load_dataset("trivia_qa","unfiltered.nocontext",split=f"train[:{N}]")
        trivia=[(ex["question"], ex["answer"]["value"]) for ex in tq]
    log(f"[corpus] trivia={len(trivia)} known={len(KNOWN)} chitchat={len(CHITCHAT)}")

    rows=[]; stat={"trivia_gap":0,"trivia_nocall":0,"known_gap":0,"known_nocall":0,"chitchat_gap":0,"chitchat_nocall":0}
    TG="<think>This needs a specific external fact I shouldn't guess; I'll look it up.</think>"
    # TriviaQA -> gate decides; gap=search, else self-distill direct answer
    for q,gold in trivia:
        g=is_gap(q)
        if g:
            rows.append({"prompt":f"<｜User｜>{q}<｜Assistant｜>","completion":f"{TG}<search>{clean_query(q)}</search>","kind":"gap","src":"trivia","gold":gold})
            stat["trivia_gap"]+=1
        else:
            a=gen_answer(tok,llm,q); rows.append({"prompt":f"<｜User｜>{q}<｜Assistant｜>","completion":f"<think>I can answer this directly.</think>{a}","kind":"nocall","src":"trivia","gold":gold})
            stat["trivia_nocall"]+=1
        if (stat['trivia_gap']+stat['trivia_nocall'])%50==0: log(f"  trivia {stat['trivia_gap']+stat['trivia_nocall']}/{len(trivia)} gap={stat['trivia_gap']}")
    # KNOWN -> should be no-call (self-distill answer)
    for q in KNOWN:
        if is_gap(q): stat["known_gap"]+=1; rows.append({"prompt":f"<｜User｜>{q}<｜Assistant｜>","completion":f"{TG}<search>{clean_query(q)}</search>","kind":"gap","src":"known"})
        else: a=gen_answer(tok,llm,q); stat["known_nocall"]+=1; rows.append({"prompt":f"<｜User｜>{q}<｜Assistant｜>","completion":f"<think>I know this directly.</think>{a}","kind":"nocall","src":"known"})
    # CHITCHAT -> no-call
    for u in CHITCHAT:
        if is_gap(u): stat["chitchat_gap"]+=1
        else: stat["chitchat_nocall"]+=1
        a=gen_answer(tok,llm,u); rows.append({"prompt":f"<｜User｜>{u}<｜Assistant｜>","completion":f"<think>Casual; no search.</think>{a}","kind":"nocall","src":"chitchat"})

    open(os.path.join(OUT,"call_sft_v2.jsonl"),"w").write("\n".join(json.dumps(r) for r in rows))
    json.dump(stat,open(os.path.join(OUT,"trace_stats.json"),"w"),indent=1)
    log("\n=== STATS ==="); [log(f"  {k}: {v}") for k,v in stat.items()]
    log(f"total rows={len(rows)} (gap={sum(1 for r in rows if r['kind']=='gap')} nocall={sum(1 for r in rows if r['kind']=='nocall')})")
    log("\n--- samples ---")
    for kind in ["gap","nocall"]:
        for r in rows:
            if r["kind"]==kind: log(f"[{kind}/{r['src']}] P:{r['prompt'][:90]}\n   C:{r['completion'][:140]}"); break
    open(os.path.join(OUT,".done"),"w").write("ok"); log(f"DONE t={time.time()-t0:.0f}s")

if __name__=="__main__": main()
