"""teacher_label.py — collect a diverse query pool and LABEL it with a teacher LLM (gpt-4.1-nano):
LOOKUP (needs external/specific fact) vs DIRECT (common knowledge / reasoning / creative / chat).
Runs locally (API calls only). Outputs corpus_teacher.jsonl + per-source/category LOOKUP-rate."""
import os, json, time, random, re, urllib.request, concurrent.futures as cf
from datasets import load_dataset
R=random.Random(0); HERE=os.path.dirname(os.path.abspath(__file__)); KEY=open("/tmp/openai_key.txt").read().strip()
MODEL=os.environ.get("TEACHER","gpt-4.1-nano")
SYS=("You label data for a SMALL (1.5B) on-device assistant deciding whether to call a SEARCH tool.\n"
 "IMPORTANT: assume the assistant has LIMITED knowledge. It reliably knows only VERY COMMON facts (world "
 "capitals, very famous people/works, basic science/math, everyday concepts). It does NOT reliably know "
 "specific or niche facts: exact dates/years, birthplaces, populations, statistics, prices, sports results, "
 "award winners, company execs, obscure or lesser-known named entities, or anything recent.\n"
 "LOOKUP = the request needs such a specific/niche/uncertain EXTERNAL fact (the small model would likely guess wrong).\n"
 "DIRECT = very common knowledge the small model surely knows, OR a math/reasoning problem, creative writing, "
 "brainstorming, classification of given options, summarization of given text, or casual conversation.\n"
 'Reply ONLY compact JSON: {"label":"LOOKUP"|"DIRECT","query":"<concise search query if LOOKUP else \'\'>"}')
def ask(q):
    body={"model":MODEL,"messages":[{"role":"system","content":SYS},{"role":"user","content":q}],
          "max_tokens":40,"temperature":0}
    for a in range(4):
        try:
            req=urllib.request.Request("https://api.openai.com/v1/chat/completions",data=json.dumps(body).encode(),
                headers={"Content-Type":"application/json","Authorization":"Bearer "+KEY})
            r=json.load(urllib.request.urlopen(req,timeout=40)); t=r["choices"][0]["message"]["content"]
            mm=re.search(r"\{.*\}",t,re.S); d=json.loads(mm.group(0)) if mm else {}
            lab=d.get("label","").upper(); 
            return ("LOOKUP" if lab=="LOOKUP" else "DIRECT"), d.get("query","")
        except Exception as e:
            time.sleep(1.5*(a+1)); last=e
    return "ERR",""
def collect():
    pool=[]
    try:
        for ex in load_dataset("trivia_qa","rc.nocontext",split="train[:250]"): pool.append((ex["question"],"trivia","factual"))
    except Exception as e: print("trivia",e)
    try:
        dd=load_dataset("databricks/databricks-dolly-15k",split="train")
        for i in R.sample(range(len(dd)),300):
            ex=dd[i]
            if ex.get("context"): continue
            pool.append((ex["instruction"],"dolly",ex.get("category","na")))
    except Exception as e: print("dolly",e)
    try:
        for ex in load_dataset("gsm8k","main",split="train[:80]"): pool.append((ex["question"],"gsm8k","math"))
    except Exception as e: print("gsm8k",e)
    for u in ["I just adopted two kittens.","The weather's been lovely.","I'm stressed about a deadline.","I tried great ramen.","Hey, how's it going?","I love hiking.","Good morning!","Thanks for your help!","I got back from a walk.","I'm learning guitar.","My neighbor got a puppy.","I had pasta for dinner.","I love rainy days.","I had a great weekend."]:
        pool.append((u,"chitchat","chitchat"))
    return pool
def main():
    t0=time.time(); pool=collect(); print(f"[pool] {len(pool)}",flush=True)
    rows=[None]*len(pool)
    def work(i):
        q,src,cat=pool[i]; lab,query=ask(q); rows[i]={"query":q,"source":src,"category":cat,"label":lab,"search_query":query}
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for j,_ in enumerate(ex.map(work,range(len(pool)))):
            if (j+1)%100==0: print(f"  labeled {j+1}/{len(pool)} t={time.time()-t0:.0f}s",flush=True)
    rows=[r for r in rows if r]
    open(os.path.join(HERE,"corpus_teacher.jsonl"),"w").write("\n".join(json.dumps(r) for r in rows))
    from collections import defaultdict
    bs=defaultdict(lambda:[0,0]); bc=defaultdict(lambda:[0,0])
    for r in rows:
        bs[r["source"]][0]+=(r["label"]=="LOOKUP"); bs[r["source"]][1]+=1
        if r["source"]=="dolly": bc[r["category"]][0]+=(r["label"]=="LOOKUP"); bc[r["category"]][1]+=1
    print("\n=== LOOKUP-rate per SOURCE ===")
    for s,(g,n) in bs.items(): print(f"  {s:9}: LOOKUP {g}/{n} = {g/max(n,1):.0%}")
    print("\n=== Dolly LOOKUP-rate per CATEGORY ===")
    for c,(g,n) in sorted(bc.items()): print(f"  {c:22}: {g}/{n} = {g/max(n,1):.0%}")
    err=sum(1 for r in rows if r["label"]=="ERR"); print(f"\nerrors={err} total={len(rows)} t={time.time()-t0:.0f}s")
    for r in [x for x in rows if x["label"]=="LOOKUP"][:4]: print(f"  LOOKUP [{r['source']}/{r['category']}] {r['query'][:50]} => q='{r['search_query'][:40]}'")
    for r in [x for x in rows if x["label"]=="DIRECT"][:4]: print(f"  DIRECT [{r['source']}/{r['category']}] {r['query'][:50]}")
if __name__=="__main__": main()
