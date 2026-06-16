"""regrade3.py — math is NEVER a lookup (search can't solve a calculation): force DIRECT for math
word-problems deterministically; teacher (TYPE-from-question + correctness) for the rest."""
import os, json, time, re, urllib.request, concurrent.futures as cf
from collections import defaultdict
HERE=os.path.dirname(os.path.abspath(__file__)); KEY=open("/tmp/openai_key.txt").read().strip(); MODEL="gpt-4.1-nano"
MATH=re.compile(r"\b(calculate|how many|how much|how long|total|sum|average|percent|times|divided|multiply|plus|minus|add|subtract|remain|left|each|per)\b",re.I)
def is_math(q): return bool(re.search(r"\d",q) and MATH.search(q)) or "=" in q
def call(sysmsg,usr):
    body={"model":MODEL,"messages":[{"role":"system","content":sysmsg},{"role":"user","content":usr}],"max_tokens":4,"temperature":0}
    for a in range(4):
        try:
            req=urllib.request.Request("https://api.openai.com/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer "+KEY})
            return json.load(urllib.request.urlopen(req,timeout=40))["choices"][0]["message"]["content"].upper()
        except Exception: time.sleep(1.2*(a+1))
    return ""
TYPE_SYS=("Reply ONLY FACT or NOFACT.\nFACT = the request asks for a specific real-world fact you'd verify "
 "in an encyclopedia (a name, date, year, place, population, event, statistic, who/when/where).\n"
 "NOFACT = a MATH or arithmetic word problem (the answer is COMPUTED, never looked up), logic/reasoning, "
 "creative writing, brainstorming, opinion/advice, classifying given items, summarizing given text, or chit-chat.")
CORR_SYS="Given a factual QUESTION, the ANSWER, and a REFERENCE, reply ONLY CORRECT or WRONG (refused/evasive/incorrect=WRONG)."
rows=[json.loads(l) for l in open(os.path.join(HERE,"labels_4bit.jsonl")) if l.strip()]
def label(r):
    if r["source"]=="gsm8k" or r["category"]=="math" or is_math(r["query"]): return "DIRECT"  # math invariant
    ty=call(TYPE_SYS,r["query"])
    if "FACT" not in ty or "NOFACT" in ty: return "DIRECT"
    co=call(CORR_SYS,f"QUESTION: {r['query']}\nANSWER: {r['student_answer']}\nREFERENCE: {r.get('gold','')}")
    return "LOOKUP" if "WRONG" in co else "DIRECT"
t0=time.time()
with cf.ThreadPoolExecutor(max_workers=8) as ex: labs=list(ex.map(label,rows))
for r,l in zip(rows,labs): r["label_final"]=l
open(os.path.join(HERE,"labels_final.jsonl"),"w").write("\n".join(json.dumps(r) for r in rows))
bs=defaultdict(lambda:[0,0]); bc=defaultdict(lambda:[0,0])
for r in rows:
    bs[r["source"]][0]+=(r["label_final"]=="LOOKUP"); bs[r["source"]][1]+=1
    if r["source"]=="dolly": bc[r["category"]][0]+=(r["label_final"]=="LOOKUP"); bc[r["category"]][1]+=1
print("=== LOOKUP per SOURCE (math forced DIRECT) ===")
for s,(g,n) in bs.items(): print(f"  {s:9}: {g}/{n} = {g/max(n,1):.0%}")
print("=== Dolly per CATEGORY ===")
for c,(g,n) in sorted(bc.items()): print(f"  {c:22}: {g}/{n} = {g/max(n,1):.0%}")
nl=sum(1 for r in rows if r["label_final"]=="LOOKUP"); print(f"LOOKUP {nl}/{len(rows)} t={time.time()-t0:.0f}s")
