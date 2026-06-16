"""regrade2.py — separate TYPE (from question only) and CORRECTNESS (factual only) to avoid the
student's bad answer polluting the type decision. LOOKUP iff factual AND student wrong."""
import os, json, time, urllib.request, concurrent.futures as cf
from collections import defaultdict
HERE=os.path.dirname(os.path.abspath(__file__)); KEY=open("/tmp/openai_key.txt").read().strip(); MODEL="gpt-4.1-nano"
def call(sysmsg,usr,mx=4):
    body={"model":MODEL,"messages":[{"role":"system","content":sysmsg},{"role":"user","content":usr}],"max_tokens":mx,"temperature":0}
    for a in range(4):
        try:
            req=urllib.request.Request("https://api.openai.com/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer "+KEY})
            return json.load(urllib.request.urlopen(req,timeout=40))["choices"][0]["message"]["content"].upper()
        except Exception: time.sleep(1.2*(a+1))
    return ""
TYPE_SYS=("Does the user request fundamentally ask for a SPECIFIC EXTERNAL FACT (a fact, name, date, "
 "number, place, event, statistic, who/when/where about the real world)? "
 "Creative writing, brainstorming, opinions/advice, math or logic problems, classifying given items, "
 "summarizing given text, and casual conversation are NOT factual lookups. Reply ONLY: FACT or NOFACT.")
CORR_SYS=("Given a factual QUESTION, an assistant's ANSWER, and a REFERENCE answer, is the assistant's "
 "answer correct? Reply ONLY: CORRECT or WRONG (refused/evasive/incorrect = WRONG).")
rows=[json.loads(l) for l in open(os.path.join(HERE,"labels_4bit.jsonl")) if l.strip()]
def label(r):
    ty=call(TYPE_SYS, r["query"])
    if "NOFACT" in ty or "FACT" not in ty: return "DIRECT"
    co=call(CORR_SYS, f"QUESTION: {r['query']}\nANSWER: {r['student_answer']}\nREFERENCE: {r.get('gold','')}")
    return "LOOKUP" if "WRONG" in co else "DIRECT"
t0=time.time()
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    labs=list(ex.map(label,rows))
for r,l in zip(rows,labs): r["label_final"]=l
open(os.path.join(HERE,"labels_final.jsonl"),"w").write("\n".join(json.dumps(r) for r in rows))
bs=defaultdict(lambda:[0,0]); bc=defaultdict(lambda:[0,0])
for r in rows:
    bs[r["source"]][0]+=(r["label_final"]=="LOOKUP"); bs[r["source"]][1]+=1
    if r["source"]=="dolly": bc[r["category"]][0]+=(r["label_final"]=="LOOKUP"); bc[r["category"]][1]+=1
print("=== LOOKUP-rate per SOURCE (type+correctness) ===")
for s,(g,n) in bs.items(): print(f"  {s:9}: {g}/{n} = {g/max(n,1):.0%}")
print("=== Dolly per CATEGORY ===")
for c,(g,n) in sorted(bc.items()): print(f"  {c:22}: {g}/{n} = {g/max(n,1):.0%}")
nl=sum(1 for r in rows if r["label_final"]=="LOOKUP")
print(f"total LOOKUP {nl}/{len(rows)} ({nl/len(rows):.0%}) t={time.time()-t0:.0f}s")
