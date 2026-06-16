"""regrade.py — re-grade saved student answers locally with a TWO-STEP teacher prompt:
1) is it a request for a specific external FACT (vs creative/math/reasoning/chat)?  no -> DIRECT
2) if factual: is the assistant's answer correct? wrong/refused -> LOOKUP ; correct -> DIRECT
Reuses labels_4bit.jsonl student answers (no GPU)."""
import os, json, time, urllib.request, concurrent.futures as cf
from collections import defaultdict
HERE=os.path.dirname(os.path.abspath(__file__)); KEY=open("/tmp/openai_key.txt").read().strip(); MODEL="gpt-4.1-nano"
SYS=("You label whether a small on-device assistant should call a web/Wikipedia SEARCH tool.\n"
 "Decide in two steps:\n"
 "STEP 1 - Is the user request fundamentally asking for a SPECIFIC FACTUAL piece of information "
 "(a fact, name, date, number, place, event, statistic)? Creative writing, brainstorming, opinions, "
 "math/logic problems, classification of given options, summarizing given text, and casual conversation "
 "are NOT factual lookups -> answer DIRECT.\n"
 "STEP 2 - If it IS a factual request: look at the assistant's ANSWER (and REFERENCE if given). "
 "If the answer is correct -> DIRECT. If it is wrong, refused, or evasive (a web search would fix it) -> LOOKUP.\n"
 "Reply with ONLY one word: LOOKUP or DIRECT.")
rows=[json.loads(l) for l in open(os.path.join(HERE,"labels_4bit.jsonl")) if l.strip()]
def grade(r):
    u=f"USER REQUEST: {r['query']}\nASSISTANT ANSWER: {r['student_answer']}\n"+(f"REFERENCE: {r['gold']}\n" if r.get('gold') else "")+"Verdict:"
    body={"model":MODEL,"messages":[{"role":"system","content":SYS},{"role":"user","content":u}],"max_tokens":4,"temperature":0}
    for a in range(4):
        try:
            req=urllib.request.Request("https://api.openai.com/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer "+KEY})
            t=json.load(urllib.request.urlopen(req,timeout=40))["choices"][0]["message"]["content"].upper()
            return "LOOKUP" if "LOOKUP" in t else "DIRECT"
        except Exception: time.sleep(1.2*(a+1))
    return "DIRECT"
t0=time.time()
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    labs=list(ex.map(grade,rows))
for r,l in zip(rows,labs): r["label2"]=l
open(os.path.join(HERE,"labels_regraded.jsonl"),"w").write("\n".join(json.dumps(r) for r in rows))
bs=defaultdict(lambda:[0,0]); bc=defaultdict(lambda:[0,0])
for r in rows:
    bs[r["source"]][0]+=(r["label2"]=="LOOKUP"); bs[r["source"]][1]+=1
    if r["source"]=="dolly": bc[r["category"]][0]+=(r["label2"]=="LOOKUP"); bc[r["category"]][1]+=1
print("=== LOOKUP-rate per SOURCE (2-step regrade) ===")
for s,(g,n) in bs.items(): print(f"  {s:9}: {g}/{n} = {g/max(n,1):.0%}")
print("=== Dolly per CATEGORY ===")
for c,(g,n) in sorted(bc.items()): print(f"  {c:22}: {g}/{n} = {g/max(n,1):.0%}")
print(f"t={time.time()-t0:.0f}s total={len(rows)}")
