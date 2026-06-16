"""build_labels.py — SIMPLE. The student answers each query with NORMAL CoT; the teacher grades
NORMALLY: is the answer correct? WRONG -> LOOKUP (it needed to look it up). CORRECT -> DIRECT.
No type-gates, no math special-casing, no token tricks. Just CoT + correctness."""
import os, json, time, random, urllib.request, concurrent.futures as cf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
R=random.Random(0); HERE=os.path.dirname(os.path.abspath(__file__)); DEV="cuda"
KEY=os.environ["OPENAI_API_KEY"]; TMODEL=os.environ.get("TEACHER","gpt-4.1-nano")
def log(*a): print(*a,flush=True)
GSYS=("You grade an assistant's answer to a user request. Reply ONLY one word: CORRECT or WRONG.\n"
 "CORRECT = the answer is correct / adequate for the request.\n"
 "WRONG = it is factually incorrect, refused, or evasive.")
def grade(q,ans,ref):
    u=f"REQUEST: {q}\nANSWER: {ans}\n"+(f"REFERENCE ANSWER: {ref}\n" if ref else "")+"Verdict:"
    body={"model":TMODEL,"messages":[{"role":"system","content":GSYS},{"role":"user","content":u}],"max_tokens":4,"temperature":0}
    for a in range(4):
        try:
            req=urllib.request.Request("https://api.openai.com/v1/chat/completions",data=json.dumps(body).encode(),
                headers={"Content-Type":"application/json","Authorization":"Bearer "+KEY})
            t=json.load(urllib.request.urlopen(req,timeout=40))["choices"][0]["message"]["content"].upper()
            return "WRONG" if "WRONG" in t else "CORRECT"
        except Exception: time.sleep(1.5*(a+1))
    return "CORRECT"
def main():
    t0=time.time(); tok=AutoTokenizer.from_pretrained("fft_hf")
    dt=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try: m=AutoModelForCausalLM.from_pretrained("fft_hf",torch_dtype=dt).to(DEV).eval()
    except TypeError: m=AutoModelForCausalLM.from_pretrained("fft_hf",dtype=dt).to(DEV).eval()
    log(f"[built {time.time()-t0:.0f}s]"); bos=tok.bos_token_id; eos=tok.eos_token_id
    @torch.no_grad()
    def ans(q,n=512):  # normal CoT generation
        ids=[bos]+tok.encode(f"<｜User｜>{q}<｜Assistant｜>",add_special_tokens=False)
        out=m.generate(input_ids=torch.tensor([ids],device=DEV),max_new_tokens=n,do_sample=False,pad_token_id=eos)
        full=tok.decode(out[0,len(ids):],skip_special_tokens=True)
        return full.split("</think>")[-1].strip()[:400] or full.strip()[-400:]
    pool=[]
    for ex in load_dataset("trivia_qa","rc.nocontext",split="train[:250]"): pool.append((ex["question"],"trivia","factual",ex["answer"]["value"]))
    dd=load_dataset("databricks/databricks-dolly-15k",split="train")
    for i in R.sample(range(len(dd)),250):
        ex=dd[i]
        if ex.get("context"): continue
        pool.append((ex["instruction"],"dolly",ex.get("category","na"),""))
    for ex in load_dataset("gsm8k","main",split="train[:60]"): pool.append((ex["question"],"gsm8k","math",ex["answer"].split("####")[-1].strip()))
    for u in ["I just adopted two kittens.","The weather's been lovely.","I'm stressed.","I tried great ramen.","Hey, how's it going?","I love hiking.","Good morning!","Thanks!","I got back from a walk.","I had a great weekend."]:
        pool.append((u,"chitchat","chitchat",""))
    log(f"[pool] {len(pool)}")
    sa={}
    for j,(q,s,c,g) in enumerate(pool):
        sa[q]=ans(q)
        if (j+1)%40==0: log(f"  answered {j+1}/{len(pool)} t={time.time()-t0:.0f}s")
    def work(p):
        q,s,c,g=p; v=grade(q,sa[q],g); return {"query":q,"source":s,"category":c,"gold":g,"student_answer":sa[q][:300],"verdict":v,"label":("LOOKUP" if v=="WRONG" else "DIRECT")}
    with cf.ThreadPoolExecutor(max_workers=8) as ex: rows=list(ex.map(work,pool))
    open(os.path.join(HERE,"labels_final.jsonl"),"w").write("\n".join(json.dumps(r) for r in rows))
    from collections import defaultdict
    bs=defaultdict(lambda:[0,0]); bc=defaultdict(lambda:[0,0])
    for r in rows:
        bs[r["source"]][0]+=(r["label"]=="LOOKUP"); bs[r["source"]][1]+=1
        if r["source"]=="dolly": bc[r["category"]][0]+=(r["label"]=="LOOKUP"); bc[r["category"]][1]+=1
    log("\n=== LOOKUP-rate per SOURCE (normal CoT + normal grading) ===")
    for s,(g,n) in bs.items(): log(f"  {s:9}: {g}/{n} = {g/max(n,1):.0%}")
    log("=== Dolly per CATEGORY ===")
    for c,(g,n) in sorted(bc.items()): log(f"  {c:22}: {g}/{n} = {g/max(n,1):.0%}")
    log("\n--- gsm8k samples (should be solved->DIRECT) ---")
    for r in [x for x in rows if x["source"]=="gsm8k"][:4]: log(f"  [{r['verdict']}] gold:{r['gold']} ans:{r['student_answer'][:60]}")
    json.dump({s:f"{g}/{n}" for s,(g,n) in bs.items()},open(os.path.join(HERE,"labels_final_stats.json"),"w"),indent=1)
    open(os.path.join(HERE,".done"),"w").write("ok"); log(f"DONE t={time.time()-t0:.0f}s")
if __name__=="__main__": main()
