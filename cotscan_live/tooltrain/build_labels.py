"""build_labels.py (GPU pod) — the user's design: the 4-bit full-KV STUDENT answers each query,
an API teacher GRADES it. Wrong factual (search would fix) -> LOOKUP; correct / non-factual -> DIRECT.
Labels reflect what the DEPLOYED (4-bit) model actually knows. Outputs labels_4bit.jsonl + stats."""
import os, json, time, random, re, urllib.request, concurrent.futures as cf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
R=random.Random(0); HERE=os.path.dirname(os.path.abspath(__file__)); DEV="cuda"
KEY=os.environ["OPENAI_API_KEY"]; TMODEL=os.environ.get("TEACHER","gpt-4.1-nano")
def log(*a): print(*a,flush=True)
NONFACT={"creative_writing","brainstorming","classification","summarization","math","chitchat"}
GSYS=("You GRADE a small assistant's answer. Given QUESTION, the assistant's ANSWER, and (if available) the REFERENCE. "
 "Reply ONLY one word:\n"
 "LOOKUP = the answer is factually wrong, evasive, or uncertain AND a web/Wikipedia search would supply the correct fact.\n"
 "DIRECT = the answer is correct, OR the task is creative/opinion/reasoning/chat where search would not help.")
def grade(q,ans,ref):
    u=f"QUESTION: {q}\nASSISTANT ANSWER: {ans}\n"+(f"REFERENCE: {ref}\n" if ref else "")+"Verdict:"
    body={"model":TMODEL,"messages":[{"role":"system","content":GSYS},{"role":"user","content":u}],"max_tokens":4,"temperature":0}
    for a in range(4):
        try:
            req=urllib.request.Request("https://api.openai.com/v1/chat/completions",data=json.dumps(body).encode(),
                headers={"Content-Type":"application/json","Authorization":"Bearer "+KEY})
            t=json.load(urllib.request.urlopen(req,timeout=40))["choices"][0]["message"]["content"].upper()
            return "LOOKUP" if "LOOKUP" in t else "DIRECT"
        except Exception: time.sleep(1.5*(a+1))
    return "DIRECT"
def main():
    t0=time.time(); tok=AutoTokenizer.from_pretrained("fft_hf")
    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16)
    m=AutoModelForCausalLM.from_pretrained("fft_hf",quantization_config=bnb,device_map="auto").eval()
    log(f"[built 4bit {time.time()-t0:.0f}s]"); bos=tok.bos_token_id; eos=tok.eos_token_id
    @torch.no_grad()
    def ans(q,n=64):
        ids=[bos]+tok.encode(f"<｜User｜>{q}<｜Assistant｜>",add_special_tokens=False)+tok.encode("<think>\n\n</think>\n\n",add_special_tokens=False)
        out=m.generate(input_ids=torch.tensor([ids],device=DEV),max_new_tokens=n,do_sample=False,pad_token_id=eos)
        return tok.decode(out[0,len(ids):],skip_special_tokens=True).strip()
    # pool
    pool=[]
    for ex in load_dataset("trivia_qa","rc.nocontext",split="train[:300]"): pool.append((ex["question"],"trivia","factual",ex["answer"]["value"]))
    dd=load_dataset("databricks/databricks-dolly-15k",split="train")
    for i in R.sample(range(len(dd)),300):
        ex=dd[i]
        if ex.get("context"): continue
        pool.append((ex["instruction"],"dolly",ex.get("category","na"),""))
    for ex in load_dataset("gsm8k","main",split="train[:60]"): pool.append((ex["question"],"gsm8k","math",""))
    for u in ["I just adopted two kittens.","The weather's been lovely.","I'm stressed.","I tried great ramen.","Hey, how's it going?","I love hiking.","Good morning!","Thanks!","I got back from a walk.","I had a great weekend."]:
        pool.append((u,"chitchat","chitchat",""))
    log(f"[pool] {len(pool)}")
    # student answers ALL queries; teacher grades ALL (no category fiat).
    sa={}
    for j,(q,s,c,g) in enumerate(pool):
        sa[q]=ans(q)
        if (j+1)%50==0: log(f"  student answered {j+1}/{len(pool)} t={time.time()-t0:.0f}s")
    def work(p):
        q,s,c,g=p; return {"query":q,"source":s,"category":c,"gold":g,"student_answer":sa[q][:200],"label":grade(q,sa[q],g)}
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        rows=list(ex.map(work,pool))
    open(os.path.join(HERE,"labels_4bit.jsonl"),"w").write("\n".join(json.dumps(r) for r in rows))
    from collections import defaultdict
    bs=defaultdict(lambda:[0,0])
    for r in rows: bs[r["source"]][0]+=(r["label"]=="LOOKUP"); bs[r["source"]][1]+=1
    log("\n=== LOOKUP-rate per SOURCE (4bit-student graded) ===")
    for s,(g,n) in bs.items(): log(f"  {s:9}: LOOKUP {g}/{n} = {g/max(n,1):.0%}")
    log("\n--- sample LOOKUP (student got it wrong) ---")
    for r in [x for x in rows if x["label"]=="LOOKUP"][:5]: log(f"  Q:{r['query'][:46]} | gold:{r['gold'][:24]} | student:{r['student_answer'][:40]}")
    log("--- sample DIRECT (student correct) ---")
    for r in [x for x in rows if x["label"]=="DIRECT" and x["source"]=="trivia"][:5]: log(f"  Q:{r['query'][:46]} | gold:{r['gold'][:24]} | student:{r['student_answer'][:40]}")
    json.dump({s:f"{g}/{n}" for s,(g,n) in bs.items()},open(os.path.join(HERE,"labels_4bit_stats.json"),"w"),indent=1)
    open(os.path.join(HERE,".done"),"w").write("ok"); log(f"DONE t={time.time()-t0:.0f}s")
if __name__=="__main__": main()
