"""sft_train.py (Phase 2, GPU) — LoRA cold-start SFT to teach the SEARCH-TIMING decision.
Trains on call_sft_v2.jsonl (prompt+completion, loss on completion only): gap -> emit
<search>QUERY</search>; known/personal/chitchat -> no <search>. bf16 + LoRA. Then evaluates
on a HELD set: does it emit <search> on gap turns and NOT on known/chitchat?
"""
import os, json, time, re, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
DEV="cuda" if torch.cuda.is_available() else "cpu"; HERE=os.path.dirname(os.path.abspath(__file__))
def log(*a): print(*a,flush=True)

HELD_GAP=["Who won the FIFA World Cup in 1998?","What is the population of Reykjavik?",
  "When was the Brooklyn Bridge opened?","Who painted Guernica and in what year?",
  "What is the height of Mount Kilimanjaro?","Which company manufactured the first mass-produced car?"]
HELD_KNOWN=["What is the capital of France?","What is 2 plus 2?","Who wrote Hamlet?",
  "What is the chemical symbol for oxygen?","Which planet is closest to the sun?"]
HELD_CHIT=["I had a really great weekend.","How are you doing today?","I just made some coffee.","Thanks for the help!"]

def main():
    t0=time.time()
    tok=AutoTokenizer.from_pretrained("fft_hf")
    dt=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try: m=AutoModelForCausalLM.from_pretrained("fft_hf",torch_dtype=dt)
    except TypeError: m=AutoModelForCausalLM.from_pretrained("fft_hf",dtype=dt)
    m=m.to(DEV); m.config.use_cache=False
    lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    m=get_peft_model(m,lora); m.print_trainable_parameters()
    bos=tok.bos_token_id; eos=tok.eos_token_id
    rows=[json.loads(l) for l in open("call_sft_v2.jsonl") if l.strip()]
    log(f"[data] {len(rows)} traces")
    def enc(r):
        p=[bos]+tok.encode(r["prompt"],add_special_tokens=False)
        c=tok.encode(r["completion"],add_special_tokens=False)+[eos]
        ids=p+c; lab=[-100]*len(p)+c
        return ids[:1024], lab[:1024]
    data=[enc(r) for r in rows]
    opt=torch.optim.AdamW([q for q in m.parameters() if q.requires_grad],lr=1e-4)
    EP=int(os.environ.get("EPOCHS","3"))
    m.train()
    for ep in range(EP):
        random.Random(ep).shuffle(data); tot=0.0
        for i,(ids,lab) in enumerate(data):
            x=torch.tensor([ids],device=DEV); y=torch.tensor([lab],device=DEV)
            out=m(input_ids=x,labels=y); loss=out.loss
            loss.backward(); torch.nn.utils.clip_grad_norm_([q for q in m.parameters() if q.requires_grad],1.0)
            opt.step(); opt.zero_grad(); tot+=loss.item()
        log(f"[ep {ep+1}/{EP}] loss={tot/len(data):.4f} t={time.time()-t0:.0f}s")
    m.save_pretrained(os.path.join(HERE,"lora_adapter"))
    log("[saved adapter]")

    # ---- eval: search-timing fire rates ----
    m.eval(); m.config.use_cache=True
    @torch.no_grad()
    def fires(q):
        ids=[bos]+tok.encode(f"<｜User｜>{q}<｜Assistant｜>",add_special_tokens=False)
        out=m.generate(input_ids=torch.tensor([ids],device=DEV),max_new_tokens=64,do_sample=False,pad_token_id=eos)
        txt=tok.decode(out[0,len(ids):],skip_special_tokens=False)
        return ("<search>" in txt), txt
    res={"gap":[],"known":[],"chitchat":[]}
    for q in HELD_GAP: f,t=fires(q); res["gap"].append(f); log(f"[gap   ] fire={int(f)} | {q[:40]} -> {t[:80]}")
    for q in HELD_KNOWN: f,t=fires(q); res["known"].append(f); log(f"[known ] fire={int(f)} | {q[:40]} -> {t[:80]}")
    for q in HELD_CHIT: f,t=fires(q); res["chitchat"].append(f); log(f"[chit  ] fire={int(f)} | {q[:40]} -> {t[:60]}")
    summ={"gap_fire":f"{sum(res['gap'])}/{len(res['gap'])}","known_fire":f"{sum(res['known'])}/{len(res['known'])}",
          "chitchat_fire":f"{sum(res['chitchat'])}/{len(res['chitchat'])}"}
    json.dump(summ,open(os.path.join(HERE,"sft_eval.json"),"w"),indent=1)
    log("\n=== SFT EVAL (search-timing) ===")
    log(f"  gap_fire (want high): {summ['gap_fire']}")
    log(f"  known_fire (want 0):  {summ['known_fire']}")
    log(f"  chitchat_fire(want 0):{summ['chitchat_fire']}")
    open(os.path.join(HERE,".done"),"w").write("ok"); log(f"DONE t={time.time()-t0:.0f}s")

if __name__=="__main__": main()
