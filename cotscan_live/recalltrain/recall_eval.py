"""recall_eval.py — needle operand_recovered: BASE vs recall-trained adapter (RAW recall injection).
If recall-aware SFT worked, the trained model should USE the raw recalled block better than base."""
import os, json, time, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from attn_export3_torch import load_pooler
from block_recall import BlockArchive
DEV="cuda" if torch.cuda.is_available() else "cpu"; HERE=os.path.dirname(os.path.abspath(__file__))
RW=int(os.environ.get("RW","256")); CAP=260; C=64
FACTS=[("vault code","7382"),("flight number","JL412"),("locker number","514"),("account pin","6207"),
 ("reference code","ZX9043"),("server name","atlas03"),("budget cap","47000 dollars"),("meeting room","C12"),
 ("project codename","Bluefin"),("hotel confirmation","88231")]
CASUAL=["How's your day?","Had great ramen.","Grey weather all week.","Learning guitar.","Traffic was awful.",
 "New coffee place downtown.","Listening to jazz.","Sunset was unreal.","Neighbor got a puppy.","Might take a trip."]
EOS="<｜end▁of▁sentence｜>"
def log(*a): print(*a,flush=True)
def build():
    tok=AutoTokenizer.from_pretrained("fft_hf")
    dt=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try: base=AutoModelForCausalLM.from_pretrained("fft_hf",torch_dtype=dt,attn_implementation="eager")
    except TypeError: base=AutoModelForCausalLM.from_pretrained("fft_hf",dtype=dt,attn_implementation="eager")
    base=base.to(DEV).eval(); m=PeftModel.from_pretrained(base,os.path.join(HERE,"recall_lora")).eval()
    pooler=load_pooler("fft_out/pooler.pt").to(DEV).eval()
    return tok,m,pooler
def scen(tok,kw,val,seed):
    r=random.Random(seed); turns=[(f"Please note: the {kw} is {val}.","Got it, noted.")]
    fills=r.sample(CASUAL,len(CASUAL)); i=0
    while sum(len(u+a) for u,a in turns)//4 < RW+400: turns.append((fills[i%len(fills)],"Sounds good!")); i+=1
    ids=[tok.bos_token_id]
    for u,a in turns: ids+=tok.encode((EOS if len(ids)>1 else "")+f"<｜User｜>{u}<｜Assistant｜>{a}",add_special_tokens=False)
    q=f"What is the {kw}?"
    ids+=tok.encode(EOS+f"<｜User｜>{q}<｜Assistant｜>",add_special_tokens=False)+tok.encode("<think>\n\n</think>\n\n",add_special_tokens=False)
    return ids,tok.encode(q,add_special_tokens=False)
@torch.no_grad()
def run(tok,m,pooler,kw,val,seed,use_adapter):
    try: from transformers import DynamicCache
    except Exception: from transformers.cache_utils import DynamicCache
    embT=m.get_input_embeddings(); feed,qids=scen(tok,kw,val,seed)
    arch=BlockArchive(m.base_model.model if use_adapter else m, tok, mode="qk")  # raw-QK on the LLM
    gen=list(feed); kept=[]; absorbed=0; new=0; done=False; rec=None
    def emb(ids): return embT(torch.tensor([ids],device=DEV)) if ids else torch.zeros(1,0,embT.weight.shape[1],dtype=embT.weight.dtype,device=DEV)
    cache=DynamicCache(); 
    ctxmgr = (lambda: m.disable_adapter()) if not use_adapter else None
    def fwd_logits_lasttok(inp):
        if use_adapter: return m(inputs_embeds=inp,past_key_values=cache,use_cache=True).logits[:,-1,:].float()
        with m.disable_adapter(): return m(inputs_embeds=inp,past_key_values=cache,use_cache=True).logits[:,-1,:].float()
    # prime BOS
    if use_adapter: m(input_ids=torch.tensor([[tok.bos_token_id]],device=DEV),past_key_values=cache,use_cache=True)
    else:
        with m.disable_adapter(): m(input_ids=torch.tensor([[tok.bos_token_id]],device=DEV),past_key_values=cache,use_cache=True)
    MQ=cache.get_seq_length()
    while not done:
        c0=len(gen); R=min(c0,RW); nd=c0-R
        if nd>absorbed: arch.extend(gen[absorbed:nd]); kept.extend(gen[absorbed:nd]); absorbed=nd
        if rec is None and (arch.blocks or len(arch.buf)>=16):
            rid=arch.retrieve(qids,k=1)
            if rid: rec=emb(rid)
        parts=[]
        if kept: parts.append(pooler(emb(kept).float()).to(embT.weight.dtype))
        if rec is not None: parts.append(rec.to(embT.weight.dtype))
        if R>0: parts.append(emb(gen[c0-R:c0]))
        cache.crop(MQ)
        last=fwd_logits_lasttok(torch.cat(parts,1))
        for _ in range(C):
            t=int(last[0].argmax())
            if t==tok.eos_token_id or new>=CAP: done=True; break
            new+=1; gen.append(t); last=fwd_logits_lasttok(emb([t]))
        if done: break
    ans=tok.decode(gen[len(feed):],skip_special_tokens=True)
    return val.lower() in ans.lower(), ans[:70]
def main():
    t0=time.time(); tok,m,pooler=build(); log(f"[built {time.time()-t0:.0f}s]")
    res={"BASE":[], "RECALL":[]}; LOG=[]
    def out(x): print(x,flush=True); LOG.append(x); open(os.path.join(HERE,"recall_eval.log"),"w").write("\n".join(LOG))
    for si,(kw,val) in enumerate(FACTS):
        b,ba=run(tok,m,pooler,kw,val,si,False); res["BASE"].append(b)
        r,ra=run(tok,m,pooler,kw,val,si,True); res["RECALL"].append(r)
        out(f"[{kw:18}={val:14}] base={int(b)} recall={int(r)} | base:{ba[:34]} || recall:{ra[:34]}")
    out(f"\n=== operand_recovered === BASE {sum(res['BASE'])}/{len(res['BASE'])} | RECALL-TRAINED {sum(res['RECALL'])}/{len(res['RECALL'])}")
    json.dump({"base":f"{sum(res['BASE'])}/{len(res['BASE'])}","recall":f"{sum(res['RECALL'])}/{len(res['RECALL'])}"},open(os.path.join(HERE,"recall_eval.json"),"w"))
    open(os.path.join(HERE,".evaldone"),"w").write("ok"); out(f"EVAL_DONE t={time.time()-t0:.0f}s")
if __name__=="__main__": main()
