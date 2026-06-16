"""recall_tag_ab.py — does WRAPPING the recalled block in a tag help the model USE it?
Needle test: plant a value, bury it past the raw window (rw), recall it during generation,
inject RAW (verbatim ids) vs TAGGED (natural-language wrapper). Metric = gold value appears in
the answer (operand_recovered). Training-free raw-QK retrieval (block clearly matches the needle).
"""
import os, json, time, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from attn_export3_torch import load_pooler
from block_recall import BlockArchive
DEV="cuda" if torch.cuda.is_available() else "cpu"; HERE=os.path.dirname(os.path.abspath(__file__))
RW=int(os.environ.get("RW","256")); CAP=int(os.environ.get("CAP","260")); C=64
def log(*a): print(*a,flush=True)
FACTS=[("vault code","7382"),("flight number","JL412"),("locker number","514"),("account pin","6207"),
 ("reference code","ZX9043"),("server name","atlas03"),("budget cap","47000 dollars"),("meeting room","C12")]
CASUAL=["How's your day going so far?","I had great ramen for lunch.","The weather's been grey all week.",
 "I started learning the guitar.","Traffic this morning was awful.","I tried a new coffee place downtown.",
 "I've been listening to jazz lately.","The sunset yesterday was unreal.","My neighbor got a puppy.","I might take a trip."]
EOS="<｜end▁of▁sentence｜>"
def build():
    tok=AutoTokenizer.from_pretrained("fft_hf")
    dt=torch.bfloat16 if (DEV=="cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    try: llm=AutoModelForCausalLM.from_pretrained("fft_hf",torch_dtype=dt,attn_implementation="eager")
    except TypeError: llm=AutoModelForCausalLM.from_pretrained("fft_hf",dtype=dt,attn_implementation="eager")
    return tok, llm.eval().to(DEV)
def scenario_ids(tok,kw,val,seed):
    r=random.Random(seed); turns=[(f"Please note: the {kw} is {val}.","Got it, noted.")]
    fills=r.sample(CASUAL,len(CASUAL)); i=0
    while sum(len(u+a) for u,a in turns)//4 < RW+400: turns.append((fills[i%len(fills)],"Sounds good!")); i+=1
    ids=[tok.bos_token_id]
    for u,a in turns: ids+=tok.encode((EOS if len(ids)>1 else "")+f"<｜User｜>{u}<｜Assistant｜>{a}",add_special_tokens=False)
    q=f"What is the {kw}?"
    ids+=tok.encode(EOS+f"<｜User｜>{q}<｜Assistant｜>",add_special_tokens=False)+tok.encode("<think>\n\n</think>\n\n",add_special_tokens=False)
    return ids, tok.encode(q,add_special_tokens=False)
@torch.no_grad()
def run(tok,llm,kw,val,seed,arm):
    try: from transformers import DynamicCache
    except Exception: from transformers.cache_utils import DynamicCache
    embT=llm.get_input_embeddings(); pooler=load_pooler("fft_out/pooler.pt").to(DEV).eval()
    feed,qids=scenario_ids(tok,kw,val,seed)
    arch=BlockArchive(llm,tok,mode="qk"); gen=list(feed); kept=[]; absorbed=0; new=0; done=False; rec_emb=None
    def emb(ids): return embT(torch.tensor([ids],device=DEV)) if ids else torch.zeros(1,0,embT.weight.shape[1],dtype=embT.weight.dtype,device=DEV)
    cache=DynamicCache(); llm(input_ids=torch.tensor([[tok.bos_token_id]],device=DEV),past_key_values=cache,use_cache=True); MQ=cache.get_seq_length()
    while not done:
        c0=len(gen); R=min(c0,RW); nd=c0-R
        if nd>absorbed: arch.extend(gen[absorbed:nd]); kept.extend(gen[absorbed:nd]); absorbed=nd
        if rec_emb is None and (arch.blocks or len(arch.buf)>=16):
            rid=arch.retrieve(qids,k=1)
            if rid:
                if arm=="TAGGED": rid=tok.encode('(Recalled note from earlier in this conversation: "',add_special_tokens=False)+rid+tok.encode('")',add_special_tokens=False)
                rec_emb=emb(rid)
        parts=[]
        if kept: parts.append(pooler(emb(kept).float()).to(embT.weight.dtype))
        if rec_emb is not None: parts.append(rec_emb.to(embT.weight.dtype))
        if R>0: parts.append(emb(gen[c0-R:c0]))
        cache.crop(MQ)
        last=llm(inputs_embeds=torch.cat(parts,1),past_key_values=cache,use_cache=True).logits[:,-1,:].float()
        for _ in range(C):
            t=int(last[0].argmax())
            if t==tok.eos_token_id or new>=CAP: done=True; break
            new+=1; gen.append(t)
            last=llm(inputs_embeds=emb([t]),past_key_values=cache,use_cache=True).logits[:,-1,:].float()
    ans=tok.decode(gen[len(feed):],skip_special_tokens=True)
    return val.lower() in ans.lower(), ans[:80]
def main():
    t0=time.time(); tok,llm=build(); log(f"[built {time.time()-t0:.0f}s] RW={RW}")
    res={"RAW":[], "TAGGED":[]}; LOG=[]
    def out(x): print(x,flush=True); LOG.append(x); open(os.path.join(HERE,"recall_tag_ab.log"),"w").write("\n".join(LOG))
    for si,(kw,val) in enumerate(FACTS):
        for arm in ["RAW","TAGGED"]:
            hit,ans=run(tok,llm,kw,val,si,arm); res[arm].append(hit)
            out(f"[{arm:6}] {kw}={val} recovered={int(hit)} | {ans[:60]}")
    out(f"\n=== operand_recovered ===  RAW {sum(res['RAW'])}/{len(res['RAW'])}  |  TAGGED {sum(res['TAGGED'])}/{len(res['TAGGED'])}")
    json.dump({"RAW":f"{sum(res['RAW'])}/{len(res['RAW'])}","TAGGED":f"{sum(res['TAGGED'])}/{len(res['TAGGED'])}"},open(os.path.join(HERE,"recall_tag_ab.json"),"w"))
    open(os.path.join(HERE,".done"),"w").write("ok"); out(f"DONE t={time.time()-t0:.0f}s")
if __name__=="__main__": main()
